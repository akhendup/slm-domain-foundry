#!/usr/bin/env python3
"""
Learning agent with canary testing and circuit breaker (Priority 7).

Implements loom's learning agent safety architecture for autonomous
pattern improvement:

  CircuitBreaker  — prevents cascading failures from bad pattern changes.
                    States: CLOSED (normal) → OPEN (blocked) → HALF_OPEN
                    (probing).  Opens after N consecutive failures; requires
                    M successes to re-close.

  CanaryTest      — safely rolls out a pattern change to a small traffic
                    fraction (default 10 %) and uses a two-sample
                    Welch t-test to validate improvement before promotion.
                    Falls back to a proportion-based comparison when the
                    sample count is too small for a t-test.

  PatternMetrics  — tracks per-pattern performance (success rate, cost,
                    latency) accumulated from runtime observations.

  LearningAgent   — orchestrates: collect metrics → score patterns →
                    propose changes → canary test → apply or rollback.
                    Three autonomy levels: MANUAL, HUMAN_APPROVAL, FULL.

No external LLMs or services required.  scipy is optional; when absent
the t-test falls back to a proportion difference check.

Usage
-----
    from data.learning_agent import LearningAgent, AutonomyLevel

    agent = LearningAgent(autonomy=AutonomyLevel.FULL)
    agent.record("pattern_a", success=True,  cost_tokens=120, latency_ms=200)
    agent.record("pattern_a", success=False, cost_tokens=80,  latency_ms=500)

    report = agent.run_cycle()
    print(report["recommendations"])   # PROMOTE / DEMOTE / INVESTIGATE
    print(report["canary_results"])     # canary test outcomes
    print(report["applied"])            # changes applied in FULL mode
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Try to import scipy for t-test
# ---------------------------------------------------------------------------

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Autonomy levels
# ---------------------------------------------------------------------------

class AutonomyLevel(str, Enum):
    MANUAL          = "manual"          # observe only; never apply
    HUMAN_APPROVAL  = "human_approval"  # propose; block until approved
    FULL            = "full"            # apply automatically (with canary)


# ---------------------------------------------------------------------------
# Recommendation types
# ---------------------------------------------------------------------------

class Recommendation(str, Enum):
    KEEP        = "KEEP"
    PROMOTE     = "PROMOTE"
    DEMOTE      = "DEMOTE"
    REMOVE      = "REMOVE"
    INVESTIGATE = "INVESTIGATE"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED    = "CLOSED"     # normal operation
    OPEN      = "OPEN"       # blocked — too many failures
    HALF_OPEN = "HALF_OPEN"  # probing after cooldown


@dataclass
class CircuitBreaker:
    """
    Prevent cascading failures from repeated pattern failures.

    Parameters
    ----------
    failure_threshold  : consecutive failures needed to OPEN (default 5)
    success_threshold  : consecutive successes needed to re-CLOSE (default 3)
    cooldown_seconds   : time before OPEN → HALF_OPEN (default 1 800 s = 30 min)
    """
    failure_threshold: int   = 5
    success_threshold: int   = 3
    cooldown_seconds:  float = 1_800.0

    _state:             CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consec_failures:   int          = field(default=0, init=False)
    _consec_successes:  int          = field(default=0, init=False)
    _opened_at:         float        = field(default=0.0, init=False)
    _history:           List[str]    = field(default_factory=list, init=False)

    @property
    def state(self) -> CircuitState:
        # Auto-transition OPEN → HALF_OPEN after cooldown
        if (self._state == CircuitState.OPEN and
                time.time() - self._opened_at >= self.cooldown_seconds):
            self._state            = CircuitState.HALF_OPEN
            self._consec_successes = 0
            self._history.append("OPEN→HALF_OPEN (cooldown elapsed)")
        return self._state

    def allow(self) -> bool:
        """Return True if a request/change should be allowed through."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful execution."""
        self._consec_failures = 0
        if self._state == CircuitState.HALF_OPEN:
            self._consec_successes += 1
            if self._consec_successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._history.append("HALF_OPEN→CLOSED (success threshold met)")
        elif self._state == CircuitState.CLOSED:
            self._consec_successes += 1

    def record_failure(self) -> None:
        """Record a failed execution."""
        self._consec_successes = 0
        self._consec_failures  += 1
        if (self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN) and
                self._consec_failures >= self.failure_threshold):
            self._state    = CircuitState.OPEN
            self._opened_at = time.time()
            self._history.append(
                f"→OPEN ({self._consec_failures} consecutive failures)"
            )

    def reset(self) -> None:
        """Force-reset to CLOSED (for testing)."""
        self._state            = CircuitState.CLOSED
        self._consec_failures  = 0
        self._consec_successes = 0
        self._opened_at        = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "state":            self.state.value,
            "consec_failures":  self._consec_failures,
            "consec_successes": self._consec_successes,
            "history":          list(self._history[-10:]),
        }


# ---------------------------------------------------------------------------
# Pattern metrics
# ---------------------------------------------------------------------------

@dataclass
class PatternMetrics:
    """Accumulated runtime metrics for one pattern."""
    name:         str
    success_count: int   = 0
    failure_count: int   = 0
    total_cost:   float  = 0.0   # token cost
    total_latency: float = 0.0   # ms
    _cost_samples:   List[float] = field(default_factory=list)
    _latency_samples: List[float] = field(default_factory=list)

    @property
    def total_uses(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        n = self.total_uses
        return self.success_count / n if n else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / max(self.total_uses, 1)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.total_uses, 1)

    def record(self, success: bool, cost_tokens: float = 0.0, latency_ms: float = 0.0) -> None:
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.total_cost    += cost_tokens
        self.total_latency += latency_ms
        self._cost_samples.append(cost_tokens)
        self._latency_samples.append(latency_ms)

    def confidence(self) -> float:
        """
        Sigmoid-based confidence in this metric's reliability.
        Low at < 10 uses (0.3), saturates at 50 uses (0.9).
        """
        n = self.total_uses
        if n == 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-0.15 * (n - 20)))

    def recommend(self) -> Recommendation:
        """Derive a recommendation from this pattern's metrics."""
        conf = self.confidence()
        sr   = self.success_rate

        if conf < 0.3:
            return Recommendation.INVESTIGATE   # not enough data

        if sr >= 0.85 and conf >= 0.6:
            return Recommendation.PROMOTE
        if sr >= 0.50:
            return Recommendation.KEEP
        if sr >= 0.20:
            return Recommendation.DEMOTE
        return Recommendation.REMOVE


# ---------------------------------------------------------------------------
# Canary test
# ---------------------------------------------------------------------------

@dataclass
class CanaryConfig:
    traffic_fraction:   float = 0.10    # 10 % of calls go to treatment
    min_samples:        int   = 10      # minimum per arm before evaluation
    alpha:              float = 0.05    # significance level for t-test


@dataclass
class CanaryResult:
    pattern_name:     str
    control_sr:       float
    treatment_sr:     float
    p_value:          float
    significant:      bool
    recommendation:   str    # "promote" | "rollback" | "inconclusive"
    n_control:        int
    n_treatment:      int


class CanaryTest:
    """
    A/B test for pattern changes.

    Observations are randomly assigned to CONTROL (old pattern) or
    TREATMENT (new pattern) based on `traffic_fraction`.  After
    `min_samples` per arm, runs a Welch t-test (or proportion comparison)
    to determine whether the treatment is significantly better.
    """

    def __init__(self, pattern_name: str, config: Optional[CanaryConfig] = None) -> None:
        self.pattern_name      = pattern_name
        self._cfg              = config or CanaryConfig()
        self._control:   List[float] = []    # 1.0 = success, 0.0 = failure
        self._treatment: List[float] = []
        self.active            = True

    def observe(self, success: bool) -> str:
        """
        Route an observation to control or treatment.
        Returns "control" | "treatment".
        """
        arm = "treatment" if random.random() < self._cfg.traffic_fraction else "control"
        val = 1.0 if success else 0.0
        if arm == "control":
            self._control.append(val)
        else:
            self._treatment.append(val)
        return arm

    def ready(self) -> bool:
        """True when both arms have at least min_samples observations."""
        return (len(self._control)   >= self._cfg.min_samples and
                len(self._treatment) >= self._cfg.min_samples)

    def evaluate(self) -> CanaryResult:
        """
        Evaluate the canary.  Returns a CanaryResult.

        Uses Welch t-test when scipy is available; otherwise falls back
        to a simple proportion difference check.
        """
        ctrl_sr  = sum(self._control)   / len(self._control)   if self._control  else 0.0
        treat_sr = sum(self._treatment) / len(self._treatment) if self._treatment else 0.0

        p_value = 1.0
        if _HAS_SCIPY and len(self._control) >= 2 and len(self._treatment) >= 2:
            _, p_value = _scipy_stats.ttest_ind(
                self._treatment, self._control, equal_var=False
            )
        else:
            # Proportion difference — significant if > 5 pp
            p_value = 0.04 if abs(treat_sr - ctrl_sr) > 0.05 else 0.5

        significant = p_value < self._cfg.alpha

        if significant and treat_sr > ctrl_sr:
            rec = "promote"
        elif significant and treat_sr <= ctrl_sr:
            rec = "rollback"
        else:
            rec = "inconclusive"

        return CanaryResult(
            pattern_name=self.pattern_name,
            control_sr=round(ctrl_sr, 4),
            treatment_sr=round(treat_sr, 4),
            p_value=round(p_value, 4),
            significant=significant,
            recommendation=rec,
            n_control=len(self._control),
            n_treatment=len(self._treatment),
        )


# ---------------------------------------------------------------------------
# Learning Agent
# ---------------------------------------------------------------------------

class LearningAgent:
    """
    Orchestrates pattern-level continuous improvement with safety guardrails.

    Parameters
    ----------
    autonomy         : AutonomyLevel (MANUAL | HUMAN_APPROVAL | FULL)
    circuit_breaker  : shared CircuitBreaker instance (or new one per agent)
    canary_config    : CanaryConfig for traffic splitting
    protected        : set of pattern names that must never be auto-modified
    max_daily_changes: limit on how many patterns can be changed per cycle
    """

    def __init__(
        self,
        autonomy:          AutonomyLevel               = AutonomyLevel.MANUAL,
        circuit_breaker:   Optional[CircuitBreaker]   = None,
        canary_config:     Optional[CanaryConfig]     = None,
        protected:         Optional[List[str]]        = None,
        max_daily_changes: int                         = 10,
    ) -> None:
        self.autonomy          = autonomy
        self._cb               = circuit_breaker or CircuitBreaker()
        self._canary_cfg       = canary_config or CanaryConfig()
        self._protected        = set(protected or [])
        self._max_changes      = max_daily_changes

        self._metrics:  Dict[str, PatternMetrics] = {}
        self._canaries: Dict[str, CanaryTest]     = {}
        self._history:  List[Dict[str, Any]]      = []
        self._pending_approvals: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def record(
        self,
        pattern_name: str,
        success:      bool,
        cost_tokens:  float = 0.0,
        latency_ms:   float = 0.0,
    ) -> None:
        """Record one execution of a pattern."""
        if pattern_name not in self._metrics:
            self._metrics[pattern_name] = PatternMetrics(name=pattern_name)
        self._metrics[pattern_name].record(success, cost_tokens, latency_ms)

        # Feed active canary for this pattern
        if pattern_name in self._canaries and self._canaries[pattern_name].active:
            self._canaries[pattern_name].observe(success)

        # Update circuit breaker
        if success:
            self._cb.record_success()
        else:
            self._cb.record_failure()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(self) -> Dict[str, Recommendation]:
        """Return per-pattern recommendations based on current metrics."""
        return {name: m.recommend() for name, m in self._metrics.items()}

    # ------------------------------------------------------------------
    # Canary management
    # ------------------------------------------------------------------

    def start_canary(self, pattern_name: str) -> bool:
        """
        Start a canary test for *pattern_name*.
        Returns False if the circuit is OPEN or pattern is protected.
        """
        if pattern_name in self._protected:
            return False
        if not self._cb.allow():
            return False
        self._canaries[pattern_name] = CanaryTest(pattern_name, self._canary_cfg)
        return True

    def evaluate_canaries(self) -> List[CanaryResult]:
        """Evaluate all ready canaries and return results."""
        results: List[CanaryResult] = []
        for name, ct in list(self._canaries.items()):
            if ct.ready():
                result = ct.evaluate()
                results.append(result)
                ct.active = False   # canary concluded
        return results

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one improvement cycle:
          1. Analyse pattern metrics.
          2. Evaluate any ready canaries.
          3. Propose changes (PROMOTE / DEMOTE / REMOVE patterns).
          4. Apply based on autonomy level.

        Returns a report dict.
        """
        recommendations  = self.analyse()
        canary_results   = self.evaluate_canaries()
        proposed_changes = self._build_proposals(recommendations, canary_results)
        applied          = self._apply(proposed_changes)

        report: Dict[str, Any] = {
            "recommendations":  {k: v.value for k, v in recommendations.items()},
            "canary_results":   [vars(r) for r in canary_results],
            "proposed_changes": proposed_changes,
            "applied":          applied,
            "circuit_breaker":  self._cb.summary(),
            "autonomy":         self.autonomy.value,
        }
        self._history.append(report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_proposals(
        self,
        recommendations: Dict[str, Recommendation],
        canary_results:  List[CanaryResult],
    ) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []

        for name, rec in recommendations.items():
            if name in self._protected:
                continue
            m   = self._metrics[name]
            if rec in (Recommendation.PROMOTE, Recommendation.DEMOTE, Recommendation.REMOVE):
                proposals.append({
                    "pattern":        name,
                    "action":         rec.value,
                    "success_rate":   round(m.success_rate, 4),
                    "confidence":     round(m.confidence(), 4),
                    "total_uses":     m.total_uses,
                    "source":         "metrics",
                })

        for cr in canary_results:
            if cr.recommendation == "promote":
                proposals.append({
                    "pattern":  cr.pattern_name,
                    "action":   "PROMOTE",
                    "source":   "canary",
                    "p_value":  cr.p_value,
                    "treatment_sr": cr.treatment_sr,
                    "control_sr":   cr.control_sr,
                })
            elif cr.recommendation == "rollback":
                proposals.append({
                    "pattern":  cr.pattern_name,
                    "action":   "ROLLBACK",
                    "source":   "canary",
                    "p_value":  cr.p_value,
                })

        return proposals[:self._max_changes]

    def _apply(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply proposals according to autonomy level.

        MANUAL          → nothing applied; everything logged as pending
        HUMAN_APPROVAL  → added to pending_approvals queue
        FULL            → applied if circuit allows; logged in history
        """
        applied: List[Dict[str, Any]] = []

        if self.autonomy == AutonomyLevel.MANUAL:
            return applied

        if self.autonomy == AutonomyLevel.HUMAN_APPROVAL:
            self._pending_approvals.extend(proposals)
            return applied

        # FULL autonomy
        if not self._cb.allow():
            return applied   # circuit open — don't apply anything

        for p in proposals:
            # Simulate applying — in production this would update YAML files
            p["applied_at"] = time.time()
            applied.append(p)

        return applied

    def approve(self, pattern_name: str) -> bool:
        """
        (HUMAN_APPROVAL mode) Approve a pending proposal for *pattern_name*.
        Returns True if found and moved to applied.
        """
        for i, p in enumerate(self._pending_approvals):
            if p.get("pattern") == pattern_name:
                p["applied_at"] = time.time()
                self._history.append({"approved": p})
                self._pending_approvals.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def pending_approvals(self) -> List[Dict[str, Any]]:
        return list(self._pending_approvals)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def pattern_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Return a summary dict for one pattern."""
        m = self._metrics.get(name)
        if not m:
            return None
        return {
            "name":        name,
            "total_uses":  m.total_uses,
            "success_rate": round(m.success_rate, 4),
            "avg_cost":    round(m.avg_cost, 2),
            "avg_latency": round(m.avg_latency, 2),
            "confidence":  round(m.confidence(), 4),
            "recommend":   m.recommend().value,
        }

    def all_summaries(self) -> List[Dict[str, Any]]:
        """Return summaries for all tracked patterns, sorted by success rate."""
        summaries = [self.pattern_summary(n) for n in self._metrics]
        summaries.sort(key=lambda x: x["success_rate"], reverse=True)
        return summaries
