"""
SwarmManager: load and manage a pool of SLMs; route queries to one or all models.

A swarm lets multiple fine-tuned (or pretrained) SLMs run simultaneously.
Queries can be sent to a single named model or fanned out to every loaded model
in parallel threads, returning a {name: response} dict.

Typical usage via the module-level singleton::

    from app.swarm import get_swarm
    swarm = get_swarm()
    swarm.load("analyst-v1", Path("saved_models/analyst-v1"))
    swarm.load("general-v2", Path("saved_models/general-v2"))
    results = swarm.generate_all([{"role": "user", "content": "What is RANK?"}])
    # results == {"analyst-v1": "...", "general-v2": "..."}
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.model_loader import generate_response, load_model


class SwarmManager:
    """Thread-safe pool of loaded SLMs.

    Each model is identified by a user-chosen *name* (e.g. ``"td-analyst-v1"``).
    Models can be loaded from any local directory that contains a trained
    HuggingFace model (fine-tuned or a plain pretrained checkpoint).

    The swarm exposes two inference modes:

    * ``generate_one`` — query a single named model.
    * ``generate_all`` — fan out to every loaded model in parallel and return all
      responses as a ``{name: answer}`` dict.  Models that fail return an
      ``"[Error: ...]"`` string so one bad model never blocks the others.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def load(self, name: str, model_dir: Path) -> str:
        """Load *model_dir* into the swarm under *name*.  Returns a status string."""
        name = name.strip()
        if not name:
            return "Model name must not be empty."
        model_dir = Path(model_dir)
        if not model_dir.exists():
            return f"Directory not found: {model_dir}"
        with self._lock:
            if name in self._models:
                return f"'{name}' is already loaded in the swarm. Unload it first to reload."
        try:
            model, tokenizer = load_model(model_dir)
            with self._lock:
                self._models[name] = (model, tokenizer)
            return f"Loaded '{name}' into swarm."
        except Exception as exc:
            return f"Failed to load '{name}': {exc}"

    def unload(self, name: str) -> str:
        """Remove *name* from the swarm and release its Python reference."""
        with self._lock:
            if name not in self._models:
                return f"'{name}' is not in the swarm."
            del self._models[name]
        return f"Unloaded '{name}' from swarm."

    def clear(self) -> int:
        """Unload all models.  Returns the number of models removed."""
        with self._lock:
            n = len(self._models)
            self._models.clear()
        return n

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def names(self) -> List[str]:
        """Alphabetically sorted list of currently loaded model names."""
        with self._lock:
            return sorted(self._models.keys())

    def is_loaded(self, name: str) -> bool:
        """Return True if *name* is currently in the swarm."""
        with self._lock:
            return name in self._models

    def size(self) -> int:
        """Return the number of models currently loaded."""
        with self._lock:
            return len(self._models)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_one(self, name: str, messages: List[dict], **kwargs: Any) -> str:
        """Generate a response from the named model.

        Returns an error string if *name* is not loaded.
        """
        with self._lock:
            pair = self._models.get(name)
        if pair is None:
            return f"['{name}' is not in the swarm]"
        model, tokenizer = pair
        return generate_response(model, tokenizer, messages, **kwargs)

    def generate_all(
        self,
        messages: List[dict],
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Fan out *messages* to every loaded model in parallel.

        Returns ``{name: response}`` for all models.  Models that raise an
        exception return ``"[Error: <message>]"`` so one failure never blocks
        the rest.  Models that do not respond within 120 seconds receive a
        timeout message.
        """
        with self._lock:
            pairs = dict(self._models)  # snapshot — don't hold lock during inference
        if not pairs:
            return {}

        results: Dict[str, str] = {}
        lock = threading.Lock()

        def _run(name: str, model: Any, tokenizer: Any) -> None:
            try:
                resp = generate_response(model, tokenizer, messages, **kwargs)
            except Exception as exc:
                resp = f"[Error: {exc}]"
            with lock:
                results[name] = resp

        threads = [
            threading.Thread(target=_run, args=(n, m, t), daemon=True)
            for n, (m, t) in pairs.items()
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=120)

        # Fill in any threads that timed out without producing a result
        for name in pairs:
            if name not in results:
                results[name] = "[Timeout: model did not respond within 120 s]"

        return results


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the Gradio UI process
# ---------------------------------------------------------------------------
_swarm: Optional[SwarmManager] = None


def get_swarm() -> SwarmManager:
    """Return the module-level SwarmManager singleton (created on first call)."""
    global _swarm
    if _swarm is None:
        _swarm = SwarmManager()
    return _swarm
