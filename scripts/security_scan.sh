#!/usr/bin/env bash
# Pre-release security checks for SLM Domain Foundry.
# Usage: ./scripts/security_scan.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== pip-audit (installed packages in requirements.txt) ==="
python -m pip install -q pip-audit safety 2>/dev/null || true
pip-audit -r requirements.txt || true

echo ""
echo "=== safety scan (requirements files) ==="
safety scan -r requirements.txt -r requirements-core.txt -r requirements-train.txt -r requirements-inference.txt || true

echo ""
echo "=== git history secret pattern scan ==="
if git log -p --all | rg -i '(\bpassword\s*[:=]\s*["\x27][^"\x27]{4,}|api[_-]?key\s*[:=]\s*["\x27]|secret\s*[:=]\s*["\x27]|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|BEGIN (RSA |OPENSSH )?PRIVATE KEY)' | head -1; then
  echo "WARNING: Possible secrets found in git history (review output above)."
  exit 1
else
  echo "No suspicious secret patterns found in git history."
fi

echo ""
echo "Done. Review pip-audit/safety warnings for unpinned dependency ranges."
