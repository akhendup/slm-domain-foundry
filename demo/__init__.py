"""Backward-compatible shim for the renamed ``app`` package."""

import warnings

warnings.warn(
    "The demo package is deprecated; use app instead (e.g. python -m app.chat).",
    DeprecationWarning,
    stacklevel=2,
)
