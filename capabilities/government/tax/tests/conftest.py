"""Pytest configuration for APG Tax Administration tests."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure the package root is on sys.path for all tests
PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))

# Pre-load canonical modules under their bare names so _load() helpers in
# legacy tests cannot replace them with stale copies.
def _preload(name: str, path: Path) -> None:
	if name not in sys.modules:
		spec = importlib.util.spec_from_file_location(name, path)
		if spec and spec.loader:
			mod = importlib.util.module_from_spec(spec)
			sys.modules[name] = mod
			spec.loader.exec_module(mod)  # type: ignore[union-attr]

_preload("capability_contract", PKG / "capability_contract.py")
_preload("models", PKG / "models.py")
