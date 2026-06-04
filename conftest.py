"""Root conftest — cleans generic module names from sys.modules around each test.

Capability test_package_contract.py files load modules via importlib using
short names like "service", "capability_contract", "models", etc.  When
multiple capabilities are tested in the same process, stale entries in
sys.modules cause cross-capability import collisions.

This fixture:
  - runs before AND after every test (both-sides cleanup)
  - uses a fast intersection check to skip the loop when nothing to clean
  - adds negligible overhead per test (< 0.1ms when no generic modules exist)
"""
from __future__ import annotations

import sys
import pytest

_GENERIC_CAPS_MODULES = frozenset({
    "service",
    "capability_contract",
    "models",
    "api",
    "views",
    "app",
    "alerts_runtime",
})


def _clean():
    found = _GENERIC_CAPS_MODULES.intersection(sys.modules)
    for name in found:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _clean_generic_capability_modules():
    """Remove short-name capability modules before and after each test."""
    _clean()
    yield
    _clean()
