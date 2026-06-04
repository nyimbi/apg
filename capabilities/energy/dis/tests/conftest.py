"""Path isolation for energy_dis tests.

Forces this capability's modules to be importable as unique names by
pre-loading them into sys.modules under capability-scoped keys before
pytest collects this package's test files.
"""
import importlib.util
import sys
import os

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAP = "dis"


def _load(name: str) -> None:
    """Load _CAP_DIR/<name>.py into sys.modules as 'energy_dis.<name>'."""
    scoped = f"energy_dis.{name}"
    if scoped in sys.modules:
        return
    path = os.path.join(_CAP_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(scoped, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[scoped] = mod
    # Also register under bare name so plain 'from service import X' works
    # — but only if not already claimed by another cap
    if name not in sys.modules or os.path.dirname(
            getattr(sys.modules[name], "__file__", "")) != _CAP_DIR:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)


for _mod in ("capability_contract", "models", "service", "views", "api"):
    _load(_mod)
