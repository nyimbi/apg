"""Database schema for APG Master Data Management."""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

_parent_pkg = "capabilities.common.mdm"
_mod_name = _parent_pkg + "._database_module"
_spec = _ilu.spec_from_file_location(
    _mod_name,
    _pl.Path(__file__).parent.parent / "database.py",
    submodule_search_locations=[],
)
_mod = _ilu.module_from_spec(_spec)
_mod.__package__ = _parent_pkg
_sys.modules[_mod_name] = _mod
try:
    _spec.loader.exec_module(_mod)
    MDMDatabaseManager = _mod.MDMDatabaseManager
except Exception as _e:
    class MDMDatabaseManager:
        """Stub — real class unavailable: see mdm/database.py"""
        _IMPORT_ERROR = _e

__all__ = ["MDMDatabaseManager"]
