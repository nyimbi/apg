"""Database schema for APG Metadata Management."""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

_parent_pkg = "capabilities.common.meta"
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
    MetaDatabaseManager = _mod.MetaDatabaseManager
    create_database_manager = _mod.create_database_manager
except Exception as _e:
    # Pre-existing issue (e.g. missing neo4j package). Provide stubs.
    async def create_database_manager(config=None):
        raise RuntimeError(f"meta database unavailable: {_e}")

    class MetaDatabaseManager:
        """Stub — real class unavailable due to import error: see meta/database.py"""
        _IMPORT_ERROR = _e

__all__ = ["MetaDatabaseManager", "create_database_manager"]
