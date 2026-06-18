"""Database schema for APG Import/Export."""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

_parent_pkg = "capabilities.common.imex"
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
    DatabaseManager = _mod.DatabaseManager
    DatabaseConfig = _mod.DatabaseConfig
    TransactionContext = _mod.TransactionContext
    DatabaseError = _mod.DatabaseError
except Exception as _e:
    class DatabaseError(Exception):
        """Stub — real class unavailable: see imex/database.py"""

    class DatabaseConfig:
        """Stub — real class unavailable: see imex/database.py"""
        _IMPORT_ERROR = _e

    class TransactionContext:
        """Stub — real class unavailable: see imex/database.py"""
        _IMPORT_ERROR = _e

    class DatabaseManager:
        """Stub — real class unavailable: see imex/database.py"""
        _IMPORT_ERROR = _e

__all__ = ["DatabaseManager", "DatabaseConfig", "TransactionContext", "DatabaseError"]
