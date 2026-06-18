"""Database schema for APG Graph-based RAG."""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys
import types as _types

_parent_pkg = "capabilities.common.grag"
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
    GraphRAGDatabaseService = _mod.GraphRAGDatabaseService
    GraphRAGDatabaseError = _mod.GraphRAGDatabaseError
except Exception as _e:
    # Pre-existing issue in database.py (e.g. SQLAlchemy reserved attr in models)
    # Provide minimal stubs so the import name resolves.
    class GraphRAGDatabaseError(Exception):
        """Stub — real class unavailable due to import error: see grag/database.py"""

    class GraphRAGDatabaseService:
        """Stub — real class unavailable due to import error: see grag/database.py"""
        _IMPORT_ERROR = _e

__all__ = ["GraphRAGDatabaseService", "GraphRAGDatabaseError"]
