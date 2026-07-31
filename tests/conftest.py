"""Shared pytest fixtures for migrated APG root tests."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ── sys.path bootstrap ──────────────────────────────────────────────────────
# Some capability packages contain bare-import fallbacks (e.g.
# capabilities/fintech/lending/service.py has `from capability_contract import
# ...`) that only work when the package directory is on sys.path.  Add these
# directories unconditionally so test-ordering doesn't affect import success.
_REPO = Path(__file__).resolve().parents[1]
for _bare_import_dir in (
    _REPO / "capabilities" / "fintech" / "lending",
):
    _s = str(_bare_import_dir)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from templates.composable.composition_engine import CompositionEngine


@dataclass
class _FixtureProperty:
	name: str
	type_annotation: str = "str"


@dataclass
class _FixtureEntityType:
	name: str


@dataclass
class _FixtureEntity:
	name: str
	entity_type: _FixtureEntityType
	properties: list[_FixtureProperty] = field(default_factory=list)
	methods: list[object] = field(default_factory=list)


@dataclass
class _FixtureAST:
	entities: list[_FixtureEntity]


@pytest.fixture
def engine() -> CompositionEngine:
	composable_root = Path(__file__).resolve().parents[1] / "templates" / "composable"
	return CompositionEngine(composable_root)


@pytest.fixture
def context(engine: CompositionEngine):
	ast = _FixtureAST(entities=[
		_FixtureEntity(
			name="User",
			entity_type=_FixtureEntityType("AGENT"),
			properties=[
				_FixtureProperty("email"),
				_FixtureProperty("password"),
			],
		),
		_FixtureEntity(
			name="AppDatabase",
			entity_type=_FixtureEntityType("DATABASE"),
			properties=[_FixtureProperty("connection")],
		),
	])
	return engine.compose_application(
		ast,
		project_name="APG Fixture App",
		project_description="Fixture-backed generated application",
		author="APG Test Suite",
	)


@pytest.fixture
def apg_source_minimal() -> str:
	return """
module fixture_app version 1.0.0 {}

table Customer {
	name: str;
}
"""


@pytest.fixture
def compiled_app_namespace(apg_source_minimal: str) -> Callable[..., dict[str, object]]:
	def _compile(
		source: str | None = None,
		*,
		filename: str = "fixture_app.apg",
		app_filename: str = "generated_fixture_app.py",
	) -> dict[str, object]:
		from compiler.compiler import APGCompiler

		result = APGCompiler().compile_string(source or apg_source_minimal, filename)
		assert result.success, result.errors
		namespace: dict[str, object] = {"__file__": app_filename}
		exec(compile(result.generated_files["app.py"], app_filename, "exec"), namespace)
		return namespace

	return _compile


@pytest.fixture
def minimal_apg_source(apg_source_minimal: str) -> str:
	"""Alias for apg_source_minimal — Wave V canonical name."""
	return apg_source_minimal


@pytest.fixture
def compiled_namespace(compiled_app_namespace: Callable[..., dict[str, object]]) -> Callable[..., dict[str, object]]:
	"""Alias for compiled_app_namespace — Wave V canonical name."""
	return compiled_app_namespace


@pytest.fixture
def compiled_flask_client(compiled_app_namespace: Callable[..., dict[str, object]]):
	def _client(source: str | None = None, **compile_kwargs):
		namespace = compiled_app_namespace(source, **compile_kwargs)
		app = namespace["_flask_app"]
		app.config["TESTING"] = True
		return app.test_client()

	return _client


def _pop_flask_contexts() -> None:
	"""Reset Flask ContextVars to None so has_request_context() returns False.

	Uses two strategies:
	1. Call ctx.pop() to trigger proper Flask teardown (teardown_request, etc.)
	2. Force-reset the ContextVar to None regardless, using _cv_request.set(None)
	"""
	try:
		from flask.globals import _cv_request, _cv_app
		ctx = _cv_request.get(None)
		if ctx is not None:
			try:
				ctx.pop()
			except Exception:
				pass
			# Force-reset even if pop() silently failed
			_cv_request.set(None)  # type: ignore[arg-type]
		app_ctx = _cv_app.get(None)
		if app_ctx is not None:
			try:
				app_ctx.pop()
			except Exception:
				pass
			_cv_app.set(None)  # type: ignore[arg-type]
	except Exception:
		pass


def _evict_stale_modules() -> None:
	"""Remove stale or stub module entries from sys.modules after each test.

	Some tests replace real packages with types.ModuleType stubs (e.g. flask,
	capabilities.fintech.*) and don't restore them.  This causes subsequent
	tests to fail with ModuleNotFoundError or ImportError.

	Stubs created via types.ModuleType have __spec__ = None and __file__ = None;
	real installed packages always have both.  We evict:
	  1. Known real packages (flask, capabilities.*) that have no __spec__/__file__
	  2. capabilities.* modules pointing to deleted temp files
	"""
	import importlib
	# Packages that should always have __spec__ and __file__ when genuine
	_REAL_PREFIXES = ("flask", "werkzeug", "sqlalchemy", "pydantic", "fastapi", "capabilities")

	stale: list[str] = []
	for name, mod in list(sys.modules.items()):
		is_real_prefix = any(name == p or name.startswith(p + ".") or name.startswith(p + "_") for p in _REAL_PREFIXES)
		if not is_real_prefix:
			continue

		mod_file = getattr(mod, "__file__", None)
		spec = getattr(mod, "__spec__", None)

		# Stub: real package replaced with types.ModuleType (no __spec__, no __file__)
		if mod_file is None and spec is None:
			stale.append(name)
			continue

		# Deleted file: __spec__.origin points to a temp dir that was cleaned up
		origin = getattr(spec, "origin", None) if spec else None
		if origin is not None and not Path(origin).exists():
			stale.append(name)

	# Also evict any capabilities.fintech/kyc/lending stubs that have __path__=[]
	for name, mod in list(sys.modules.items()):
		if name.startswith("capabilities."):
			path = getattr(mod, "__path__", None)
			if path is not None and list(path) == []:
				if name not in stale:
					stale.append(name)

	stale_set = set(stale)
	for name in stale_set:
		sys.modules.pop(name, None)
	if stale_set:
		importlib.invalidate_caches()


@pytest.fixture(autouse=True)
def _flush_stale_flask_contexts():
	"""Ensure clean Flask context and module state around each test."""
	_pop_flask_contexts()
	yield
	_pop_flask_contexts()
	_evict_stale_modules()
