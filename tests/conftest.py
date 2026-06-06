"""Shared pytest fixtures for migrated APG root tests."""

from __future__ import annotations

import sys
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


@pytest.fixture(autouse=True)
def _flush_stale_flask_contexts():
	"""Pop any lingering Flask request/app contexts left by previous tests.

	Flask 3.x uses ContextVar — request context leaks between tests when a
	test_request_context() is pushed but not popped (e.g. on test failure).
	"""
	yield
	try:
		from flask.globals import _cv_request, _cv_app
		ctx = _cv_request.get(None)
		if ctx is not None:
			try:
				ctx.pop()
			except Exception:
				pass
		app_ctx = _cv_app.get(None)
		if app_ctx is not None:
			try:
				app_ctx.pop()
			except Exception:
				pass
	except Exception:
		pass
