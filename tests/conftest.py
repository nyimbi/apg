"""Shared pytest fixtures for migrated APG root tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

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
