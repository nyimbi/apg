from types import SimpleNamespace

import pytest

from capabilities.composition.registry.composition_engine import (
	ConflictSeverity,
	IntelligentCompositionEngine,
)


class _FakeScalarResult:
	def __init__(self, records):
		self._records = records

	def all(self):
		return self._records


class _FakeResult:
	def __init__(self, records):
		self._records = records

	def scalars(self):
		return _FakeScalarResult(self._records)


class _FakeSession:
	def __init__(self, records):
		self._records = records

	async def execute(self, _query):
		return _FakeResult(self._records)


def _capability(capability_id: str, **kwargs):
	defaults = {
		"tenant_id": "tenant-1234",
		"capability_id": capability_id,
		"capability_code": capability_id,
		"capability_name": capability_id,
		"api_endpoints": [],
		"provides_services": [],
		"data_models": [],
		"metadata_json": {},
	}
	defaults.update(kwargs)
	return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_composition_engine_detects_resource_conflicts() -> None:
	records = [
		_capability(
			"orders",
			api_endpoints=[{"method": "POST", "path": "/api/orders", "port": 8080}],
			provides_services=["order-service"],
			data_models=["Order"],
			metadata_json={"ports": [9000]},
		),
		_capability(
			"fulfillment",
			api_endpoints=[{"method": "POST", "path": "/api/orders", "port": 8080}],
			provides_services=["order-service"],
			data_models=["Order"],
			metadata_json={"ports": [9000]},
		),
	]
	engine = IntelligentCompositionEngine(_FakeSession(records), "tenant-1234", "user-1")

	conflicts = await engine._detect_resource_conflicts(["orders", "fulfillment"])

	conflict_types = {conflict.conflict_type for conflict in conflicts}
	assert conflict_types == {
		"api_endpoint_conflict",
		"service_name_conflict",
		"data_model_ownership_conflict",
		"port_conflict",
	}
	assert {
		conflict.conflict_type
		for conflict in conflicts
		if conflict.severity == ConflictSeverity.HIGH
	} == {"api_endpoint_conflict", "port_conflict"}
	assert all(set(conflict.conflicting_capabilities) == {"orders", "fulfillment"} for conflict in conflicts)


@pytest.mark.asyncio
async def test_composition_engine_detects_configuration_conflicts() -> None:
	records = [
		_capability(
			"billing",
			metadata_json={
				"configuration": {
					"database": {"schema": "billing"},
					"feature_flags": {"strict_mode": True},
				}
			},
		),
		_capability(
			"payments",
			metadata_json={
				"configuration": {
					"database": {"schema": "payments"},
					"feature_flags": {"strict_mode": True},
				}
			},
		),
	]
	engine = IntelligentCompositionEngine(_FakeSession(records), "tenant-1234", "user-1")

	conflicts = await engine._detect_configuration_conflicts(["billing", "payments"])

	assert len(conflicts) == 1
	assert conflicts[0].conflict_type == "configuration_value_conflict"
	assert conflicts[0].severity == ConflictSeverity.MEDIUM
	assert conflicts[0].conflicting_capabilities == ["billing", "payments"]
	assert conflicts[0].auto_resolvable is False
