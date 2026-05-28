from types import SimpleNamespace

import pytest

from capabilities.composition.registry.composition_engine import (
	ConflictSeverity,
	IntelligentCompositionEngine,
	PerformanceImpact,
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


@pytest.mark.asyncio
async def test_composition_engine_rejects_empty_composition_without_database_query() -> None:
	engine = IntelligentCompositionEngine(_FakeSession([]), "tenant-1234", "user-1")

	result = await engine.validate_composition([])

	assert result.is_valid is False
	assert result.validation_score == 0.0
	assert result.conflicts[0].conflict_type == "empty_composition"
	assert result.cost_analysis["cost_per_capability"] == 0.0
	assert result.deployment_strategy["phases"] == []


@pytest.mark.asyncio
async def test_composition_engine_cost_analysis_uses_resource_impact() -> None:
	engine = IntelligentCompositionEngine(_FakeSession([]), "tenant-1234", "user-1")

	cost = await engine._generate_cost_analysis(
		["orders", "payments"],
		PerformanceImpact(
			memory_usage_mb=1024,
			cpu_usage_pct=25,
			network_bandwidth_mbps=10,
			disk_io_ops=100,
			startup_time_ms=500,
			response_time_ms=120,
			scalability_score=0.9,
		),
	)

	assert cost == {
		"monthly_cost_usd": 27.5,
		"cost_breakdown": {
			"base_cost": 20.0,
			"memory_cost": 5.0,
			"cpu_cost": 2.5,
		},
		"cost_per_capability": 13.75,
		"optimization_potential": 0.15,
	}
