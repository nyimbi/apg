"""Service-mesh federation startup regressions."""

from __future__ import annotations

import json

import pytest

from capabilities.composition.gateway.service_mesh_federation import (
	ClusterRole,
	ServiceMeshFederation,
)


class _RedisRecorder:
	def __init__(self) -> None:
		self.values: dict[str, str] = {}
		self.published: list[tuple[str, str]] = []

	async def set(self, key: str, value: str) -> None:
		self.values[key] = value

	async def publish(self, channel: str, value: str) -> None:
		self.published.append((channel, value))


@pytest.mark.asyncio
async def test_federation_startup_records_runtime_services():
	redis = _RedisRecorder()
	federation = ServiceMeshFederation(
		cluster_id="cluster-a",
		cluster_name="Cluster A",
		region="ke-central",
		zone="ke-central-1a",
		db_session=None,
		redis_client=redis,
		cert_manager=None,
		federation_endpoint="https://cluster-a.mesh.local",
		role=ClusterRole.PRIMARY,
	)

	await federation._start_federation_services()

	assert federation._federation_started_at is not None
	assert set(federation._federation_service_status) == {
		"federation_api",
		"federation_routing",
		"certificate_rotation",
		"metrics_collector",
	}
	record = json.loads(redis.values["federation:services:cluster-a"])
	assert record["cluster_id"] == "cluster-a"
	assert record["health_status"] == "running"
	assert record["services"]["federation_api"]["endpoint"] == "https://cluster-a.mesh.local"

	channel, payload = redis.published[-1]
	event = json.loads(payload)
	assert channel == "federation:events"
	assert event["type"] == "federation_services_started"
	assert event["cluster_id"] == "cluster-a"
	assert "metrics_collector" in event["services"]
