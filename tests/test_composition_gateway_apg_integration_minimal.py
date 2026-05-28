"""Minimal-runtime regressions for gateway APG integration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from capabilities.composition.gateway.apg_integration import (
	CapabilityRegistryIntegration,
	EventStreamingIntegration,
	EventType,
	ServiceMeshEvent,
	create_apg_integration,
	redis,
)


class _ASMService:
	async def discover_services(self):
		return []


@pytest.mark.asyncio
async def test_gateway_apg_integration_factory_uses_in_memory_redis_fallback():
	"""Gateway APG integration should be constructible without a Redis package/server."""
	integration = await create_apg_integration(_ASMService(), "redis://memory")

	assert integration.redis_client.__class__.__name__ == "_InMemoryRedis"
	await integration.composition_engine.start()
	cached = await integration.redis_client.get("apg:composition_engines:service_mesh")
	await integration.composition_engine.stop()

	assert cached is not None
	assert await integration.redis_client.get("apg:composition_engines:service_mesh") is None


@pytest.mark.asyncio
async def test_gateway_capability_registry_catalog_works_with_in_memory_redis():
	client = redis.from_url("redis://memory")
	registry = CapabilityRegistryIntegration(client)
	event = ServiceMeshEvent(
		event_id="evt_1",
		event_type=EventType.SERVICE_REGISTERED.value,
		service_id="svc_1",
		route_id=None,
		data={
			"service_name": "ledger-service",
			"service_version": "1.0.0",
			"endpoints": ["/ledger"],
		},
		timestamp=datetime.now(timezone.utc),
		tenant_id="tenant123",
	)

	await registry.register_capability()
	await registry.update_service_catalog(event)
	services = await registry.get_registered_services("tenant123")

	assert await client.get("apg:capabilities:api_service_mesh") is not None
	assert services == [{
		"service_id": "svc_1",
		"service_name": "ledger-service",
		"service_version": "1.0.0",
		"endpoints": ["/ledger"],
		"mesh_managed": True,
		"registered_at": event.timestamp.isoformat(),
	}]


@pytest.mark.asyncio
async def test_gateway_event_streaming_publishes_to_stream_and_channel():
	client = redis.from_url("redis://memory")
	streaming = EventStreamingIntegration(client)
	event = ServiceMeshEvent(
		event_id="evt_traffic",
		event_type=EventType.TRAFFIC_SPIKE.value,
		service_id="svc_traffic",
		route_id="route_1",
		data={"current_rps": 180},
		timestamp=datetime.now(timezone.utc),
		tenant_id="tenant123",
	)

	await streaming.publish_event(event)

	stream_key = streaming.event_streams["service_mesh.traffic"]
	assert client._streams[stream_key][0]["fields"]["event_id"] == "evt_traffic"
	assert client._published[0]["channel"] == "apg:events:service_mesh"
