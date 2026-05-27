"""Composition gateway health-monitoring regressions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from capabilities.composition.gateway.apg_integration import CompositionEngineIntegration


class _RedisRecorder:
	def __init__(self) -> None:
		self.setex_calls: list[tuple[str, int, str]] = []
		self.publish_calls: list[tuple[str, str]] = []

	async def setex(self, key: str, ttl: int, payload: str) -> None:
		self.setex_calls.append((key, ttl, payload))

	async def publish(self, channel: str, payload: str) -> None:
		self.publish_calls.append((channel, payload))


class _ASMService:
	def __init__(self, services: list[SimpleNamespace]) -> None:
		self.services = services

	async def discover_services(self) -> list[SimpleNamespace]:
		return self.services


def _composition() -> dict:
	return {
		"composition_id": "comp-1",
		"composition_name": "payments",
		"services": ["auth-service", "payment-service"],
		"status": "active",
	}


@pytest.mark.asyncio
async def test_composition_health_check_degrades_unhealthy_service():
	redis = _RedisRecorder()
	engine = CompositionEngineIntegration(
		_ASMService([
			SimpleNamespace(service_id="svc-auth", service_name="auth-service", health_status="healthy"),
			SimpleNamespace(service_id="svc-pay", service_name="payment-service", health_status="unhealthy"),
		]),
		redis,
	)
	composition = _composition()

	await engine._check_composition_health(composition)

	assert composition["status"] == "degraded"
	assert composition["health"]["services"] == {
		"auth-service": "healthy",
		"payment-service": "unhealthy",
	}
	assert composition["current_unhealthy_services"] == ["payment-service"]
	assert composition["failed_services"] == [
		{
			"service_id": "payment-service",
			"failed_at": composition["last_health_check"],
			"detected_by": "composition_health_check",
		}
	]

	key, ttl, cached_payload = redis.setex_calls[-1]
	assert key == "apg:compositions:service_mesh:comp-1"
	assert ttl == 86400
	assert json.loads(cached_payload)["status"] == "degraded"

	channel, event_payload = redis.publish_calls[-1]
	assert channel == "apg:compositions:health"
	assert json.loads(event_payload) == {
		"composition_id": "comp-1",
		"status": "degraded",
		"services": {
			"auth-service": "healthy",
			"payment-service": "unhealthy",
		},
		"unhealthy_services": ["payment-service"],
		"timestamp": composition["last_health_check"],
	}


@pytest.mark.asyncio
async def test_composition_health_check_recovers_when_services_are_healthy():
	redis = _RedisRecorder()
	engine = CompositionEngineIntegration(
		_ASMService([
			SimpleNamespace(service_id="svc-auth", service_name="auth-service", health_status="healthy"),
			SimpleNamespace(service_id="svc-pay", service_name="payment-service", health_status="healthy"),
		]),
		redis,
	)
	composition = _composition()
	composition["status"] = "degraded"
	composition["current_unhealthy_services"] = ["payment-service"]

	await engine._check_composition_health(composition)

	assert composition["status"] == "active"
	assert composition["current_unhealthy_services"] == []
	assert composition["health"]["healthy_count"] == 2
	assert composition["health"]["unhealthy_count"] == 0
	assert composition["recovered_at"] == composition["last_health_check"]
