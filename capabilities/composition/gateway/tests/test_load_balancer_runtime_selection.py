"""Gateway load balancer selection should be deterministic and runtime-aware."""

from __future__ import annotations

import pytest

from capabilities.composition.gateway.service import LoadBalancerService


class FakeRedis:
	def __init__(self, values: dict[str, object] | None = None):
		self.values = values or {}

	async def get(self, key: str):
		return self.values.get(key)


def make_balancer(values: dict[str, object] | None = None) -> LoadBalancerService:
	return LoadBalancerService(db_session=None, redis_client=FakeRedis(values))


def endpoints() -> list[dict[str, object]]:
	return [
		{"endpoint_id": "ep-b", "host": "b.internal", "port": 8080, "weight": 100, "service_weight": 100},
		{"endpoint_id": "ep-a", "host": "a.internal", "port": 8080, "weight": 100, "service_weight": 100},
		{"endpoint_id": "ep-c", "host": "c.internal", "port": 8080, "weight": 100, "service_weight": 100},
	]


@pytest.mark.asyncio
async def test_least_connections_uses_endpoint_connection_counts() -> None:
	balancer = make_balancer()
	candidates = endpoints()
	candidates[0]["active_connections"] = 10
	candidates[1]["active_connections"] = 1
	candidates[2]["active_connections"] = 5

	selected = await balancer._least_connections_selection(candidates)

	assert selected["endpoint_id"] == "ep-a"


@pytest.mark.asyncio
async def test_least_connections_reads_redis_metrics_and_breaks_ties_by_weight() -> None:
	balancer = make_balancer({
		"lb:connections:ep-a": b"4",
		"lb:connections:ep-b": "4",
		"lb:connections:ep-c": "9",
	})
	candidates = endpoints()
	candidates[0]["weight"] = 300
	candidates[1]["weight"] = 100

	selected = await balancer._least_connections_selection(candidates)

	assert selected["endpoint_id"] == "ep-b"


@pytest.mark.asyncio
async def test_weighted_least_connections_accounts_for_capacity_weight() -> None:
	balancer = make_balancer()
	candidates = endpoints()
	candidates[0]["active_connections"] = 10
	candidates[0]["weight"] = 500
	candidates[1]["active_connections"] = 3
	candidates[1]["weight"] = 100
	candidates[2]["active_connections"] = 5
	candidates[2]["weight"] = 100

	selected = await balancer._weighted_least_connections_selection(candidates)

	assert selected["endpoint_id"] == "ep-b"


@pytest.mark.asyncio
async def test_ip_hash_without_client_ip_uses_stable_endpoint_order() -> None:
	balancer = make_balancer()

	first = await balancer._ip_hash_selection(endpoints(), "")
	second = await balancer._ip_hash_selection(list(reversed(endpoints())), "")

	assert first["endpoint_id"] == "ep-a"
	assert second["endpoint_id"] == "ep-a"
