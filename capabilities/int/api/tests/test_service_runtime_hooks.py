"""Executable runtime hooks for API management services."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capabilities.int.api.service import APILifecycleService, AnalyticsService


class MemoryCache:
	def __init__(self):
		self.store = {}
		self.deleted = []

	async def get(self, key):
		return self.store.get(key)

	async def set(self, key, value, ttl=None):
		self.store[key] = value
		return True

	async def delete(self, key):
		self.deleted.append(key)
		self.store.pop(key, None)
		return True

	async def delete_pattern(self, pattern):
		prefix = pattern.rstrip("*")
		for key in list(self.store):
			if key.startswith(prefix):
				self.deleted.append(key)
				self.store.pop(key, None)
		return True


class ScalarListResult:
	def __init__(self, value=None, values=None):
		self.value = value
		self.values = list(values or [])

	def scalar_one_or_none(self):
		return self.value

	def scalars(self):
		return self

	def all(self):
		return self.values


class FakeSession:
	def __init__(self, *results):
		self.results = list(results)
		self.executed = []
		self.commit_count = 0

	async def execute(self, statement):
		self.executed.append(statement)
		return self.results.pop(0)

	async def commit(self):
		self.commit_count += 1


def make_api():
	return SimpleNamespace(
		api_id="api-1",
		api_name="orders",
		api_title="Orders API",
		api_description="Order operations",
		version="1.2.3",
		status="active",
		base_path="/orders",
		upstream_url="https://orders.internal",
		auth_type="api_key",
		default_rate_limit=500,
		tenant_id="tenant-a",
		openapi_spec=None,
		updated_at=None,
	)


def make_endpoint(path="/items", method="GET"):
	return SimpleNamespace(
		endpoint_id="endpoint-1",
		path=path,
		method=method,
		operation_id=None,
		summary="List items",
		description="List order items",
		parameters=[{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
		request_schema=None,
		response_schema={"type": "object"},
		auth_required=True,
		scopes_required=["orders:read"],
		cache_enabled=True,
		deprecated=False,
	)


@pytest.mark.asyncio
async def test_openapi_spec_is_regenerated_from_endpoint_state():
	api = make_api()
	endpoint = make_endpoint()
	cache = MemoryCache()
	session = FakeSession(
		ScalarListResult(value=api),
		ScalarListResult(values=[endpoint]),
	)
	service = APILifecycleService(session, cache)

	await service._update_openapi_spec(api.api_id)

	assert session.commit_count == 1
	assert api.openapi_spec["openapi"] == "3.0.3"
	operation = api.openapi_spec["paths"]["/items"]["get"]
	assert operation["operationId"] == "get_items"
	assert operation["security"] == [{"apiKeyAuth": ["orders:read"]}]
	assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {"type": "object"}
	assert cache.store["api:openapi:api-1"] == api.openapi_spec
	assert "api:tenant-a:api-1" in cache.deleted


@pytest.mark.asyncio
async def test_lifecycle_notifications_write_cache_backed_event_ledgers():
	api = make_api()
	endpoint = make_endpoint(path="/orders/{id}", method="POST")
	cache = MemoryCache()
	service = APILifecycleService(
		FakeSession(
			ScalarListResult(value=api),
			ScalarListResult(values=[endpoint]),
			ScalarListResult(values=[
				SimpleNamespace(consumer_id="consumer-a"),
				SimpleNamespace(consumer_id="consumer-b"),
			]),
		),
		cache,
	)

	await service._notify_gateway_update(api.api_id)
	await service._notify_api_deprecation(api.api_id, "Use v2 before 2026-07-01")

	gateway_event = cache.store["gateway:update:api-1"]
	assert gateway_event["event_type"] == "api_gateway_update"
	assert gateway_event["endpoint_count"] == 1
	assert gateway_event["endpoints"][0]["path"] == "/orders/{id}"
	assert cache.store["gateway:update_events"][-1] == gateway_event

	deprecation_event = cache.store["api:deprecation:api-1"]
	assert deprecation_event["event_type"] == "api_deprecated"
	assert deprecation_event["consumer_ids"] == ["consumer-a", "consumer-b"]
	assert cache.store["api:deprecation_events"][-1] == deprecation_event


@pytest.mark.asyncio
async def test_realtime_metrics_accumulate_api_and_consumer_state():
	cache = MemoryCache()
	service = AnalyticsService(FakeSession(), cache)

	await service._update_realtime_metrics("api-1", "consumer-a", 200, 120)
	await service._update_realtime_metrics("api-1", "consumer-a", 503, 240)

	api_metrics = cache.store["metrics:realtime:api:api-1"]
	assert api_metrics["total_requests"] == 2
	assert api_metrics["error_count"] == 1
	assert api_metrics["status_counts"] == {"200": 1, "503": 1}
	assert api_metrics["consumer_counts"] == {"consumer-a": 2}
	assert api_metrics["average_response_time_ms"] == 180
	assert api_metrics["error_rate_percent"] == 50

	consumer_metrics = cache.store["metrics:realtime:consumer:consumer-a"]
	assert consumer_metrics["total_requests"] == 2
	assert consumer_metrics["api_counts"] == {"api-1": 2}
	assert consumer_metrics["error_rate_percent"] == 50
