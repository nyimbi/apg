"""REGY cache metric regressions for executable service statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from capabilities.common.regy.models import (
	ServiceDiscoveryResult,
	ServiceHealthStatus,
	ServiceStatus,
)
from capabilities.common.regy.service import ServiceRegistryService


TENANT_ID = "tenant-regy-cache"


def _discovery_result() -> ServiceDiscoveryResult:
	return ServiceDiscoveryResult(
		total_count=0,
		returned_count=0,
		query_time_ms=1.0,
		services=[],
		tenant_id=TENANT_ID,
	)


def _health_status() -> ServiceHealthStatus:
	return ServiceHealthStatus(
		service_id="svc-cache",
		instance_id="svc-cache-1",
		overall_status=ServiceStatus.HEALTHY,
		health_score=1.0,
		status_message="healthy",
		tenant_id=TENANT_ID,
	)


def test_discovery_cache_hit_rate_uses_observed_hits_and_misses() -> None:
	service = ServiceRegistryService(TENANT_ID, {"cache_ttl_seconds": 60})
	result = _discovery_result()

	assert service._calculate_cache_hit_rate() == 0.0

	service.discovery_cache["query"] = (result, datetime.now(timezone.utc))

	assert service._get_cached_discovery_result("query") is result
	assert result.cached_result is True
	assert service.cache_hits == 1
	assert service.cache_misses == 0
	assert service._calculate_cache_hit_rate() == 1.0

	assert service._get_cached_discovery_result("missing") is None
	assert service.cache_hits == 1
	assert service.cache_misses == 1
	assert service._calculate_cache_hit_rate() == 0.5


def test_expired_discovery_cache_entries_count_as_misses_and_are_evicted() -> None:
	service = ServiceRegistryService(TENANT_ID, {"cache_ttl_seconds": 60})
	expired_at = datetime.now(timezone.utc) - timedelta(seconds=61)
	service.discovery_cache["expired"] = (_discovery_result(), expired_at)

	assert service._get_cached_discovery_result("expired") is None
	assert "expired" not in service.discovery_cache
	assert service.cache_hits == 0
	assert service.cache_misses == 1
	assert service._calculate_cache_hit_rate() == 0.0


def test_health_cache_contributes_to_same_cache_hit_rate_counters() -> None:
	service = ServiceRegistryService(TENANT_ID)
	health = _health_status()

	service._cache_health_status("svc-cache", health)

	assert service._get_cached_health_status("svc-cache") is health
	assert service.cache_hits == 1
	assert service.cache_misses == 0

	assert service._get_cached_health_status("unknown") is None
	assert service.cache_hits == 1
	assert service.cache_misses == 1
	assert service._calculate_cache_hit_rate() == 0.5
