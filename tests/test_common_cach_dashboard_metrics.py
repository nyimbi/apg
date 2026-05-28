"""CACH dashboard metrics should reflect runtime cache state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from capabilities.common.cach.dashboard import CacheDashboardView


def make_view(cache_service: object | None = None) -> CacheDashboardView:
	view = CacheDashboardView.__new__(CacheDashboardView)
	view._cache_service = cache_service
	return view


def test_dashboard_metrics_empty_state_has_zeroes_not_demo_values() -> None:
	view = make_view()
	view._get_cache_service = lambda: None

	metrics = view._get_dashboard_metrics()

	assert metrics.total_entries == 0
	assert metrics.hit_rate == 0.0
	assert metrics.miss_rate == 0.0
	assert metrics.memory_usage_mb == 0.0
	assert metrics.tier_distribution == {}
	assert metrics.top_keys == []
	assert metrics.recent_operations == []


def test_dashboard_metrics_are_derived_from_cache_service_state() -> None:
	now = datetime(2026, 5, 29, 1, 55, 0)
	metrics = SimpleNamespace(
		cache_hits=8,
		cache_misses=2,
		p50_latency_ms=1.2,
		p95_latency_ms=3.4,
		p99_latency_ms=5.6,
		operations_per_second=42.0,
		used_memory_bytes=0,
		cpu_usage_percent=17.5,
		hit_rate=lambda: 0.8,
		error_rate=lambda: 0.05,
	)
	service = SimpleNamespace(
		_metrics=metrics,
		_cache_store={
			"user:1": SimpleNamespace(
				key="user:1",
				hit_count=5,
				access_count=7,
				size_bytes=2048,
				tier_recommendation=SimpleNamespace(value="l1"),
				last_accessed=now,
			),
			"product:list": SimpleNamespace(
				key="product:list",
				hit_count=2,
				access_count=3,
				size_bytes=1024,
				tier_recommendation="l2",
				last_accessed=now - timedelta(seconds=10),
			),
		},
	)
	view = make_view(service)

	dashboard_metrics = view._get_dashboard_metrics()

	assert dashboard_metrics.total_entries == 2
	assert dashboard_metrics.hit_rate == 0.8
	assert dashboard_metrics.miss_rate == 0.2
	assert dashboard_metrics.latency_p50 == 1.2
	assert dashboard_metrics.latency_p95 == 3.4
	assert dashboard_metrics.latency_p99 == 5.6
	assert dashboard_metrics.throughput_qps == 42.0
	assert dashboard_metrics.error_rate == 0.05
	assert dashboard_metrics.memory_usage_mb == pytest.approx(3072 / (1024 * 1024))
	assert dashboard_metrics.cpu_usage_percent == 17.5
	assert dashboard_metrics.tier_distribution == {"L1": 1, "L2": 1}
	assert dashboard_metrics.top_keys[0] == {"key": "user:1", "hits": 5, "size_kb": 2.0}
	assert dashboard_metrics.recent_operations[0] == {
		"timestamp": now.isoformat(),
		"operation": "GET",
		"key": "user:1",
		"result": "HIT",
	}


def test_dashboard_metrics_normalize_explicit_operation_history() -> None:
	service = SimpleNamespace(
		_metrics=SimpleNamespace(cache_hits=1, cache_misses=1),
		_cache_store={},
		_operation_history=[
			{"timestamp": datetime(2026, 5, 29, 2, 0, 0), "operation": "set", "key": "a", "result": "success"},
			SimpleNamespace(timestamp="2026-05-29T02:00:01", operation="get", key="a", result="hit"),
		],
	)
	view = make_view(service)

	dashboard_metrics = view._get_dashboard_metrics()

	assert dashboard_metrics.recent_operations == [
		{"timestamp": "2026-05-29T02:00:00", "operation": "SET", "key": "a", "result": "SUCCESS"},
		{"timestamp": "2026-05-29T02:00:01", "operation": "GET", "key": "a", "result": "HIT"},
	]


def test_tier_distribution_chart_serializes_without_plotly_dependency() -> None:
	view = make_view()

	chart = json.loads(view._create_tier_distribution_chart({"L1": 2, "L2": 1}))

	assert chart["data"][0]["type"] == "pie"
	assert chart["data"][0]["labels"] == ["L1", "L2"]
	assert chart["data"][0]["values"] == [2, 1]
