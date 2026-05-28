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


def test_dashboard_charts_are_derived_from_service_history_and_entries() -> None:
	first = SimpleNamespace(
		timestamp=datetime(2026, 5, 29, 2, 5, 0),
		hit_rate=lambda: 0.5,
		average_latency_ms=10.0,
		p50_latency_ms=5.0,
		p95_latency_ms=15.0,
		p99_latency_ms=25.0,
		operations_per_second=5.0,
		memory_utilization_percent=30.0,
	)
	second = SimpleNamespace(
		timestamp=datetime(2026, 5, 29, 2, 6, 0),
		hit_rate=lambda: 0.75,
		average_latency_ms=4.0,
		p50_latency_ms=2.0,
		p95_latency_ms=6.0,
		p99_latency_ms=9.0,
		operations_per_second=9.0,
		memory_utilization_percent=35.0,
	)
	service = SimpleNamespace(
		_metrics=second,
		_performance_history=[first, second],
		_prefetch_predictions={"order:2": 0.4, "order:1": 0.9},
		_cache_store={
			"default:orders:1": SimpleNamespace(namespace="orders", access_pattern=SimpleNamespace(value="read_heavy")),
			"default:orders:2": SimpleNamespace(namespace="orders", access_pattern="random"),
			"default:users:1": SimpleNamespace(namespace="users", access_pattern=SimpleNamespace(value="read_heavy")),
		},
	)
	view = make_view(service)

	performance = json.loads(view._create_performance_chart())
	latency = json.loads(view._create_latency_histogram())
	throughput = json.loads(view._create_throughput_timeline())
	access_patterns = json.loads(view._create_access_pattern_chart())
	predictions = json.loads(view._create_predictive_analytics_chart())
	efficiency = json.loads(view._create_efficiency_trends_chart())
	namespaces = json.loads(view._create_geo_distribution_chart())

	assert performance["data"][0]["y"] == [0.5, 0.75]
	assert performance["data"][1]["y"] == [10.0, 4.0]
	assert latency["data"][0]["x"] == [5.0, 15.0, 25.0, 2.0, 6.0, 9.0]
	assert throughput["data"][0]["y"] == [5.0, 9.0]
	assert access_patterns["data"][0]["x"] == ["Read Heavy", "Random"]
	assert access_patterns["data"][0]["y"] == [2, 1]
	assert predictions["data"][0]["x"] == ["order:1", "order:2"]
	assert predictions["data"][0]["y"] == [0.9, 0.4]
	assert efficiency["data"][0]["y"] == [50.0, 75.0]
	assert namespaces["data"][0]["x"] == ["orders", "users"]
	assert namespaces["data"][0]["y"] == [2, 1]


def test_dashboard_operational_panels_use_service_state() -> None:
	metrics = SimpleNamespace(
		timestamp=datetime(2026, 5, 29, 2, 10, 0),
		total_operations=10,
		cache_hits=7,
		cache_misses=3,
		memory_utilization_percent=91.0,
		operations_per_second=12.0,
		average_latency_ms=4.0,
		bytes_per_second=2048.0,
		hit_rate=lambda: 0.7,
		error_rate=lambda: 0.06,
	)
	config = SimpleNamespace(
		max_memory_mb=512,
		eviction_policy="adaptive",
		security_level="enterprise",
		metrics_enabled=True,
		ai_optimization_enabled=True,
		encryption_enabled=True,
	)
	service = SimpleNamespace(
		running=True,
		config=config,
		_metrics=metrics,
		_performance_history=[
			SimpleNamespace(timestamp=datetime(2026, 5, 29, 2, 9, 0), hit_rate=lambda: 0.5, average_latency_ms=8.0, operations_per_second=6.0),
			metrics,
		],
		_cache_store={
			"default:orders:1": SimpleNamespace(namespace="orders", size_bytes=1024, tier_recommendation=SimpleNamespace(value="l1")),
			"default:users:1": SimpleNamespace(namespace="users", size_bytes=2048, tier_recommendation=SimpleNamespace(value="l2")),
		},
		_clusters={
			"cluster-a": SimpleNamespace(
				name="cluster-a",
				alert_thresholds={"memory_utilization": 90},
				is_healthy=lambda: True,
			)
		},
		_ai_optimization_results=[
			SimpleNamespace(
				timestamp=datetime(2026, 5, 29, 2, 8, 0),
				confidence_score=0.83,
				recommendations=[{"type": "increase_cache_size", "description": "Add capacity", "impact": "high"}],
			)
		],
		_prefetch_predictions={"orders:next": 0.8},
	)
	view = make_view(service)

	health = view._get_system_health_status()
	alerts = view._get_recent_alerts()
	recommendations = view._get_optimization_recommendations()
	analytics = view._get_analytics_data()
	configuration = view._get_current_configuration()
	optimization_status = view._get_optimization_status()
	system_metrics = view._get_system_metrics()

	assert health["overall_health"] == "Degraded"
	assert health["services_status"]["cache_service"] == "Running"
	assert health["resource_usage"]["memory"] == 91.0
	assert [alert["message"] for alert in alerts] == [
		"Cache memory utilization above 90%",
		"Cache error rate above 5%",
		"Cache hit rate below 80%",
	]
	assert recommendations == [
		{
			"type": "increase_cache_size",
			"title": "increase_cache_size",
			"description": "Add capacity",
			"impact": "High",
			"confidence": 0.83,
		}
	]
	assert analytics["summary"]["total_requests"] == 10
	assert analytics["summary"]["data_served_gb"] == pytest.approx(3072 / (1024 ** 3))
	assert analytics["trends"]["hit_rate_trend"] == "+40.0%"
	assert configuration["cache_size_mb"] == 512
	assert configuration["tier_distribution"] == {"L1": 1, "L2": 1}
	assert optimization_status["status"] == "Active"
	assert optimization_status["last_run"] == "2026-05-29T02:08:00"
	assert system_metrics["total_entries"] == 2
	assert view._get_alert_rules() == [{"cluster": "cluster-a", "metric": "memory_utilization", "threshold": 90}]
	assert view._get_performance_predictions() == {"prefetch_candidates": [{"key": "orders:next", "probability": 0.8}]}
