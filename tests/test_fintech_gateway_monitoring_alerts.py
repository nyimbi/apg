"""Executable alert evaluation tests for the fintech gateway monitor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_monitoring_module():
	module_name = "fintech_gateway_monitoring_service_under_test"
	module_path = (
		Path(__file__).resolve().parents[1]
		/ "capabilities"
		/ "fintech"
		/ "gateway"
		/ "monitoring_service.py"
	)
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


@pytest.mark.asyncio
async def test_monitoring_module_imports_without_optional_observability_packages() -> None:
	module = _load_monitoring_module()
	monitor = module.PaymentGatewayMonitoring()

	await monitor.record_transaction(
		amount=100.0,
		currency="USD",
		status="success",
		processor="sandbox",
		merchant_id="merchant-a",
		duration=0.2,
		payment_method="card",
	)

	metrics = monitor.get_metrics()
	assert "payment_transactions_total" in metrics
	assert monitor.get_metrics_content_type().startswith("text/plain")


@pytest.mark.asyncio
async def test_error_rate_condition_uses_recorded_transaction_metrics() -> None:
	module = _load_monitoring_module()
	monitor = module.PaymentGatewayMonitoring()

	await monitor.record_transaction(100.0, "USD", "success", "sandbox", "merchant-a", 0.2, "card")
	await monitor.record_transaction(100.0, "USD", "failed", "sandbox", "merchant-a", 0.2, "card")

	assert await monitor._evaluate_alert_condition(monitor.alerts["high_error_rate"]) is True
	assert await monitor._evaluate_alert_condition(monitor.alerts["low_success_rate"]) is True


@pytest.mark.asyncio
async def test_check_alerts_triggers_and_resolves_database_connection_alert() -> None:
	module = _load_monitoring_module()
	monitor = module.PaymentGatewayMonitoring()
	monitor.alerts = {
		"database_connection_issue": monitor.alerts["database_connection_issue"]
	}
	notifications: list[str] = []

	async def record_triggered(alert):
		notifications.append(f"triggered:{alert.id}")

	async def record_resolved(alert):
		notifications.append(f"resolved:{alert.id}")

	monitor._send_alert_notification = record_triggered
	monitor._send_resolution_notification = record_resolved

	await monitor.update_system_metrics(db_connections=0, cache_hit_rate=0.9, active_merchants=5)
	triggered = await monitor.check_alerts()

	alert = monitor.alerts["database_connection_issue"]
	assert [item.id for item in triggered] == ["database_connection_issue"]
	assert alert.active is True
	assert alert.triggered_at is not None
	assert notifications == ["triggered:database_connection_issue"]

	await monitor.update_system_metrics(db_connections=2, cache_hit_rate=0.9, active_merchants=5)
	assert await monitor.check_alerts() == []
	assert alert.active is False
	assert alert.resolved_at is not None
	assert notifications == [
		"triggered:database_connection_issue",
		"resolved:database_connection_issue",
	]


@pytest.mark.asyncio
async def test_latency_condition_uses_recorded_duration_histogram() -> None:
	module = _load_monitoring_module()
	monitor = module.PaymentGatewayMonitoring()

	await monitor.record_transaction(100.0, "USD", "success", "slowpay", "merchant-a", 6.0, "card")

	assert await monitor._evaluate_alert_condition(monitor.alerts["high_latency"]) is True
