"""Process-local API helpers for APG Performance Management."""

from __future__ import annotations

from .service import TelecomPerService

_SERVICE = TelecomPerService()


def service() -> TelecomPerService:
	return _SERVICE


def record_kpi(payload: dict) -> dict:
	return _SERVICE.record_kpi(payload["kpi_id"], payload.get("tenant_id", "default"), payload["kpi_category"], payload["kpi_name"], payload["value"], payload.get("baseline_value", 0.0), payload.get("unit", ""), payload.get("network_layer", "core"), payload.get("recorded_at", ""), payload.get("policy_attached", True))


def record_sla_compliance(payload: dict) -> dict:
	return _SERVICE.record_sla_compliance(payload["compliance_id"], payload.get("tenant_id", "default"), payload["sla_type"], payload.get("customer_id"), payload["target_value"], payload["actual_value"], payload.get("period", ""), payload.get("notification_sent", False))


def record_capacity(payload: dict) -> dict:
	return _SERVICE.record_capacity(payload["record_id"], payload.get("tenant_id", "default"), payload["resource_reference"], payload["capacity_state"], payload["utilisation_pct"], payload.get("forecast_horizon_days", 90), payload.get("recorded_at", ""))


def record_trend(payload: dict) -> dict:
	return _SERVICE.record_trend(payload["trend_id"], payload.get("tenant_id", "default"), payload["kpi_id"], payload["trend_direction"], payload.get("lookback_days", 30), payload.get("forecast_value"), payload.get("recorded_at", ""))


def set_threshold(payload: dict) -> dict:
	return _SERVICE.set_threshold(payload["threshold_id"], payload.get("tenant_id", "default"), payload["kpi_name"], payload.get("network_layer", "core"), payload["warning_value"], payload["critical_value"], payload["action"], payload["approval_reference"], payload.get("set_by", ""))


def record_benchmark(payload: dict) -> dict:
	return _SERVICE.record_benchmark(payload["benchmark_id"], payload.get("tenant_id", "default"), payload["benchmark_type"], payload["kpi_name"], payload["benchmark_value"], payload["current_value"], payload.get("recorded_at", ""))


def generate_report(payload: dict) -> dict:
	return _SERVICE.generate_report(payload["report_id"], payload.get("tenant_id", "default"), payload["report_period"], payload.get("format", "json"), payload["approval_reference"], payload.get("generated_by", ""), payload.get("generated_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "performance management operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
