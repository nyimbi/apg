"""Process-local API helpers for APG Telecom Analytics."""

from __future__ import annotations

from .service import TelecomAnaService

_SERVICE = TelecomAnaService()


def service() -> TelecomAnaService:
	return _SERVICE


def record_analysis_run(payload: dict) -> dict:
	return _SERVICE.record_analysis_run(payload["run_id"], payload.get("tenant_id", "default"), payload["analysis_type"], payload["owner_id"], payload.get("time_granularity", "daily"), payload.get("start_time", ""), payload.get("end_time", ""), payload.get("evidence_reference", "evidence"), payload.get("policy_attached", True))


def record_metric(payload: dict) -> dict:
	return _SERVICE.record_metric(payload["metric_id"], payload.get("tenant_id", "default"), payload["metric_type"], payload["metric_name"], payload["value"], payload.get("unit", ""), payload.get("baseline_value", 0.0), payload.get("aggregation_type", "avg"), payload.get("recorded_at", ""))


def record_churn_prediction(payload: dict) -> dict:
	return _SERVICE.record_churn_prediction(payload["prediction_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["risk_level"], payload["confidence_score"], payload["model_id"], payload.get("predicted_at", ""), payload.get("features_reference", ""))


def record_revenue_event(payload: dict) -> dict:
	return _SERVICE.record_revenue_event(payload["event_id"], payload.get("tenant_id", "default"), payload["category"], payload["amount"], payload.get("currency", "KES"), payload.get("period", ""), payload["evidence_reference"])


def record_segment(payload: dict) -> dict:
	return _SERVICE.record_segment(payload["segment_id"], payload.get("tenant_id", "default"), payload["segment_name"], payload.get("segment_type", "custom"), payload["criteria"], payload.get("customer_count", 0), payload.get("created_by", ""))


def record_network_analytics(payload: dict) -> dict:
	return _SERVICE.record_network_analytics(payload["record_id"], payload.get("tenant_id", "default"), payload["network_layer"], payload["metric_name"], payload["value"], payload.get("threshold", 0.0), payload.get("recorded_at", ""))


def record_anomaly(payload: dict) -> dict:
	return _SERVICE.record_anomaly(payload["anomaly_id"], payload.get("tenant_id", "default"), payload["anomaly_type"], payload["confidence_score"], payload["description"], payload["evidence_reference"], payload.get("detected_at", ""))


def register_model(payload: dict) -> dict:
	return _SERVICE.register_model(payload["model_id"], payload.get("tenant_id", "default"), payload["model_type"], payload["model_name"], payload.get("version", "1.0.0"), payload["validation_reference"], payload.get("registered_by", ""))


def generate_report(payload: dict) -> dict:
	return _SERVICE.generate_report(payload["report_id"], payload.get("tenant_id", "default"), payload["report_format"], payload.get("analysis_id", ""), payload["approval_reference"], payload.get("generated_by", ""), payload.get("generated_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "analytics operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
