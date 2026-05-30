"""API helpers for the APG Digital Twin Framework capability."""

from __future__ import annotations

from typing import Any

from .service import DtwnService


SERVICE = DtwnService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"twin_count": summary["twin_count"],
		"model_count": summary["model_count"],
		"simulation_count": summary["simulation_count"],
		"review_required_prediction_count": summary["review_required_prediction_count"],
		"twin_agent_count": summary["twin_agent_count"],
		"audit_event_count": summary["audit_event_count"],
	}


def create_twin(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_twin(
		twin_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		asset_id=str(payload.get("asset_id") or ""),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		twin_type=str(payload.get("twin_type") or "asset"),
		location=dict(payload.get("location") or {}),
		initial_state=dict(payload.get("initial_state") or {}),
	)


def register_simulation_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_simulation_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		version=str(payload.get("version") or "1.0.0"),
		owner=str(payload.get("owner") or "model-owner"),
		model_type=str(payload.get("model_type") or "physics"),
		calibration_evidence=str(payload.get("calibration_evidence") or ""),
		approved_by=payload.get("approved_by"),
		confidence=float(payload.get("confidence") or 0.75),
	)


def ingest_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.ingest_telemetry(
		sample_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		twin_id=str(payload["twin_id"]),
		source_id=str(payload["source_id"]),
		source_type=str(payload.get("source_type") or "iot"),
		authenticated=bool(payload.get("authenticated", False)),
		measurements=dict(payload.get("measurements") or {}),
		geospatial_context=dict(payload.get("geospatial_context") or {}),
		vision_signals=dict(payload.get("vision_signals") or {}),
	)


def link_topology(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.link_topology(
		link_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_twin_id=str(payload["source_twin_id"]),
		target_twin_id=str(payload["target_twin_id"]),
		relationship=str(payload.get("relationship") or "depends_on"),
		metadata=dict(payload.get("metadata") or {}),
	)


def run_simulation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_simulation(
		run_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		twin_id=str(payload["twin_id"]),
		model_id=str(payload["model_id"]),
		scenario=str(payload.get("scenario") or "baseline"),
		environment=str(payload.get("environment") or "sandbox"),
		approved_by=payload.get("approved_by"),
	)


def record_prediction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_prediction(
		prediction_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		twin_id=str(payload["twin_id"]),
		model_id=str(payload["model_id"]),
		risk_score=float(payload.get("risk_score") or 0.0),
		confidence=float(payload.get("confidence") or 0.75),
		horizon=str(payload.get("horizon") or "24h"),
		recommendation=str(payload.get("recommendation") or "continue_normal_operation"),
		reviewed_by=payload.get("reviewed_by"),
	)


def review_prediction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.review_prediction(
		prediction_id=str(payload["prediction_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
	)


def register_twin_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_twin_agent(
		tenant_id=str(payload.get("tenant_id") or "default"),
		agent_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", False)),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=bool(payload.get("registered", True)),
	)


def change_twin_status(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_twin_status(
		tenant_id=str(payload.get("tenant_id") or "default"),
		twin_id=str(payload["twin_id"]),
		status=str(payload["status"]),
		reason=str(payload.get("reason") or ""),
		actor=str(payload.get("actor") or "twin-operator"),
		audit_recorded=bool(payload.get("audit_recorded", True)),
	)


def validate_batch_twin_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_twin_mutation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or ""),
		actor=str(payload.get("actor") or "twin-operator"),
	)


def twin_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"twins": SERVICE.list_twins(tenant_id),
		"models": SERVICE.list_models(tenant_id),
		"telemetry": SERVICE.list_telemetry(tenant_id),
		"topology": SERVICE.list_topology(tenant_id),
		"simulations": SERVICE.list_simulations(tenant_id),
		"predictions": SERVICE.list_predictions(tenant_id),
		"twin_agents": SERVICE.list_twin_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
