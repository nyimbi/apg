"""API helpers for the Federated Learning capability."""

from __future__ import annotations

from typing import Any

from .service import FedlService


SERVICE = FedlService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def create_federation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_federation(
		federation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		coordinator=str(payload.get("coordinator") or "fedl-coordinator"),
		model_family=str(payload.get("model_family") or "tabular"),
		objective_metric=str(payload.get("objective_metric") or "accuracy"),
		privacy_epsilon_limit=float(payload.get("privacy_epsilon_limit") or 8.0),
		data_residency_regions=list(payload.get("data_residency_regions") or []),
	)


def register_participant(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_participant(
		participant_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		federation_id=str(payload["federation_id"]),
		name=str(payload["name"]),
		region=str(payload["region"]),
		contract_ref=str(payload.get("contract_ref") or ""),
		attested=bool(payload.get("attested", False)),
		compute_profile=str(payload.get("compute_profile") or "standard"),
	)


def start_round(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_round(
		round_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		federation_id=str(payload["federation_id"]),
		round_number=int(payload.get("round_number") or 1),
		privacy_epsilon=float(payload.get("privacy_epsilon") or 1.0),
		approval_ref=str(payload.get("approval_ref") or ""),
		secure_aggregation=bool(payload.get("secure_aggregation", True)),
		privacy_review_recorded=bool(payload.get("privacy_review_recorded", True)),
	)


def submit_update(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_update(
		update_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		round_id=str(payload["round_id"]),
		participant_id=str(payload["participant_id"]),
		payload=dict(payload.get("payload") or {}),
		sample_count=int(payload.get("sample_count") or 0),
		quality_score=float(payload.get("quality_score") or 0.0),
		poisoning_signal=bool(payload.get("poisoning_signal", False)),
	)


def aggregate_updates(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.aggregate_updates(
		aggregation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		round_id=str(payload["round_id"]),
		secure_aggregation_enabled=bool(payload.get("secure_aggregation_enabled", False)),
		privacy_review_recorded=bool(payload.get("privacy_review_recorded", True)),
	)


def release_model(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.release_model(
		release_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		mlcm_model_ref=str(payload.get("mlcm_model_ref") or ""),
		release_approval_ref=str(payload.get("release_approval_ref") or ""),
		privacy_review_ref=str(payload.get("privacy_review_ref") or ""),
		artifact_ref=str(payload.get("artifact_ref") or ""),
	)


def retire_federation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_federation(
		federation_id=str(payload["federation_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		impact_review_ref=str(payload.get("impact_review_ref") or ""),
		retired_by=str(payload.get("retired_by") or ""),
	)


def list_federations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_federations(tenant_id)


def list_participants(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_participants(tenant_id)


def list_rounds(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_rounds(tenant_id)


def list_updates(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_updates(tenant_id)


def list_aggregations(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_aggregations(tenant_id)


def list_models(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_models(tenant_id)


def list_releases(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_releases(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)
