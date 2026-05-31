"""Dependency-light FREC API helpers for generated APG applications."""

from __future__ import annotations

from typing import Any

from .face_runtime import FrecGuardrailError, FrecService


SERVICE = FrecService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe()
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		**SERVICE.dashboard_summary(tenant_id),
	}


def record_face_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.record_face_consent(
		consent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		purpose=str(payload["purpose"]),
		evidence=str(payload["evidence"]),
		actor=str(payload.get("actor") or payload["subject_id"]),
	))


def enroll_face(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.enroll_face(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		consent_id=str(payload["consent_id"]),
		template_hash=str(payload["template_hash"]),
		face_quality=float(payload["face_quality"]),
		template_encrypted=_payload_bool(payload, "template_encrypted", True),
		retention_policy=str(payload.get("retention_policy") or "face-template-365d"),
		recapture_completed=_payload_bool(payload, "recapture_completed", True),
	))


def record_liveness(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.record_liveness(
		liveness_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		liveness_score=float(payload["liveness_score"]),
		spoof_detected=_payload_bool(payload, "spoof_detected", False),
		deepfake_detected=_payload_bool(payload, "deepfake_detected", False),
	))


def verify_face(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.verify_face(
		verification_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		template_id=str(payload["template_id"]),
		liveness_id=str(payload["liveness_id"]),
		match_confidence=float(payload["match_confidence"]),
		review_recorded=_payload_bool(payload, "review_recorded", False),
	))


def create_watchlist(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.create_watchlist(
		watchlist_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		policy_id=str(payload["policy_id"]),
		owner=str(payload["owner"]),
		reason=str(payload["reason"]),
	))


def add_watchlist_subject(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.add_watchlist_subject(
		watchlist_id=str(payload["watchlist_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		template_id=str(payload["template_id"]),
		added_by=str(payload["added_by"]),
		reason=str(payload["reason"]),
	))


def identify_face(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.identify_face(
		identification_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		watchlist_id=str(payload["watchlist_id"]),
		candidate_subject_id=str(payload["candidate_subject_id"]),
		identification_confidence=float(payload["identification_confidence"]),
		review_recorded=_payload_bool(payload, "review_recorded", False),
	))


def register_facial_recognition_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.register_facial_recognition_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=_payload_bool(payload, "contribution_disclosed", True),
		human_approval_required=_payload_bool(payload, "human_approval_required", False),
	))


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: SERVICE.validate_frec_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count", 1)),
		operation=str(payload.get("operation") or "facial_recognition_agent_batch"),
		batch_id=payload.get("id"),
	))


def list_facial_recognition_agents(tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_facial_recognition_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def dashboard_payload(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "data": SERVICE.package(tenant_id)}


def _wrap(operation) -> dict[str, Any]:
	try:
		return {"ok": True, "data": operation()}
	except FrecGuardrailError as exc:
		return {"ok": False, "error": exc.result}
	except ValueError as exc:
		return {"ok": False, "error": {"decision": "deny", "reason": str(exc), "required_action": "correct_frec_request"}}


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
