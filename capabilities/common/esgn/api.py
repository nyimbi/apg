"""API helpers for the Digital Forms and eSign capability."""

from __future__ import annotations

from typing import Any

from .service import EsgnService


SERVICE = EsgnService()


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


def create_form_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_template(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		schema_fields=list(payload.get("schema_fields") or payload.get("fields") or []),
		compliance_framework=str(payload.get("compliance_framework") or ""),
		dlp_policy=str(payload.get("dlp_policy") or ""),
		retention_policy=str(payload.get("retention_policy") or ""),
		regulated_form=bool(payload.get("regulated_form", False)),
		compliance_review_recorded=bool(payload.get("compliance_review_recorded", True)),
	)


def publish_form(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_template(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		approved_by=str(payload.get("approved_by") or "forms-admin"),
		publication_approved=bool(payload.get("publication_approved", False)),
	)


def submit_form(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_form(
		submission_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		template_id=str(payload["template_id"]),
		submitted_by=str(payload.get("submitted_by") or "system"),
		data=dict(payload.get("data") or {}),
		evidence_ref=str(payload.get("evidence_ref") or ""),
	)


def create_envelope(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_envelope(
		envelope_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		submission_id=str(payload["submission_id"]),
		subject=str(payload.get("subject") or "Signature request"),
		recipients=list(payload.get("recipients") or []),
		sender=str(payload.get("sender") or "system"),
		signature_intent=str(payload.get("signature_intent") or "approval"),
		compliance_review_recorded=bool(payload.get("compliance_review_recorded", True)),
		document_hash=str(payload.get("document_hash") or ""),
		expires_at=str(payload.get("expires_at") or ""),
	)


def sign_envelope(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.sign_envelope(
		ceremony_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		envelope_id=str(payload["envelope_id"]),
		recipient_id=str(payload["recipient_id"]),
		signature_intent=str(payload.get("signature_intent") or ""),
		identity_verified=bool(payload.get("identity_verified", False)),
		signature_intent_recorded=bool(payload.get("signature_intent_recorded", True)),
		signed_at=payload.get("signed_at"),
	)


def create_evidence_package(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_evidence_package(
		evidence_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		envelope_id=str(payload["envelope_id"]),
		encrypted=bool(payload.get("encrypted", False)),
		retention_policy=str(payload.get("retention_policy") or ""),
		audit_trail_ref=str(payload.get("audit_trail_ref") or ""),
	)


def cancel_envelope(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.cancel_envelope(
		envelope_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload.get("actor") or "system"),
		reason=str(payload.get("reason") or ""),
	)


def reject_envelope(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.reject_envelope(
		envelope_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		recipient_id=str(payload["recipient_id"]),
		reason=str(payload.get("reason") or ""),
	)


def register_signing_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_signing_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or ""),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope_ref=str(payload.get("scope_ref") or ""),
		registered_by=str(payload.get("registered_by") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", False)),
	)


def verify_tamper_seal(payload: dict[str, Any]) -> dict[str, Any]:
	envelope_id = str(payload["id"])
	tenant_id = str(payload.get("tenant_id") or "default")
	return {"id": envelope_id, "tenant_id": tenant_id, "valid": SERVICE.verify_tamper_seal(envelope_id, tenant_id)}


def validate_batch_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_mutation(str(payload.get("event_stream") or ""))


def list_form_templates(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_templates(tenant_id)


def list_submissions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_submissions(tenant_id)


def list_envelopes(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_envelopes(tenant_id)


def list_signing_ceremonies(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_ceremonies(tenant_id)


def list_evidence_packages(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_evidence_packages(tenant_id)


def list_signing_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_signing_agents(tenant_id)


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
