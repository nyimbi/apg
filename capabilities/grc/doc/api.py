"""Dependency-light API helpers for GRC Document Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import GrcDocService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import GrcDocService  # type: ignore


SERVICE = GrcDocService()


def service() -> GrcDocService:
	"""Return the process-local document service."""
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def create_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_document(
		payload.get("document_id", payload.get("id", "document")),
		payload["tenant_id"],
		payload["title"],
		payload["owner_id"],
		payload.get("content"),
		payload.get("document_type", "record"),
		payload.get("classification", "internal"),
		payload.get("template_id"),
		payload.get("reviewed_by"),
		payload.get("metadata", {}),
	)


def register_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_template(
		payload.get("template_id", payload.get("id", "template")),
		payload["tenant_id"],
		payload["name"],
		payload["body"],
		payload["owner_id"],
		payload.get("classification", "internal"),
	)


def create_revision(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_revision(
		payload.get("revision_id", payload.get("id", "revision")),
		payload["tenant_id"],
		payload["document_id"],
		payload["editor_id"],
		payload["content"],
		payload["change_summary"],
		payload.get("reviewed_by"),
	)


def approve_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_document(
		payload["document_id"],
		payload["tenant_id"],
		payload["approver_id"],
		payload["approval_note"],
	)


def publish_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_document(payload["document_id"], payload["tenant_id"], payload["published_by"])


def assign_retention_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assign_retention_policy(
		payload.get("policy_id", payload.get("id", "retention")),
		payload["tenant_id"],
		payload["document_id"],
		int(payload.get("retention_days", 2555)),
		bool(payload.get("legal_hold", False)),
	)


def grant_access(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.grant_access(
		payload.get("grant_id", payload.get("id", "grant")),
		payload["tenant_id"],
		payload["document_id"],
		payload["principal_id"],
		payload.get("permission", "view"),
		payload.get("expires_on"),
	)


def register_processing_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_processing_job(
		payload.get("job_id", payload.get("id", "job")),
		payload["tenant_id"],
		payload["document_id"],
		payload.get("job_type", "classification"),
		payload.get("processor", "bytewax"),
	)


def complete_processing_job(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_processing_job(payload["job_id"], payload["tenant_id"], payload.get("result", {}))


def register_doc_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_doc_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review document governance operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package probes."""
	return SERVICE.create_record(
		str(payload.get("id", "api-document")),
		str(payload.get("tenant_id") or "default"),
		{
			"title": payload.get("title", "API Document"),
			"owner_id": payload.get("owner_id", "api-owner"),
			"content": payload.get("content", "API document content"),
			"document_type": payload.get("document_type", "record"),
			"classification": payload.get("classification", "internal"),
			"reviewed_by": payload.get("reviewed_by"),
		},
		str(payload.get("status") or "draft"),
	)


def list_records(collection: str | None = None, tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
