"""Dependency-light API helpers for Risk and Compliance Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import GrcRcmService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import GrcRcmService  # type: ignore


_SERVICE = GrcRcmService()


def service() -> GrcRcmService:
	"""Return the process-local RCM service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "grc_rcm", "summary": _SERVICE.dashboard_summary(tenant_id)}


def register_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_risk(
		payload.get("risk_id", payload.get("id", "risk")),
		payload["tenant_id"],
		payload["title"],
		payload.get("category", "operational"),
		payload["owner_id"],
		float(payload.get("likelihood", payload.get("probability", 0.2))),
		float(payload["impact"]),
		payload.get("reviewed_by"),
		payload.get("metadata", {}),
	)


def register_control(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_control(
		payload.get("control_id", payload.get("id", "control")),
		payload["tenant_id"],
		payload["name"],
		payload["owner_id"],
		payload.get("control_type", "preventive"),
		list(payload.get("mapped_risk_ids") or []),
		int(payload.get("test_frequency_days", 90)),
	)


def register_obligation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_obligation(
		payload.get("obligation_id", payload.get("id", "obligation")),
		payload["tenant_id"],
		payload["framework"],
		payload["requirement"],
		payload["owner_id"],
		payload.get("jurisdiction", "global"),
		payload["due_date"],
		list(payload.get("mapped_control_ids") or []),
	)


def assess_control(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.assess_control(
		payload.get("assessment_id", payload.get("id", "assessment")),
		payload["tenant_id"],
		payload["control_id"],
		payload["assessor_id"],
		payload.get("result", "effective"),
		list(payload.get("evidence_ids") or []),
		list(payload.get("findings") or []),
	)


def collect_evidence(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.collect_evidence(
		payload.get("evidence_id", payload.get("id", "evidence")),
		payload["tenant_id"],
		payload["source"],
		payload["linked_record_type"],
		payload["linked_record_id"],
		bool(payload.get("encrypted", True)),
		int(payload.get("retention_days", 2555)),
	)


def open_issue(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_issue(
		payload.get("issue_id", payload.get("id", "issue")),
		payload["tenant_id"],
		payload["title"],
		payload.get("severity", "medium"),
		payload["owner_id"],
		payload["remediation_plan"],
		payload.get("linked_assessment_id"),
		payload.get("reviewed_by"),
	)


def remediate_issue(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.remediate_issue(payload["issue_id"], payload["tenant_id"], payload["remediation_evidence_id"])


def record_governance_decision(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_governance_decision(
		payload.get("decision_id", payload.get("id", "decision")),
		payload["tenant_id"],
		payload["title"],
		payload["approver_id"],
		payload["rationale"],
		list(payload.get("related_risk_ids") or []),
		payload.get("reviewed_by"),
	)


def register_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_exception(
		payload.get("exception_id", payload.get("id", "exception")),
		payload["tenant_id"],
		payload["exception_type"],
		payload["linked_risk_id"],
		payload["expiration_date"],
		payload["approved_by"],
	)


def register_rcm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_rcm_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review risk and compliance operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return _SERVICE.create_record(
		payload.get("id", "api-risk"),
		payload["tenant_id"],
		{
			"title": payload.get("title", "API Risk"),
			"owner_id": payload.get("owner_id", "api-owner"),
			"category": payload.get("category", "operational"),
			"likelihood": payload.get("likelihood", 0.2),
			"impact": payload.get("impact", 0.2),
		},
	)


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
