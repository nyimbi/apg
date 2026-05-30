"""Screen-model helpers for the Risk and Compliance Management capability."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import GrcRcmService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import GrcRcmService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/grc-rcm/dashboard", "icon": "layout-dashboard"},
	{"name": "Risks", "route": "/grc-rcm/risks", "icon": "shield-alert"},
	{"name": "Controls", "route": "/grc-rcm/controls", "icon": "list-checks"},
	{"name": "Obligations", "route": "/grc-rcm/obligations", "icon": "scroll-text"},
	{"name": "Assessments", "route": "/grc-rcm/assessments", "icon": "clipboard-check"},
	{"name": "Evidence", "route": "/grc-rcm/evidence", "icon": "archive"},
	{"name": "Issues", "route": "/grc-rcm/issues", "icon": "octagon-alert"},
	{"name": "Governance", "route": "/grc-rcm/governance", "icon": "landmark"},
	{"name": "Exceptions", "route": "/grc-rcm/exceptions", "icon": "file-warning"},
	{"name": "Agents", "route": "/grc-rcm/agents", "icon": "bot"},
	{"name": "Settings", "route": "/grc-rcm/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"high_risks": len([record for record in service.risks.values() if record["tenant_id"] == tenant_id and record["risk_level"] in {"high", "critical"}]),
		"open_issues": len([record for record in service.issues.values() if record["tenant_id"] == tenant_id and record["status"] == "open"]),
		"expiring_exceptions": len([record for record in service.exceptions.values() if record["tenant_id"] == tenant_id and record["status"] == "approved"]),
	}
	return model


def risk_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("risks", tenant_id)
	model["records"] = service.list_records("risks", tenant_id)
	model["columns"] = ["title", "category", "owner_id", "likelihood", "impact", "risk_level", "status"]
	return model


def control_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("controls", tenant_id)
	model["records"] = service.list_records("controls", tenant_id)
	model["columns"] = ["name", "owner_id", "control_type", "mapped_risk_ids", "test_frequency_days", "last_assessment_result", "status"]
	return model


def obligation_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("obligations", tenant_id)
	model["records"] = service.list_records("obligations", tenant_id)
	model["columns"] = ["framework", "requirement", "owner_id", "jurisdiction", "due_date", "status"]
	return model


def assessment_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("assessments", tenant_id)
	model["records"] = service.list_records("assessments", tenant_id)
	model["columns"] = ["control_id", "assessor_id", "result", "evidence_ids", "findings", "status"]
	return model


def evidence_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("evidence", tenant_id)
	model["records"] = service.list_records("evidence", tenant_id)
	model["columns"] = ["source", "linked_record_type", "linked_record_id", "encrypted", "retention_days", "status"]
	return model


def issue_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("issues", tenant_id)
	model["records"] = service.list_records("issues", tenant_id)
	model["columns"] = ["title", "severity", "owner_id", "remediation_plan", "linked_assessment_id", "status"]
	return model


def governance_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("governance", tenant_id)
	model["records"] = service.list_records("governance_decisions", tenant_id)
	model["columns"] = ["title", "approver_id", "related_risk_ids", "reviewed_by", "status"]
	return model


def exception_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("exceptions", tenant_id)
	model["records"] = service.list_records("exceptions", tenant_id)
	model["columns"] = ["exception_type", "linked_risk_id", "expiration_date", "approved_by", "status"]
	return model


def agent_workbench_model(service: GrcRcmService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_risk", "review_control", "review_evidence", "review_issue", "review_governance_decision"]
	return model
