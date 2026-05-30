"""View models for the SCM Vendor Management capability."""

from __future__ import annotations

from typing import Any


def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Vendor Management",
		"tenant_id": tenant_id,
		"cards": [
			{"label": "Vendors", "value": summary["vendor_count"], "icon": "building-2"},
			{"label": "Performance", "value": summary["performance_count"], "icon": "activity"},
			{"label": "Risks", "value": summary["risk_count"], "icon": "shield-alert"},
			{"label": "Compliance", "value": summary["compliance_count"], "icon": "clipboard-check"},
			{"label": "Contracts", "value": summary["contract_count"], "icon": "file-signature"},
			{"label": "Agents", "value": summary["agent_count"], "icon": "bot"},
		],
		"streaming": summary["streaming"],
	}


def vendor_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Vendors", "records": service.list_records(tenant_id, "vendor")}


def qualification_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Qualification", "records": service.list_records(tenant_id, "qualification")}


def onboarding_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Onboarding", "records": service.list_records(tenant_id, "onboarding")}


def performance_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Performance", "records": service.list_records(tenant_id, "performance")}


def risk_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Risk", "records": service.list_records(tenant_id, "risk")}


def compliance_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Compliance", "records": service.list_records(tenant_id, "compliance")}


def contract_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Contracts", "records": service.list_records(tenant_id, "contract")}


def portal_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Portal Users", "records": service.list_records(tenant_id, "portal_user")}


def scorecard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Scorecards", "records": service.list_records(tenant_id, "scorecard")}


def agent_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Vendor Agents", "records": service.list_records(tenant_id, "agent"), "policy": {"max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True}}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "Vendor Rules", "rule_count": len(contract["rule_engine"]["rules"]), "rules": contract["rule_engine"]["rules"]}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "Vendor Settings", "configuration": contract["configuration"], "theme": contract["theme"], "routes": contract["ui"]["routes"]}
