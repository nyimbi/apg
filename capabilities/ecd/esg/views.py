"""View models for Sustainability and ESG Management."""

from __future__ import annotations

from typing import Any


def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Sustainability and ESG",
		"tenant_id": tenant_id,
		"cards": [
			{"label": "Profiles", "value": summary["profile_count"], "icon": "building-2"},
			{"label": "Metrics", "value": summary["metric_count"], "icon": "ruler"},
			{"label": "Measurements", "value": summary["measurement_count"], "icon": "database"},
			{"label": "Targets", "value": summary["target_count"], "icon": "target"},
			{"label": "Reports", "value": summary["report_count"], "icon": "file-text"},
			{"label": "Agents", "value": summary["agent_count"], "icon": "bot"},
		],
		"streaming": summary["streaming"],
	}


def profile_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "ESG Profiles", "records": service.list_records(tenant_id, "profile")}


def framework_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Frameworks", "records": service.list_records(tenant_id, "framework")}


def metric_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Metrics", "records": service.list_records(tenant_id, "metric")}


def measurement_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Measurements", "records": service.list_records(tenant_id, "measurement")}


def target_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Targets", "records": service.list_records(tenant_id, "target")}


def report_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Reports", "records": service.list_records(tenant_id, "report")}


def stakeholder_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Stakeholders", "records": service.list_records(tenant_id, "stakeholder")}


def agent_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "ESG Agents", "records": service.list_records(tenant_id, "agent"), "policy": {"max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True}}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "ESG Rules", "rule_count": len(contract["rule_engine"]["rules"]), "rules": contract["rule_engine"]["rules"]}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {"name": "ESG Settings", "configuration": contract["configuration"], "theme": contract["theme"], "routes": contract["ui"]["routes"]}
