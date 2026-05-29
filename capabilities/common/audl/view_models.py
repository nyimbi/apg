"""Dependency-light AUDL view models for package-composed UIs."""

from __future__ import annotations

from .audit_runtime import AudlService
from .capability_contract import get_capability_contract


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	contract = get_capability_contract(tenant_id)
	return [
		{
			"name": route["name"],
			"path": route["path"],
			"component": route["component"],
			"permission": route["permission"],
			"nav_group": route["nav_group"],
		}
		for route in contract["ui"]["routes"]
	]


def dashboard_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	contract = get_capability_contract(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.audit_summary(tenant_id),
		"events": service.list_events(tenant_id),
		"legal_holds": service.list_legal_holds(tenant_id),
		"exports": service.list_exports(tenant_id),
		"purges": service.list_purges(tenant_id),
		"investigations": service.list_investigations(tenant_id),
		"governance_events": service.list_governance_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def event_explorer_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"events": service.list_events(tenant_id),
		"filters": ["actor", "action", "resource_type", "resource_id", "severity", "contains_pii"],
		"required_fields": ["id", "actor", "action", "resource_type", "resource_id"],
	}


def timeline_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"events": sorted(service.list_events(tenant_id), key=lambda item: item["timestamp"]),
		"governance_events": service.list_governance_events(tenant_id),
	}


def legal_hold_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"legal_holds": service.list_legal_holds(tenant_id),
		"required_fields": ["id", "scope", "reason", "approver"],
		"release_required_fields": ["released_by", "release_evidence"],
	}


def export_review_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"exports": service.list_exports(tenant_id),
		"pending_exports": [item for item in service.list_exports(tenant_id) if item["decision"] == "pending"],
		"required_controls": ["masking_enabled", "reviewer", "notes"],
	}


def purge_review_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"purges": service.list_purges(tenant_id),
		"pending_purges": [item for item in service.list_purges(tenant_id) if item["decision"] == "pending"],
		"required_controls": ["dual_control_reviewer", "notes", "legal_hold_check"],
	}


def investigation_workbench_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"investigations": service.list_investigations(tenant_id),
		"events": service.list_events(tenant_id),
		"required_fields": ["id", "event_ids", "owner"],
		"closure_required_fields": ["closed_by", "resolution", "evidence"],
	}


def compliance_center_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	contract = get_capability_contract(tenant_id)
	return {
		"summary": service.audit_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"frameworks": contract["configuration"]["compliance"]["enabled_frameworks"],
		"legal_holds": service.list_legal_holds(tenant_id),
		"exports": service.list_exports(tenant_id),
		"purges": service.list_purges(tenant_id),
	}


def reporting_studio_model(
	service: AudlService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudlService()
	return {
		"summary": service.audit_summary(tenant_id),
		"exports": service.list_exports(tenant_id),
		"report_types": ["chain_of_custody", "pii_export", "retention", "investigation"],
	}


def rule_workbench_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"rules": contract["rule_engine"]["rules"],
		"configuration_schema": contract["configuration_schema"],
		"theme": contract["theme"],
	}
