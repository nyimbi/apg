"""UI metadata and view models for APG Accessibility Services."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import AccsService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.compliance_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"targets": service.list_targets(tenant_id),
		"findings": service.list_findings(tenant_id),
		"remediations": service.list_remediations(tenant_id),
		"reviews": service.list_reviews(tenant_id),
		"accessibility_exceptions": service.list_accessibility_exceptions(tenant_id),
		"accessibility_agents": service.list_accessibility_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def audit_console_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	return {
		"standards": service.list_standards(tenant_id),
		"targets": service.list_targets(tenant_id),
		"audits": service.list_audits(tenant_id),
		"audit_fields": ["id", "standard_id", "target_ids", "remediation_owner"],
	}


def findings_board_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	findings = service.list_findings(tenant_id)
	return {
		"findings": findings,
		"columns": ["critical", "high", "medium", "low"],
		"status_groups": ["open", "in_progress", "blocked", "closed"],
	}


def remediation_queue_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	return {
		"remediations": service.list_remediations(tenant_id),
		"reviews": service.list_reviews(tenant_id),
		"accessibility_exceptions": service.list_accessibility_exceptions(tenant_id),
		"actions": ["assign", "start", "record_review", "close"],
		"required_fields": ["owner", "status", "review_recorded"],
	}


def review_queue_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	approved_review_finding_ids = {
		item["finding_id"] for item in service.list_reviews(tenant_id)
		if item["decision"] == "approved"
	}
	findings = [
		item for item in service.list_findings(tenant_id)
		if item.get("review_required") and item["id"] not in approved_review_finding_ids
	]
	return {
		"findings_requiring_review": findings,
		"recorded_reviews": service.list_reviews(tenant_id),
		"actions": ["approve", "reject", "needs_work"],
		"required_fields": ["reviewer", "decision", "notes"],
	}


def compliance_evidence_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	return {
		"summary": service.compliance_summary(tenant_id),
		"audits": service.list_audits(tenant_id),
		"findings": service.list_findings(tenant_id),
		"reviews": service.list_reviews(tenant_id),
		"accessibility_exceptions": service.list_accessibility_exceptions(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def accessibility_exceptions_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	return {
		"accessibility_exceptions": service.list_accessibility_exceptions(tenant_id),
		"open_findings": [item for item in service.list_findings(tenant_id) if item["status"] != "closed"],
		"actions": ["record_exception", "review_expiry", "revoke"],
		"required_fields": ["approver", "reason", "expires_on", "compensating_controls"],
	}


def accessibility_agents_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	contract = service.describe(tenant_id)
	return {
		"agents": service.list_accessibility_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["accessibility_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["accessibility_agents"]["allowed_roles"],
		"actions": ["register", "scope", "review_contribution", "deactivate"],
		"required_fields": ["name", "runtime", "role", "scope", "contribution_disclosed"],
	}


def audit_trail_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	return {
		"audit_events": service.list_audit_events(tenant_id),
		"event_types": [
			"standard_registered",
			"target_registered",
			"audit_completed",
			"finding_recorded",
			"remediation_updated",
			"finding_review_recorded",
			"finding_closed",
			"accessibility_exception_recorded",
			"accessibility_agent_registered",
		],
	}


def analytics_model(
	service: AccsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AccsService()
	summary = service.compliance_summary(tenant_id)
	return {
		"summary": summary,
		"closure_rate": _safe_ratio(summary["finding_count"] - summary["open_finding_count"], summary["finding_count"]),
		"review_rate": _safe_ratio(summary["review_count"], summary["critical_or_high_count"]),
		"exception_rate": _safe_ratio(summary["exception_count"], summary["open_finding_count"]),
		"agent_coverage": _safe_ratio(summary["accessibility_agent_count"], max(summary["target_count"], 1)),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def assistive_preview_model(target: dict[str, object]) -> dict[str, object]:
	return {
		"target_id": target["id"],
		"surface": target["surface"],
		"semantic_tree_ready": bool(target.get("semantic_labels_present")),
		"keyboard_ready": bool(target.get("keyboard_navigation_present")),
		"media_ready": not target.get("media_content_present") or bool(target.get("captions_available")),
		"preview_sections": ["landmarks", "labels", "keyboard_order", "media_alternatives"],
	}


def _safe_ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
