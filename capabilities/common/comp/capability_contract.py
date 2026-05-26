"""Executable capability contract for APG Compliance Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"frameworks": {
		"enabled": ["soc2", "iso27001", "gdpr", "hipaa", "pci_dss", "sox"],
		"framework_owner_required": True,
		"obligation_mapping_required": True,
		"policy_versioning_enabled": True
	},
	"controls": {
		"control_owner_required": True,
		"testing_frequency_days": 90,
		"automated_control_testing": True,
		"exception_approval_required": True
	},
	"evidence": {
		"evidence_freshness_days": 30,
		"immutable_audit_required": True,
		"encrypted_evidence_required": True,
		"retention_years": 7
	},
	"reporting": {
		"approval_required": True,
		"attestation_required": True,
		"finding_remediation_sla_days": 30,
		"regulatory_export_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_control_changes": True,
		"dlp_for_regulated_data_required": True,
		"role_separation_required": True
	},
	"ui": {
		"enable_compliance_dashboard": True,
		"enable_control_library": True,
		"enable_evidence_vault": True,
		"enable_report_builder": True
	},
	"theme": {
		"default_theme": "comp_compliance_command_center",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "frameworks", "controls", "evidence", "reporting", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["frameworks", "controls", "evidence", "reporting", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All compliance operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "control_requires_owner", "description": "Controls require accountable owners.", "condition": {"operation": "create_control", "control_owner_assigned": False}, "effect": {"decision": "deny", "reason": "control_owner_required", "required_action": "assign_control_owner"}},
	{"name": "stale_evidence_requires_refresh", "description": "Stale evidence requires refresh before attestation.", "condition": {"evidence_age_days_gt": 30, "evidence_refresh_completed": False}, "effect": {"decision": "deny", "reason": "evidence_refresh_required", "required_action": "refresh_evidence"}},
	{"name": "regulated_data_requires_dlp", "description": "Regulated data controls require linked DLP policy evidence.", "condition": {"regulated_data_scope": True, "dlp_policy_linked": False}, "effect": {"decision": "deny", "reason": "dlp_policy_required", "required_action": "link_dlp_policy"}},
	{"name": "report_requires_approval", "description": "Compliance reports require approval before release.", "condition": {"operation": "publish_report", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "record_report_approval"}},
	{"name": "overdue_finding_requires_escalation", "description": "Overdue findings require escalation.", "condition": {"finding_age_days_gt": 30, "escalation_recorded": False}, "effect": {"decision": "require_review", "reason": "finding_escalation_required", "required_action": "escalate_finding"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/comp/dashboard", "component": "COMPDashboard", "permission": "comp:view", "nav_group": "Overview"},
	{"name": "frameworks", "path": "/comp/frameworks", "component": "FrameworkManager", "permission": "comp:manage_controls", "nav_group": "Frameworks"},
	{"name": "controls", "path": "/comp/controls", "component": "ControlLibrary", "permission": "comp:manage_controls", "nav_group": "Controls"},
	{"name": "evidence", "path": "/comp/evidence", "component": "EvidenceVault", "permission": "comp:collect_evidence", "nav_group": "Evidence"},
	{"name": "findings", "path": "/comp/findings", "component": "FindingTracker", "permission": "comp:remediate", "nav_group": "Remediation"},
	{"name": "reports", "path": "/comp/reports", "component": "ReportBuilder", "permission": "comp:approve_reports", "nav_group": "Reporting"},
	{"name": "attestations", "path": "/comp/attestations", "component": "AttestationCenter", "permission": "comp:approve_reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/comp/settings", "component": "COMPSettings", "permission": "comp:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "comp_compliance_command_center",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#805AD5",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"framework_matrix": {"icon": "clipboard-check", "status_indicator": "framework-pill", "risk_style": "coverage-band"},
		"control_card": {"visual": "control-status-stack", "highlight": "owner-chip"},
		"evidence_vault": {"visual": "evidence-list", "status_style": "freshness-chip"},
		"finding_board": {"visual": "remediation-lanes", "status_style": "sla-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable COMP capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "comp",
		"display_name": "Compliance Management",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/comp/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default COMP governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
