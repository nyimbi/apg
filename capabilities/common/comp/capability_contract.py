"""Executable capability contract for APG Compliance Management."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"frameworks": {
		"enabled": ["soc2", "iso27001", "gdpr", "hipaa", "pci_dss", "sox"],
		"framework_owner_required": True,
		"obligation_mapping_required": True,
		"policy_versioning_enabled": True,
		"duplicate_framework_blocked": True,
	},
	"controls": {
		"control_owner_required": True,
		"testing_frequency_days": 90,
		"automated_control_testing": True,
		"exception_approval_required": True,
		"independent_testing_required": True,
		"owner_separation_required": True,
	},
	"evidence": {
		"evidence_freshness_days": 30,
		"immutable_audit_required": True,
		"encrypted_evidence_required": True,
		"retention_years": 7,
		"source_required": True,
		"collector_required": True,
	},
	"assessments": {
		"tester_required": True,
		"fresh_evidence_required": True,
		"failed_control_opens_finding": True,
		"assessment_audit_required": True,
	},
	"findings": {
		"owner_required": True,
		"remediation_plan_required_for_high": True,
		"sla_days": 30,
		"escalation_required": True,
		"resolution_evidence_required": True,
	},
	"reporting": {
		"approval_required": True,
		"attestation_required": True,
		"finding_remediation_sla_days": 30,
		"regulatory_export_enabled": True,
		"open_critical_findings_block_publish": True,
		"independent_approval_required": True,
	},
	"exceptions": {
		"exception_owner_required": True,
		"exception_expiry_required": True,
		"expired_exception_blocks_report": True,
	},
	"security": {
		"tenant_isolation_required": True,
		"encrypted_evidence_required": True,
		"dlp_for_regulated_data_required": True,
		"audit_required": True,
		"role_separation_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_control_changes": True,
		"dlp_for_regulated_data_required": True,
		"role_separation_required": True,
		"policy_mutation_audit_required": True,
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.CompService",
		"helper_runtime": "compliance_engine.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"data_loss_prevention": "dlpd",
		"encryption": "encr",
		"authentication": "auth",
		"security_framework": "secu",
		"multi_tenancy": "mten",
		"identity_federation": "idfd",
		"zero_trust_access": "ztna",
		"document_management": "docm",
		"workflow": "wflo",
		"notification": "ntfy",
		"message_bus": "mqeb",
		"cache": "cach",
	},
	"ui": {
		"enable_compliance_dashboard": True,
		"enable_framework_manager": True,
		"enable_control_library": True,
		"enable_evidence_vault": True,
		"enable_assessment_workbench": True,
		"enable_finding_tracker": True,
		"enable_exception_register": True,
		"enable_report_builder": True,
		"enable_attestation_center": True,
		"enable_regulatory_exports": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "comp_compliance_command_center", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"frameworks",
		"controls",
		"evidence",
		"assessments",
		"findings",
		"reporting",
		"exceptions",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"frameworks",
		"controls",
		"evidence",
		"assessments",
		"findings",
		"reporting",
		"exceptions",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All compliance operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "framework_requires_owner", "description": "Framework registration requires an accountable owner.", "condition": {"operation": "register_framework", "framework_owner_assigned": False}, "effect": {"decision": "deny", "reason": "framework_owner_required", "required_action": "assign_framework_owner"}},
	{"name": "framework_requires_obligations", "description": "Frameworks require mapped obligations.", "condition": {"operation": "register_framework", "obligations_mapped": False}, "effect": {"decision": "deny", "reason": "obligation_mapping_required", "required_action": "map_framework_obligations"}},
	{"name": "framework_requires_policy_version", "description": "Frameworks require policy version evidence.", "condition": {"operation": "register_framework", "policy_version_present": False}, "effect": {"decision": "deny", "reason": "policy_version_required", "required_action": "attach_policy_version"}},
	{"name": "duplicate_framework_blocked", "description": "Duplicate tenant framework keys are blocked.", "condition": {"operation": "register_framework", "duplicate_framework": True}, "effect": {"decision": "deny", "reason": "duplicate_framework", "required_action": "use_existing_framework"}},
	{"name": "control_requires_framework", "description": "Controls require a tenant-local framework.", "condition": {"operation": "create_control", "framework_present": False}, "effect": {"decision": "deny", "reason": "control_framework_required", "required_action": "select_framework"}},
	{"name": "control_requires_name", "description": "Controls require a name.", "condition": {"operation": "create_control", "control_name_present": False}, "effect": {"decision": "deny", "reason": "control_name_required", "required_action": "name_control"}},
	{"name": "control_requires_owner", "description": "Controls require accountable owners.", "condition": {"operation": "create_control", "control_owner_assigned": False}, "effect": {"decision": "deny", "reason": "control_owner_required", "required_action": "assign_control_owner"}},
	{"name": "control_frequency_requires_positive_days", "description": "Control testing frequency must be positive.", "condition": {"operation": "create_control", "testing_frequency_days_lte": 0}, "effect": {"decision": "deny", "reason": "control_testing_frequency_required", "required_action": "set_testing_frequency"}},
	{"name": "regulated_data_requires_dlp", "description": "Regulated data controls require linked DLP policy evidence.", "condition": {"regulated_data_scope": True, "dlp_policy_linked": False}, "effect": {"decision": "deny", "reason": "dlp_policy_required", "required_action": "link_dlp_policy"}},
	{"name": "evidence_requires_control", "description": "Evidence requires a tenant-local control.", "condition": {"operation": "record_evidence", "control_present": False}, "effect": {"decision": "deny", "reason": "evidence_control_required", "required_action": "select_control"}},
	{"name": "evidence_requires_source", "description": "Evidence records require a source.", "condition": {"operation": "record_evidence", "evidence_source_present": False}, "effect": {"decision": "deny", "reason": "evidence_source_required", "required_action": "record_evidence_source"}},
	{"name": "evidence_requires_collector", "description": "Evidence records require a collector.", "condition": {"operation": "record_evidence", "evidence_collector_present": False}, "effect": {"decision": "deny", "reason": "evidence_collector_required", "required_action": "record_collector"}},
	{"name": "evidence_requires_encryption", "description": "Compliance evidence must be encrypted.", "condition": {"operation": "record_evidence", "evidence_encrypted": False}, "effect": {"decision": "deny", "reason": "encrypted_evidence_required", "required_action": "encrypt_evidence"}},
	{"name": "evidence_requires_immutable_reference", "description": "Compliance evidence requires immutable reference metadata.", "condition": {"operation": "record_evidence", "immutable_reference_present": False}, "effect": {"decision": "deny", "reason": "immutable_evidence_reference_required", "required_action": "record_immutable_reference"}},
	{"name": "stale_evidence_requires_refresh", "description": "Stale evidence requires refresh before attestation.", "condition": {"evidence_age_days_gt": 30, "evidence_refresh_completed": False}, "effect": {"decision": "deny", "reason": "evidence_refresh_required", "required_action": "refresh_evidence"}},
	{"name": "assessment_requires_tester", "description": "Control assessment requires a tester.", "condition": {"operation": "assess_control", "tester_present": False}, "effect": {"decision": "deny", "reason": "control_tester_required", "required_action": "assign_control_tester"}},
	{"name": "assessment_requires_independent_tester", "description": "Control owner cannot self-test the control.", "condition": {"operation": "assess_control", "tester_is_control_owner": True}, "effect": {"decision": "require_review", "reason": "independent_control_test_required", "required_action": "route_to_independent_tester"}},
	{"name": "failed_assessment_requires_finding", "description": "Failed assessments require finding linkage.", "condition": {"operation": "assess_control", "assessment_failed": True, "finding_linked": False}, "effect": {"decision": "require_review", "reason": "assessment_finding_required", "required_action": "open_control_finding"}},
	{"name": "finding_requires_owner", "description": "Findings require accountable owners.", "condition": {"operation": "open_finding", "finding_owner_assigned": False}, "effect": {"decision": "deny", "reason": "finding_owner_required", "required_action": "assign_finding_owner"}},
	{"name": "finding_requires_description", "description": "Findings require a description.", "condition": {"operation": "open_finding", "finding_description_present": False}, "effect": {"decision": "deny", "reason": "finding_description_required", "required_action": "describe_finding"}},
	{"name": "high_severity_finding_requires_plan", "description": "High and critical findings require remediation plans.", "condition": {"operation": "open_finding", "high_severity_finding": True, "remediation_plan_present": False}, "effect": {"decision": "require_review", "reason": "remediation_plan_required", "required_action": "record_remediation_plan"}},
	{"name": "overdue_finding_requires_escalation", "description": "Overdue findings require escalation.", "condition": {"finding_age_days_gt": 30, "escalation_recorded": False}, "effect": {"decision": "require_review", "reason": "finding_escalation_required", "required_action": "escalate_finding"}},
	{"name": "finding_resolution_requires_evidence", "description": "Resolved findings require evidence.", "condition": {"operation": "resolve_finding", "resolution_evidence_present": False}, "effect": {"decision": "deny", "reason": "finding_resolution_evidence_required", "required_action": "attach_resolution_evidence"}},
	{"name": "report_requires_framework", "description": "Reports require tenant-local frameworks.", "condition": {"operation": "prepare_report", "framework_present": False}, "effect": {"decision": "deny", "reason": "report_framework_required", "required_action": "select_framework"}},
	{"name": "report_requires_period", "description": "Reports require a reporting period.", "condition": {"operation": "prepare_report", "report_period_present": False}, "effect": {"decision": "deny", "reason": "report_period_required", "required_action": "set_report_period"}},
	{"name": "report_requires_preparer", "description": "Reports require a preparer.", "condition": {"operation": "prepare_report", "report_preparer_present": False}, "effect": {"decision": "deny", "reason": "report_preparer_required", "required_action": "assign_report_preparer"}},
	{"name": "report_approval_requires_independent_approver", "description": "Report approver must differ from preparer.", "condition": {"operation": "approve_report", "approver_is_preparer": True}, "effect": {"decision": "deny", "reason": "independent_report_approval_required", "required_action": "route_report_to_independent_approver"}},
	{"name": "attestation_requires_statement", "description": "Attestations require a statement.", "condition": {"operation": "attest_report", "attestation_statement_present": False}, "effect": {"decision": "deny", "reason": "attestation_statement_required", "required_action": "record_attestation_statement"}},
	{"name": "attestation_requires_approved_report", "description": "Attestations require approved reports.", "condition": {"operation": "attest_report", "report_approved": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "approve_report"}},
	{"name": "report_requires_approval", "description": "Compliance reports require approval before release.", "condition": {"operation": "publish_report", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "record_report_approval"}},
	{"name": "report_requires_attestation", "description": "Compliance reports require attestation before release.", "condition": {"operation": "publish_report", "attestation_recorded": False}, "effect": {"decision": "deny", "reason": "report_attestation_required", "required_action": "record_report_attestation"}},
	{"name": "critical_findings_block_publish", "description": "Open critical findings block report publishing.", "condition": {"operation": "publish_report", "open_critical_findings": True}, "effect": {"decision": "deny", "reason": "critical_findings_open", "required_action": "resolve_critical_findings"}},
	{"name": "cross_tenant_compliance_access_denied", "description": "Compliance records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_compliance_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "compliance_state_change_requires_audit", "description": "Compliance state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "compliance_audit_event_required", "required_action": "record_compliance_audit"}},
	{"name": "batch_compliance_mutation_requires_bytewax", "description": "Batch compliance mutations must use Bytewax event streams.", "condition": {"operation": "batch_compliance_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/comp/dashboard", "component": "COMPDashboard", "permission": "comp:view", "nav_group": "Overview"},
	{"name": "frameworks", "path": "/comp/frameworks", "component": "FrameworkManager", "permission": "comp:manage_controls", "nav_group": "Frameworks"},
	{"name": "controls", "path": "/comp/controls", "component": "ControlLibrary", "permission": "comp:manage_controls", "nav_group": "Controls"},
	{"name": "evidence", "path": "/comp/evidence", "component": "EvidenceVault", "permission": "comp:collect_evidence", "nav_group": "Evidence"},
	{"name": "assessments", "path": "/comp/assessments", "component": "AssessmentWorkbench", "permission": "comp:manage_controls", "nav_group": "Assurance"},
	{"name": "findings", "path": "/comp/findings", "component": "FindingTracker", "permission": "comp:remediate", "nav_group": "Remediation"},
	{"name": "exceptions", "path": "/comp/exceptions", "component": "ExceptionRegister", "permission": "comp:remediate", "nav_group": "Remediation"},
	{"name": "reports", "path": "/comp/reports", "component": "ReportBuilder", "permission": "comp:approve_reports", "nav_group": "Reporting"},
	{"name": "attestations", "path": "/comp/attestations", "component": "AttestationCenter", "permission": "comp:approve_reports", "nav_group": "Reporting"},
	{"name": "exports", "path": "/comp/exports", "component": "RegulatoryExportCenter", "permission": "comp:approve_reports", "nav_group": "Reporting"},
	{"name": "audit", "path": "/comp/audit", "component": "ComplianceAuditTrail", "permission": "comp:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/comp/settings", "component": "COMPSettings", "permission": "comp:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"framework_matrix": {"icon": "clipboard-check", "status_indicator": "framework-pill", "risk_style": "coverage-band"},
		"control_card": {"visual": "control-status-stack", "highlight": "owner-chip"},
		"evidence_vault": {"visual": "evidence-list", "status_style": "freshness-chip"},
		"assessment_workbench": {"visual": "testing-queue", "status_style": "assurance-chip"},
		"finding_board": {"visual": "remediation-lanes", "status_style": "sla-chip"},
		"exception_register": {"visual": "exception-table", "status_style": "expiry-chip"},
		"report_builder": {"visual": "report-stage-list", "status_style": "approval-chip"},
		"attestation_center": {"visual": "attestation-list", "status_style": "signature-chip"},
		"regulatory_export": {"visual": "export-table", "status_style": "delivery-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "hash-chip"},
	},
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
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/comp/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
