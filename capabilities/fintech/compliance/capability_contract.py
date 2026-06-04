"""Executable capability contract for APG FinTech Compliance Automation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_compliance"
CAPABILITY_NAME = "FinTech Compliance Automation"
CAPABILITY_VERSION = "1.1.0"
COMPLIANCE_EVENT_STREAM = "apg.fintech.compliance.lifecycle"

SUPPORTED_REGULATORY_FRAMEWORKS = ["pci_dss", "psd2", "open_banking", "gdpr", "sox", "basel_iii", "mifid_ii", "aml", "kyc", "data_privacy"]
SUPPORTED_OBLIGATION_TYPES = ["policy", "control", "reporting", "retention", "disclosure", "monitoring", "approval", "training"]
SUPPORTED_CONTROL_TYPES = ["preventive", "detective", "corrective", "automated", "manual", "compensating"]
SUPPORTED_CHECK_TYPES = ["transaction", "customer", "account", "merchant", "policy", "control", "report", "agent_action"]
SUPPORTED_EVIDENCE_TYPES = ["document", "transaction_sample", "control_log", "attestation", "system_export", "approval_record", "training_record"]
SUPPORTED_STATUSES = ["draft", "active", "compliant", "non_compliant", "waived", "remediated", "closed"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_REPORT_TYPES = ["regulatory", "board", "audit", "management", "exception", "incident"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["obligation_reviewer", "control_testing_agent", "evidence_reviewer", "attestation_reviewer", "regulatory_report_reviewer", "issue_remediation_agent"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"obligations": {"supported_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS, "supported_types": SUPPORTED_OBLIGATION_TYPES, "owner_required": True, "evidence_required": True, "effective_date_required": True},
	"controls": {"supported_types": SUPPORTED_CONTROL_TYPES, "obligation_required": True, "owner_required": True, "evidence_required": True, "frequency_required": True},
	"checks": {"supported_types": SUPPORTED_CHECK_TYPES, "obligation_required": True, "control_required": True, "subject_required": True, "result_required": True, "evidence_required_for_failure": True},
	"evidence": {"supported_types": SUPPORTED_EVIDENCE_TYPES, "reference_required": True, "retention_required": True, "source_required": True},
	"attestations": {"obligation_required": True, "attestor_required": True, "status_required": True, "evidence_required": True},
	"issues": {"supported_severities": SUPPORTED_SEVERITIES, "obligation_required": True, "owner_required": True, "evidence_required": True, "due_date_required": True},
	"remediation": {"issue_required": True, "owner_required": True, "plan_required": True, "approval_required_for_high_impact": True},
	"reports": {"supported_types": SUPPORTED_REPORT_TYPES, "framework_required": True, "period_required": True, "evidence_required": True, "approver_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": COMPLIANCE_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "risk": "fintech_risk", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_obligations": True, "enable_controls": True, "enable_checks": True, "enable_evidence": True, "enable_attestations": True, "enable_issues": True, "enable_reports": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_compliance_control", "allow_tenant_overrides": True},
}

PROVIDES = ["compliance_obligation_workflow", "compliance_control_workflow", "compliance_check_workflow", "compliance_evidence_workflow", "compliance_attestation_workflow", "compliance_issue_workflow", "compliance_remediation_workflow", "compliance_report_workflow", "compliance_review_workflow", "compliance_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_risk", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-compliance/dashboard", "component": "ComplianceDashboard", "permission": "fintech_compliance:view", "nav_group": "Overview"},
	{"name": "obligations", "path": "/fintech-compliance/obligations", "component": "ObligationConsole", "permission": "fintech_compliance:obligations", "nav_group": "Obligations"},
	{"name": "controls", "path": "/fintech-compliance/controls", "component": "ComplianceControlConsole", "permission": "fintech_compliance:controls", "nav_group": "Controls"},
	{"name": "checks", "path": "/fintech-compliance/checks", "component": "ComplianceCheckWorkbench", "permission": "fintech_compliance:checks", "nav_group": "Testing"},
	{"name": "evidence", "path": "/fintech-compliance/evidence", "component": "ComplianceEvidenceVault", "permission": "fintech_compliance:evidence", "nav_group": "Evidence"},
	{"name": "attestations", "path": "/fintech-compliance/attestations", "component": "AttestationConsole", "permission": "fintech_compliance:attestations", "nav_group": "Governance"},
	{"name": "issues", "path": "/fintech-compliance/issues", "component": "ComplianceIssueQueue", "permission": "fintech_compliance:issues", "nav_group": "Issues"},
	{"name": "reports", "path": "/fintech-compliance/reports", "component": "ComplianceReportConsole", "permission": "fintech_compliance:reports", "nav_group": "Reporting"},
	{"name": "reviews", "path": "/fintech-compliance/reviews", "component": "ComplianceReviewConsole", "permission": "fintech_compliance:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-compliance/agents", "component": "ComplianceAgentWorkbench", "permission": "fintech_compliance:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-compliance/settings", "component": "ComplianceSettings", "permission": "fintech_compliance:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_compliance_control",
	"tokens": {"color.primary": "#155E75", "color.accent": "#4F46E5", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"obligations": {"icon": "scroll-text", "status_indicator": "obligation-chip"}, "controls": {"icon": "shield-check", "status_indicator": "control-chip"}, "checks": {"icon": "list-checks", "status_indicator": "check-chip"}, "evidence": {"icon": "archive", "status_indicator": "evidence-chip"}, "attestations": {"icon": "signature", "status_indicator": "attestation-chip"}, "issues": {"icon": "triangle-alert", "status_indicator": "issue-chip"}, "reports": {"icon": "file-bar-chart", "status_indicator": "report-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": COMPLIANCE_EVENT_STREAM, "key": "tenant_id", "events": ["compliance_obligation_registered", "compliance_control_mapped", "compliance_check_recorded", "compliance_evidence_attached", "compliance_attestation_recorded", "compliance_issue_opened", "compliance_remediation_recorded", "compliance_report_published", "compliance_review_recorded", "compliance_agent_registered"], "guardrails": ["compliance_batch_requires_bytewax", "privileged_compliance_agent_action_requires_human_approval"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "compliance_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "compliance_policy_required", "required_action": "attach_compliance_policy"}},
	{"name": "obligation_framework_supported", "condition": {"operation": "register_obligation", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "obligation_type_supported", "condition": {"operation": "register_obligation", "obligation_type_supported": False}, "effect": {"decision": "deny", "reason": "obligation_type_not_supported", "required_action": "select_supported_obligation_type"}},
	{"name": "obligation_owner_required", "condition": {"operation": "register_obligation", "owner_present": False}, "effect": {"decision": "deny", "reason": "obligation_owner_required", "required_action": "assign_owner"}},
	{"name": "obligation_evidence_required", "condition": {"operation": "register_obligation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "obligation_evidence_required", "required_action": "attach_evidence"}},
	{"name": "obligation_effective_date_required", "condition": {"operation": "register_obligation", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "control_obligation_required", "condition": {"operation": "map_control", "obligation_present": False}, "effect": {"decision": "deny", "reason": "obligation_required", "required_action": "select_obligation"}},
	{"name": "control_type_supported", "condition": {"operation": "map_control", "control_type_supported": False}, "effect": {"decision": "deny", "reason": "control_type_not_supported", "required_action": "select_supported_control_type"}},
	{"name": "control_owner_required", "condition": {"operation": "map_control", "owner_present": False}, "effect": {"decision": "deny", "reason": "control_owner_required", "required_action": "assign_control_owner"}},
	{"name": "control_evidence_required", "condition": {"operation": "map_control", "evidence_present": False}, "effect": {"decision": "deny", "reason": "control_evidence_required", "required_action": "attach_control_evidence"}},
	{"name": "control_frequency_required", "condition": {"operation": "map_control", "frequency_present": False}, "effect": {"decision": "deny", "reason": "control_frequency_required", "required_action": "set_frequency"}},
	{"name": "check_obligation_required", "condition": {"operation": "record_check", "obligation_present": False}, "effect": {"decision": "deny", "reason": "obligation_required", "required_action": "select_obligation"}},
	{"name": "check_control_required", "condition": {"operation": "record_check", "control_present": False}, "effect": {"decision": "deny", "reason": "control_required", "required_action": "select_control"}},
	{"name": "check_type_supported", "condition": {"operation": "record_check", "check_type_supported": False}, "effect": {"decision": "deny", "reason": "check_type_not_supported", "required_action": "select_supported_check_type"}},
	{"name": "check_subject_required", "condition": {"operation": "record_check", "subject_present": False}, "effect": {"decision": "deny", "reason": "check_subject_required", "required_action": "attach_subject"}},
	{"name": "check_result_required", "condition": {"operation": "record_check", "result_present": False}, "effect": {"decision": "deny", "reason": "check_result_required", "required_action": "record_result"}},
	{"name": "failed_check_requires_evidence", "condition": {"operation": "record_check", "failed_check": True, "evidence_present": False}, "effect": {"decision": "deny", "reason": "failed_check_evidence_required", "required_action": "attach_failure_evidence"}},
	{"name": "evidence_reference_required", "condition": {"operation": "attach_evidence", "reference_present": False}, "effect": {"decision": "deny", "reason": "evidence_reference_required", "required_action": "attach_reference"}},
	{"name": "evidence_type_supported", "condition": {"operation": "attach_evidence", "evidence_type_supported": False}, "effect": {"decision": "deny", "reason": "evidence_type_not_supported", "required_action": "select_supported_evidence_type"}},
	{"name": "evidence_source_required", "condition": {"operation": "attach_evidence", "source_present": False}, "effect": {"decision": "deny", "reason": "evidence_source_required", "required_action": "attach_source"}},
	{"name": "evidence_retention_required", "condition": {"operation": "attach_evidence", "retention_present": False}, "effect": {"decision": "deny", "reason": "retention_period_required", "required_action": "set_retention_period"}},
	{"name": "attestation_obligation_required", "condition": {"operation": "record_attestation", "obligation_present": False}, "effect": {"decision": "deny", "reason": "obligation_required", "required_action": "select_obligation"}},
	{"name": "attestor_required", "condition": {"operation": "record_attestation", "attestor_present": False}, "effect": {"decision": "deny", "reason": "attestor_required", "required_action": "assign_attestor"}},
	{"name": "attestation_status_supported", "condition": {"operation": "record_attestation", "status_supported": False}, "effect": {"decision": "deny", "reason": "attestation_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "attestation_evidence_required", "condition": {"operation": "record_attestation", "evidence_present": False}, "effect": {"decision": "deny", "reason": "attestation_evidence_required", "required_action": "attach_attestation_evidence"}},
	{"name": "issue_obligation_required", "condition": {"operation": "open_issue", "obligation_present": False}, "effect": {"decision": "deny", "reason": "obligation_required", "required_action": "select_obligation"}},
	{"name": "issue_severity_supported", "condition": {"operation": "open_issue", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "issue_owner_required", "condition": {"operation": "open_issue", "owner_present": False}, "effect": {"decision": "deny", "reason": "issue_owner_required", "required_action": "assign_issue_owner"}},
	{"name": "issue_evidence_required", "condition": {"operation": "open_issue", "evidence_present": False}, "effect": {"decision": "deny", "reason": "issue_evidence_required", "required_action": "attach_issue_evidence"}},
	{"name": "issue_due_date_required", "condition": {"operation": "open_issue", "due_date_present": False}, "effect": {"decision": "deny", "reason": "issue_due_date_required", "required_action": "set_due_date"}},
	{"name": "remediation_issue_required", "condition": {"operation": "record_remediation", "issue_present": False}, "effect": {"decision": "deny", "reason": "issue_required", "required_action": "select_issue"}},
	{"name": "remediation_owner_required", "condition": {"operation": "record_remediation", "owner_present": False}, "effect": {"decision": "deny", "reason": "remediation_owner_required", "required_action": "assign_remediation_owner"}},
	{"name": "remediation_plan_required", "condition": {"operation": "record_remediation", "plan_present": False}, "effect": {"decision": "deny", "reason": "remediation_plan_required", "required_action": "attach_remediation_plan"}},
	{"name": "high_impact_remediation_requires_approval", "condition": {"operation": "record_remediation", "high_impact": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "remediation_approval_required", "required_action": "attach_approval"}},
	{"name": "report_type_supported", "condition": {"operation": "publish_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "report_framework_supported", "condition": {"operation": "publish_report", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "report_period_required", "condition": {"operation": "publish_report", "period_present": False}, "effect": {"decision": "deny", "reason": "report_period_required", "required_action": "set_report_period"}},
	{"name": "report_evidence_required", "condition": {"operation": "publish_report", "evidence_present": False}, "effect": {"decision": "deny", "reason": "report_evidence_required", "required_action": "attach_report_evidence"}},
	{"name": "report_approver_required", "condition": {"operation": "publish_report", "approver_present": False}, "effect": {"decision": "deny", "reason": "report_approver_required", "required_action": "assign_approver"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "compliance_batch_requires_bytewax", "condition": {"operation": "compliance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_compliance_batch_to_bytewax"}},
	{"name": "compliance_agent_runtime_supported", "condition": {"operation": "register_compliance_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "compliance_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "compliance_agent_role_supported", "condition": {"operation": "register_compliance_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "compliance_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_compliance_agent_action_requires_human_approval", "condition": {"operation": "compliance_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_compliance_access_denied", "description": "Compliance resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Compliance privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific compliance rules
	{"name": "ke_cbk_prudential_returns_required", "description": "Kenya CBK prudential returns must be filed on schedule.", "condition": {"operation": "file_regulatory_return", "jurisdiction": "KE", "cbk_return_overdue": True}, "effect": {"decision": "deny", "reason": "ke_cbk_return_overdue", "required_action": "file_cbk_prudential_return_immediately"}},
	{"name": "ke_frc_aml_return_required", "description": "Kenya FRC AML/CFT compliance returns are mandatory.", "condition": {"operation": "file_regulatory_return", "jurisdiction": "KE", "frc_aml_return_filed": False}, "effect": {"decision": "deny", "reason": "ke_frc_aml_return_required", "required_action": "file_frc_aml_return"}},
	{"name": "ng_cbn_compliance_return_required", "description": "Nigeria CBN compliance returns must be filed on schedule.", "condition": {"operation": "file_regulatory_return", "jurisdiction": "NG", "cbn_return_overdue": True}, "effect": {"decision": "deny", "reason": "ng_cbn_return_overdue", "required_action": "file_cbn_compliance_return"}},
	{"name": "mobile_money_regulatory_cap_enforced", "description": "Mobile money regulatory transaction caps must be enforced.", "condition": {"operation": "process_mobile_money", "regulatory_cap_exceeded": True}, "effect": {"decision": "deny", "reason": "mobile_money_regulatory_cap_exceeded", "required_action": "enforce_regulatory_transaction_cap"}},
	{"name": "ke_data_protection_act_compliance", "description": "Kenya Data Protection Act compliance is required for personal data processing.", "condition": {"operation": "process_personal_data", "country": "KE", "dpa_compliant": False}, "effect": {"decision": "deny", "reason": "ke_data_protection_act_compliance_required", "required_action": "comply_with_ke_data_protection_act"}},
	{"name": "cbk_cbdc_pilot_compliance", "description": "CBK e-Shilling CBDC pilot compliance is required for digital currency operations in Kenya.", "condition": {"operation": "digital_currency_operation", "jurisdiction": "KE", "cbdc_pilot_compliant": False}, "effect": {"decision": "deny", "reason": "cbk_cbdc_pilot_compliance_required", "required_action": "comply_with_cbk_cbdc_pilot_guidelines"}},
]



def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-compliance/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
