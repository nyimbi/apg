"""
Executable capability contract for APG Security Framework.

SECU is a first-class APG capability: composition tooling can inspect its
tenant configuration, deterministic rules, UI surfaces, and visual theme
without initializing the full security runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_SECU_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_SECU_AGENT_ROLES = [
	"risk_reviewer",
	"threat_triage",
	"incident_responder",
	"compliance_reviewer",
	"exception_reviewer",
]
PRIVILEGED_SECU_AGENT_ROLES = {"incident_responder", "compliance_reviewer", "exception_reviewer"}


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped SECU configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"zero_trust": {
			"enabled": True,
			"default_security_level": "confidential",
			"continuous_verification": True,
			"deny_unknown_devices": True
		},
		"risk": {
			"critical_threshold": 90,
			"high_threshold": 70,
			"challenge_threshold": 50,
			"auto_quarantine_threshold": 85
		},
		"threat_detection": {
			"enabled": True,
			"ai_detection_enabled": True,
			"indicator_ttl_hours": 24,
			"alert_on_unknown_threats": True
		},
		"compliance": {
			"frameworks": ["iso_27001", "soc2", "nist"],
			"assessment_interval_days": 30,
			"require_audit_evidence": True
		},
		"incident_response": {
			"require_containment_for_critical": True,
			"require_resolution_evidence": True,
			"require_independent_exception_review": True,
			"policy_exception_max_days": 30
		},
		"agents": {
			"enabled": True,
			"supported_runtimes": list(SUPPORTED_SECU_AGENT_RUNTIMES),
			"supported_roles": list(SUPPORTED_SECU_AGENT_ROLES),
			"privileged_roles": sorted(PRIVILEGED_SECU_AGENT_ROLES),
			"privileged_roles_require_human_approval": True,
			"scope_required": True,
			"contribution_disclosure_required": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "secu.lifecycle",
			"required_for_operations": ["security_lifecycle_batch", "threat_batch_triage"],
			"topics": ["secu.risk", "secu.threats", "secu.incidents", "secu.agents"],
			"watermark_field": "event_time"
		},
		"ui": {
			"enable_security_dashboard": True,
			"enable_policy_workbench": True,
			"enable_threat_console": True,
			"enable_compliance_console": True,
			"enable_exception_queue": True,
			"enable_incident_response": True,
			"enable_quarantine_console": True,
			"enable_audit_timeline": True
		},
		"theme": {
			"default_theme": "secu_zero_trust",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": ["tenant_id", "zero_trust", "risk", "threat_detection", "compliance", "incident_response", "agents", "streaming", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"zero_trust": {"type": "object"},
			"risk": {"type": "object"},
			"threat_detection": {"type": "object"},
			"compliance": {"type": "object"},
			"incident_response": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"}
		}
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return configuration with tenant-specific overrides applied."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	"""Simple SECU policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic SECU rule engine for security posture decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a security context."""
		assert isinstance(context, dict), "context must be a dictionary"
		matched: list[str] = []
		actions: list[dict[str, Any]] = []
		decision = "allow"

		for rule in self.rules:
			if _matches(rule.condition, context):
				matched.append(rule.name)
				actions.append(rule.effect)
				if rule.effect.get("decision") == "deny":
					decision = "deny"
				elif rule.effect.get("decision") == "quarantine" and decision not in {"deny"}:
					decision = "quarantine"
				elif rule.effect.get("decision") == "challenge" and decision == "allow":
					decision = "challenge"
				elif rule.effect.get("decision") == "require_review" and decision == "allow":
					decision = "require_review"

		return {
			"decision": decision,
			"matched_rules": matched,
			"actions": actions,
			"context": context
		}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by SECU."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for SECU UI surfaces."""

	name: str = "secu_zero_trust"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#1C3D5A",
		"color.accent": "#C2410C",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#102A43",
		"text.secondary": "#52616B",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"risk_score_meter": {
			"visual": "threshold-gauge",
			"critical_band": "color.danger",
			"safe_band": "color.success"
		},
		"threat_indicator": {
			"icon": "shield-alert",
			"severity_style": "left-border"
		},
		"policy_card": {
			"icon": "lock-keyhole",
			"status_indicator": "top-right"
		},
		"compliance_badge": {
			"icon": "badge-check",
			"variant": "subtle"
		},
		"policy_exception_queue": {
			"icon": "clipboard-check",
			"status_indicator": "left-bar",
			"variant": "review"
		},
		"incident_response_panel": {
			"icon": "siren",
			"status_indicator": "top-bar",
			"variant": "critical"
		},
		"device_quarantine_list": {
			"icon": "shield-x",
			"status_indicator": "lock",
			"variant": "containment"
		},
		"security_audit_timeline": {
			"icon": "scroll-text",
			"line_style": "segmented",
			"variant": "evidence"
		},
		"security_agent_roster": {
			"icon": "bot",
			"status_indicator": "approval-chip",
			"variant": "governed"
		},
		"bytewax_stream_indicator": {
			"icon": "activity",
			"status_indicator": "stream-chip",
			"variant": "lifecycle"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default SECU rules available to every tenant."""
	return [
		CapabilityRule(
			name="known_malicious_network_denied",
			description="Known malicious network origins are denied.",
			condition={"is_known_malicious": True},
			effect={
				"decision": "deny",
				"reason": "malicious_network_origin",
				"required_action": "block_request"
			}
		),
		CapabilityRule(
			name="compromised_device_quarantined",
			description="Compromised devices require quarantine.",
			condition={"device_trust": "compromised"},
			effect={
				"decision": "quarantine",
				"reason": "compromised_device",
				"required_action": "isolate_device"
			}
		),
		CapabilityRule(
			name="critical_risk_denied",
			description="Critical risk scores are denied by default.",
			condition={"risk_score_gte": 90},
			effect={
				"decision": "deny",
				"reason": "critical_risk_score",
				"required_action": "investigate_security_event"
			}
		),
		CapabilityRule(
			name="high_risk_requires_challenge",
			description="High risk requests require challenge before access.",
			condition={"risk_score_gte": 70, "challenge_completed": False},
			effect={
				"decision": "challenge",
				"reason": "step_up_required",
				"required_action": "complete_security_challenge"
			}
		),
		CapabilityRule(
			name="compliance_violation_alert",
			description="Compliance violations require audit evidence and alerting.",
			condition={"compliance_violation": True, "audit_evidence_attached": False},
			effect={
				"decision": "challenge",
				"reason": "audit_evidence_required",
				"required_action": "attach_audit_evidence"
			}
		),
		CapabilityRule(
			name="policy_exception_requires_independent_reviewer",
			description="Policy exceptions require independent review.",
			condition={"operation": "approve_policy_exception", "exception_reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_exception_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="expired_policy_exception_denied",
			description="Expired policy exceptions cannot be approved.",
			condition={"operation": "approve_policy_exception", "policy_exception_expired": True},
			effect={
				"decision": "deny",
				"reason": "policy_exception_expired",
				"required_action": "request_new_exception"
			}
		),
		CapabilityRule(
			name="critical_incident_requires_containment",
			description="Critical incidents require a containment plan when opened.",
			condition={"operation": "open_incident", "incident_severity": "critical", "containment_plan_attached": False},
			effect={
				"decision": "deny",
				"reason": "critical_incident_containment_required",
				"required_action": "attach_containment_plan"
			}
		),
		CapabilityRule(
			name="incident_resolution_requires_containment",
			description="Incidents require containment evidence before resolution.",
			condition={"operation": "resolve_incident", "containment_evidence_attached": False},
			effect={
				"decision": "deny",
				"reason": "incident_containment_evidence_required",
				"required_action": "record_containment_evidence"
			}
		),
		CapabilityRule(
			name="security_agent_runtime_supported",
			description="SECU agents must run on an approved runtime.",
			condition={"operation": "register_security_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "security_agent_runtime_unsupported",
				"required_action": "select_supported_security_agent_runtime"
			}
		),
		CapabilityRule(
			name="security_agent_role_supported",
			description="SECU agents must use an approved security role.",
			condition={"operation": "register_security_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "security_agent_role_unsupported",
				"required_action": "select_supported_security_agent_role"
			}
		),
		CapabilityRule(
			name="security_agent_requires_scope",
			description="SECU agents require explicit operating scope.",
			condition={"operation": "register_security_agent", "agent_scope_present": False},
			effect={
				"decision": "deny",
				"reason": "security_agent_scope_required",
				"required_action": "set_security_agent_scope"
			}
		),
		CapabilityRule(
			name="security_agent_privileged_role_requires_human_approval",
			description="Privileged SECU agent roles require human approval evidence or review.",
			condition={
				"operation": "register_security_agent",
				"agent_privileged_role": True,
				"human_approval_required": False
			},
			effect={
				"decision": "require_review",
				"reason": "security_agent_human_approval_required",
				"required_action": "enable_human_approval_for_privileged_security_agent"
			}
		),
		CapabilityRule(
			name="bytewax_security_stream_required",
			description="Security lifecycle batch operations must use Bytewax.",
			condition={"operation": "security_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_security_stream_required",
				"required_action": "route_security_lifecycle_batch_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return SECU UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/secu/dashboard", "SecurityDashboard", "secu:view", "Operations"),
		CapabilityUIRoute("risk", "/secu/risk", "RiskAssessmentConsole", "secu:view_risk", "Operations"),
		CapabilityUIRoute("threats", "/secu/threats", "ThreatDetectionConsole", "secu:view_threats", "Operations"),
		CapabilityUIRoute("policies", "/secu/policies", "SecurityPolicyWorkbench", "secu:manage_policies", "Governance"),
		CapabilityUIRoute("exceptions", "/secu/exceptions", "PolicyExceptionQueue", "secu:approve_exception", "Governance"),
		CapabilityUIRoute("incidents", "/secu/incidents", "IncidentResponseConsole", "secu:respond", "Operations"),
		CapabilityUIRoute("quarantine", "/secu/quarantine", "DeviceQuarantineConsole", "secu:respond", "Operations"),
		CapabilityUIRoute("compliance", "/secu/compliance", "ComplianceConsole", "secu:view_compliance", "Governance"),
		CapabilityUIRoute("audit", "/secu/audit", "SecurityAuditTimeline", "secu:view", "Governance"),
		CapabilityUIRoute("agents", "/secu/agents", "SecurityAgentRoster", "secu:admin", "Governance"),
		CapabilityUIRoute("rules", "/secu/rules", "SecurityRuleWorkbench", "secu:admin", "Governance"),
		CapabilityUIRoute("settings", "/secu/settings", "SecuritySettings", "secu:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"frontend_bundle": "views.py",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class SECU security-agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_SECU_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_SECU_AGENT_ROLES),
		"privileged_roles": sorted(PRIVILEGED_SECU_AGENT_ROLES),
		"composition_points": [
			"risk_review",
			"threat_triage",
			"incident_response",
			"compliance_evidence_review",
			"policy_exception_review"
		],
		"guardrails": [
			"security_agent_runtime_supported",
			"security_agent_role_supported",
			"security_agent_requires_scope",
			"security_agent_privileged_role_requires_human_approval"
		]
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return SECU lifecycle stream metadata."""
	return {
		"engine": "bytewax",
		"processor": "bytewax",
		"lifecycle_stream": "secu.lifecycle",
		"topics": ["secu.risk", "secu.threats", "secu.incidents", "secu.agents"],
		"watermark_field": "event_time",
		"state": [
			"policies",
			"devices",
			"threats",
			"assessments",
			"controls",
			"policy_exceptions",
			"incidents",
			"security_agents",
			"audit_events"
		],
		"events": [
			"policy_created",
			"device_posture_recorded",
			"device_quarantined",
			"threat_indicator_registered",
			"access_challenge",
			"access_quarantine",
			"access_deny",
			"compliance_gap_recorded",
			"policy_exception_requested",
			"policy_exception_decided",
			"security_incident_opened",
			"security_incident_contained",
			"security_incident_resolved",
			"security_agent_registered"
		],
		"guardrails": ["bytewax_security_stream_required"]
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SECU capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "secu",
		"display_name": "Security Framework",
		"provides": [
			"risk_assessment",
			"threat_detection",
			"security_policies",
			"compliance_automation",
			"incident_response_governance",
			"security_agents",
			"review_evidence"
		],
		"requires": ["auth", "conf", "audl"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"theme": {
			"name": theme.name,
			"tokens": theme.tokens,
			"components": theme.components
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"review_evidence": {
			"durable_statuses": [
				"pending",
				"pending_review",
				"evidence_required",
				"non_compliant",
				"denied",
				"approved",
				"rejected",
				"accepted",
			],
			"policy_fields": ["policy_decision", "matched_rules", "review_reasons", "review_evidence"],
			"pending_queues": ["policy_exceptions", "compliance_controls", "security_agents", "security_lifecycle_batches"],
			"deny_behavior": "Denied security lifecycle batches persist evidence before PermissionError",
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default SECU rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gte"):
			field_name = key[:-4]
			if not context.get(field_name, 0) >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
	copied: dict[str, Any] = {}
	for key, item in value.items():
		if isinstance(item, dict):
			copied[key] = _deep_copy(item)
		elif isinstance(item, list):
			copied[key] = list(item)
		else:
			copied[key] = item
	return copied


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
