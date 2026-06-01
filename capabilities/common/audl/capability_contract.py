"""
Executable capability contract for APG Audit Logging.

AUDL is a first-class APG capability: it exposes tenant-scoped audit
configuration, deterministic governance rules, UI surfaces, and theme tokens
that composition and administration tooling can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_AUDIT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AUDIT_AGENT_ROLES = [
	"audit_reviewer",
	"evidence_reviewer",
	"legal_hold_reviewer",
	"export_reviewer",
	"purge_reviewer",
	"investigation_reviewer",
	"compliance_reviewer",
]
PRIVILEGED_AUDIT_AGENT_ROLES = {
	"legal_hold_reviewer",
	"export_reviewer",
	"purge_reviewer",
	"investigation_reviewer",
	"compliance_reviewer",
}


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped AUDL configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"ingestion": {
			"default_batch_size": 1000,
			"max_batch_size": 50000,
			"stream_processing_enabled": True,
			"lifecycle_stream": "bytewax",
			"immutable_storage_required": True,
			"checksum_verification_required": True
		},
		"agents": {
			"enabled": True,
			"supported_runtimes": list(SUPPORTED_AUDIT_AGENT_RUNTIMES),
			"supported_roles": list(SUPPORTED_AUDIT_AGENT_ROLES),
			"privileged_roles_require_human_approval": True
		},
		"retention": {
			"default_retention_days": 2555,
			"legal_hold_enabled": True,
			"purge_requires_dual_control": True,
			"archive_after_days": 365
		},
		"compliance": {
			"enabled_frameworks": ["SOX", "GDPR", "HIPAA", "PCI-DSS"],
			"evidence_chain_of_custody": True,
			"pii_masking_enabled": True,
			"regulated_export_requires_approval": True
		},
		"investigations": {
			"collaborative_cases_enabled": True,
			"timeline_reconstruction_enabled": True,
			"default_case_priority": "high",
			"auto_assign_critical_incidents": True
		},
		"notifications": {
			"real_time_alerting": True,
			"critical_event_channels": ["email", "webhook"],
			"daily_digest_enabled": True
		},
		"ui": {
			"enable_investigation_workbench": True,
			"enable_compliance_center": True,
			"enable_live_timeline": True,
			"enable_natural_language_search": True
		},
		"theme": {
			"default_theme": "audl_forensics",
			"allow_tenant_overrides": True
		},
		"streaming": {
			"engine": "bytewax",
			"required_for_operations": ["audit_batch", "high_volume_ingestion"],
			"topics": ["audl.events", "audl.governance", "audl.agents"],
			"watermark_field": "timestamp"
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"ingestion",
			"agents",
			"retention",
			"compliance",
			"investigations",
			"notifications",
			"ui",
			"theme",
			"streaming"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"ingestion": {"type": "object"},
			"agents": {"type": "object"},
			"retention": {"type": "object"},
			"compliance": {"type": "object"},
			"investigations": {"type": "object"},
			"notifications": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
			"streaming": {"type": "object"}
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
	"""Simple AUDL policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic AUDL rule engine for governance and evidence controls."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against an audit workload context."""
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
				elif rule.effect.get("decision") == "require_review" and decision != "deny":
					decision = "require_review"

		return {
			"decision": decision,
			"matched_rules": matched,
			"actions": actions,
			"context": context
		}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by AUDL."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for AUDL UI surfaces."""

	name: str = "audl_forensics"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#0F4C5C",
		"color.accent": "#C97A2B",
		"color.success": "#2E7D32",
		"color.warning": "#A16207",
		"color.danger": "#B42318",
		"surface.canvas": "#F3F5F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#13232F",
		"text.secondary": "#526371",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"audit_timeline": {
			"orientation": "vertical",
			"event_marker": "severity-dot",
			"chain_of_custody_badge": "inline"
		},
		"investigation_case_card": {
			"icon": "folder-search",
			"shape": "rounded-rectangle",
			"priority_indicator": "left-rail"
		},
		"compliance_scorecard": {
			"trend_style": "sparkline",
			"status_variant": "stacked"
		},
		"severity_badge": {
			"icon": "shield-alert",
			"variant": "contrast"
		},
		"legal_hold_indicator": {
			"icon": "lock-keyhole",
			"status_style": "hold-ribbon"
		},
		"export_review_panel": {
			"icon": "file-output",
			"masking_state": "privacy-chip"
		},
		"purge_approval_warning": {
			"icon": "trash-2",
			"confirmation_style": "dual-control-banner"
		},
		"audit_agent_roster": {
			"icon": "bot",
			"runtime_badge": "inline",
			"approval_state": "human-review-chip"
		},
		"bytewax_stream_indicator": {
			"icon": "activity",
			"variant": "compact-meter",
			"status_style": "stream-health"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default AUDL rules available to every tenant."""
	return [
		CapabilityRule(
			name="require_tenant_context",
			description="All audit workloads must execute with tenant context.",
			condition={"tenant_id_missing": True},
			effect={
				"decision": "deny",
				"reason": "tenant_id_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="immutable_events_require_checksum",
			description="Immutable audit storage requires checksum verification.",
			condition={"immutable_storage": True, "checksum_verified": False},
			effect={
				"decision": "deny",
				"reason": "checksum_verification_required",
				"required_action": "verify_event_checksum"
			}
		),
		CapabilityRule(
			name="legal_hold_blocks_purge",
			description="Audit data under legal hold cannot be purged.",
			condition={"requested_operation": "purge", "legal_hold_active": True},
			effect={
				"decision": "deny",
				"reason": "legal_hold_active",
				"required_action": "release_legal_hold_or_abort"
			}
		),
		CapabilityRule(
			name="regulated_exports_require_masking",
			description="PII-bearing exports require masking before release.",
			condition={"requested_operation": "export", "contains_pii": True, "masking_enabled": False},
			effect={
				"decision": "deny",
				"reason": "pii_masking_required",
				"required_action": "enable_export_masking"
			}
		),
		CapabilityRule(
			name="critical_events_require_escalation",
			description="Critical security findings require an escalation route.",
			condition={"event_severity": "critical", "escalation_configured": False},
			effect={
				"decision": "deny",
				"reason": "critical_escalation_required",
				"required_action": "configure_critical_event_escalation"
			}
		),
		CapabilityRule(
			name="high_volume_ingestion_requires_stream_processing",
			description="Large audit batches require stream processing safeguards.",
			condition={"batch_size_gt": 10000, "stream_processing_enabled": False},
			effect={
				"decision": "deny",
				"reason": "stream_processing_required",
				"required_action": "enable_stream_processing"
			}
		),
		CapabilityRule(
			name="bytewax_event_stream_required",
			description="Audit batch ingestion must use the Bytewax lifecycle stream.",
			condition={"requested_operation": "audit_batch", "event_stream": "non_bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_event_stream_required",
				"required_action": "route_batch_through_bytewax"
			}
		),
		CapabilityRule(
			name="audit_agent_runtime_supported",
			description="Audit agents must run on an approved runtime.",
			condition={"requested_operation": "register_audit_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "audit_agent_runtime_unsupported",
				"required_action": "select_supported_audit_agent_runtime"
			}
		),
		CapabilityRule(
			name="audit_agent_role_supported",
			description="Audit agents must use an approved AUDL role.",
			condition={"requested_operation": "register_audit_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "audit_agent_role_unsupported",
				"required_action": "select_supported_audit_agent_role"
			}
		),
		CapabilityRule(
			name="audit_agent_privileged_action_requires_approval",
			description="Privileged audit-agent roles require human approval evidence or review.",
			condition={
				"requested_operation": "register_audit_agent",
				"privileged_action": True,
				"human_approval_required": False
			},
			effect={
				"decision": "require_review",
				"reason": "audit_agent_human_approval_required",
				"required_action": "enable_human_approval_for_privileged_audit_agent"
			}
		),
		CapabilityRule(
			name="regulated_export_requires_review",
			description="Masked PII-bearing exports must enter review before release.",
			condition={"requested_operation": "export", "contains_pii": True, "masking_enabled": True},
			effect={
				"decision": "require_review",
				"reason": "regulated_export_review_required",
				"required_action": "review_regulated_export"
			}
		),
		CapabilityRule(
			name="audit_purge_requires_dual_control_review",
			description="Audit purge requests must enter dual-control review before execution.",
			condition={"requested_operation": "purge", "legal_hold_active": False},
			effect={
				"decision": "require_review",
				"reason": "audit_purge_review_required",
				"required_action": "review_audit_purge"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return AUDL UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/audit/dashboard", "AuditDashboard", "audl:view", "Operations"),
		CapabilityUIRoute("events", "/audit/events", "AuditEventExplorer", "audl:view", "Operations"),
		CapabilityUIRoute("timeline", "/audit/timeline", "AuditTimelineWorkbench", "audl:view", "Investigations"),
		CapabilityUIRoute("investigations", "/audit/investigations", "InvestigationWorkbench", "audl:investigate", "Investigations"),
		CapabilityUIRoute("legal_holds", "/audit/legal-holds", "LegalHoldConsole", "audl:hold", "Governance"),
		CapabilityUIRoute("exports", "/audit/exports", "AuditExportReviewQueue", "audl:export", "Governance"),
		CapabilityUIRoute("purges", "/audit/purges", "AuditPurgeReviewQueue", "audl:purge", "Governance"),
		CapabilityUIRoute("compliance", "/audit/compliance", "ComplianceControlCenter", "audl:compliance", "Governance"),
		CapabilityUIRoute("agents", "/audit/agents", "AuditAgentRoster", "audl:admin", "Governance"),
		CapabilityUIRoute("reports", "/audit/reports", "AuditReportingStudio", "audl:report", "Governance"),
		CapabilityUIRoute("rules", "/audit/rules", "AuditRuleWorkbench", "audl:admin", "Governance"),
		CapabilityUIRoute("settings", "/audit/settings", "AuditCapabilitySettings", "audl:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"blueprint_module": "blueprint.py",
		"api_prefix": "/api/v1/audit",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "frontend/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class AUDL agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_AUDIT_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_AUDIT_AGENT_ROLES),
		"privileged_roles": sorted(PRIVILEGED_AUDIT_AGENT_ROLES),
		"composition_points": [
			"event_review",
			"evidence_review",
			"legal_hold_review",
			"export_review",
			"purge_review",
			"investigation_review",
			"compliance_review"
		],
		"guardrails": [
			"audit_agent_runtime_supported",
			"audit_agent_role_supported",
			"audit_agent_privileged_action_requires_approval"
		]
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return AUDL lifecycle stream metadata."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "audl.lifecycle",
		"required_for_operations": ["audit_batch", "high_volume_ingestion"],
		"topics": ["audl.events", "audl.governance", "audl.agents"],
		"watermark_field": "timestamp",
		"guardrails": [
			"bytewax_event_stream_required",
			"high_volume_ingestion_requires_stream_processing"
		]
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable AUDL capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "audl",
		"display_name": "Audit Logging",
		"provides": ["audl_operations", "audit_agents", "review_evidence"],
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
			"durable_statuses": ["review_required", "pending_review", "denied", "accepted"],
			"policy_fields": ["policy_decision", "matched_rules", "review_reasons", "audit_evidence"],
			"pending_queues": ["audit_exports", "audit_purges", "audit_agents", "audit_batches"],
			"deny_behavior": "Denied audit exports, purges, and batches persist evidence before PermissionError",
		},
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default AUDL rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
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
