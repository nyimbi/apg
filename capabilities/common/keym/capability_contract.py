"""
Executable capability contract for APG Key Management.

KEYM is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic key-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with KEYM without initializing the
full key-management runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_KEYM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_KEYM_AGENT_ROLES = [
	"key_policy_reviewer",
	"key_lifecycle_reviewer",
	"key_custody_reviewer",
	"export_reviewer",
	"rotation_exception_reviewer",
	"compromise_responder",
	"hsm_attestation_reviewer",
]
PRIVILEGED_KEYM_AGENT_ROLES = {
	"export_reviewer",
	"rotation_exception_reviewer",
	"compromise_responder",
	"hsm_attestation_reviewer",
}


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped KEYM configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"key_domains": {
			"default_domain": "tenant-root",
			"tenant_isolation_required": True,
			"root_keys_require_hsm": True,
			"allow_cross_region_replication": True
		},
		"lifecycle": {
			"default_rotation_days": 90,
			"auto_rotation_enabled": True,
			"compromise_response": "disable_and_rotate",
			"backup_required": True
		},
		"access": {
			"require_tenant_context": True,
			"require_policy_for_key_creation": True,
			"require_dual_control_for_export": True,
			"max_failed_attempts": 3
		},
		"hsm": {
			"software_hsm_enabled": True,
			"hardware_hsm_preferred": True,
			"attestation_required_for_root_keys": True
		},
		"compliance": {
			"frameworks": ["FIPS_140_2", "GDPR", "HIPAA", "PCI_DSS", "ISO_27001"],
			"immutable_audit_required": True,
			"audit_retention_days": 2555
		},
		"automation": {
			"ai_lifecycle_recommendations": True,
			"anomaly_detection_enabled": True,
			"notify_on_policy_violation": True
		},
		"operation_governance": {
			"require_independent_export_review": True,
			"require_independent_rotation_exception_review": True,
			"require_rotation_evidence": True,
			"record_key_lifecycle_audit": True
		},
		"ui": {
			"enable_inventory": True,
			"enable_policy_manager": True,
			"enable_hsm_console": True,
			"enable_audit_viewer": True,
			"enable_export_approvals": True,
			"enable_rotation_exceptions": True,
			"enable_compromise_console": True
		},
		"theme": {
			"default_theme": "keym_vault_console",
			"allow_tenant_overrides": True
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_KEYM_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_KEYM_AGENT_ROLES,
			"privileged_roles": sorted(PRIVILEGED_KEYM_AGENT_ROLES),
			"require_owner": True,
			"require_declared_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "keym.lifecycle",
			"required_operations": [
				"key_lifecycle_batch",
				"key_rotation_batch",
				"key_agent_batch"
			],
			"topics": [
				"keym.keys",
				"keym.operations",
				"keym.approvals",
				"keym.rotations",
				"keym.agents"
			],
			"watermark": "event_time"
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"key_domains",
			"lifecycle",
			"access",
			"hsm",
			"compliance",
			"automation",
			"operation_governance",
			"ui",
			"theme",
			"agents",
			"streaming"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"key_domains": {"type": "object"},
			"lifecycle": {"type": "object"},
			"access": {"type": "object"},
			"hsm": {"type": "object"},
			"compliance": {"type": "object"},
			"automation": {"type": "object"},
			"operation_governance": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
			"agents": {"type": "object"},
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
	"""Simple KEYM policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic KEYM rule engine for key lifecycle decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching key-management governance rules."""
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
	"""UI route exposed by KEYM."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for KEYM UI surfaces."""

	name: str = "keym_vault_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#24415F",
		"color.accent": "#B7791F",
		"color.success": "#2F855A",
		"color.warning": "#C05621",
		"color.danger": "#B83232",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#16202A",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"key_inventory_row": {
			"icon": "key-round",
			"status_indicator": "lifecycle-pill",
			"risk_style": "right-aligned-score"
		},
		"rotation_timeline": {
			"visual": "deadline-track",
			"threshold_style": "expiry-bands"
		},
		"hsm_attestation_panel": {
			"visual": "signed-attestation-stack",
			"status_style": "seal-chip"
		},
		"policy_violation_trace": {
			"visual": "rule-ladder",
			"highlight": "deny-marker"
		},
		"export_approval_queue": {
			"icon": "shield-check",
			"status_indicator": "dual-control",
			"variant": "approval"
		},
		"rotation_exception_queue": {
			"icon": "clock-alert",
			"status_indicator": "review-bar",
			"variant": "review"
		},
		"compromise_response_panel": {
			"icon": "shield-x",
			"status_indicator": "danger-rail",
			"variant": "incident"
		},
		"key_audit_timeline": {
			"icon": "scroll-text",
			"line_style": "segmented",
			"variant": "evidence"
		},
		"key_agent_roster": {
			"icon": "bot",
			"status_indicator": "approval-chip",
			"variant": "agent-governance"
		},
		"bytewax_stream_indicator": {
			"icon": "activity",
			"status_indicator": "stream-health",
			"variant": "streaming"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default KEYM rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All key operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="key_creation_requires_policy",
			description="Key creation requires an attached key policy.",
			condition={"operation": "create_key", "policy_attached": False},
			effect={
				"decision": "deny",
				"reason": "key_policy_required",
				"required_action": "attach_key_policy"
			}
		),
		CapabilityRule(
			name="root_key_requires_hsm_attestation",
			description="Root keys require HSM attestation before activation.",
			condition={"key_class": "root", "hsm_attested": False},
			effect={
				"decision": "deny",
				"reason": "hsm_attestation_required",
				"required_action": "complete_hsm_attestation"
			}
		),
		CapabilityRule(
			name="export_requires_dual_control",
			description="Key export requires dual-control approval and wrapping.",
			condition={"operation": "export_key", "dual_control_approved": False},
			effect={
				"decision": "deny",
				"reason": "dual_control_required",
				"required_action": "record_dual_control_approval"
			}
		),
		CapabilityRule(
			name="overdue_rotation_requires_review",
			description="Overdue key rotation requires review before continued use.",
			condition={"rotation_age_days_gt": 90, "rotation_exception_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "rotation_overdue",
				"required_action": "rotate_key_or_record_exception"
			}
		),
		CapabilityRule(
			name="compromised_key_blocks_use",
			description="Compromised keys cannot be used for cryptographic operations.",
			condition={"key_status": "compromised", "operation_is_cryptographic": True},
			effect={
				"decision": "deny",
				"reason": "key_compromised",
				"required_action": "disable_and_rotate_key"
			}
		),
		CapabilityRule(
			name="disabled_key_blocks_use",
			description="Disabled keys cannot be used for cryptographic operations.",
			condition={"key_status": "disabled", "operation_is_cryptographic": True},
			effect={
				"decision": "deny",
				"reason": "key_disabled",
				"required_action": "reactivate_or_rotate_key"
			}
		),
		CapabilityRule(
			name="destroyed_key_blocks_use",
			description="Destroyed keys cannot be used for cryptographic operations.",
			condition={"key_status": "destroyed", "operation_is_cryptographic": True},
			effect={
				"decision": "deny",
				"reason": "key_destroyed",
				"required_action": "provision_replacement_key"
			}
		),
		CapabilityRule(
			name="review_requires_independent_reviewer",
			description="KEYM approvals and exceptions require independent review.",
			condition={"reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="review_requires_notes",
			description="KEYM approvals and exceptions require reviewer notes.",
			condition={"review_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		),
		CapabilityRule(
			name="rotation_completion_requires_evidence",
			description="Key rotation completion requires evidence.",
			condition={"operation": "complete_rotation", "key_rotation_evidence_attached": False},
			effect={
				"decision": "deny",
				"reason": "key_rotation_evidence_required",
				"required_action": "attach_key_rotation_evidence"
			}
		),
		CapabilityRule(
			name="key_agent_runtime_supported",
			description="Key-management agents must use an approved APG agent runtime.",
			condition={"operation": "register_key_agent", "key_agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "key_agent_runtime_not_supported",
				"required_action": "select_supported_key_agent_runtime"
			}
		),
		CapabilityRule(
			name="key_agent_role_supported",
			description="Key-management agents must use an approved KEYM composition role.",
			condition={"operation": "register_key_agent", "key_agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "key_agent_role_not_supported",
				"required_action": "select_supported_key_agent_role"
			}
		),
		CapabilityRule(
			name="key_agent_requires_scope",
			description="Key-management agents require a declared operating scope.",
			condition={"operation": "register_key_agent", "key_agent_scope_attached": False},
			effect={
				"decision": "deny",
				"reason": "key_agent_scope_required",
				"required_action": "attach_key_agent_scope"
			}
		),
		CapabilityRule(
			name="key_agent_privileged_role_requires_human_approval",
			description="Privileged key-management agent roles require human approval.",
			condition={
				"operation": "register_key_agent",
				"key_agent_privileged_role": True,
				"human_approval_required": False
			},
			effect={
				"decision": "deny",
				"reason": "key_agent_privileged_role_requires_human_approval",
				"required_action": "require_human_key_approval"
			}
		),
		CapabilityRule(
			name="bytewax_key_stream_required",
			description="Key lifecycle batch mutations must be routed through Bytewax.",
			condition={"operation": "validate_key_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_key_stream_required",
				"required_action": "route_key_lifecycle_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return KEYM UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/keym/dashboard", "KeyManagementDashboard", "keym.read_key", "Overview"),
		CapabilityUIRoute("inventory", "/keym/keys", "KeyInventoryView", "keym.read_key", "Operations"),
		CapabilityUIRoute("lifecycle", "/keym/lifecycle", "KeyLifecycleWorkbench", "keym.rotate_key", "Operations"),
		CapabilityUIRoute("export_approvals", "/keym/export-approvals", "ExportApprovalQueue", "keym.export_key", "Governance"),
		CapabilityUIRoute("rotation_exceptions", "/keym/rotation-exceptions", "RotationExceptionQueue", "keym.rotate_key", "Governance"),
		CapabilityUIRoute("compromise", "/keym/compromise", "CompromiseResponseConsole", "keym.admin", "Security"),
		CapabilityUIRoute("policies", "/keym/policies", "PolicyManagerView", "keym.manage_policies", "Governance"),
		CapabilityUIRoute("hsm", "/keym/hsm", "HSMConsole", "keym.manage_hsm", "Security"),
		CapabilityUIRoute("audit", "/keym/audit", "AuditLogsView", "keym.view_audit_logs", "Governance"),
		CapabilityUIRoute("analytics", "/keym/analytics", "SecurityAnalyticsView", "keym.admin", "Intelligence"),
		CapabilityUIRoute("agents", "/keym/agents", "KeyAgentRoster", "keym.admin", "Administration"),
		CapabilityUIRoute("settings", "/keym/settings", "KeyManagementSettings", "keym.admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/keym/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return KEYM first-class AI agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_KEYM_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_KEYM_AGENT_ROLES),
		"privileged_roles": sorted(PRIVILEGED_KEYM_AGENT_ROLES),
		"guardrails": [
			"runtime_supported",
			"role_supported",
			"scope_required",
			"owner_required",
			"purpose_required",
			"human_approval_for_privileged_roles",
			"contribution_disclosure_required"
		]
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return KEYM Bytewax lifecycle stream metadata."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "keym.lifecycle",
		"required_operations": [
			"key_lifecycle_batch",
			"key_rotation_batch",
			"key_agent_batch"
		],
		"topics": [
			"keym.keys",
			"keym.operations",
			"keym.approvals",
			"keym.rotations",
			"keym.agents"
		],
		"state": [
			"keys",
			"operations",
			"export_approvals",
			"rotation_exceptions",
			"rotations",
			"key_agents"
		],
		"events": [
			"managed_key_created",
			"key_operation_allowed",
			"key_operation_denied",
			"export_approval_decided",
			"rotation_exception_decided",
			"key_rotation_completed",
			"managed_key_compromised",
			"key_agent_registered"
		],
		"watermark": "event_time"
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable KEYM capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "keym",
		"display_name": "Key Management",
		"provides": ["keym_operations", "key_lifecycle_governance", "key_agent_composition"],
		"requires": ["conf", "auth", "audl", "mten", "secu"],
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
		"streaming": streaming_manifest()
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default KEYM rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
				return False
		elif key.endswith("_ne"):
			field_name = key[:-3]
			if context.get(field_name) == expected:
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
