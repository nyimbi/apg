"""
Executable capability contract for APG Encryption Services.

ENCR is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic cryptographic governance rules, UI surfaces, and
theme tokens so composition tooling can integrate with ENCR without loading the
full encryption runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_ENCR_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ENCR_AGENT_ROLES = [
	"crypto_policy_reviewer",
	"key_lifecycle_reviewer",
	"entropy_reviewer",
	"exception_reviewer",
	"threat_rotation_reviewer",
	"homomorphic_compute_reviewer",
]
PRIVILEGED_ENCR_AGENT_ROLES = {
	"exception_reviewer",
	"threat_rotation_reviewer",
	"homomorphic_compute_reviewer",
}


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped ENCR configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"cryptography": {
			"default_symmetric_algorithm": "AES-256-GCM",
			"post_quantum_enabled": True,
			"zero_knowledge_enabled": True,
			"homomorphic_computation_enabled": True,
			"minimum_entropy_quality": 0.95
		},
		"key_lifecycle": {
			"autonomous_rotation_enabled": True,
			"default_rotation_days": 90,
			"tenant_key_domains_required": True,
			"external_key_manager": "keym"
		},
		"policy": {
			"require_tenant_context": True,
			"deny_plaintext_exports": True,
			"require_quantum_safe_for_restricted_data": True,
			"require_review_for_legacy_algorithms": True
		},
		"threat_adaptive": {
			"enabled": True,
			"escalate_on_active_threat": True,
			"rotate_keys_on_compromise_signal": True
		},
		"operation_governance": {
			"require_key_domain": True,
			"require_independent_exception_review": True,
			"require_rotation_evidence": True,
			"record_crypto_audit_events": True
		},
		"compliance": {
			"frameworks": ["GDPR", "HIPAA", "PCI_DSS", "FIPS_140_2"],
			"audit_all_crypto_operations": True,
			"evidence_retention_days": 2555
		},
		"ui": {
			"enable_dashboard": True,
			"enable_policy_designer": True,
			"enable_entropy_console": True,
			"enable_homomorphic_workspace": True,
			"enable_operation_queue": True,
			"enable_exception_queue": True,
			"enable_rotation_console": True,
			"enable_audit_timeline": True
		},
		"theme": {
			"default_theme": "encr_quantum_guard",
			"allow_tenant_overrides": True
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_ENCR_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_ENCR_AGENT_ROLES,
			"privileged_roles": sorted(PRIVILEGED_ENCR_AGENT_ROLES),
			"require_owner": True,
			"require_declared_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "encr.lifecycle",
			"required_operations": [
				"crypto_lifecycle_batch",
				"key_rotation_batch",
				"crypto_agent_batch"
			],
			"topics": [
				"encr.key_domains",
				"encr.operations",
				"encr.rotations",
				"encr.agents"
			],
			"watermark": "event_time"
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"cryptography",
			"key_lifecycle",
			"policy",
			"threat_adaptive",
			"operation_governance",
			"compliance",
			"ui",
			"theme",
			"agents",
			"streaming"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"cryptography": {"type": "object"},
			"key_lifecycle": {"type": "object"},
			"policy": {"type": "object"},
			"threat_adaptive": {"type": "object"},
			"operation_governance": {"type": "object"},
			"compliance": {"type": "object"},
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
	"""Simple ENCR policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic ENCR rule engine for cryptographic control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching cryptographic governance rules."""
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
	"""UI route exposed by ENCR."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for ENCR UI surfaces."""

	name: str = "encr_quantum_guard"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#143C5C",
		"color.accent": "#23A6A6",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F4F7FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#53627A",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"crypto_posture_card": {
			"icon": "shield-check",
			"status_indicator": "algorithm-badge",
			"risk_style": "left-rail"
		},
		"entropy_quality_meter": {
			"visual": "segmented-meter",
			"threshold_style": "quality-bands"
		},
		"policy_decision_trace": {
			"visual": "stacked-rule-list",
			"highlight": "decision-chip"
		},
		"homomorphic_workspace": {
			"visual": "locked-data-flow",
			"result_style": "sealed-output"
		},
		"key_domain_card": {
			"icon": "key-round",
			"status_indicator": "rotation-chip",
			"variant": "domain"
		},
		"crypto_operation_queue": {
			"icon": "list-checks",
			"status_indicator": "decision-badge",
			"variant": "governance"
		},
		"crypto_exception_queue": {
			"icon": "clipboard-check",
			"status_indicator": "review-bar",
			"variant": "review"
		},
		"key_rotation_timeline": {
			"icon": "rotate-cw",
			"status_indicator": "evidence-lock",
			"variant": "rotation"
		},
		"crypto_audit_timeline": {
			"icon": "scroll-text",
			"line_style": "segmented",
			"variant": "evidence"
		},
		"crypto_agent_roster": {
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
	"""Default ENCR rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All encryption operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="restricted_data_requires_quantum_safe_algorithm",
			description="Restricted data must use quantum-safe encryption.",
			condition={"data_classification": "restricted", "algorithm_quantum_safe": False},
			effect={
				"decision": "deny",
				"reason": "quantum_safe_algorithm_required",
				"required_action": "select_quantum_safe_algorithm"
			}
		),
		CapabilityRule(
			name="plaintext_export_blocked",
			description="Plaintext export requests are blocked by default.",
			condition={"plaintext_export_requested": True},
			effect={
				"decision": "deny",
				"reason": "plaintext_export_blocked",
				"required_action": "use_wrapped_or_encrypted_export"
			}
		),
		CapabilityRule(
			name="low_entropy_blocks_key_generation",
			description="Key generation requires high-quality entropy.",
			condition={"entropy_quality_lt": 0.95, "operation": "generate_key"},
			effect={
				"decision": "deny",
				"reason": "entropy_quality_too_low",
				"required_action": "refresh_entropy_source"
			}
		),
		CapabilityRule(
			name="legacy_algorithm_requires_review",
			description="Legacy cryptographic algorithms require explicit review.",
			condition={"algorithm_family": "legacy", "security_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "legacy_algorithm_review_required",
				"required_action": "record_crypto_exception"
			}
		),
		CapabilityRule(
			name="active_threat_requires_key_rotation",
			description="Active threat signals require key rotation before sensitive operations.",
			condition={"active_threat_signal": True, "key_rotation_completed": False},
			effect={
				"decision": "deny",
				"reason": "threat_adaptive_rotation_required",
				"required_action": "rotate_affected_keys"
			}
		),
		CapabilityRule(
			name="crypto_exception_requires_independent_reviewer",
			description="Crypto exception review requires an independent reviewer.",
			condition={"operation": "decide_crypto_exception", "crypto_exception_reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_crypto_reviewer_required",
				"required_action": "assign_independent_crypto_reviewer"
			}
		),
		CapabilityRule(
			name="crypto_exception_requires_notes",
			description="Crypto exception review requires reviewer notes.",
			condition={"operation": "decide_crypto_exception", "crypto_exception_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "crypto_exception_notes_required",
				"required_action": "attach_crypto_review_notes"
			}
		),
		CapabilityRule(
			name="key_rotation_completion_requires_evidence",
			description="Key rotation completion requires evidence.",
			condition={"operation": "complete_key_rotation", "key_rotation_evidence_attached": False},
			effect={
				"decision": "deny",
				"reason": "key_rotation_evidence_required",
				"required_action": "attach_key_rotation_evidence"
			}
		),
		CapabilityRule(
			name="crypto_agent_runtime_supported",
			description="Crypto agents must use an approved APG agent runtime.",
			condition={"operation": "register_crypto_agent", "crypto_agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "crypto_agent_runtime_not_supported",
				"required_action": "select_supported_crypto_agent_runtime"
			}
		),
		CapabilityRule(
			name="crypto_agent_role_supported",
			description="Crypto agents must use an approved ENCR composition role.",
			condition={"operation": "register_crypto_agent", "crypto_agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "crypto_agent_role_not_supported",
				"required_action": "select_supported_crypto_agent_role"
			}
		),
		CapabilityRule(
			name="crypto_agent_requires_scope",
			description="Crypto agents require a declared operating scope.",
			condition={"operation": "register_crypto_agent", "crypto_agent_scope_attached": False},
			effect={
				"decision": "deny",
				"reason": "crypto_agent_scope_required",
				"required_action": "attach_crypto_agent_scope"
			}
		),
		CapabilityRule(
			name="crypto_agent_privileged_role_requires_human_approval",
			description="Privileged crypto-agent roles require human approval evidence or review.",
			condition={
				"operation": "register_crypto_agent",
				"crypto_agent_privileged_role": True,
				"human_approval_required": False
			},
			effect={
				"decision": "require_review",
				"reason": "crypto_agent_privileged_role_requires_human_approval",
				"required_action": "require_human_crypto_approval"
			}
		),
		CapabilityRule(
			name="bytewax_crypto_stream_required",
			description="Crypto lifecycle batch mutations must be routed through Bytewax.",
			condition={"operation": "validate_crypto_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_crypto_stream_required",
				"required_action": "route_crypto_lifecycle_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return ENCR UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/encr/dashboard", "EncryptionDashboard", "encr:view", "Overview"),
		CapabilityUIRoute("operations", "/encr/operations", "CryptoOperationsConsole", "encr:operate", "Operations"),
		CapabilityUIRoute("keys", "/encr/keys", "EncryptionKeyDomains", "encr:view_keys", "Operations"),
		CapabilityUIRoute("policies", "/encr/policies", "CryptoPolicyDesigner", "encr:manage_policies", "Governance"),
		CapabilityUIRoute("entropy", "/encr/entropy", "EntropyQualityConsole", "encr:view_entropy", "Governance"),
		CapabilityUIRoute("exceptions", "/encr/exceptions", "CryptoExceptionQueue", "encr:review", "Governance"),
		CapabilityUIRoute("rotations", "/encr/rotations", "KeyRotationConsole", "encr:rotate", "Operations"),
		CapabilityUIRoute("homomorphic", "/encr/homomorphic", "HomomorphicWorkspace", "encr:compute", "Advanced"),
		CapabilityUIRoute("analytics", "/encr/analytics", "CryptoAnalytics", "encr:view_analytics", "Intelligence"),
		CapabilityUIRoute("audit", "/encr/audit", "CryptoAuditTimeline", "encr:view", "Governance"),
		CapabilityUIRoute("agents", "/encr/agents", "CryptoAgentRoster", "encr:admin", "Administration"),
		CapabilityUIRoute("settings", "/encr/settings", "EncryptionSettings", "encr:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "web_ui.py",
		"api_prefix": "/encr/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return ENCR first-class AI agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_ENCR_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_ENCR_AGENT_ROLES),
		"privileged_roles": sorted(PRIVILEGED_ENCR_AGENT_ROLES),
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
	"""Return ENCR Bytewax lifecycle stream metadata."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "encr.lifecycle",
		"required_operations": [
			"crypto_lifecycle_batch",
			"key_rotation_batch",
			"crypto_agent_batch"
		],
		"topics": [
			"encr.key_domains",
			"encr.operations",
			"encr.rotations",
			"encr.agents"
		],
		"state": [
			"key_domains",
			"operations",
			"exception_reviews",
			"rotations",
			"crypto_agents"
		],
		"events": [
			"key_domain_registered",
			"crypto_operation_allowed",
			"crypto_operation_denied",
			"crypto_operation_review_required",
			"crypto_exception_decided",
			"key_rotation_completed",
			"crypto_agent_registered"
		],
		"watermark": "event_time"
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ENCR capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "encr",
		"display_name": "Encryption Services",
		"provides": ["encr_operations", "crypto_governance", "crypto_agent_composition", "review_evidence"],
		"requires": ["conf", "auth", "secu", "audl"],
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
				"review_required",
				"scheduled",
				"denied",
				"allowed",
				"approved",
				"rejected",
				"completed",
				"accepted",
			],
			"policy_fields": ["policy_decision", "matched_rules", "review_reasons", "review_evidence"],
			"pending_queues": ["operations", "exception_reviews", "rotations", "crypto_agents", "crypto_lifecycle_batches"],
			"deny_behavior": "Denied crypto lifecycle batches persist evidence before PermissionError",
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default ENCR rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
				return False
		elif key.endswith("_lt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) < expected:
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
