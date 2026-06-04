"""
Executable capability contract for APG Cache Management.

CACH is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic cache-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with CACH without loading optional
compression or UI runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_CACH_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CACH_AGENT_ROLES = [
	"namespace_policy_reviewer",
	"warming_reviewer",
	"eviction_reviewer",
	"freshness_reviewer",
	"tier_optimization_reviewer",
	"adapter_health_reviewer",
	"lifecycle_auditor",
]
PRIVILEGED_CACH_AGENT_ROLES = [
	"warming_reviewer",
	"eviction_reviewer",
	"tier_optimization_reviewer",
	"adapter_health_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CACH configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"hierarchy": {
			"default_tier": "memory",
			"tiers": ["memory", "distributed", "edge"],
			"max_memory_mb": 1024,
			"multi_tenant_isolation": True
		},
		"policy": {
			"default_ttl_seconds": 3600,
			"max_ttl_seconds": 86400,
			"default_eviction_policy": "adaptive_lru",
			"namespace_required": True,
			"critical_reads_require_freshness": True,
			"allow_stale_while_revalidate": True
		},
		"warming": {
			"predictive_prefetching_enabled": True,
			"source_registration_required": True,
			"max_warming_batch_size": 10000,
			"require_reason": True,
			"require_review_above_batch_limit": True
		},
		"security": {
			"require_tenant_context": True,
			"sensitive_entries_require_encryption": True,
			"regulated_entries_require_encryption": True,
			"restricted_entries_require_encryption": True,
			"cross_tenant_access_allowed": False,
			"quantum_ready_security": True
		},
		"optimization": {
			"ai_optimization_enabled": True,
			"memory_pressure_review_threshold_percent": 90,
			"auto_evict_on_pressure": True,
			"require_independent_eviction_review": True
		},
		"adapters": {
			"default_backend": "memory",
			"supported_backends": ["memory", "redis", "valkey", "edge", "cdn", "query_cache"],
			"backend_binding_required_for_production": True,
			"emit_mqeb_events": True
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_CACH_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_CACH_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_CACH_AGENT_ROLES,
			"require_owner": True,
			"require_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "cach.lifecycle",
			"watermark": "event_time",
			"required_operations": [
				"cache_policy_batch",
				"cache_warming_batch",
				"cache_agent_batch",
				"cache_eviction_batch"
			],
			"topics": [
				"cach.namespaces",
				"cach.entries",
				"cach.warming",
				"cach.evictions",
				"cach.agents"
			]
		},
		"telemetry": {
			"metrics_enabled": True,
			"track_access_patterns": True,
			"emit_cache_events": True,
			"record_rule_decisions": True,
			"record_lifecycle_audit": True
		},
		"ui": {
			"enable_dashboard": True,
			"enable_namespace_inventory": True,
			"enable_policy_manager": True,
			"enable_warming_console": True,
			"enable_eviction_review_queue": True,
			"enable_hierarchy_map": True,
			"enable_adapter_health": True,
			"enable_audit_timeline": True,
			"enable_cache_agent_roster": True,
			"enable_lifecycle_batch_monitor": True
		},
		"theme": {
			"default_theme": "cach_memory_fabric",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"hierarchy",
			"policy",
			"warming",
			"security",
			"optimization",
			"adapters",
			"agents",
			"streaming",
			"telemetry",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"hierarchy": {"type": "object"},
			"policy": {"type": "object"},
			"warming": {"type": "object"},
			"security": {"type": "object"},
			"optimization": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"telemetry": {"type": "object"},
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
	"""Simple CACH policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic CACH rule engine for cache governance decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching cache governance rules."""
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
	"""UI route exposed by CACH."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for CACH UI surfaces."""

	name: str = "cach_memory_fabric"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#31572C",
		"color.accent": "#4D908E",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7FAF8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#152017",
		"text.secondary": "#53635A",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"cache_hit_card": {
			"icon": "gauge",
			"status_indicator": "hit-rate-pill",
			"risk_style": "trend-sparkline"
		},
		"tier_hierarchy_map": {
			"visual": "layered-topology",
			"edge_style": "promotion-path"
		},
		"warming_plan_timeline": {
			"visual": "batch-timeline",
			"threshold_style": "source-readiness"
		},
		"namespace_policy_trace": {
			"visual": "rule-ladder",
			"highlight": "ttl-chip"
		},
		"entry_freshness_badge": {
			"visual": "age-band",
			"status_indicator": "freshness-state"
		},
		"eviction_review_queue": {
			"visual": "review-queue",
			"status_indicator": "pressure-band",
			"variant": "capacity-control"
		},
		"adapter_health_panel": {
			"visual": "backend-grid",
			"status_indicator": "adapter-state",
			"variant": "runtime-binding"
		},
		"audit_event_timeline": {
			"visual": "decision-timeline",
			"highlight": "matched-rule-chip"
		},
		"cache_agent_roster": {
			"icon": "bot",
			"status_indicator": "approval-state",
			"variant": "agent-governance"
		},
		"bytewax_lifecycle_panel": {
			"icon": "activity",
			"status_indicator": "processor-state",
			"variant": "stream-lifecycle"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CACH rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All cache operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="write_requires_namespace",
			description="Cache writes require an explicit namespace.",
			condition={"operation": "write", "namespace_present": False},
			effect={
				"decision": "deny",
				"reason": "namespace_required",
				"required_action": "select_cache_namespace"
			}
		),
		CapabilityRule(
			name="sensitive_entry_requires_encryption",
			description="Sensitive cache entries require encryption at rest.",
			condition={"data_classification": "sensitive", "entry_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "cache_entry_encryption_required",
				"required_action": "enable_entry_encryption"
			}
		),
		CapabilityRule(
			name="regulated_entry_requires_encryption",
			description="Regulated cache entries require encryption at rest.",
			condition={"data_classification": "regulated", "entry_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "cache_entry_encryption_required",
				"required_action": "enable_entry_encryption"
			}
		),
		CapabilityRule(
			name="restricted_entry_requires_encryption",
			description="Restricted cache entries require encryption at rest.",
			condition={"data_classification": "restricted", "entry_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "cache_entry_encryption_required",
				"required_action": "enable_entry_encryption"
			}
		),
		CapabilityRule(
			name="cross_tenant_cache_access_denied",
			description="Cross-tenant cache access is denied by default.",
			condition={"cross_tenant_access": True},
			effect={
				"decision": "deny",
				"reason": "cross_tenant_cache_access_denied",
				"required_action": "use_tenant_scoped_namespace"
			}
		),
		CapabilityRule(
			name="disabled_namespace_blocks_cache_writes",
			description="Disabled namespaces cannot accept writes or warming plans.",
			condition={"operation": "write", "namespace_status": "disabled"},
			effect={
				"decision": "deny",
				"reason": "namespace_disabled",
				"required_action": "reactivate_or_select_namespace"
			}
		),
		CapabilityRule(
			name="disabled_namespace_blocks_cache_warming",
			description="Disabled namespaces cannot accept warming plans.",
			condition={"operation": "warm", "namespace_status": "disabled"},
			effect={
				"decision": "deny",
				"reason": "namespace_disabled",
				"required_action": "reactivate_or_select_namespace"
			}
		),
		CapabilityRule(
			name="ttl_above_namespace_limit_requires_review",
			description="Entries above the namespace TTL limit require review.",
			condition={"ttl_above_namespace_limit": True},
			effect={
				"decision": "require_review",
				"reason": "ttl_review_required",
				"required_action": "request_ttl_exception"
			}
		),
		CapabilityRule(
			name="critical_stale_read_requires_refresh",
			description="Critical stale reads require refresh before serving.",
			condition={"operation": "read", "data_criticality": "critical", "entry_stale": True},
			effect={
				"decision": "deny",
				"reason": "critical_entry_refresh_required",
				"required_action": "refresh_cache_entry"
			}
		),
		CapabilityRule(
			name="warming_requires_registered_source",
			description="Cache warming requires registered source evidence.",
			condition={"operation": "warm", "source_registered": False},
			effect={
				"decision": "deny",
				"reason": "warming_source_required",
				"required_action": "register_warming_source"
			}
		),
		CapabilityRule(
			name="warming_requires_namespace",
			description="Cache warming requires a registered namespace.",
			condition={"operation": "warm", "namespace_present": False},
			effect={
				"decision": "deny",
				"reason": "namespace_required",
				"required_action": "create_or_select_namespace"
			}
		),
		CapabilityRule(
			name="warming_batch_limit_requires_review",
			description="Large cache warming batches require review.",
			condition={"warming_batch_above_limit": True},
			effect={
				"decision": "require_review",
				"reason": "warming_batch_review_required",
				"required_action": "request_warming_review"
			}
		),
		CapabilityRule(
			name="high_memory_pressure_requires_review",
			description="High memory pressure requires eviction or review.",
			condition={"memory_utilization_percent_gt": 90, "eviction_plan_ready": False},
			effect={
				"decision": "require_review",
				"reason": "memory_pressure_review_required",
				"required_action": "prepare_eviction_or_capacity_plan"
			}
		),
		CapabilityRule(
			name="eviction_review_requires_independent_reviewer",
			description="Eviction and capacity approvals require independent review.",
			condition={"reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="review_notes_required",
			description="Eviction, capacity, and warming reviews require notes.",
			condition={"review_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		),
		CapabilityRule(
			name="cache_agent_runtime_supported",
			description="Cache agents must use a supported runtime adapter.",
			condition={"operation": "register_cache_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_cache_agent_runtime",
				"required_action": "select_supported_agent_runtime"
			}
		),
		CapabilityRule(
			name="cache_agent_role_supported",
			description="Cache agents must use a supported cache governance role.",
			condition={"operation": "register_cache_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_cache_agent_role",
				"required_action": "select_supported_agent_role"
			}
		),
		CapabilityRule(
			name="cache_agent_requires_scope",
			description="Cache agents require an explicit operating scope.",
			condition={"operation": "register_cache_agent", "agent_scope_present": False},
			effect={
				"decision": "deny",
				"reason": "cache_agent_scope_required",
				"required_action": "attach_agent_scope"
			}
		),
		CapabilityRule(
			name="cache_agent_requires_owner",
			description="Cache agents require an accountable owner.",
			condition={"operation": "register_cache_agent", "agent_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "cache_agent_owner_required",
				"required_action": "attach_agent_owner"
			}
		),
		CapabilityRule(
			name="cache_agent_requires_purpose",
			description="Cache agents require a declared purpose.",
			condition={"operation": "register_cache_agent", "agent_purpose_present": False},
			effect={
				"decision": "deny",
				"reason": "cache_agent_purpose_required",
				"required_action": "attach_agent_purpose"
			}
		),
		CapabilityRule(
			name="cache_agent_requires_contribution_disclosure",
			description="Cache agents must disclose machine contribution in cache decisions.",
			condition={"operation": "register_cache_agent", "contribution_disclosed": False},
			effect={
				"decision": "deny",
				"reason": "cache_agent_contribution_disclosure_required",
				"required_action": "enable_agent_contribution_disclosure"
			}
		),
		CapabilityRule(
			name="cache_agent_privileged_role_requires_human_approval",
			description="Privileged cache-agent roles require human approval evidence or review.",
			condition={"operation": "register_cache_agent", "privileged_agent_role": True, "human_approval_required": False},
			effect={
				"decision": "require_review",
				"reason": "cache_agent_human_approval_required",
				"required_action": "require_human_approval_for_agent"
			}
		),
		CapabilityRule(
			name="bytewax_cache_stream_required",
			description="CACH lifecycle batches must declare Bytewax as the cache lifecycle processor.",
			condition={"operation": "validate_cache_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_cache_stream_required",
				"required_action": "route_batch_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CACH UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/cach/dashboard", "CacheDashboard", "cach:view", "Overview"),
		CapabilityUIRoute("namespaces", "/cach/namespaces", "CacheNamespaceInventory", "cach:manage_namespaces", "Operations"),
		CapabilityUIRoute("entries", "/cach/entries", "CacheEntryExplorer", "cach:read", "Operations"),
		CapabilityUIRoute("policies", "/cach/policies", "CachePolicyManager", "cach:manage_policies", "Governance"),
		CapabilityUIRoute("warming", "/cach/warming", "CacheWarmingConsole", "cach:warm", "Operations"),
		CapabilityUIRoute("evictions", "/cach/evictions", "CacheEvictionReviewQueue", "cach:review_eviction", "Governance"),
		CapabilityUIRoute("hierarchy", "/cach/hierarchy", "CacheHierarchyMap", "cach:view", "Architecture"),
		CapabilityUIRoute("analytics", "/cach/analytics", "CacheAnalytics", "cach:view_analytics", "Intelligence"),
		CapabilityUIRoute("security", "/cach/security", "CacheSecurityView", "cach:admin", "Governance"),
		CapabilityUIRoute("adapters", "/cach/adapters", "CacheAdapterHealth", "cach:admin", "Runtime"),
		CapabilityUIRoute("agents", "/cach/agents", "CacheAgentRoster", "cach:admin", "Administration"),
		CapabilityUIRoute("lifecycle", "/cach/lifecycle", "CacheLifecycleBatchMonitor", "cach:admin", "Runtime"),
		CapabilityUIRoute("audit", "/cach/audit", "CacheAuditTimeline", "cach:admin", "Governance"),
		CapabilityUIRoute("settings", "/cach/settings", "CacheSettings", "cach:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "dashboard.py",
		"api_prefix": "/cach/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class CACH agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_CACH_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_CACH_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_CACH_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles"
		]
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return CACH lifecycle stream-processing contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "cach.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"cache_policy_batch",
			"cache_warming_batch",
			"cache_agent_batch",
			"cache_eviction_batch"
		],
		"topics": [
			"cach.namespaces",
			"cach.entries",
			"cach.warming",
			"cach.evictions",
			"cach.agents"
		],
		"broker_core_dependency_allowed": False
	}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.cach.lifecycle",
	"key": "tenant_id",
	"events": [
		"namespace_created",
		"namespace_updated",
		"namespace_evicted",
		"cache_warmed",
		"cache_invalidated",
		"tier_promoted",
		"tier_demoted",
		"eviction_policy_changed",
		"freshness_review_required",
		"adapter_health_changed",
		"agent_registered",
	],
	"guardrails": [
		"cach_batch_requires_bytewax",
		"cach_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CACH capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "cach",
		"display_name": "Cache Management",
		"provides": ["cache_governance", "cache_runtime_adapters", "cache_agent_composition", "review_evidence"],
		"requires": ["conf", "auth", "audl"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"agents": agent_manifest(),
		"streaming": STREAMING,
		"review_evidence": {
			"durable_statuses": [
				"pending",
				"pending_review",
				"review_required",
				"denied",
				"active",
				"expired",
				"invalidated",
				"refresh_required",
				"ready",
				"approved",
				"rejected",
				"accepted"
			],
			"policy_fields": [
				"policy_decision",
				"matched_rules",
				"review_reasons",
				"review_evidence"
			],
			"pending_queues": [
				"entries",
				"warming_plans",
				"eviction_reviews",
				"cache_agents",
				"lifecycle_batches"
			],
			"deny_behavior": "Denied CACH lifecycle batches persist evidence before PermissionError"
		},
		"theme": {
			"name": theme.name,
			"tokens": theme.tokens,
			"components": theme.components
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default CACH rule evaluation."""
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
