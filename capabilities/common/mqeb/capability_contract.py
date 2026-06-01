"""
Executable capability contract for APG Message Queue Event Bus.

MQEB is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic message-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with MQEB without loading the full
messaging UI runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_MQEB_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_MQEB_AGENT_ROLES = [
	"routing_reviewer",
	"delivery_reliability_reviewer",
	"quota_reviewer",
	"replay_reviewer",
	"schema_governance_reviewer",
	"bytewax_topology_reviewer",
	"dead_letter_triage",
]
PRIVILEGED_MQEB_AGENT_ROLES = [
	"quota_reviewer",
	"replay_reviewer",
	"bytewax_topology_reviewer",
	"dead_letter_triage",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped MQEB configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"broker": {
			"default_protocol": "http_rest",
			"max_message_size_mb": 100,
			"max_topics_per_tenant": 1000,
			"dead_letter_queues_enabled": True
		},
		"delivery": {
			"default_mode": "at_least_once",
			"retention_days": 7,
			"require_dead_letter_for_guaranteed_delivery": True,
			"enable_idempotency_keys": True,
			"require_idempotency_for_exactly_once": True
		},
		"routing": {
			"ai_routing_enabled": True,
			"schema_registry_required": True,
			"cross_tenant_publish_allowed": False,
			"priority_quota_review_threshold": 10000
		},
		"security": {
			"require_tenant_context": True,
			"restricted_topics_require_encryption": True,
			"quantum_safe_encryption_enabled": True,
			"sign_messages": True
		},
		"compliance": {
			"audit_publish_and_subscribe": True,
			"frameworks": ["GDPR", "HIPAA", "PCI_DSS", "SOX"],
			"retention_policy_required": True
		},
		"scaling": {
			"predictive_scaling_enabled": True,
			"edge_federation_enabled": True,
			"max_concurrent_connections": 1000000
		},
		"operation_governance": {
			"bytewax_first_runtime": True,
			"broker_core_dependency_allowed": False,
			"require_independent_quota_review": True,
			"require_independent_replay_review": True,
			"record_delivery_audit": True
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_MQEB_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_MQEB_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_MQEB_AGENT_ROLES,
			"require_owner": True,
			"require_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "mqeb.lifecycle",
			"watermark": "event_time",
			"required_operations": [
				"event_fabric_batch",
				"message_delivery_batch",
				"event_agent_batch"
			],
			"topics": [
				"mqeb.topics",
				"mqeb.messages",
				"mqeb.subscriptions",
				"mqeb.replays",
				"mqeb.agents"
			]
		},
		"ui": {
			"enable_dashboard": True,
			"enable_topic_manager": True,
			"enable_routing_designer": True,
			"enable_scaling_console": True,
			"enable_dead_letter_console": True,
			"enable_replay_console": True,
			"enable_quota_exception_queue": True,
			"enable_bytewax_bridge": True,
			"enable_event_agent_roster": True
		},
		"theme": {
			"default_theme": "mqeb_event_fabric",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"broker",
			"delivery",
			"routing",
			"security",
			"compliance",
			"scaling",
			"operation_governance",
			"agents",
			"streaming",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"broker": {"type": "object"},
			"delivery": {"type": "object"},
			"routing": {"type": "object"},
			"security": {"type": "object"},
			"compliance": {"type": "object"},
			"scaling": {"type": "object"},
			"operation_governance": {"type": "object"},
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
	"""Simple MQEB policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic MQEB rule engine for message governance decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching messaging governance rules."""
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
	"""UI route exposed by MQEB."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for MQEB UI surfaces."""

	name: str = "mqeb_event_fabric"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#26547C",
		"color.accent": "#06A77D",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#53627A",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"topic_health_card": {
			"icon": "radio-tower",
			"status_indicator": "lag-pill",
			"risk_style": "throughput-band"
		},
		"message_flow_map": {
			"visual": "directed-event-graph",
			"edge_style": "delivery-mode-line"
		},
		"routing_rule_trace": {
			"visual": "stacked-rule-list",
			"highlight": "selected-route-chip"
		},
		"consumer_lag_meter": {
			"visual": "segmented-meter",
			"threshold_style": "lag-bands"
		},
		"dead_letter_queue": {
			"icon": "mail-warning",
			"status_indicator": "failure-count",
			"variant": "exception"
		},
		"replay_console": {
			"icon": "history",
			"status_indicator": "review-state",
			"variant": "evidence"
		},
		"quota_exception_queue": {
			"icon": "gauge",
			"status_indicator": "quota-band",
			"variant": "review"
		},
		"bytewax_bridge_panel": {
			"icon": "git-branch",
			"status_indicator": "adapter-state",
			"variant": "stream-runtime"
		},
		"event_agent_roster": {
			"icon": "bot",
			"status_indicator": "approval-state",
			"variant": "agent-governance"
		},
		"bytewax_stream_indicator": {
			"icon": "activity",
			"status_indicator": "processor-state",
			"variant": "stream-lifecycle"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default MQEB rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All message operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="publish_requires_topic",
			description="Publish operations require an existing topic.",
			condition={"operation": "publish", "topic_exists": False},
			effect={
				"decision": "deny",
				"reason": "topic_required",
				"required_action": "create_or_select_topic"
			}
		),
		CapabilityRule(
			name="restricted_topic_requires_encryption",
			description="Restricted topics require encrypted message transport.",
			condition={"topic_classification": "restricted", "message_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "message_encryption_required",
				"required_action": "enable_topic_encryption"
			}
		),
		CapabilityRule(
			name="regulated_topic_requires_schema",
			description="Regulated topics require schema evidence before publish.",
			condition={"topic_classification": "regulated", "schema_ref_present": False},
			effect={
				"decision": "deny",
				"reason": "schema_reference_required",
				"required_action": "attach_schema_reference"
			}
		),
		CapabilityRule(
			name="regulated_topic_requires_encryption",
			description="Regulated topics require encrypted message transport.",
			condition={"topic_classification": "regulated", "message_encrypted": False},
			effect={
				"decision": "deny",
				"reason": "message_encryption_required",
				"required_action": "enable_topic_encryption"
			}
		),
		CapabilityRule(
			name="cross_tenant_publish_denied",
			description="Cross-tenant publish is denied by default.",
			condition={"cross_tenant_publish": True},
			effect={
				"decision": "deny",
				"reason": "cross_tenant_publish_denied",
				"required_action": "route_through_authorized_exchange"
			}
		),
		CapabilityRule(
			name="guaranteed_delivery_requires_dead_letter_queue",
			description="Guaranteed delivery requires a configured dead-letter queue.",
			condition={"delivery_mode": "exactly_once", "dead_letter_queue_configured": False},
			effect={
				"decision": "deny",
				"reason": "dead_letter_queue_required",
				"required_action": "configure_dead_letter_queue"
			}
		),
		CapabilityRule(
			name="exactly_once_requires_idempotency_key",
			description="Exactly-once publish requires an idempotency key.",
			condition={"delivery_mode": "exactly_once", "idempotency_key_present": False},
			effect={
				"decision": "deny",
				"reason": "idempotency_key_required",
				"required_action": "attach_idempotency_key"
			}
		),
		CapabilityRule(
			name="disabled_topic_blocks_publish",
			description="Disabled topics cannot accept new messages.",
			condition={"operation": "publish", "topic_status": "disabled"},
			effect={
				"decision": "deny",
				"reason": "topic_disabled",
				"required_action": "reactivate_or_select_topic"
			}
		),
		CapabilityRule(
			name="paused_subscription_blocks_delivery",
			description="Paused subscriptions cannot receive deliveries.",
			condition={"operation": "deliver", "subscription_status": "paused"},
			effect={
				"decision": "deny",
				"reason": "subscription_paused",
				"required_action": "resume_subscription"
			}
		),
		CapabilityRule(
			name="priority_quota_exhaustion_requires_review",
			description="High priority publish volume above quota requires review.",
			condition={"priority_messages_per_minute_gt": 10000, "quota_exception_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "priority_quota_review_required",
				"required_action": "record_priority_quota_exception"
			}
		),
		CapabilityRule(
			name="replay_requires_bounded_range",
			description="Replay requests require a bounded range and reason.",
			condition={"operation": "replay", "replay_range_bounded": False},
			effect={
				"decision": "deny",
				"reason": "replay_range_required",
				"required_action": "attach_replay_range"
			}
		),
		CapabilityRule(
			name="replay_requires_reason",
			description="Replay requests require an operational reason.",
			condition={"operation": "replay", "replay_reason_present": False},
			effect={
				"decision": "deny",
				"reason": "replay_reason_required",
				"required_action": "attach_replay_reason"
			}
		),
		CapabilityRule(
			name="review_requires_independent_reviewer",
			description="MQEB exceptions and replay approvals require independent review.",
			condition={"reviewer_same_as_requester": True},
			effect={
				"decision": "deny",
				"reason": "independent_reviewer_required",
				"required_action": "assign_independent_reviewer"
			}
		),
		CapabilityRule(
			name="review_requires_notes",
			description="MQEB reviews require notes or evidence.",
			condition={"review_notes_attached": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		),
		CapabilityRule(
			name="event_agent_runtime_supported",
			description="MQEB event agents must use a supported runtime adapter.",
			condition={"operation": "register_event_agent", "agent_runtime_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_event_agent_runtime",
				"required_action": "select_supported_agent_runtime"
			}
		),
		CapabilityRule(
			name="event_agent_role_supported",
			description="MQEB event agents must use a supported event-fabric role.",
			condition={"operation": "register_event_agent", "agent_role_supported": False},
			effect={
				"decision": "deny",
				"reason": "unsupported_event_agent_role",
				"required_action": "select_supported_agent_role"
			}
		),
		CapabilityRule(
			name="event_agent_requires_scope",
			description="MQEB event agents require an explicit operating scope.",
			condition={"operation": "register_event_agent", "agent_scope_present": False},
			effect={
				"decision": "deny",
				"reason": "event_agent_scope_required",
				"required_action": "attach_agent_scope"
			}
		),
		CapabilityRule(
			name="event_agent_requires_owner",
			description="MQEB event agents require an accountable owner.",
			condition={"operation": "register_event_agent", "agent_owner_present": False},
			effect={
				"decision": "deny",
				"reason": "event_agent_owner_required",
				"required_action": "attach_agent_owner"
			}
		),
		CapabilityRule(
			name="event_agent_requires_purpose",
			description="MQEB event agents require a declared purpose.",
			condition={"operation": "register_event_agent", "agent_purpose_present": False},
			effect={
				"decision": "deny",
				"reason": "event_agent_purpose_required",
				"required_action": "attach_agent_purpose"
			}
		),
		CapabilityRule(
			name="event_agent_requires_contribution_disclosure",
			description="MQEB event agents must disclose machine contribution in event decisions.",
			condition={"operation": "register_event_agent", "contribution_disclosed": False},
			effect={
				"decision": "deny",
				"reason": "event_agent_contribution_disclosure_required",
				"required_action": "enable_agent_contribution_disclosure"
			}
		),
		CapabilityRule(
			name="event_agent_privileged_role_requires_human_approval",
			description="Privileged MQEB event-agent roles require human approval evidence or review.",
			condition={"operation": "register_event_agent", "privileged_agent_role": True, "human_approval_required": False},
			effect={
				"decision": "require_review",
				"reason": "event_agent_human_approval_required",
				"required_action": "require_human_approval_for_agent"
			}
		),
		CapabilityRule(
			name="bytewax_event_stream_required",
			description="MQEB lifecycle batches must declare Bytewax as the event stream processor.",
			condition={"operation": "validate_event_lifecycle_batch", "event_stream_ne": "bytewax"},
			effect={
				"decision": "deny",
				"reason": "bytewax_event_stream_required",
				"required_action": "route_batch_through_bytewax"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return MQEB UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/mqeb/dashboard", "MQEBDashboard", "mqeb:view", "Overview"),
		CapabilityUIRoute("topics", "/mqeb/topics", "TopicManagementView", "mqeb:manage_topics", "Operations"),
		CapabilityUIRoute("publish", "/mqeb/publish", "MessagePublishingView", "mqeb:publish", "Operations"),
		CapabilityUIRoute("subscriptions", "/mqeb/subscriptions", "SubscriptionManagementView", "mqeb:subscribe", "Operations"),
		CapabilityUIRoute("delivery", "/mqeb/delivery", "DeliveryAttemptConsole", "mqeb:view_metrics", "Reliability"),
		CapabilityUIRoute("dead_letters", "/mqeb/dead-letters", "DeadLetterQueue", "mqeb:manage_routing", "Reliability"),
		CapabilityUIRoute("quota_exceptions", "/mqeb/quota-exceptions", "PriorityQuotaExceptionQueue", "mqeb:admin", "Governance"),
		CapabilityUIRoute("replays", "/mqeb/replays", "ReplayConsole", "mqeb:admin", "Governance"),
		CapabilityUIRoute("agents", "/mqeb/agents", "EventAgentRoster", "mqeb:admin", "Administration"),
		CapabilityUIRoute("bytewax", "/mqeb/bytewax", "BytewaxBridgeStatus", "mqeb:admin", "Runtime"),
		CapabilityUIRoute("routing", "/mqeb/routing", "RoutingDesigner", "mqeb:manage_routing", "Governance"),
		CapabilityUIRoute("scaling", "/mqeb/scaling", "PredictiveScalingConsole", "mqeb:admin", "Reliability"),
		CapabilityUIRoute("monitoring", "/mqeb/monitoring", "MonitoringView", "mqeb:view_metrics", "Reliability"),
		CapabilityUIRoute("settings", "/mqeb/settings", "MQEBSettings", "mqeb:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/mqeb/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class MQEB event-agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_MQEB_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_MQEB_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_MQEB_AGENT_ROLES),
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
	"""Return MQEB stream-processing contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "mqeb.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"event_fabric_batch",
			"message_delivery_batch",
			"event_agent_batch"
		],
		"topics": [
			"mqeb.topics",
			"mqeb.messages",
			"mqeb.subscriptions",
			"mqeb.replays",
			"mqeb.agents"
		],
		"broker_core_dependency_allowed": False
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MQEB capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "mqeb",
		"display_name": "Message Queue Event Bus",
		"provides": ["mqeb_event_fabric", "message_governance", "event_agent_composition", "review_evidence"],
		"requires": ["conf", "auth", "audl", "secu"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {
			"type": "deterministic",
			"rules": [rule.__dict__ for rule in default_rules()]
		},
		"ui": ui_manifest(),
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": {
			"name": theme.name,
			"tokens": theme.tokens,
			"components": theme.components
		},
		"review_evidence": {
			"durable_statuses": [
				"pending",
				"pending_review",
				"review_required",
				"denied",
				"published",
				"delivered",
				"retry",
				"dead_letter",
				"approved",
				"rejected",
				"accepted",
			],
			"policy_fields": ["policy_decision", "matched_rules", "review_reasons", "review_evidence"],
			"pending_queues": [
				"messages",
				"priority_exceptions",
				"replay_requests",
				"event_agents",
				"lifecycle_batches",
			],
			"deny_behavior": "Denied MQEB lifecycle batches persist evidence before PermissionError",
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default MQEB rule evaluation."""
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
