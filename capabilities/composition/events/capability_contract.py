"""Executable capability contract for APG event streaming."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_EVENT_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_EVENT_AGENT_ROLES = [
	"stream_architect",
	"schema_reviewer",
	"processor_reviewer",
	"subscription_reviewer",
	"dead_letter_reviewer",
	"replay_reviewer",
]
EVENT_BUS_STREAM = "apg.composition.events.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"streams": {
		"owner_required": True,
		"schema_required_for_pii": True,
		"retention_policy_required": True,
		"partition_key_required": True,
		"bytewax_stream_required": True,
	},
	"schemas": {
		"compatibility_required": True,
		"review_required_for_breaking_change": True,
		"versioning_enabled": True,
	},
	"publishing": {
		"source_capability_required": True,
		"correlation_required": True,
		"bytewax_required": True,
		"batch_size_limit": 1000,
	},
	"subscriptions": {
		"consumer_owner_required": True,
		"dead_letter_required_for_retrying": True,
		"delivery_mode_required": True,
	},
	"processors": {
		"bytewax_required": True,
		"review_required_for_stateful": True,
		"checkpoint_required": True,
	},
	"event_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_EVENT_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_EVENT_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_and_validate",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"replay_requires_approval": True,
	},
	"observability": {
		"event_stream": EVENT_BUS_STREAM,
		"stream_processor": "bytewax",
		"emit_stream_events": True,
		"emit_schema_events": True,
		"emit_subscription_events": True,
		"emit_processor_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_stream_console": True,
		"enable_schema_registry": True,
		"enable_subscription_console": True,
		"enable_processor_topology": True,
		"enable_dead_letter_console": True,
		"enable_agent_workbench": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_events_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"streams",
		"schemas",
		"publishing",
		"subscriptions",
		"processors",
		"event_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"streams": {"type": "object"},
		"schemas": {"type": "object"},
		"publishing": {"type": "object"},
		"subscriptions": {"type": "object"},
		"processors": {"type": "object"},
		"event_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All event-bus operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "event_write_requires_policy", "description": "Event-bus write operations require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "stream_requires_owner", "description": "Streams require an accountable owner.", "condition": {"operation": "create_stream", "stream_owner_assigned": False}, "effect": {"decision": "deny", "reason": "stream_owner_required", "required_action": "assign_stream_owner"}},
	{"name": "stream_requires_retention_policy", "description": "Streams require a retention policy.", "condition": {"operation": "create_stream", "retention_policy_present": False}, "effect": {"decision": "deny", "reason": "stream_retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "pii_stream_requires_schema", "description": "Streams carrying PII require a schema.", "condition": {"operation": "create_stream", "pii_stream": True, "schema_attached": False}, "effect": {"decision": "deny", "reason": "stream_schema_required", "required_action": "attach_event_schema"}},
	{"name": "stream_requires_bytewax", "description": "Stream lifecycle events must use Bytewax.", "condition": {"operation": "create_stream", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_stream_lifecycle_to_bytewax"}},
	{"name": "breaking_schema_requires_review", "description": "Breaking schema changes require review.", "condition": {"operation": "register_schema", "breaking_change": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "schema_review_required", "required_action": "record_schema_review"}},
	{"name": "publish_requires_source_capability", "description": "Published events require source-capability attribution.", "condition": {"operation": "publish_event", "source_capability_present": False}, "effect": {"decision": "deny", "reason": "source_capability_required", "required_action": "attach_source_capability"}},
	{"name": "publish_requires_correlation", "description": "Published events require correlation or causation context.", "condition": {"operation": "publish_event", "correlation_present": False}, "effect": {"decision": "deny", "reason": "event_correlation_required", "required_action": "attach_correlation_context"}},
	{"name": "publish_requires_bytewax", "description": "Published events must be appended through Bytewax.", "condition": {"operation": "publish_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "append_event_to_bytewax"}},
	{"name": "batch_publish_limit", "description": "Batch publishing must stay within configured limits.", "condition": {"operation": "batch_publish", "batch_size_gt": 1000}, "effect": {"decision": "deny", "reason": "batch_size_limit_exceeded", "required_action": "split_event_batch"}},
	{"name": "batch_publish_requires_bytewax", "description": "Batch publishing requires Bytewax.", "condition": {"operation": "batch_publish", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "append_batch_to_bytewax"}},
	{"name": "subscription_requires_owner", "description": "Subscriptions require a consumer owner.", "condition": {"operation": "create_subscription", "consumer_owner_assigned": False}, "effect": {"decision": "deny", "reason": "consumer_owner_required", "required_action": "assign_consumer_owner"}},
	{"name": "retry_subscription_requires_dead_letter", "description": "Retrying subscriptions require a dead-letter stream.", "condition": {"operation": "create_subscription", "retry_enabled": True, "dead_letter_attached": False}, "effect": {"decision": "deny", "reason": "dead_letter_required", "required_action": "attach_dead_letter_stream"}},
	{"name": "stateful_processor_requires_review", "description": "Stateful processors require review.", "condition": {"operation": "register_processor", "stateful_processor": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "processor_review_required", "required_action": "record_processor_review"}},
	{"name": "processor_requires_checkpoint", "description": "Processors require checkpoint configuration.", "condition": {"operation": "register_processor", "checkpoint_configured": False}, "effect": {"decision": "deny", "reason": "processor_checkpoint_required", "required_action": "configure_processor_checkpoint"}},
	{"name": "processor_requires_bytewax", "description": "Processors must run on Bytewax.", "condition": {"operation": "register_processor", "processor_runtime_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_processor_required", "required_action": "select_bytewax_processor"}},
	{"name": "replay_requires_approval", "description": "Event replay requires approval.", "condition": {"operation": "replay_events", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "event_replay_approval_required", "required_action": "record_replay_approval"}},
	{"name": "event_agent_runtime_supported", "description": "Event agents must use an approved runtime.", "condition": {"operation": "register_event_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "event_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "event_agent_role_supported", "description": "Event agents must use an approved role.", "condition": {"operation": "register_event_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "event_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_event_action_requires_human_approval", "description": "Privileged event actions proposed by agents require human approval.", "condition": {"operation": "agent_event_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-events/dashboard", "component": "EventBusDashboard", "permission": "composition_events:view", "nav_group": "Overview"},
	{"name": "streams", "path": "/composition-events/streams", "component": "EventStreamConsole", "permission": "composition_events:manage_streams", "nav_group": "Streams"},
	{"name": "schemas", "path": "/composition-events/schemas", "component": "EventSchemaRegistry", "permission": "composition_events:govern", "nav_group": "Governance"},
	{"name": "subscriptions", "path": "/composition-events/subscriptions", "component": "EventSubscriptionConsole", "permission": "composition_events:operate", "nav_group": "Consumers"},
	{"name": "processors", "path": "/composition-events/processors", "component": "EventProcessorTopology", "permission": "composition_events:operate", "nav_group": "Processing"},
	{"name": "dead_letters", "path": "/composition-events/dead-letters", "component": "DeadLetterConsole", "permission": "composition_events:operate", "nav_group": "Operations"},
	{"name": "agents", "path": "/composition-events/agents", "component": "EventAgentWorkbench", "permission": "composition_events:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/composition-events/settings", "component": "EventBusSettings", "permission": "composition_events:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_events_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"stream_console": {"icon": "route", "status_indicator": "stream-pill", "risk_style": "throughput-band"},
		"schema_registry": {"visual": "schema-grid", "status_style": "compatibility-chip"},
		"subscription_console": {"visual": "consumer-lanes", "status_style": "lag-chip"},
		"processor_topology": {"visual": "dataflow-map", "status_style": "checkpoint-chip"},
		"dead_letter_console": {"visual": "error-queue", "status_style": "retry-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_events",
		"display_name": "Event Streaming Bus",
		"provides": [
			"event_stream_registry",
			"bytewax_event_publishing",
			"event_schema_registry",
			"subscription_lifecycle",
			"stream_processor_topology",
			"dead_letter_operations",
			"event_agents",
		],
		"requires": ["auth", "audl", "ntfy", "registry", "composition_access"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/composition-events/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": EVENT_BUS_STREAM,
		"key": "tenant_id",
		"events": [
			"stream_created",
			"schema_registered",
			"event_published",
			"event_batch_published",
			"subscription_created",
			"processor_registered",
			"dead_letter_recorded",
			"events_replayed",
			"event_agent_registered",
		],
		"states": ["draft", "active", "paused", "review_required", "processing", "degraded", "blocked", "retired"],
		"guardrails": [
			"stream_requires_bytewax",
			"publish_requires_bytewax",
			"batch_publish_requires_bytewax",
			"processor_requires_bytewax",
			"privileged_agent_event_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return EVENT_BUS_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
