"""Executable capability contract for APG event streaming bus."""

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
SUPPORTED_DELIVERY_MODES = ["at_least_once", "at_most_once", "exactly_once"]
SUPPORTED_SCHEMA_COMPATIBILITY_MODES = ["backward", "forward", "full", "none"]
SUPPORTED_PARTITION_KEY_STRATEGIES = ["tenant_id", "entity_id", "event_type", "custom"]
SUPPORTED_RETENTION_UNITS = ["hours", "days", "weeks", "months", "forever"]
SUPPORTED_PROCESSOR_STATES = ["registered", "running", "paused", "degraded", "failed", "retired"]
SUPPORTED_CIRCUIT_BREAKER_STATES = ["closed", "open", "half_open"]
SUPPORTED_STREAM_TIERS = ["standard", "priority", "critical"]
SUPPORTED_DEAD_LETTER_STRATEGIES = ["discard", "retry", "requeue", "alert"]
SUPPORTED_REPLAY_MODES = ["full", "partial", "from_offset", "from_timestamp"]
SUPPORTED_COMPRESSION_TYPES = ["none", "gzip", "snappy", "lz4"]
SUPPORTED_ENCODING_FORMATS = ["json", "avro", "protobuf", "msgpack"]
SUPPORTED_BACKPRESSURE_STRATEGIES = ["drop", "block", "buffer", "shed"]

EVENT_BUS_STREAM = "apg.composition.events.lifecycle"

PROVIDES = [
	"event_stream_registry",
	"bytewax_event_publishing",
	"event_schema_registry",
	"subscription_lifecycle",
	"stream_processor_topology",
	"dead_letter_operations",
	"event_agents",
	"cross_tenant_event_isolation",
	"circuit_breaker_event_gate",
	"cascading_failure_stream_containment",
	"event_replay_governance",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"moni",
	"conf",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"streams": {
		"owner_required": True,
		"schema_required_for_pii": True,
		"retention_policy_required": True,
		"partition_key_required": True,
		"bytewax_stream_required": True,
		"tier_required": True,
		"cross_tenant_isolation_enforced": True,
	},
	"schemas": {
		"compatibility_required": True,
		"review_required_for_breaking_change": True,
		"versioning_enabled": True,
		"encoding_format_required": True,
	},
	"publishing": {
		"source_capability_required": True,
		"correlation_required": True,
		"bytewax_required": True,
		"batch_size_limit": 1000,
		"backpressure_strategy": "buffer",
	},
	"subscriptions": {
		"consumer_owner_required": True,
		"dead_letter_required_for_retrying": True,
		"delivery_mode_required": True,
		"max_lag_threshold_enabled": True,
	},
	"processors": {
		"bytewax_required": True,
		"review_required_for_stateful": True,
		"checkpoint_required": True,
		"backpressure_strategy_required": True,
	},
	"circuit_breaker": {
		"enabled": True,
		"failure_threshold": 5,
		"recovery_timeout_seconds": 30,
		"half_open_probe_count": 3,
		"cascade_isolation_enabled": True,
		"per_stream_breaker_enabled": True,
	},
	"cascading_failure": {
		"dependency_health_check_enabled": True,
		"bulkhead_isolation_enabled": True,
		"max_downstream_stream_failures": 3,
		"quarantine_stream_on_cascade": True,
		"shed_load_on_cascade": True,
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
		"privilege_escalation_blocked": True,
		"cross_tenant_publish_blocked": True,
	},
	"observability": {
		"event_stream": EVENT_BUS_STREAM,
		"stream_processor": "bytewax",
		"emit_stream_events": True,
		"emit_schema_events": True,
		"emit_subscription_events": True,
		"emit_processor_events": True,
		"emit_circuit_breaker_events": True,
		"emit_cascade_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"monitoring": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_stream_console": True,
		"enable_schema_registry": True,
		"enable_subscription_console": True,
		"enable_processor_topology": True,
		"enable_dead_letter_console": True,
		"enable_circuit_breaker_console": True,
		"enable_replay_console": True,
		"enable_agent_workbench": True,
		"enable_audit_console": True,
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
		"circuit_breaker",
		"cascading_failure",
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
		"circuit_breaker": {"type": "object"},
		"cascading_failure": {"type": "object"},
		"event_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	# --- Tenant context (hard gate) ---
	{
		"name": "tenant_context_required",
		"description": "All event-bus operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	# --- Write-requires-policy ---
	{
		"name": "event_write_requires_policy",
		"description": "Event-bus write operations require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	# --- Cross-tenant isolation ---
	{
		"name": "cross_tenant_publish_blocked",
		"description": "Events may not be published to a stream owned by a different tenant.",
		"condition": {"cross_tenant_publish_attempted": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_publish_forbidden", "required_action": "reject_cross_tenant_publish"},
	},
	{
		"name": "cross_tenant_subscription_blocked",
		"description": "Consumers may not subscribe to streams owned by a different tenant without explicit federation approval.",
		"condition": {"operation": "create_subscription", "cross_tenant_stream": True, "federation_approved": False},
		"effect": {"decision": "deny", "reason": "cross_tenant_subscription_forbidden", "required_action": "request_stream_federation_approval"},
	},
	# --- Privilege escalation prevention ---
	{
		"name": "stream_privilege_escalation_blocked",
		"description": "A principal may not create a stream with a tier higher than their authorised scope.",
		"condition": {"operation": "create_stream", "stream_tier_exceeds_principal_scope": True},
		"effect": {"decision": "deny", "reason": "stream_privilege_escalation_forbidden", "required_action": "reduce_stream_tier"},
	},
	# --- Circuit breaker rules ---
	{
		"name": "circuit_breaker_open_blocks_publish",
		"description": "When the stream circuit breaker is open, publish operations are denied.",
		"condition": {"circuit_breaker_state": "open", "operation": "publish_event"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_open", "required_action": "wait_for_circuit_recovery"},
	},
	{
		"name": "circuit_breaker_half_open_limits_publish",
		"description": "In half-open state only probe publishes are permitted; excess is shed.",
		"condition": {"circuit_breaker_state": "half_open", "probe_budget_exhausted": True},
		"effect": {"decision": "deny", "reason": "circuit_breaker_half_open_budget_exhausted", "required_action": "shed_publish_load"},
	},
	{
		"name": "circuit_breaker_trip_requires_event",
		"description": "Circuit breaker state transitions must emit a Bytewax lifecycle event.",
		"condition": {"operation": "trip_circuit_breaker", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_event_required", "required_action": "emit_circuit_breaker_event_to_bytewax"},
	},
	# --- Cascading failure containment ---
	{
		"name": "cascade_isolation_on_stream_failure",
		"description": "When downstream stream failures exceed threshold, quarantine the stream.",
		"condition": {"downstream_failure_count_gt": 3, "stream_quarantine_active": False},
		"effect": {"decision": "require_review", "reason": "stream_cascade_isolation_required", "required_action": "quarantine_failing_stream"},
	},
	{
		"name": "bulkhead_overflow_sheds_stream_load",
		"description": "Publish requests exceeding per-tenant bulkhead capacity are denied.",
		"condition": {"bulkhead_capacity_exceeded": True, "operation": "publish_event"},
		"effect": {"decision": "deny", "reason": "bulkhead_capacity_exceeded", "required_action": "shed_excess_publish_load"},
	},
	{
		"name": "processor_backpressure_activates_shed",
		"description": "When a processor signals backpressure, new publishes to its input stream are shed.",
		"condition": {"processor_backpressure_active": True, "operation": "publish_event"},
		"effect": {"decision": "deny", "reason": "processor_backpressure_active", "required_action": "apply_backpressure_strategy"},
	},
	# --- Stream lifecycle ---
	{
		"name": "stream_requires_owner",
		"description": "Streams require an accountable owner.",
		"condition": {"operation": "create_stream", "stream_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "stream_owner_required", "required_action": "assign_stream_owner"},
	},
	{
		"name": "stream_requires_retention_policy",
		"description": "Streams require a retention policy.",
		"condition": {"operation": "create_stream", "retention_policy_present": False},
		"effect": {"decision": "deny", "reason": "stream_retention_policy_required", "required_action": "attach_retention_policy"},
	},
	{
		"name": "pii_stream_requires_schema",
		"description": "Streams carrying PII require a schema.",
		"condition": {"operation": "create_stream", "pii_stream": True, "schema_attached": False},
		"effect": {"decision": "deny", "reason": "stream_schema_required", "required_action": "attach_event_schema"},
	},
	{
		"name": "stream_requires_bytewax",
		"description": "Stream lifecycle events must use Bytewax.",
		"condition": {"operation": "create_stream", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_stream_lifecycle_to_bytewax"},
	},
	# --- Schema lifecycle ---
	{
		"name": "breaking_schema_requires_review",
		"description": "Breaking schema changes require review.",
		"condition": {"operation": "register_schema", "breaking_change": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "schema_review_required", "required_action": "record_schema_review"},
	},
	# --- Publish lifecycle ---
	{
		"name": "publish_requires_source_capability",
		"description": "Published events require source-capability attribution.",
		"condition": {"operation": "publish_event", "source_capability_present": False},
		"effect": {"decision": "deny", "reason": "source_capability_required", "required_action": "attach_source_capability"},
	},
	{
		"name": "publish_requires_correlation",
		"description": "Published events require correlation or causation context.",
		"condition": {"operation": "publish_event", "correlation_present": False},
		"effect": {"decision": "deny", "reason": "event_correlation_required", "required_action": "attach_correlation_context"},
	},
	{
		"name": "publish_requires_bytewax",
		"description": "Published events must be appended through Bytewax.",
		"condition": {"operation": "publish_event", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "append_event_to_bytewax"},
	},
	{
		"name": "batch_publish_limit",
		"description": "Batch publishing must stay within configured limits.",
		"condition": {"operation": "batch_publish", "batch_size_gt": 1000},
		"effect": {"decision": "deny", "reason": "batch_size_limit_exceeded", "required_action": "split_event_batch"},
	},
	{
		"name": "batch_publish_requires_bytewax",
		"description": "Batch publishing requires Bytewax.",
		"condition": {"operation": "batch_publish", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "append_batch_to_bytewax"},
	},
	# --- Subscription lifecycle ---
	{
		"name": "subscription_requires_owner",
		"description": "Subscriptions require a consumer owner.",
		"condition": {"operation": "create_subscription", "consumer_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "consumer_owner_required", "required_action": "assign_consumer_owner"},
	},
	{
		"name": "retry_subscription_requires_dead_letter",
		"description": "Retrying subscriptions require a dead-letter stream.",
		"condition": {"operation": "create_subscription", "retry_enabled": True, "dead_letter_attached": False},
		"effect": {"decision": "deny", "reason": "dead_letter_required", "required_action": "attach_dead_letter_stream"},
	},
	# --- Processor lifecycle ---
	{
		"name": "stateful_processor_requires_review",
		"description": "Stateful processors require review.",
		"condition": {"operation": "register_processor", "stateful_processor": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "processor_review_required", "required_action": "record_processor_review"},
	},
	{
		"name": "processor_requires_checkpoint",
		"description": "Processors require checkpoint configuration.",
		"condition": {"operation": "register_processor", "checkpoint_configured": False},
		"effect": {"decision": "deny", "reason": "processor_checkpoint_required", "required_action": "configure_processor_checkpoint"},
	},
	{
		"name": "processor_requires_bytewax",
		"description": "Processors must run on Bytewax.",
		"condition": {"operation": "register_processor", "processor_runtime_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_processor_required", "required_action": "select_bytewax_processor"},
	},
	{
		"name": "processor_requires_backpressure_strategy",
		"description": "Processors must declare a backpressure strategy to prevent unbounded queue growth.",
		"condition": {"operation": "register_processor", "backpressure_strategy_present": False},
		"effect": {"decision": "deny", "reason": "backpressure_strategy_required", "required_action": "declare_backpressure_strategy"},
	},
	# --- Replay governance ---
	{
		"name": "replay_requires_approval",
		"description": "Event replay requires approval.",
		"condition": {"operation": "replay_events", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "event_replay_approval_required", "required_action": "record_replay_approval"},
	},
	# --- Agent governance ---
	{
		"name": "event_agent_runtime_supported",
		"description": "Event agents must use an approved runtime.",
		"condition": {"operation": "register_event_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "event_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "event_agent_role_supported",
		"description": "Event agents must use an approved role.",
		"condition": {"operation": "register_event_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "event_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_agent_event_action_requires_human_approval",
		"description": "Privileged event actions proposed by agents require human approval.",
		"condition": {"operation": "agent_event_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# --- Service mesh integrity ---
	{
		"name": "service_mesh_identity_required_for_stream_admin",
		"description": "Intra-mesh callers performing stream admin must present a verified mesh identity.",
		"condition": {"operation": "stream_admin", "mesh_identity_verified": False},
		"effect": {"decision": "deny", "reason": "mesh_identity_required", "required_action": "attach_verified_mesh_identity"},
	},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-events/dashboard", "component": "EventBusDashboard", "permission": "composition_events:view", "nav_group": "Overview"},
	{"name": "streams", "path": "/composition-events/streams", "component": "EventStreamConsole", "permission": "composition_events:manage_streams", "nav_group": "Streams"},
	{"name": "schemas", "path": "/composition-events/schemas", "component": "EventSchemaRegistry", "permission": "composition_events:govern", "nav_group": "Governance"},
	{"name": "subscriptions", "path": "/composition-events/subscriptions", "component": "EventSubscriptionConsole", "permission": "composition_events:operate", "nav_group": "Consumers"},
	{"name": "processors", "path": "/composition-events/processors", "component": "EventProcessorTopology", "permission": "composition_events:operate", "nav_group": "Processing"},
	{"name": "dead_letters", "path": "/composition-events/dead-letters", "component": "DeadLetterConsole", "permission": "composition_events:operate", "nav_group": "Operations"},
	{"name": "replay", "path": "/composition-events/replay", "component": "EventReplayConsole", "permission": "composition_events:govern", "nav_group": "Governance"},
	{"name": "circuit_breaker", "path": "/composition-events/circuit-breaker", "component": "EventCircuitBreakerConsole", "permission": "composition_events:operate", "nav_group": "Resilience"},
	{"name": "agents", "path": "/composition-events/agents", "component": "EventAgentWorkbench", "permission": "composition_events:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/composition-events/audit", "component": "EventAuditConsole", "permission": "composition_events:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/composition-events/settings", "component": "EventBusSettings", "permission": "composition_events:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_events_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
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
		"stream_console": {"icon": "route", "status_indicator": "stream-pill", "risk_style": "throughput-band"},
		"schema_registry": {"visual": "schema-grid", "status_style": "compatibility-chip"},
		"subscription_console": {"visual": "consumer-lanes", "status_style": "lag-chip"},
		"processor_topology": {"visual": "dataflow-map", "status_style": "checkpoint-chip"},
		"dead_letter_console": {"visual": "error-queue", "status_style": "retry-chip"},
		"replay_console": {"visual": "offset-timeline", "status_style": "replay-chip"},
		"circuit_breaker_console": {"visual": "breaker-gauge", "status_style": "breaker-chip"},
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
		"version": "1.2.0",
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/composition-events/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
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
			"stream_quarantined",
			"schema_registered",
			"schema_breaking_change_reviewed",
			"event_published",
			"event_batch_published",
			"subscription_created",
			"subscription_lag_threshold_exceeded",
			"processor_registered",
			"processor_backpressure_activated",
			"dead_letter_recorded",
			"events_replayed",
			"circuit_breaker_tripped",
			"circuit_breaker_recovered",
			"cascade_isolation_triggered",
			"event_agent_registered",
		],
		"states": ["draft", "active", "paused", "review_required", "processing", "degraded", "quarantined", "blocked", "retired"],
		"guardrails": [
			"stream_requires_bytewax",
			"publish_requires_bytewax",
			"batch_publish_requires_bytewax",
			"processor_requires_bytewax",
			"privileged_agent_event_action_requires_human_approval",
			"circuit_breaker_trip_requires_event",
			"cross_tenant_publish_blocked",
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
