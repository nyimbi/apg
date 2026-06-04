"""Executable capability contract for APG Connection Management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_CONN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CONN_AGENT_ROLES = [
	"connector_reviewer",
	"connection_reviewer",
	"flow_designer",
	"sync_operator",
	"quality_reviewer",
	"lineage_reviewer",
	"marketplace_reviewer",
	"credential_reviewer",
	"tap_steward",
]
PRIVILEGED_CONN_AGENT_ROLES = [
	"connector_reviewer",
	"connection_reviewer",
	"sync_operator",
	"quality_reviewer",
	"lineage_reviewer",
	"marketplace_reviewer",
	"credential_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CONN configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"connectors": {
			"owner_required": True,
			"runtime_required": True,
			"source_required": True,
			"checksum_required": True,
			"local_singer_runtime_required": True,
			"marketplace_review_required_for_unverified": True,
			"supported_runtimes": ["singer", "apg", "http", "database", "file", "stream", "webhook"],
			"max_connectors_per_tenant": 500,
		},
		"connections": {
			"owner_required": True,
			"credential_vault_required": True,
			"test_required_before_activation": True,
			"production_activation_review_required": True,
			"secret_rotation_required": True,
			"cross_tenant_connections_allowed": False,
		},
		"flows": {
			"source_and_target_required": True,
			"mapping_required": True,
			"lineage_required": True,
			"quality_gate_required": True,
			"idempotency_required_for_replay": True,
		},
		"sync": {
			"default_batch_size": 1000,
			"max_batch_size": 100000,
			"review_batch_threshold": 10000,
			"default_mode": "incremental",
			"allowed_modes": ["full_refresh", "incremental", "cdc", "event_stream"],
			"monitoring_required_for_large_batch": True,
		},
		"security": {
			"encrypt_credentials": True,
			"credential_vault": "keym",
			"pii_policy_required": True,
			"webhook_auth_required": True,
			"destructive_delete_review_required": True,
		},
		"quality": {
			"quality_gates_enabled": True,
			"default_min_quality_score": 0.95,
			"schema_change_review_required": True,
			"sample_validation_required": True,
		},
		"governance": {
			"require_tenant_context": True,
			"audit_enabled": True,
			"lineage_capture_required": True,
			"retirement_impact_review_required": True,
			"owner_transfer_review_required": True,
		},
		"observability": {
			"health_check_interval_seconds": 60,
			"metrics_required": True,
			"alerts_required_for_production": True,
			"sync_audit_required": True,
		},
		"adapters": {
			"production_runtime": "service.ConnectionManager",
			"generated_app_runtime": "conn_runtime.ConnService",
			"http_api": "api.app",
			"singer_runtime": "singer_runtime.SingerRuntime",
			"singer_tap_registry": "singer_taps.erp_registry",
			"auth_provider": "auth",
			"credential_vault": "keym",
			"encryption_provider": "encr",
			"audit_sink": "audl",
			"metrics_sink": "moni",
			"lineage_store": "meta",
			"data_quality": "dqol",
			"event_stream": "bytewax",
			"registry": "regy",
			"gateway": "apig",
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_CONN_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_CONN_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_CONN_AGENT_ROLES,
			"scope_required": True,
			"owner_required": True,
			"purpose_required": True,
			"contribution_disclosure_required": True,
			"human_approval_required_for_privileged_roles": True,
		},
		"streaming": {
			"engine": "bytewax",
			"required_processor": "bytewax",
			"lifecycle_stream": "conn.lifecycle",
			"watermark": "event_time",
			"operations": [
				"connector_batch",
				"connection_batch",
				"flow_batch",
				"sync_batch",
				"schedule_batch",
				"review_batch",
				"connector_agent_batch",
			],
			"topics": [
				"conn.connectors",
				"conn.connections",
				"conn.flows",
				"conn.sync_runs",
				"conn.schedules",
				"conn.reviews",
				"conn.agents",
			],
		},
		"ui": {
			"enable_dashboard": True,
			"enable_connector_catalog": True,
			"enable_connection_workbench": True,
			"enable_visual_designer": True,
			"enable_sync_monitor": True,
			"enable_quality_console": True,
			"enable_lineage_view": True,
			"enable_marketplace": True,
			"enable_security_console": True,
			"enable_audit_timeline": True,
			"enable_connector_agent_roster": True,
			"enable_lifecycle_batch_monitor": True,
			"enable_settings": True,
		},
		"theme": {
			"default_theme": "conn_integration_console",
			"allow_tenant_overrides": True,
		},
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"connectors",
			"connections",
			"flows",
			"sync",
			"security",
			"quality",
			"governance",
			"observability",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme",
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"connectors": {"type": "object"},
			"connections": {"type": "object"},
			"flows": {"type": "object"},
			"sync": {"type": "object"},
			"security": {"type": "object"},
			"quality": {"type": "object"},
			"governance": {"type": "object"},
			"observability": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"},
		},
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
	"""CONN policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic rule engine for CONN policy and workflow decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a connection context."""
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

		return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


@dataclass(frozen=True)
class CapabilityUIRoute:
	"""UI route exposed by the capability."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for CONN UI surfaces."""

	name: str = "conn_integration_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#176B87",
		"color.accent": "#C86F2D",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"connector_catalog_row": {"icon": "plug", "status_indicator": "runtime-pill", "risk_style": "verification-band"},
		"connection_workbench": {"visual": "connection-table", "status_indicator": "test-chip"},
		"flow_designer_canvas": {"visual": "node-graph", "edge_style": "lineage-edge"},
		"sync_monitor": {"visual": "run-timeline", "status_indicator": "batch-chip"},
		"quality_gate_panel": {"visual": "score-grid", "highlight": "threshold-chip"},
		"lineage_map": {"visual": "dependency-graph", "edge_style": "dataset-flow"},
		"marketplace_review_queue": {"visual": "review-list", "status_indicator": "source-chip"},
		"security_console": {"visual": "credential-matrix", "status_indicator": "rotation-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"connector_agent_roster": {"visual": "agent-roster", "status_indicator": "approval-chip"},
		"bytewax_lifecycle_panel": {"visual": "stream-batches", "status_indicator": "processor-chip"},
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CONN rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All connector operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("connector_requires_owner", "Connectors require an accountable owner.", {"operation": "register_connector", "owner_assigned": False}, {"decision": "deny", "reason": "connector_owner_required", "required_action": "assign_connector_owner"}),
		CapabilityRule("connector_requires_runtime", "Connectors require a declared runtime.", {"operation": "register_connector", "runtime_present": False}, {"decision": "deny", "reason": "connector_runtime_required", "required_action": "declare_connector_runtime"}),
		CapabilityRule("connector_runtime_must_be_supported", "Connectors require a supported runtime.", {"operation": "register_connector", "connector_runtime_supported": False}, {"decision": "deny", "reason": "unsupported_connector_runtime", "required_action": "choose_supported_connector_runtime"}),
		CapabilityRule("connector_requires_source", "Connectors require a source package or local tap reference.", {"operation": "register_connector", "source_present": False}, {"decision": "deny", "reason": "connector_source_required", "required_action": "attach_connector_source"}),
		CapabilityRule("connector_requires_checksum", "Connector packages require checksum evidence.", {"operation": "register_connector", "checksum_present": False}, {"decision": "deny", "reason": "connector_checksum_required", "required_action": "attach_connector_checksum"}),
		CapabilityRule("unverified_connector_requires_review", "Unverified connector sources require marketplace review.", {"operation": "register_connector", "verified_source": False, "marketplace_review_recorded": False}, {"decision": "require_review", "reason": "marketplace_review_required", "required_action": "record_marketplace_review"}),
		CapabilityRule("connection_requires_owner", "Connections require an accountable owner.", {"operation": "register_connection", "owner_assigned": False}, {"decision": "deny", "reason": "connection_owner_required", "required_action": "assign_connection_owner"}),
		CapabilityRule("connection_requires_registered_connector", "Connections require a registered connector.", {"operation": "register_connection", "connector_registered": False}, {"decision": "deny", "reason": "registered_connector_required", "required_action": "register_connector"}),
		CapabilityRule("credential_vault_required", "Credential-bearing connections require vault references.", {"operation": "register_connection", "contains_credentials": True, "credential_vault_ref_present": False}, {"decision": "deny", "reason": "credential_vault_required", "required_action": "store_credentials_in_vault"}),
		CapabilityRule("encrypt_credentials", "Credential-bearing connections require encrypted storage.", {"operation": "register_connection", "contains_credentials": True, "credentials_encrypted": False}, {"decision": "deny", "reason": "credentials_must_be_encrypted", "required_action": "enable_encryption"}),
		CapabilityRule("secret_rotation_required", "Connections require secret rotation evidence.", {"operation": "activate_connection", "secret_rotation_recorded": False}, {"decision": "deny", "reason": "secret_rotation_required", "required_action": "record_secret_rotation"}),
		CapabilityRule("require_connection_test_before_activation", "Connections must pass a test before activation.", {"operation": "activate_connection", "last_test_passed": False}, {"decision": "deny", "reason": "connection_test_required", "required_action": "run_connection_test"}),
		CapabilityRule("production_activation_requires_review", "Production connection activation requires review evidence.", {"operation": "activate_connection", "environment": "production", "activation_review_recorded": False}, {"decision": "require_review", "reason": "activation_review_required", "required_action": "record_activation_review"}),
		CapabilityRule("cross_tenant_connection_denied", "Cross-tenant connections are denied by default.", {"operation": "register_connection", "cross_tenant_connection": True}, {"decision": "deny", "reason": "cross_tenant_connection_denied", "required_action": "use_tenant_scoped_connection"}),
		CapabilityRule("flow_requires_source_connection", "Flows require an active source connection.", {"operation": "create_flow", "source_connection_active": False}, {"decision": "deny", "reason": "source_connection_required", "required_action": "activate_source_connection"}),
		CapabilityRule("flow_requires_target_connection", "Flows require an active target connection.", {"operation": "create_flow", "target_connection_active": False}, {"decision": "deny", "reason": "target_connection_required", "required_action": "activate_target_connection"}),
		CapabilityRule("flow_requires_mapping", "Flows require field mapping evidence.", {"operation": "create_flow", "mapping_present": False}, {"decision": "deny", "reason": "field_mapping_required", "required_action": "attach_field_mapping"}),
		CapabilityRule("flow_requires_lineage", "Flows require lineage capture configuration.", {"operation": "create_flow", "lineage_enabled": False}, {"decision": "deny", "reason": "lineage_required", "required_action": "enable_lineage_capture"}),
		CapabilityRule("flow_requires_quality_gate", "Flows require a data quality gate.", {"operation": "create_flow", "quality_gate_present": False}, {"decision": "deny", "reason": "quality_gate_required", "required_action": "attach_quality_gate"}),
		CapabilityRule("large_batch_requires_monitoring", "Large synchronization batches require monitoring.", {"operation": "start_sync", "batch_size_gt": 10000, "monitoring_enabled": False}, {"decision": "deny", "reason": "large_batch_requires_monitoring", "required_action": "enable_monitoring"}),
		CapabilityRule("oversized_batch_denied", "Synchronization batch size cannot exceed the tenant maximum.", {"operation": "start_sync", "batch_size_gt": 100000}, {"decision": "deny", "reason": "batch_size_limit_exceeded", "required_action": "reduce_batch_size"}),
		CapabilityRule("sync_mode_must_be_supported", "Synchronization runs require a supported mode.", {"operation": "start_sync", "sync_mode_supported": False}, {"decision": "deny", "reason": "unsupported_sync_mode", "required_action": "choose_supported_sync_mode"}),
		CapabilityRule("schema_change_requires_review", "Schema changes require review before sync.", {"operation": "start_sync", "schema_change_detected": True, "schema_review_recorded": False}, {"decision": "require_review", "reason": "schema_review_required", "required_action": "record_schema_review"}),
		CapabilityRule("pii_requires_policy", "PII-bearing flows require data protection policy evidence.", {"operation": "create_flow", "pii_detected": True, "pii_policy_attached": False}, {"decision": "deny", "reason": "pii_policy_required", "required_action": "attach_pii_policy"}),
		CapabilityRule("webhook_requires_auth_policy", "Webhook connectors require authentication policy.", {"operation": "register_connector", "connector_runtime": "webhook", "auth_policy_attached": False}, {"decision": "deny", "reason": "webhook_auth_policy_required", "required_action": "attach_webhook_auth_policy"}),
		CapabilityRule("schedule_requires_timezone", "Scheduled flows require a timezone.", {"operation": "schedule_flow", "timezone_present": False}, {"decision": "deny", "reason": "timezone_required", "required_action": "declare_schedule_timezone"}),
		CapabilityRule("replay_requires_idempotency", "Replay operations require idempotency evidence.", {"operation": "replay_sync", "idempotency_key_present": False}, {"decision": "deny", "reason": "idempotency_required", "required_action": "attach_idempotency_key"}),
		CapabilityRule("destructive_delete_requires_review", "Destructive connector deletes require review.", {"operation": "delete_connection", "destructive": True, "delete_review_recorded": False}, {"decision": "deny", "reason": "delete_review_required", "required_action": "record_delete_review"}),
		CapabilityRule("connection_retirement_requires_impact_review", "Retiring a connection requires impact review.", {"operation": "retire_connection", "impact_review_recorded": False}, {"decision": "deny", "reason": "impact_review_required", "required_action": "record_retirement_impact_review"}),
		CapabilityRule("owner_transfer_requires_review", "Connection owner transfer requires review.", {"operation": "transfer_owner", "owner_transfer_review_recorded": False}, {"decision": "require_review", "reason": "owner_transfer_review_required", "required_action": "record_owner_transfer_review"}),
		CapabilityRule("connector_agent_runtime_supported", "Connector agents must use a supported runtime adapter.", {"operation": "register_connector_agent", "agent_runtime_supported": False}, {"decision": "deny", "reason": "unsupported_connector_agent_runtime", "required_action": "choose_supported_connector_agent_runtime"}),
		CapabilityRule("connector_agent_role_supported", "Connector agents must use a supported connector role.", {"operation": "register_connector_agent", "agent_role_supported": False}, {"decision": "deny", "reason": "unsupported_connector_agent_role", "required_action": "choose_supported_connector_agent_role"}),
		CapabilityRule("connector_agent_requires_scope", "Connector agents require an explicit bounded scope.", {"operation": "register_connector_agent", "scope_present": False}, {"decision": "deny", "reason": "connector_agent_scope_required", "required_action": "declare_connector_agent_scope"}),
		CapabilityRule("connector_agent_requires_owner", "Connector agents require an accountable owner.", {"operation": "register_connector_agent", "owner_present": False}, {"decision": "deny", "reason": "connector_agent_owner_required", "required_action": "assign_connector_agent_owner"}),
		CapabilityRule("connector_agent_requires_purpose", "Connector agents require a documented purpose.", {"operation": "register_connector_agent", "purpose_present": False}, {"decision": "deny", "reason": "connector_agent_purpose_required", "required_action": "document_connector_agent_purpose"}),
		CapabilityRule("connector_agent_requires_contribution_disclosure", "Connector agents must disclose machine-authored connector contributions.", {"operation": "register_connector_agent", "contribution_disclosed": False}, {"decision": "deny", "reason": "connector_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}),
		CapabilityRule("connector_agent_privileged_role_requires_human_approval", "Privileged connector-agent roles require human approval evidence.", {"operation": "register_connector_agent", "privileged_role": True, "human_approval_required": False}, {"decision": "require_review", "reason": "connector_agent_human_approval_required", "required_action": "record_human_connector_agent_approval"}),
		CapabilityRule("bytewax_conn_stream_required", "CONN lifecycle batches must be routed through Bytewax.", {"operation": "validate_conn_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_conn_lifecycle_batch_to_bytewax"}),
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CONN UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/conn/dashboard", "ConnectionDashboard", "conn:view", "Operations"),
		CapabilityUIRoute("connectors", "/conn/connectors", "ConnectorCatalog", "conn:view", "Catalog"),
		CapabilityUIRoute("connections", "/conn/connections", "ConnectionWorkbench", "conn:create", "Catalog"),
		CapabilityUIRoute("designer", "/conn/designer", "VisualFlowDesigner", "conn:create", "Build"),
		CapabilityUIRoute("sync_runs", "/conn/sync-runs", "SyncRunMonitor", "conn:view", "Operations"),
		CapabilityUIRoute("quality", "/conn/quality", "DataQualityConsole", "conn:view", "Governance"),
		CapabilityUIRoute("lineage", "/conn/lineage", "DataLineageView", "conn:view", "Governance"),
		CapabilityUIRoute("marketplace", "/conn/marketplace", "ConnectorMarketplace", "conn:admin", "Extend"),
		CapabilityUIRoute("security", "/conn/security", "ConnectionSecurityConsole", "conn:admin", "Security"),
		CapabilityUIRoute("audit", "/conn/audit", "ConnectionAuditTimeline", "conn:view", "Governance"),
		CapabilityUIRoute("agents", "/conn/agents", "ConnectorAgentRoster", "conn:admin", "Governance"),
		CapabilityUIRoute("lifecycle", "/conn/lifecycle", "ConnectorLifecycleBatchMonitor", "conn:view", "Operations"),
		CapabilityUIRoute("rules", "/conn/rules", "CapabilityRuleWorkbench", "conn:admin", "Governance"),
		CapabilityUIRoute("settings", "/conn/settings", "CapabilitySettings", "conn:admin", "Administration"),
	]
	return {"shell": "apg_python", "frontend_bundle": "frontend/src/App.tsx", "view_module": "view_models.py", "api_prefix": "/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "frontend/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class connector-agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_CONN_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_CONN_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_CONN_AGENT_ROLES),
		
		"approval": "human approval is required before privileged connector agents mutate connector state",
		"external_runtimes_are_adapters": True,
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return CONN lifecycle streaming metadata."""
	return {
		"engine": "bytewax",
		"required_processor": "bytewax",
		"lifecycle_stream": "conn.lifecycle",
		"watermark": "event_time",
		"operations": [
			"connector_batch",
			"connection_batch",
			"flow_batch",
			"sync_batch",
			"schedule_batch",
			"review_batch",
			"connector_agent_batch",
		],
		"topics": [
			"conn.connectors",
			"conn.connections",
			"conn.flows",
			"conn.sync_runs",
			"conn.schedules",
			"conn.reviews",
			"conn.agents",
		],
	}


def review_evidence_manifest() -> dict[str, Any]:
	"""Return durable CONN review-evidence metadata."""
	return {
		"durable_statuses": [
			"pending",
			"pending_review",
			"review_required",
			"review_denied",
			"denied",
			"accepted",
			"registered",
			"active",
			"running",
			"queued",
			"completed",
			"retired",
		],
		"policy_fields": [
			"policy_decision",
			"matched_rules",
			"review_reasons",
			"review_evidence",
		],
		"pending_queues": [
			"connectors",
			"connections",
			"flows",
			"sync_runs",
			"schedules",
			"reviews",
			"connector_agents",
			"lifecycle_batches",
		],
		"deny_behavior": "Denied CONN lifecycle batches persist evidence before PermissionError",
	}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.conn.lifecycle",
	"key": "tenant_id",
	"events": [
		"connector_registered",
		"connector_updated",
		"connector_retired",
		"connection_created",
		"connection_activated",
		"connection_deactivated",
		"flow_created",
		"flow_executed",
		"sync_started",
		"sync_completed",
		"sync_failed",
		"quality_gate_passed",
		"quality_gate_failed",
		"agent_registered",
	],
	"guardrails": [
		"conn_batch_requires_bytewax",
		"conn_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CONN capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "conn",
		"display_name": "Connection Management",
		"provides": ["connector_management", "connection_orchestration", "connector_agent_composition", "review_evidence"],
		"requires": ["apig", "auth", "encr", "audl"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"agents": agent_manifest(),
		"streaming": STREAMING,
		"review_evidence": review_evidence_manifest(),
		"ui": ui_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components},
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default CONN rule evaluation."""
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
