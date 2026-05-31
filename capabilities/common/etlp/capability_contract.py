"""
Executable capability contract for APG ETL/ELT Processing.

ETLP is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic pipeline-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with ETLP without instantiating the
pipeline API/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_ETLP_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ETLP_AGENT_ROLES = [
	"pipeline_designer_reviewer",
	"datasource_reviewer",
	"mapping_reviewer",
	"quality_gate_reviewer",
	"lineage_reviewer",
	"execution_reviewer",
	"publish_gate_reviewer",
	"replay_reviewer",
]
PRIVILEGED_ETLP_AGENT_ROLES = [
	"datasource_reviewer",
	"quality_gate_reviewer",
	"execution_reviewer",
	"publish_gate_reviewer",
	"replay_reviewer",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped ETLP configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"pipelines": {
			"visual_designer_enabled": True,
			"max_concurrent_executions": 10,
			"schedule_required_for_production": True,
			"owner_required": True,
			"supported_modes": ["elt", "etl", "batch", "streaming", "hybrid"],
			"default_status": "draft",
			"retire_requires_impact_review": True
		},
		"datasources": {
			"approval_required": True,
			"owner_required": True,
			"secret_reference_required": True,
			"embedded_secret_denied": True,
			"supported_types": ["database", "file", "api", "stream", "warehouse", "lakehouse", "object_store"]
		},
		"mappings": {
			"field_mapping_required": True,
			"schema_validation_required": True,
			"lineage_required": True
		},
		"processing": {
			"default_mode": "elt",
			"streaming_enabled": True,
			"batch_enabled": True,
			"federated_processing_enabled": True,
			"bytewax_streams_enabled": True
		},
		"quality": {
			"quality_gate_enabled": True,
			"minimum_publish_score": 80.0,
			"quarantine_failed_records": True,
			"required_dimensions": ["completeness", "validity", "freshness", "lineage", "mapping"]
		},
		"governance": {
			"require_tenant_context": True,
			"lineage_emission_required": True,
			"audit_all_executions": True,
			"production_approval_required": True,
			"publish_approval_required": True,
			"destructive_delete_review_required": True
		},
		"optimization": {
			"ai_optimization_enabled": True,
			"self_healing_enabled": True,
			"cost_guardrail_enabled": True,
			"max_estimated_cost": 1000.0
		},
		"execution": {
			"default_retry_limit": 3,
			"max_retry_limit": 5,
			"max_replay_window_hours": 72,
			"backfill_requires_review": True,
			"idempotency_key_required": True
		},
		"adapters": {
			"production_runtime": "service.ETLPService",
			"generated_app_runtime": "service.ETLPLifecycleService",
			"execution_engine": "adapter",
			"connector_registry": "adapter",
			"metadata_catalog": "meta",
			"event_stream": "bytewax",
			"quality_engine": "adapter",
			"lineage_emitter": "adapter",
			"secret_store": "adapter"
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_ETLP_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_ETLP_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_ETLP_AGENT_ROLES,
			"require_owner": True,
			"require_purpose": True,
			"require_scope": True,
			"require_contribution_disclosure": True,
			"require_human_approval_for_privileged_roles": True
		},
		"streaming": {
			"engine": "bytewax",
			"lifecycle_stream": "etlp.lifecycle",
			"watermark": "event_time",
			"required_operations": [
				"pipeline_batch",
				"datasource_batch",
				"mapping_batch",
				"execution_batch",
				"quality_batch",
				"publish_batch",
				"replay_batch",
				"pipeline_agent_batch"
			],
			"topics": [
				"etlp.pipelines",
				"etlp.datasources",
				"etlp.mappings",
				"etlp.executions",
				"etlp.quality",
				"etlp.publish",
				"etlp.replay",
				"etlp.agents"
			]
		},
		"ui": {
			"enable_pipeline_designer": True,
			"enable_execution_monitor": True,
			"enable_quality_console": True,
			"enable_field_mapper": True,
			"enable_publish_review": True,
			"enable_replay_console": True,
			"enable_adapter_health": True,
			"enable_audit_timeline": True,
			"enable_pipeline_agent_roster": True,
			"enable_lifecycle_batch_monitor": True
		},
		"theme": {
			"default_theme": "etlp_pipeline_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"pipelines",
			"datasources",
			"mappings",
			"processing",
			"quality",
			"governance",
			"optimization",
			"execution",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"pipelines": {"type": "object"},
			"datasources": {"type": "object"},
			"mappings": {"type": "object"},
			"processing": {"type": "object"},
			"quality": {"type": "object"},
			"governance": {"type": "object"},
			"optimization": {"type": "object"},
			"execution": {"type": "object"},
			"adapters": {"type": "object"},
			"agents": {"type": "object"},
			"streaming": {"type": "object"},
			"ui": {"type": "object"},
			"theme": {"type": "object"}
		}
	})

	def for_tenant(self, tenant_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id is required"
		merged = _deep_copy(self.defaults)
		merged["tenant_id"] = tenant_id
		if overrides:
			_deep_merge(merged, overrides)
		return merged


@dataclass(frozen=True)
class CapabilityRule:
	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic ETLP rule engine for pipeline control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
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
	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	name: str = "etlp_pipeline_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#2C5282",
		"color.accent": "#38A169",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"pipeline_status_card": {"icon": "workflow", "status_indicator": "run-state-pill", "risk_style": "quality-band"},
		"field_mapping_canvas": {"visual": "source-target-map", "edge_style": "transform-line"},
		"execution_timeline": {"visual": "stage-timeline", "status_style": "checkpoint-pill"},
		"quality_gate_panel": {"visual": "rule-stack", "highlight": "score-chip"},
		"datasource_inventory": {"visual": "connector-table", "status_indicator": "approval-pill"},
		"publish_review_queue": {"visual": "quality-lineage-review", "highlight": "gate-chip"},
		"replay_console": {"visual": "bounded-time-window", "status_style": "replay-state-pill"},
		"adapter_health_panel": {"visual": "adapter-grid", "status_indicator": "health-pill"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"pipeline_agent_roster": {"icon": "bot", "status_indicator": "approval-state", "variant": "pipeline-agent-governance"},
		"bytewax_lifecycle_panel": {"icon": "activity", "status_indicator": "processor-state", "variant": "pipeline-stream-lifecycle"}
	})


def default_rules() -> list[CapabilityRule]:
	return [
		CapabilityRule("tenant_context_required", "All pipeline operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("pipeline_registration_requires_owner", "Pipeline registration requires an assigned owner.", {"operation": "register_pipeline", "owner_assigned": False}, {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}),
		CapabilityRule("pipeline_mode_must_be_supported", "Pipeline mode must be configured for the tenant.", {"operation": "register_pipeline", "unsupported_mode": True}, {"decision": "deny", "reason": "unsupported_pipeline_mode", "required_action": "choose_supported_mode"}),
		CapabilityRule("pipeline_execution_requires_owner", "Pipeline execution requires an assigned owner.", {"operation": "execute_pipeline", "owner_assigned": False}, {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}),
		CapabilityRule("production_execution_requires_approval", "Production executions require approval.", {"operation": "execute_pipeline", "environment": "production", "approval_recorded": False}, {"decision": "deny", "reason": "production_approval_required", "required_action": "record_execution_approval"}),
		CapabilityRule("execution_requires_idempotency_key", "Pipeline executions require idempotency evidence.", {"operation": "execute_pipeline", "idempotency_key_present": False}, {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"}),
		CapabilityRule("datasource_registration_requires_owner", "Data source registration requires an assigned owner.", {"operation": "register_datasource", "datasource_owner_assigned": False}, {"decision": "deny", "reason": "datasource_owner_required", "required_action": "assign_datasource_owner"}),
		CapabilityRule("datasource_requires_secret_reference", "Data source registration requires a secret-store reference.", {"operation": "register_datasource", "secret_reference_present": False}, {"decision": "deny", "reason": "secret_reference_required", "required_action": "attach_secret_reference"}),
		CapabilityRule("datasource_requires_approval", "Data sources require approval before execution.", {"operation": "register_datasource", "datasource_approved": False}, {"decision": "require_review", "reason": "datasource_approval_required", "required_action": "approve_datasource"}),
		CapabilityRule("datasource_type_must_be_supported", "Data source type must be configured for the tenant.", {"operation": "register_datasource", "unsupported_datasource_type": True}, {"decision": "deny", "reason": "unsupported_datasource_type", "required_action": "choose_supported_datasource_type"}),
		CapabilityRule("datasource_secrets_must_use_reference", "Data source secret material must use secret-store references.", {"operation": "register_datasource", "embedded_secret_present": True}, {"decision": "deny", "reason": "embedded_secret_denied", "required_action": "store_secret_reference"}),
		CapabilityRule("mapping_requires_schema_validation", "Field mappings require schema validation before use.", {"operation": "register_mapping", "schema_validated": False}, {"decision": "deny", "reason": "schema_validation_required", "required_action": "validate_mapping_schema"}),
		CapabilityRule("mapping_requires_registered_datasources", "Field mappings require registered source and target data sources.", {"operation": "register_mapping", "source_and_target_registered": False}, {"decision": "deny", "reason": "registered_datasources_required", "required_action": "register_mapping_datasources"}),
		CapabilityRule("publish_requires_quality_gate", "Publishing transformed data requires passing quality gates.", {"operation": "publish_output", "quality_gate_passed": False}, {"decision": "deny", "reason": "quality_gate_required", "required_action": "resolve_quality_failures"}),
		CapabilityRule("publish_requires_minimum_quality", "Publishing transformed data requires the configured minimum quality score.", {"operation": "publish_output", "quality_score_lt": 80.0}, {"decision": "deny", "reason": "minimum_quality_required", "required_action": "raise_quality_score"}),
		CapabilityRule("publish_requires_approval", "Publishing production outputs requires an approval record.", {"operation": "publish_output", "publish_approval_recorded": False}, {"decision": "deny", "reason": "publish_approval_required", "required_action": "record_publish_approval"}),
		CapabilityRule("production_schedule_requires_review", "Production schedules require current review evidence.", {"operation": "schedule_pipeline", "environment": "production", "schedule_review_recorded": False}, {"decision": "require_review", "reason": "schedule_review_required", "required_action": "record_schedule_review"}),
		CapabilityRule("lineage_required_for_transformations", "Transformations require lineage emission.", {"operation": "execute_pipeline", "transformation_present": True, "lineage_emitted": False}, {"decision": "deny", "reason": "lineage_emission_required", "required_action": "emit_lineage_event"}),
		CapabilityRule("high_cost_execution_requires_review", "High estimated execution cost requires review.", {"operation": "execute_pipeline", "estimated_cost_gt": 1000.0, "cost_review_recorded": False}, {"decision": "require_review", "reason": "cost_review_required", "required_action": "record_cost_review"}),
		CapabilityRule("retry_limit_requires_review", "Retry requests above tenant limits require review.", {"operation": "retry_execution", "retry_count_gt": 3, "retry_review_recorded": False}, {"decision": "require_review", "reason": "retry_review_required", "required_action": "record_retry_review"}),
		CapabilityRule("replay_requires_reason", "Replay and backfill requests require a business reason.", {"operation": "replay_execution", "reason_present": False}, {"decision": "deny", "reason": "replay_reason_required", "required_action": "attach_replay_reason"}),
		CapabilityRule("replay_window_requires_review", "Replay windows above the tenant limit require review.", {"operation": "replay_execution", "replay_window_hours_gt": 72, "replay_review_recorded": False}, {"decision": "require_review", "reason": "replay_window_review_required", "required_action": "record_replay_review"}),
		CapabilityRule("destructive_delete_requires_review", "Destructive pipeline deletion requires impact review.", {"operation": "retire_pipeline", "impact_review_recorded": False}, {"decision": "deny", "reason": "impact_review_required", "required_action": "record_impact_review"}),
		CapabilityRule("pipeline_agent_runtime_supported", "Pipeline agents must use a supported runtime adapter.", {"operation": "register_pipeline_agent", "agent_runtime_supported": False}, {"decision": "deny", "reason": "unsupported_pipeline_agent_runtime", "required_action": "select_supported_agent_runtime"}),
		CapabilityRule("pipeline_agent_role_supported", "Pipeline agents must use a supported pipeline governance role.", {"operation": "register_pipeline_agent", "agent_role_supported": False}, {"decision": "deny", "reason": "unsupported_pipeline_agent_role", "required_action": "select_supported_agent_role"}),
		CapabilityRule("pipeline_agent_requires_scope", "Pipeline agents require an explicit operating scope.", {"operation": "register_pipeline_agent", "agent_scope_present": False}, {"decision": "deny", "reason": "pipeline_agent_scope_required", "required_action": "attach_agent_scope"}),
		CapabilityRule("pipeline_agent_requires_owner", "Pipeline agents require an accountable owner.", {"operation": "register_pipeline_agent", "agent_owner_present": False}, {"decision": "deny", "reason": "pipeline_agent_owner_required", "required_action": "attach_agent_owner"}),
		CapabilityRule("pipeline_agent_requires_purpose", "Pipeline agents require a declared purpose.", {"operation": "register_pipeline_agent", "agent_purpose_present": False}, {"decision": "deny", "reason": "pipeline_agent_purpose_required", "required_action": "attach_agent_purpose"}),
		CapabilityRule("pipeline_agent_requires_contribution_disclosure", "Pipeline agents must disclose machine contribution in pipeline decisions.", {"operation": "register_pipeline_agent", "contribution_disclosed": False}, {"decision": "deny", "reason": "pipeline_agent_contribution_disclosure_required", "required_action": "enable_agent_contribution_disclosure"}),
		CapabilityRule("pipeline_agent_privileged_role_requires_human_approval", "Privileged pipeline-agent roles require human approval.", {"operation": "register_pipeline_agent", "privileged_agent_role": True, "human_approval_required": False}, {"decision": "deny", "reason": "pipeline_agent_human_approval_required", "required_action": "require_human_approval_for_agent"}),
		CapabilityRule("bytewax_etlp_stream_required", "ETLP lifecycle batches must declare Bytewax as the pipeline lifecycle processor.", {"operation": "validate_etlp_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_etlp_stream_required", "required_action": "route_batch_through_bytewax"})
	]


def ui_manifest() -> dict[str, Any]:
	routes = [
		CapabilityUIRoute("dashboard", "/etlp/dashboard", "ETLPDashboard", "etlp:pipeline:read", "Overview"),
		CapabilityUIRoute("pipelines", "/etlp/pipelines", "PipelineWorkbench", "etlp:pipeline:read", "Pipelines"),
		CapabilityUIRoute("designer", "/etlp/designer", "PipelineDesigner", "etlp:pipeline:write", "Pipelines"),
		CapabilityUIRoute("field_mapper", "/etlp/field-mapper", "FieldMapper", "etlp:transformation:write", "Design"),
		CapabilityUIRoute("executions", "/etlp/executions", "ExecutionMonitor", "etlp:pipeline:execute", "Operations"),
		CapabilityUIRoute("quality", "/etlp/quality", "QualityGateConsole", "etlp:quality:read", "Governance"),
		CapabilityUIRoute("datasources", "/etlp/datasources", "DatasourceManager", "etlp:datasource:read", "Sources"),
		CapabilityUIRoute("schedules", "/etlp/schedules", "ScheduleConsole", "etlp:pipeline:write", "Operations"),
		CapabilityUIRoute("publish", "/etlp/publish", "PublishReviewQueue", "etlp:publish:review", "Governance"),
		CapabilityUIRoute("lineage", "/etlp/lineage", "PipelineLineage", "etlp:lineage:read", "Governance"),
		CapabilityUIRoute("replay", "/etlp/replay", "ReplayConsole", "etlp:execution:replay", "Operations"),
		CapabilityUIRoute("adapters", "/etlp/adapters", "ETLPAdapterHealth", "etlp:admin", "Administration"),
		CapabilityUIRoute("audit", "/etlp/audit", "ETLPAuditTimeline", "etlp:audit:read", "Governance"),
		CapabilityUIRoute("agents", "/etlp/agents", "PipelineAgentRoster", "etlp:admin", "Administration"),
		CapabilityUIRoute("lifecycle", "/etlp/lifecycle", "ETLPLifecycleBatchMonitor", "etlp:admin", "Runtime"),
		CapabilityUIRoute("settings", "/etlp/settings", "ETLPSettings", "etlp:pipeline:write", "Administration")
	]
	return {"shell": "apg_python", "view_module": "view_models.py", "api_prefix": "/etlp/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_ETLP_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_ETLP_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_ETLP_AGENT_ROLES),
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
	return {
		"engine": "bytewax",
		"lifecycle_stream": "etlp.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"pipeline_batch",
			"datasource_batch",
			"mapping_batch",
			"execution_batch",
			"quality_batch",
			"publish_batch",
			"replay_batch",
			"pipeline_agent_batch"
		],
		"topics": [
			"etlp.pipelines",
			"etlp.datasources",
			"etlp.mappings",
			"etlp.executions",
			"etlp.quality",
			"etlp.publish",
			"etlp.replay",
			"etlp.agents"
		],
		"broker_core_dependency_allowed": False
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "etlp",
		"display_name": "ETL/ELT Processing",
		"provides": ["pipeline_lifecycle", "data_integration_governance", "pipeline_agent_composition"],
		"requires": ["mdm", "meta", "mqeb", "moni"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"ui": ui_manifest(),
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
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
