"""Executable capability contract for APG Import/Export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_IMEX_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_IMEX_AGENT_ROLES = [
	"import_reviewer",
	"export_reviewer",
	"migration_reviewer",
	"mapping_reviewer",
	"quality_reviewer",
	"artifact_reviewer",
	"retention_reviewer",
	"transfer_operator",
	"data_steward",
]
PRIVILEGED_IMEX_AGENT_ROLES = [
	"import_reviewer",
	"export_reviewer",
	"migration_reviewer",
	"mapping_reviewer",
	"quality_reviewer",
	"artifact_reviewer",
	"retention_reviewer",
	"transfer_operator",
]


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped IMEX configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"jobs": {
			"max_concurrent_jobs": 25,
			"owner_required": True,
			"approval_required_for_production": True,
			"checkpointing_enabled": True,
			"idempotency_required_for_replay": True,
			"retention_days": 90,
		},
		"formats": {
			"supported_formats": ["csv", "json", "parquet", "xlsx", "xml", "sql"],
			"schema_mapping_required": True,
			"format_conversion_enabled": True,
			"source_profile_required": True,
			"checksum_required": True,
		},
		"validation": {
			"data_validation_enabled": True,
			"minimum_quality_score": 0.95,
			"quarantine_invalid_records": True,
			"preview_required_before_execute": True,
			"sample_size": 1000,
		},
		"security": {
			"require_tenant_context": True,
			"sensitive_exports_require_encryption": True,
			"destination_approval_required": True,
			"audit_all_transfers": True,
			"pii_policy_required": True,
			"destructive_purge_review_required": True,
		},
		"orchestration": {
			"etlp_integration_enabled": True,
			"conn_integration_enabled": True,
			"notification_enabled": True,
			"collaboration_enabled": True,
			"bytewax_events_enabled": True,
		},
		"observability": {
			"metrics_required": True,
			"large_transfer_record_threshold": 100000,
			"large_transfer_monitoring_required": True,
			"audit_enabled": True,
		},
		"adapters": {
			"generated_app_runtime": "imex_runtime.ImexService",
			"production_runtime": "service.ImportExportService",
			"http_api": "api.app",
			"event_stream": "bytewax",
			"pipeline_engine": "etlp",
			"connector_control_plane": "conn",
			"auth_provider": "auth",
			"audit_sink": "audl",
			"metrics_sink": "moni",
			"key_vault": "keym",
			"encryption_provider": "encr",
		},
		"agents": {
			"first_class": True,
			"supported_runtimes": SUPPORTED_IMEX_AGENT_RUNTIMES,
			"supported_roles": SUPPORTED_IMEX_AGENT_ROLES,
			"privileged_roles": PRIVILEGED_IMEX_AGENT_ROLES,
			"scope_required": True,
			"owner_required": True,
			"purpose_required": True,
			"contribution_disclosure_required": True,
			"human_approval_required_for_privileged_roles": True,
		},
		"streaming": {
			"engine": "bytewax",
			"required_processor": "bytewax",
			"lifecycle_stream": "imex.lifecycle",
			"watermark": "event_time",
			"operations": [
				"endpoint_batch",
				"mapping_batch",
				"job_batch",
				"run_batch",
				"artifact_batch",
				"review_batch",
				"transfer_agent_batch",
			],
			"topics": [
				"imex.endpoints",
				"imex.mappings",
				"imex.jobs",
				"imex.runs",
				"imex.artifacts",
				"imex.reviews",
				"imex.agents",
			],
		},
		"ui": {
			"enable_dashboard": True,
			"enable_job_designer": True,
			"enable_mapping_workbench": True,
			"enable_transfer_monitor": True,
			"enable_validation_console": True,
			"enable_imports": True,
			"enable_exports": True,
			"enable_approvals": True,
			"enable_artifacts": True,
			"enable_retention": True,
			"enable_audit": True,
			"enable_transfer_agent_roster": True,
			"enable_lifecycle_batch_monitor": True,
			"enable_settings": True,
		},
		"theme": {
			"default_theme": "imex_transfer_console",
			"allow_tenant_overrides": True,
		},
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"jobs",
			"formats",
			"validation",
			"security",
			"orchestration",
			"observability",
			"adapters",
			"agents",
			"streaming",
			"ui",
			"theme",
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"jobs": {"type": "object"},
			"formats": {"type": "object"},
			"validation": {"type": "object"},
			"security": {"type": "object"},
			"orchestration": {"type": "object"},
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
	"""IMEX policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic IMEX rule engine for transfer control decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching import/export governance rules."""
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
	"""UI route exposed by IMEX."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for IMEX UI surfaces."""

	name: str = "imex_transfer_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#2D5D7B",
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
		"transfer_job_row": {"icon": "arrow-left-right", "status_indicator": "job-state-pill", "risk_style": "quality-band"},
		"schema_mapping_canvas": {"visual": "source-target-map", "edge_style": "field-transform-line"},
		"validation_result_panel": {"visual": "rule-stack", "highlight": "invalid-record-chip"},
		"transfer_timeline": {"visual": "checkpoint-timeline", "status_style": "throughput-pill"},
		"approval_queue": {"visual": "review-list", "status_indicator": "risk-chip"},
		"artifact_browser": {"visual": "file-grid", "status_indicator": "retention-chip"},
		"retention_console": {"visual": "policy-table", "status_indicator": "expiry-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-pill"},
		"transfer_agent_roster": {"visual": "agent-roster", "status_indicator": "approval-chip"},
		"bytewax_lifecycle_panel": {"visual": "stream-batches", "status_indicator": "processor-chip"},
	})


def default_rules() -> list[CapabilityRule]:
	"""Default IMEX rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All import/export operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("job_requires_owner", "Transfer jobs require an accountable owner.", {"operation": "create_job", "owner_assigned": False}, {"decision": "deny", "reason": "job_owner_required", "required_action": "assign_job_owner"}),
		CapabilityRule("job_requires_direction", "Transfer jobs require import, export, or migration direction.", {"operation": "create_job", "direction_present": False}, {"decision": "deny", "reason": "job_direction_required", "required_action": "choose_transfer_direction"}),
		CapabilityRule("job_requires_source", "Transfer jobs require a source endpoint.", {"operation": "create_job", "source_registered": False}, {"decision": "deny", "reason": "source_endpoint_required", "required_action": "register_source_endpoint"}),
		CapabilityRule("job_requires_destination", "Transfer jobs require a destination endpoint.", {"operation": "create_job", "destination_registered": False}, {"decision": "deny", "reason": "destination_endpoint_required", "required_action": "register_destination_endpoint"}),
		CapabilityRule("format_must_be_supported", "Transfer jobs require a supported file or stream format.", {"operation": "create_job", "format_supported": False}, {"decision": "deny", "reason": "unsupported_transfer_format", "required_action": "choose_supported_format"}),
		CapabilityRule("source_profile_required", "Import and migration jobs require source schema profile evidence.", {"operation": "create_job", "source_profile_present": False}, {"decision": "deny", "reason": "source_profile_required", "required_action": "profile_source_schema"}),
		CapabilityRule("checksum_required", "External transfer artifacts require checksum evidence.", {"operation": "create_job", "checksum_present": False}, {"decision": "deny", "reason": "checksum_required", "required_action": "attach_checksum"}),
		CapabilityRule("mapping_required", "Transfers require schema mapping evidence.", {"operation": "create_job", "mapping_present": False}, {"decision": "deny", "reason": "schema_mapping_required", "required_action": "attach_schema_mapping"}),
		CapabilityRule("pii_policy_required", "PII-bearing transfers require a data protection policy.", {"operation": "create_job", "pii_detected": True, "pii_policy_attached": False}, {"decision": "deny", "reason": "pii_policy_required", "required_action": "attach_pii_policy"}),
		CapabilityRule("destination_approval_required", "External export destinations require destination approval.", {"operation": "create_job", "external_destination": True, "destination_approved": False}, {"decision": "require_review", "reason": "destination_approval_required", "required_action": "record_destination_approval"}),
		CapabilityRule("preview_required_before_execute", "Execution requires preview validation.", {"operation": "execute_job", "preview_validated": False}, {"decision": "deny", "reason": "preview_validation_required", "required_action": "run_preview_validation"}),
		CapabilityRule("production_transfer_requires_approval", "Production transfers require approval.", {"operation": "execute_job", "environment": "production", "approval_recorded": False}, {"decision": "deny", "reason": "production_approval_required", "required_action": "record_transfer_approval"}),
		CapabilityRule("sensitive_export_requires_encryption", "Sensitive exports require encryption.", {"operation": "execute_job", "direction": "export", "data_classification": "sensitive", "export_encrypted": False}, {"decision": "deny", "reason": "export_encryption_required", "required_action": "enable_export_encryption"}),
		CapabilityRule("large_transfer_requires_monitoring", "Large transfers require monitoring.", {"operation": "execute_job", "record_count_gt": 100000, "monitoring_enabled": False}, {"decision": "deny", "reason": "large_transfer_monitoring_required", "required_action": "enable_monitoring"}),
		CapabilityRule("checkpointing_required", "Executable transfer jobs require checkpointing.", {"operation": "execute_job", "checkpointing_enabled": False}, {"decision": "deny", "reason": "checkpointing_required", "required_action": "enable_checkpointing"}),
		CapabilityRule("quality_review_required", "Low quality transfer previews require review.", {"operation": "execute_job", "quality_score_lt": 0.95, "quality_review_recorded": False}, {"decision": "require_review", "reason": "quality_review_required", "required_action": "record_quality_review"}),
		CapabilityRule("invalid_records_require_quarantine", "Invalid records require quarantine before execution.", {"operation": "execute_job", "invalid_records_present": True, "quarantine_enabled": False}, {"decision": "deny", "reason": "invalid_record_quarantine_required", "required_action": "enable_quarantine"}),
		CapabilityRule("concurrent_limit_requires_review", "Tenant concurrency above the configured limit requires review.", {"operation": "execute_job", "active_jobs_gt": 25, "capacity_review_recorded": False}, {"decision": "require_review", "reason": "capacity_review_required", "required_action": "record_capacity_review"}),
		CapabilityRule("retry_requires_failure", "Retry operations require a failed run.", {"operation": "retry_run", "previous_run_failed": False}, {"decision": "deny", "reason": "failed_run_required", "required_action": "select_failed_run"}),
		CapabilityRule("replay_requires_idempotency", "Replay operations require an idempotency key.", {"operation": "replay_run", "idempotency_key_present": False}, {"decision": "deny", "reason": "idempotency_required", "required_action": "attach_idempotency_key"}),
		CapabilityRule("schedule_requires_timezone", "Scheduled transfers require timezone.", {"operation": "schedule_job", "timezone_present": False}, {"decision": "deny", "reason": "timezone_required", "required_action": "declare_schedule_timezone"}),
		CapabilityRule("artifact_publication_requires_checksum", "Published artifacts require checksum evidence.", {"operation": "publish_artifact", "checksum_present": False}, {"decision": "deny", "reason": "artifact_checksum_required", "required_action": "attach_artifact_checksum"}),
		CapabilityRule("retention_policy_required", "Transfer artifacts require retention policy.", {"operation": "publish_artifact", "retention_policy_present": False}, {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}),
		CapabilityRule("destructive_purge_requires_review", "Destructive artifact purges require review.", {"operation": "purge_artifact", "destructive": True, "purge_review_recorded": False}, {"decision": "deny", "reason": "purge_review_required", "required_action": "record_purge_review"}),
		CapabilityRule("owner_transfer_requires_review", "Job owner transfer requires review.", {"operation": "transfer_owner", "owner_transfer_review_recorded": False}, {"decision": "require_review", "reason": "owner_transfer_review_required", "required_action": "record_owner_transfer_review"}),
		CapabilityRule("migration_requires_etlp_plan", "Migration jobs require ETLP plan linkage.", {"operation": "create_job", "direction": "migration", "etlp_plan_present": False}, {"decision": "deny", "reason": "etlp_plan_required", "required_action": "attach_etlp_plan"}),
		CapabilityRule("connector_binding_required", "Endpoint-backed transfers require CONN connection binding.", {"operation": "create_job", "connector_binding_present": False}, {"decision": "deny", "reason": "connector_binding_required", "required_action": "attach_conn_binding"}),
		CapabilityRule("audit_evidence_required", "Executed transfers require audit evidence.", {"operation": "complete_run", "audit_evidence_present": False}, {"decision": "deny", "reason": "audit_evidence_required", "required_action": "record_audit_evidence"}),
		CapabilityRule("completion_quality_required", "Completed transfers require final quality score.", {"operation": "complete_run", "quality_score_present": False}, {"decision": "deny", "reason": "quality_score_required", "required_action": "record_quality_score"}),
		CapabilityRule("transfer_agent_runtime_supported", "Transfer agents must use a supported runtime adapter.", {"operation": "register_transfer_agent", "agent_runtime_supported": False}, {"decision": "deny", "reason": "unsupported_transfer_agent_runtime", "required_action": "choose_supported_transfer_agent_runtime"}),
		CapabilityRule("transfer_agent_role_supported", "Transfer agents must use a supported transfer role.", {"operation": "register_transfer_agent", "agent_role_supported": False}, {"decision": "deny", "reason": "unsupported_transfer_agent_role", "required_action": "choose_supported_transfer_agent_role"}),
		CapabilityRule("transfer_agent_requires_scope", "Transfer agents require an explicit bounded scope.", {"operation": "register_transfer_agent", "scope_present": False}, {"decision": "deny", "reason": "transfer_agent_scope_required", "required_action": "declare_transfer_agent_scope"}),
		CapabilityRule("transfer_agent_requires_owner", "Transfer agents require an accountable owner.", {"operation": "register_transfer_agent", "owner_present": False}, {"decision": "deny", "reason": "transfer_agent_owner_required", "required_action": "assign_transfer_agent_owner"}),
		CapabilityRule("transfer_agent_requires_purpose", "Transfer agents require a documented purpose.", {"operation": "register_transfer_agent", "purpose_present": False}, {"decision": "deny", "reason": "transfer_agent_purpose_required", "required_action": "document_transfer_agent_purpose"}),
		CapabilityRule("transfer_agent_requires_contribution_disclosure", "Transfer agents must disclose machine-authored transfer contributions.", {"operation": "register_transfer_agent", "contribution_disclosed": False}, {"decision": "deny", "reason": "transfer_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}),
		CapabilityRule("transfer_agent_privileged_role_requires_human_approval", "Privileged transfer-agent roles require human approval evidence.", {"operation": "register_transfer_agent", "privileged_role": True, "human_approval_required": False}, {"decision": "require_review", "reason": "transfer_agent_human_approval_required", "required_action": "record_human_transfer_agent_approval"}),
		CapabilityRule("bytewax_imex_stream_required", "IMEX lifecycle batches must be routed through Bytewax.", {"operation": "validate_imex_lifecycle_batch", "event_stream_ne": "bytewax"}, {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_imex_lifecycle_batch_to_bytewax"}),
	]


def ui_manifest() -> dict[str, Any]:
	"""Return IMEX UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/imex/dashboard", "IMEXDashboard", "imex:view", "Overview"),
		CapabilityUIRoute("jobs", "/imex/jobs", "TransferJobs", "imex:view", "Operations"),
		CapabilityUIRoute("designer", "/imex/designer", "JobDesigner", "imex:create", "Build"),
		CapabilityUIRoute("mappings", "/imex/mappings", "SchemaMappingWorkbench", "imex:manage", "Build"),
		CapabilityUIRoute("monitor", "/imex/monitor", "TransferMonitor", "imex:execute", "Operations"),
		CapabilityUIRoute("validation", "/imex/validation", "ValidationConsole", "imex:manage", "Governance"),
		CapabilityUIRoute("imports", "/imex/imports", "ImportWorkbench", "imex:create", "Operations"),
		CapabilityUIRoute("exports", "/imex/exports", "ExportWorkbench", "imex:create", "Operations"),
		CapabilityUIRoute("approvals", "/imex/approvals", "TransferApprovalQueue", "imex:approve", "Governance"),
		CapabilityUIRoute("artifacts", "/imex/artifacts", "TransferArtifacts", "imex:view", "Operations"),
		CapabilityUIRoute("audit", "/imex/audit", "TransferAuditTimeline", "imex:view", "Governance"),
		CapabilityUIRoute("agents", "/imex/agents", "TransferAgentRoster", "imex:admin", "Governance"),
		CapabilityUIRoute("lifecycle", "/imex/lifecycle", "TransferLifecycleBatchMonitor", "imex:view", "Operations"),
		CapabilityUIRoute("settings", "/imex/settings", "IMEXSettings", "imex:admin", "Administration"),
	]
	return {"shell": "apg_python", "view_module": "view_models.py", "api_prefix": "/imex/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class transfer-agent composition metadata."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_IMEX_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_IMEX_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_IMEX_AGENT_ROLES),
		"requires": ["scope", "owner", "purpose", "contribution_disclosure"],
		"approval": "human approval is required before privileged transfer agents mutate transfer state",
		"external_runtimes_are_adapters": True,
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return IMEX lifecycle streaming metadata."""
	return {
		"engine": "bytewax",
		"required_processor": "bytewax",
		"lifecycle_stream": "imex.lifecycle",
		"watermark": "event_time",
		"operations": [
			"endpoint_batch",
			"mapping_batch",
			"job_batch",
			"run_batch",
			"artifact_batch",
			"review_batch",
			"transfer_agent_batch",
		],
		"topics": [
			"imex.endpoints",
			"imex.mappings",
			"imex.jobs",
			"imex.runs",
			"imex.artifacts",
			"imex.reviews",
			"imex.agents",
		],
	}


def review_evidence_manifest() -> dict[str, Any]:
	"""Return durable IMEX review-evidence metadata."""
	return {
		"durable_statuses": [
			"pending",
			"pending_review",
			"review_required",
			"review_denied",
			"denied",
			"accepted",
			"draft",
			"validated",
			"running",
			"queued",
			"completed",
			"published",
			"purged",
		],
		"policy_fields": [
			"policy_decision",
			"matched_rules",
			"review_reasons",
			"review_evidence",
		],
		"pending_queues": [
			"endpoints",
			"mappings",
			"jobs",
			"runs",
			"artifacts",
			"reviews",
			"transfer_agents",
			"lifecycle_batches",
		],
		"deny_behavior": "Denied IMEX lifecycle batches persist evidence before PermissionError",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable IMEX capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "imex",
		"display_name": "Import/Export",
		"provides": ["import_export", "bulk_transfer", "transfer_agent_composition", "review_evidence"],
		"requires": ["etlp", "conn", "auth", "audl"],
		"configuration": config.for_tenant(tenant_id, overrides),
		"configuration_schema": config.schema,
		"rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"review_evidence": review_evidence_manifest(),
		"ui": ui_manifest(),
		"theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components},
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default IMEX rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
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
