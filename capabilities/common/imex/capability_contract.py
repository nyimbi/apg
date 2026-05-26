"""
Executable capability contract for APG Import/Export.

IMEX is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic import/export governance rules, UI surfaces, and
theme tokens so composition tooling can integrate with IMEX consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped IMEX configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"jobs": {
			"max_concurrent_jobs": 25,
			"owner_required": True,
			"approval_required_for_production": True,
			"checkpointing_enabled": True
		},
		"formats": {
			"supported_formats": ["csv", "json", "parquet", "xlsx", "xml", "sql"],
			"schema_mapping_required": True,
			"format_conversion_enabled": True
		},
		"validation": {
			"data_validation_enabled": True,
			"minimum_quality_score": 80.0,
			"quarantine_invalid_records": True,
			"preview_required_before_execute": True
		},
		"security": {
			"require_tenant_context": True,
			"sensitive_exports_require_encryption": True,
			"destination_approval_required": True,
			"audit_all_transfers": True
		},
		"orchestration": {
			"etlp_integration_enabled": True,
			"conn_integration_enabled": True,
			"notification_enabled": True,
			"collaboration_enabled": True
		},
		"ui": {
			"enable_job_designer": True,
			"enable_mapping_workbench": True,
			"enable_transfer_monitor": True,
			"enable_validation_console": True
		},
		"theme": {
			"default_theme": "imex_transfer_console",
			"allow_tenant_overrides": True
		}
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
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"jobs": {"type": "object"},
			"formats": {"type": "object"},
			"validation": {"type": "object"},
			"security": {"type": "object"},
			"orchestration": {"type": "object"},
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
	"""Simple IMEX policy rule definition."""

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
		"color.accent": "#F4A261",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"transfer_job_card": {"icon": "arrow-left-right", "status_indicator": "job-state-pill", "risk_style": "quality-band"},
		"schema_mapping_canvas": {"visual": "source-target-map", "edge_style": "field-transform-line"},
		"validation_result_panel": {"visual": "rule-stack", "highlight": "invalid-record-chip"},
		"migration_timeline": {"visual": "checkpoint-timeline", "status_style": "throughput-pill"}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default IMEX rules available to every tenant."""
	return [
		CapabilityRule("tenant_context_required", "All import/export operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("job_execution_requires_owner", "Transfer jobs require an owner before execution.", {"operation": "execute_job", "owner_assigned": False}, {"decision": "deny", "reason": "job_owner_required", "required_action": "assign_job_owner"}),
		CapabilityRule("production_transfer_requires_approval", "Production transfers require approval.", {"environment": "production", "approval_recorded": False}, {"decision": "deny", "reason": "production_approval_required", "required_action": "record_transfer_approval"}),
		CapabilityRule("sensitive_export_requires_encryption", "Sensitive exports require encryption.", {"operation": "export", "data_classification": "sensitive", "export_encrypted": False}, {"decision": "deny", "reason": "export_encryption_required", "required_action": "enable_export_encryption"}),
		CapabilityRule("execution_requires_preview_validation", "Execution requires preview validation.", {"operation": "execute_job", "preview_validated": False}, {"decision": "deny", "reason": "preview_validation_required", "required_action": "run_preview_validation"}),
		CapabilityRule("low_quality_transfer_requires_review", "Low quality transfer output requires review.", {"quality_score_lt": 80.0, "quality_review_recorded": False}, {"decision": "require_review", "reason": "quality_review_required", "required_action": "record_quality_review"})
	]


def ui_manifest() -> dict[str, Any]:
	"""Return IMEX UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/imex/dashboard", "IMEXDashboard", "imex.view", "Overview"),
		CapabilityUIRoute("jobs", "/imex/jobs", "TransferJobs", "imex.view", "Operations"),
		CapabilityUIRoute("designer", "/imex/designer", "JobDesigner", "imex.create", "Build"),
		CapabilityUIRoute("mappings", "/imex/mappings", "SchemaMappingWorkbench", "imex.manage", "Build"),
		CapabilityUIRoute("monitor", "/imex/monitor", "TransferMonitor", "imex.execute", "Operations"),
		CapabilityUIRoute("validation", "/imex/validation", "ValidationConsole", "imex.manage", "Governance"),
		CapabilityUIRoute("workflows", "/imex/workflows", "MigrationWorkflows", "imex.manage", "Orchestration"),
		CapabilityUIRoute("settings", "/imex/settings", "IMEXSettings", "imex.admin", "Administration")
	]
	return {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/imex/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable IMEX capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {"capability": "imex", "display_name": "Import/Export", "configuration": config.for_tenant(tenant_id, overrides), "configuration_schema": config.schema, "rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]}, "ui": ui_manifest(), "theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default IMEX rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
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
