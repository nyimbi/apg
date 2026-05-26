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


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped ETLP configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"pipelines": {
			"visual_designer_enabled": True,
			"max_concurrent_executions": 10,
			"schedule_required_for_production": True,
			"owner_required": True
		},
		"processing": {
			"default_mode": "elt",
			"streaming_enabled": True,
			"batch_enabled": True,
			"federated_processing_enabled": True
		},
		"quality": {
			"quality_gate_enabled": True,
			"minimum_publish_score": 80.0,
			"quarantine_failed_records": True
		},
		"governance": {
			"require_tenant_context": True,
			"lineage_emission_required": True,
			"audit_all_executions": True,
			"production_approval_required": True
		},
		"optimization": {
			"ai_optimization_enabled": True,
			"self_healing_enabled": True,
			"cost_guardrail_enabled": True,
			"max_estimated_cost": 1000.0
		},
		"ui": {
			"enable_pipeline_designer": True,
			"enable_execution_monitor": True,
			"enable_quality_console": True,
			"enable_field_mapper": True
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
			"processing",
			"quality",
			"governance",
			"optimization",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"pipelines": {"type": "object"},
			"processing": {"type": "object"},
			"quality": {"type": "object"},
			"governance": {"type": "object"},
			"optimization": {"type": "object"},
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
		"quality_gate_panel": {"visual": "rule-stack", "highlight": "score-chip"}
	})


def default_rules() -> list[CapabilityRule]:
	return [
		CapabilityRule("tenant_context_required", "All pipeline operations require tenant context.", {"tenant_context_present": False}, {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}),
		CapabilityRule("pipeline_execution_requires_owner", "Pipeline execution requires an assigned owner.", {"operation": "execute_pipeline", "owner_assigned": False}, {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}),
		CapabilityRule("production_execution_requires_approval", "Production executions require approval.", {"environment": "production", "approval_recorded": False}, {"decision": "deny", "reason": "production_approval_required", "required_action": "record_execution_approval"}),
		CapabilityRule("publish_requires_quality_gate", "Publishing transformed data requires passing quality gates.", {"operation": "publish_output", "quality_gate_passed": False}, {"decision": "deny", "reason": "quality_gate_required", "required_action": "resolve_quality_failures"}),
		CapabilityRule("lineage_required_for_transformations", "Transformations require lineage emission.", {"transformation_present": True, "lineage_emitted": False}, {"decision": "deny", "reason": "lineage_emission_required", "required_action": "emit_lineage_event"}),
		CapabilityRule("high_cost_execution_requires_review", "High estimated execution cost requires review.", {"estimated_cost_gt": 1000.0, "cost_review_recorded": False}, {"decision": "require_review", "reason": "cost_review_required", "required_action": "record_cost_review"})
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
		CapabilityUIRoute("settings", "/etlp/settings", "ETLPSettings", "etlp:pipeline:write", "Administration")
	]
	return {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/etlp/api/v1", "routes": [route.__dict__ for route in routes], "template_roots": ["templates/", "static/"], "requires_theme": True}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {"capability": "etlp", "display_name": "ETL/ELT Processing", "configuration": config.for_tenant(tenant_id, overrides), "configuration_schema": config.schema, "rule_engine": {"type": "deterministic", "rules": [rule.__dict__ for rule in default_rules()]}, "ui": ui_manifest(), "theme": {"name": theme.name, "tokens": theme.tokens, "components": theme.components}}


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
