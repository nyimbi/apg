"""
Executable capability contract for APG Metadata Management.

META is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic metadata-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with META without starting
database, discovery, or AI-classification runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped META configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"catalog": {
			"asset_registration_required": True,
			"default_asset_status": "draft",
			"owner_required_for_published_assets": True,
			"business_glossary_enabled": True,
			"supported_asset_types": [
				"database",
				"schema",
				"table",
				"column",
				"file",
				"api",
				"stream",
				"report",
				"dashboard",
				"model",
				"pipeline",
				"business_term"
			],
			"source_system_required": True,
			"business_key_required": True
		},
		"discovery": {
			"auto_discovery_enabled": True,
			"max_concurrent_jobs": 5,
			"connector_approval_required": True,
			"schedule_review_days": 30,
			"allowed_connector_types": ["database", "file", "api", "stream", "ml", "catalog"],
			"discovery_result_review_required": True
		},
		"classification": {
			"ai_classification_enabled": True,
			"restricted_data_requires_classification": True,
			"confidence_review_threshold": 0.75,
			"pii_detection_enabled": True,
			"review_notes_required": True,
			"sensitive_labels": ["pii", "phi", "pci", "secret", "restricted"]
		},
		"lineage": {
			"lineage_tracking_enabled": True,
			"lineage_required_for_certified_assets": True,
			"impact_analysis_enabled": True,
			"max_lineage_depth": 8,
			"source_and_target_registration_required": True,
			"retire_requires_impact_analysis": True
		},
		"quality": {
			"quality_assessment_enabled": True,
			"minimum_certification_score": 85.0,
			"stale_asset_days": 90,
			"score_range": [0.0, 100.0],
			"dimensions": ["completeness", "freshness", "accuracy", "lineage", "classification", "usage"]
		},
		"governance": {
			"require_tenant_context": True,
			"audit_catalog_mutations": True,
			"steward_approval_required": True,
			"certification_review_required": True,
			"glossary_owner_required": True,
			"publish_requires_quality": True
		},
		"adapters": {
			"production_runtime": "service.APGMetadataService",
			"generated_app_runtime": "service.MetaService",
			"discovery_engine": "adapter",
			"classification_engine": "adapter",
			"lineage_engine": "adapter",
			"search_engine": "adapter",
			"event_stream": "bytewax",
			"metadata_store": "adapter"
		},
		"ui": {
			"enable_catalog": True,
			"enable_discovery_console": True,
			"enable_lineage_viewer": True,
			"enable_classification_review": True,
			"enable_quality_console": True,
			"enable_certification_queue": True,
			"enable_glossary": True,
			"enable_impact_analysis": True,
			"enable_audit_timeline": True,
			"enable_adapter_health": True
		},
		"theme": {
			"default_theme": "meta_catalog_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"catalog",
			"discovery",
			"classification",
			"lineage",
			"quality",
			"governance",
			"adapters",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"catalog": {"type": "object"},
			"discovery": {"type": "object"},
			"classification": {"type": "object"},
			"lineage": {"type": "object"},
			"quality": {"type": "object"},
			"governance": {"type": "object"},
			"adapters": {"type": "object"},
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
	"""Simple META policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic META rule engine for metadata governance decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching metadata governance rules."""
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
	"""UI route exposed by META."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for META UI surfaces."""

	name: str = "meta_catalog_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#3D405B",
		"color.accent": "#81B29A",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1F2933",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"asset_catalog_card": {
			"icon": "database",
			"status_indicator": "certification-pill",
			"risk_style": "classification-band"
		},
		"lineage_graph_viewer": {
			"visual": "directed-lineage-graph",
			"edge_style": "transformation-line"
		},
		"classification_review_queue": {
			"visual": "confidence-stack",
			"highlight": "sensitivity-chip"
		},
		"discovery_job_timeline": {
			"visual": "connector-run-timeline",
			"status_style": "job-state-pill"
		},
		"certification_queue": {
			"visual": "evidence-checklist",
			"highlight": "certification-state"
		},
		"glossary_term_panel": {
			"visual": "term-definition-list",
			"status_indicator": "ownership-pill"
		},
		"impact_analysis_graph": {
			"visual": "downstream-impact-graph",
			"edge_style": "risk-weighted-line"
		},
		"audit_decision_timeline": {
			"visual": "decision-timeline",
			"highlight": "matched-rule-chip"
		},
		"adapter_status_panel": {
			"visual": "backend-grid",
			"status_indicator": "adapter-state"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default META rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All metadata operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="asset_type_must_be_supported",
			description="Metadata assets must use a configured asset type.",
			condition={"operation": "register_asset", "unsupported_asset_type": True},
			effect={
				"decision": "deny",
				"reason": "unsupported_asset_type",
				"required_action": "configure_asset_type"
			}
		),
		CapabilityRule(
			name="asset_registration_requires_business_key",
			description="Metadata assets require a durable business key.",
			condition={"operation": "register_asset", "business_key_present": False},
			effect={
				"decision": "deny",
				"reason": "business_key_required",
				"required_action": "attach_business_key"
			}
		),
		CapabilityRule(
			name="asset_registration_requires_source_system",
			description="Metadata assets require source-system context.",
			condition={"operation": "register_asset", "source_system_present": False},
			effect={
				"decision": "deny",
				"reason": "source_system_required",
				"required_action": "attach_source_system"
			}
		),
		CapabilityRule(
			name="published_asset_requires_owner",
			description="Published metadata assets require an assigned owner.",
			condition={"operation": "publish_asset", "asset_owner_assigned": False},
			effect={
				"decision": "deny",
				"reason": "asset_owner_required",
				"required_action": "assign_asset_owner"
			}
		),
		CapabilityRule(
			name="publish_requires_quality_assessment",
			description="Published metadata assets require current quality evidence.",
			condition={"operation": "publish_asset", "quality_assessment_present": False},
			effect={
				"decision": "deny",
				"reason": "quality_assessment_required",
				"required_action": "run_quality_assessment"
			}
		),
		CapabilityRule(
			name="restricted_asset_requires_classification",
			description="Restricted assets require completed classification.",
			condition={"operation": "publish_asset", "asset_sensitivity": "restricted", "classification_complete": False},
			effect={
				"decision": "deny",
				"reason": "classification_required",
				"required_action": "complete_asset_classification"
			}
		),
		CapabilityRule(
			name="sensitive_asset_requires_steward",
			description="Sensitive assets require an assigned data steward.",
			condition={"asset_sensitivity": "restricted", "steward_assigned": False},
			effect={
				"decision": "deny",
				"reason": "steward_required",
				"required_action": "assign_data_steward"
			}
		),
		CapabilityRule(
			name="certified_asset_requires_lineage",
			description="Certified assets require lineage evidence.",
			condition={"certification_requested": True, "lineage_available": False},
			effect={
				"decision": "deny",
				"reason": "lineage_required",
				"required_action": "capture_asset_lineage"
			}
		),
		CapabilityRule(
			name="certification_requires_quality_threshold",
			description="Certification requires metadata quality above the configured threshold.",
			condition={"operation": "certify_asset", "quality_score_lt": 85.0},
			effect={
				"decision": "deny",
				"reason": "quality_score_too_low",
				"required_action": "improve_metadata_quality"
			}
		),
		CapabilityRule(
			name="classification_review_requires_notes",
			description="Classification stewardship decisions require review notes.",
			condition={"operation": "review_classification", "review_notes_present": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		),
		CapabilityRule(
			name="low_classification_confidence_requires_review",
			description="Low confidence AI classifications require steward review.",
			condition={"classification_confidence_lt": 0.75, "steward_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "classification_review_required",
				"required_action": "complete_steward_classification_review"
			}
		),
		CapabilityRule(
			name="discovery_requires_approved_connector",
			description="Discovery jobs require approved connector configuration.",
			condition={"operation": "schedule_discovery", "connector_approved": False},
			effect={
				"decision": "deny",
				"reason": "connector_approval_required",
				"required_action": "approve_connector"
			}
		),
		CapabilityRule(
			name="discovery_schedule_requires_review",
			description="Recurring discovery schedules require current schedule review.",
			condition={"operation": "schedule_discovery", "schedule_review_current": False},
			effect={
				"decision": "require_review",
				"reason": "schedule_review_required",
				"required_action": "review_discovery_schedule"
			}
		),
		CapabilityRule(
			name="lineage_requires_registered_assets",
			description="Lineage edges require registered source and target assets.",
			condition={"operation": "capture_lineage", "source_and_target_registered": False},
			effect={
				"decision": "deny",
				"reason": "registered_assets_required",
				"required_action": "register_lineage_assets"
			}
		),
		CapabilityRule(
			name="lineage_depth_requires_review",
			description="Lineage requests above configured depth require review.",
			condition={"operation": "capture_lineage", "lineage_depth_gt": 8},
			effect={
				"decision": "require_review",
				"reason": "lineage_depth_review_required",
				"required_action": "review_lineage_depth"
			}
		),
		CapabilityRule(
			name="glossary_term_requires_owner",
			description="Business glossary terms require an accountable owner.",
			condition={"operation": "register_glossary_term", "term_owner_assigned": False},
			effect={
				"decision": "deny",
				"reason": "term_owner_required",
				"required_action": "assign_glossary_owner"
			}
		),
		CapabilityRule(
			name="retire_asset_requires_impact_analysis",
			description="Retiring metadata assets requires impact analysis evidence.",
			condition={"operation": "retire_asset", "impact_analysis_present": False},
			effect={
				"decision": "deny",
				"reason": "impact_analysis_required",
				"required_action": "complete_impact_analysis"
			}
		),
		CapabilityRule(
			name="stale_asset_requires_review",
			description="Stale metadata assets require review before certification.",
			condition={"asset_age_days_gt": 90, "freshness_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "freshness_review_required",
				"required_action": "review_asset_freshness"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return META UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/meta/dashboard", "MetadataDashboard", "meta:view", "Overview"),
		CapabilityUIRoute("catalog", "/meta/catalog", "AssetCatalog", "meta:view_assets", "Catalog"),
		CapabilityUIRoute("discovery", "/meta/discovery", "DiscoveryConsole", "meta:run_discovery", "Discovery"),
		CapabilityUIRoute("lineage", "/meta/lineage", "LineageViewer", "meta:view_lineage", "Catalog"),
		CapabilityUIRoute("classification", "/meta/classification", "ClassificationReview", "meta:classify", "Governance"),
		CapabilityUIRoute("quality", "/meta/quality", "MetadataQualityConsole", "meta:view_quality", "Governance"),
		CapabilityUIRoute("certification", "/meta/certification", "CertificationQueue", "meta:certify", "Governance"),
		CapabilityUIRoute("glossary", "/meta/glossary", "BusinessGlossary", "meta:manage_glossary", "Catalog"),
		CapabilityUIRoute("impact", "/meta/impact", "ImpactAnalysis", "meta:view_impact", "Governance"),
		CapabilityUIRoute("search", "/meta/search", "MetadataSearch", "meta:search", "Catalog"),
		CapabilityUIRoute("audit", "/meta/audit", "MetadataAuditTimeline", "meta:view_audit", "Governance"),
		CapabilityUIRoute("adapters", "/meta/adapters", "MetadataAdapterHealth", "meta:admin", "Administration"),
		CapabilityUIRoute("settings", "/meta/settings", "MetadataSettings", "meta:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "view_models.py",
		"api_prefix": "/meta/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable META capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "meta",
		"display_name": "Metadata Management",
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
		}
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Convenience wrapper for default META rule evaluation."""
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
