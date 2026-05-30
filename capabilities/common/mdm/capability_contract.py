"""
Executable capability contract for APG Master Data Management.

MDM is a first-class APG capability. This module exposes tenant-scoped
configuration, deterministic data-governance rules, UI surfaces, and theme
tokens so composition tooling can integrate with MDM without starting the
database-backed runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped MDM configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"entities": {
			"supported_entity_types": [
				"customer",
				"product",
				"supplier",
				"employee",
				"location",
				"asset",
				"account",
				"contract",
				"organization",
				"custom"
			],
			"golden_record_required": True,
			"version_history_enabled": True,
			"cross_reference_tracking": True,
			"business_key_required": True,
			"restricted_entity_types": ["customer", "employee", "account", "contract"]
		},
		"quality": {
			"quality_assessment_enabled": True,
			"minimum_quality_score": 80.0,
			"block_publish_below_score": 60.0,
			"assessment_required_before_publish": True,
			"dimension_score_range": [0.0, 100.0],
			"dimensions": ["completeness", "accuracy", "consistency", "validity", "uniqueness", "timeliness"]
		},
		"matching": {
			"ai_matching_enabled": True,
			"duplicate_detection_enabled": True,
			"auto_merge_threshold": 95.0,
			"manual_review_threshold": 70.0,
			"steward_review_required": True,
			"conflict_review_required": True
		},
		"survivorship": {
			"supported_policies": [
				"most_recent",
				"most_complete",
				"most_trusted_source",
				"highest_quality",
				"custom_rules",
				"ai_determined"
			],
			"policy_required_for_merge": True,
			"independent_steward_required_for_conflicts": True
		},
		"governance": {
			"require_tenant_context": True,
			"steward_approval_required": True,
			"audit_all_mutations": True,
			"publish_requires_data_owner": True,
			"review_notes_required": True,
			"retire_requires_lineage_evidence": True,
			"restricted_data_requires_classification_evidence": True
		},
		"integration": {
			"emit_entity_events": True,
			"use_cache": True,
			"metadata_sync_enabled": True,
			"source_system_evidence_required": True,
			"event_stream_adapter": "bytewax"
		},
		"adapters": {
			"database_runtime": "service.MDMService",
			"generated_app_runtime": "service.MdmService",
			"quality_engine": "adapter",
			"matching_engine": "adapter",
			"lineage_adapter": "adapter",
			"event_stream": "bytewax",
			"metadata_catalog": "adapter"
		},
		"ui": {
			"enable_dashboard": True,
			"enable_entity_workbench": True,
			"enable_quality_console": True,
			"enable_match_review": True,
			"enable_stewardship_queue": True,
			"enable_lineage_trace": True,
			"enable_cross_reference_console": True,
			"enable_publish_console": True,
			"enable_audit_timeline": True,
			"enable_adapter_health": True
		},
		"theme": {
			"default_theme": "mdm_golden_record_console",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": [
			"tenant_id",
			"entities",
			"quality",
			"matching",
			"survivorship",
			"governance",
			"integration",
			"adapters",
			"ui",
			"theme"
		],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"entities": {"type": "object"},
			"quality": {"type": "object"},
			"matching": {"type": "object"},
			"survivorship": {"type": "object"},
			"governance": {"type": "object"},
			"integration": {"type": "object"},
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
	"""Simple MDM policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic MDM rule engine for master-data governance decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate matching master-data governance rules."""
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
	"""UI route exposed by MDM."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for MDM UI surfaces."""

	name: str = "mdm_golden_record_console"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#345995",
		"color.accent": "#EAC435",
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
		"golden_record_card": {
			"icon": "badge-check",
			"status_indicator": "survivorship-pill",
			"risk_style": "quality-band"
		},
		"duplicate_review_queue": {
			"visual": "candidate-stack",
			"highlight": "confidence-chip"
		},
		"quality_score_panel": {
			"visual": "dimension-radar",
			"threshold_style": "score-bands"
		},
		"entity_lineage_trace": {
			"visual": "version-timeline",
			"status_style": "mutation-marker"
		},
		"stewardship_decision_panel": {
			"visual": "decision-queue",
			"highlight": "review-chip"
		},
		"cross_reference_matrix": {
			"visual": "source-system-grid",
			"status_indicator": "mapping-state"
		},
		"publish_readiness_panel": {
			"visual": "gate-checklist",
			"status_style": "readiness-band"
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
	"""Default MDM rules available to every tenant."""
	return [
		CapabilityRule(
			name="tenant_context_required",
			description="All MDM operations require tenant context.",
			condition={"tenant_context_present": False},
			effect={
				"decision": "deny",
				"reason": "tenant_context_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="entity_type_must_be_supported",
			description="Entity registration is limited to configured master-data entity types.",
			condition={"operation": "register_entity", "unsupported_entity_type": True},
			effect={
				"decision": "deny",
				"reason": "unsupported_entity_type",
				"required_action": "configure_entity_type"
			}
		),
		CapabilityRule(
			name="business_key_required_for_entity",
			description="Every master-data entity requires a durable business key.",
			condition={"operation": "register_entity", "business_key_present": False},
			effect={
				"decision": "deny",
				"reason": "business_key_required",
				"required_action": "attach_business_key"
			}
		),
		CapabilityRule(
			name="restricted_entity_requires_data_owner",
			description="Restricted master data requires an accountable data owner at registration.",
			condition={"operation": "register_entity", "entity_classification": "restricted", "data_owner_assigned": False},
			effect={
				"decision": "deny",
				"reason": "restricted_data_owner_required",
				"required_action": "assign_data_owner"
			}
		),
		CapabilityRule(
			name="entity_publish_requires_data_owner",
			description="Published master data requires an assigned data owner.",
			condition={"operation": "publish_entity", "data_owner_assigned": False},
			effect={
				"decision": "deny",
				"reason": "data_owner_required",
				"required_action": "assign_data_owner"
			}
		),
		CapabilityRule(
			name="publish_requires_latest_quality_assessment",
			description="Entities need a current quality assessment before publish.",
			condition={"operation": "publish_entity", "latest_quality_assessment_present": False},
			effect={
				"decision": "deny",
				"reason": "quality_assessment_required",
				"required_action": "run_quality_assessment"
			}
		),
		CapabilityRule(
			name="low_quality_blocks_publish",
			description="Entities below minimum quality cannot be published.",
			condition={"operation": "publish_entity", "quality_score_lt": 60.0},
			effect={
				"decision": "deny",
				"reason": "quality_score_too_low",
				"required_action": "remediate_data_quality"
			}
		),
		CapabilityRule(
			name="invalid_quality_score_blocks_assessment",
			description="Quality score inputs must be within the configured score range.",
			condition={"operation": "assess_quality", "quality_score_invalid": True},
			effect={
				"decision": "deny",
				"reason": "invalid_quality_score",
				"required_action": "normalize_quality_score"
			}
		),
		CapabilityRule(
			name="duplicate_candidates_require_review",
			description="Likely duplicate entities require stewardship review.",
			condition={"duplicate_confidence_gt": 70.0, "steward_review_recorded": False},
			effect={
				"decision": "require_review",
				"reason": "duplicate_review_required",
				"required_action": "complete_steward_review"
			}
		),
		CapabilityRule(
			name="auto_merge_requires_high_confidence",
			description="Automated golden-record merges require configured high confidence.",
			condition={"operation": "auto_merge_duplicate", "auto_merge_confidence_low": True},
			effect={
				"decision": "deny",
				"reason": "auto_merge_confidence_too_low",
				"required_action": "route_to_steward_review"
			}
		),
		CapabilityRule(
			name="golden_record_merge_requires_survivorship",
			description="Golden-record merges require a survivorship policy.",
			condition={"operation": "merge_golden_record", "survivorship_policy_present": False},
			effect={
				"decision": "deny",
				"reason": "survivorship_policy_required",
				"required_action": "attach_survivorship_policy"
			}
		),
		CapabilityRule(
			name="conflicted_merge_requires_independent_steward",
			description="Conflicted merges require an independent steward decision.",
			condition={"operation": "merge_golden_record", "conflict_present": True, "independent_steward_present": False},
			effect={
				"decision": "require_review",
				"reason": "independent_steward_required",
				"required_action": "assign_independent_steward"
			}
		),
		CapabilityRule(
			name="restricted_entity_requires_audit_trail",
			description="Restricted master data requires mutation audit evidence.",
			condition={"entity_classification": "restricted", "audit_evidence_present": False},
			effect={
				"decision": "deny",
				"reason": "audit_evidence_required",
				"required_action": "record_audit_evidence"
			}
		),
		CapabilityRule(
			name="restricted_entity_requires_classification_evidence",
			description="Restricted or sensitive attributes require classification evidence.",
			condition={"restricted_attributes_present": True, "classification_evidence_present": False},
			effect={
				"decision": "deny",
				"reason": "classification_evidence_required",
				"required_action": "attach_classification_evidence"
			}
		),
		CapabilityRule(
			name="cross_reference_requires_source_evidence",
			description="Cross-reference mapping changes require source-system evidence.",
			condition={"operation": "update_cross_reference", "source_system_evidence_present": False},
			effect={
				"decision": "deny",
				"reason": "source_system_evidence_required",
				"required_action": "attach_source_system_evidence"
			}
		),
		CapabilityRule(
			name="retire_requires_lineage_evidence",
			description="Retiring master data requires lineage and audit evidence.",
			condition={"operation": "retire_entity", "lineage_evidence_present": False},
			effect={
				"decision": "deny",
				"reason": "lineage_evidence_required",
				"required_action": "attach_lineage_evidence"
			}
		),
		CapabilityRule(
			name="review_decisions_require_notes",
			description="Stewardship decisions require review notes.",
			condition={"operation": "review", "review_notes_present": False},
			effect={
				"decision": "deny",
				"reason": "review_notes_required",
				"required_action": "attach_review_notes"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return MDM UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/mdm/dashboard", "MDMDashboard", "mdm:view", "Overview"),
		CapabilityUIRoute("entities", "/mdm/entities", "EntityWorkbench", "mdm:manage_entities", "Operations"),
		CapabilityUIRoute("golden_records", "/mdm/golden-records", "GoldenRecordManager", "mdm:manage_golden_records", "Operations"),
		CapabilityUIRoute("quality", "/mdm/quality", "QualityConsole", "mdm:view_quality", "Governance"),
		CapabilityUIRoute("duplicates", "/mdm/duplicates", "DuplicateReviewQueue", "mdm:review_duplicates", "Governance"),
		CapabilityUIRoute("stewardship", "/mdm/stewardship", "StewardshipQueue", "mdm:steward", "Governance"),
		CapabilityUIRoute("lineage", "/mdm/lineage", "EntityLineageTrace", "mdm:view_lineage", "Governance"),
		CapabilityUIRoute("cross_references", "/mdm/cross-references", "CrossReferenceConsole", "mdm:manage_cross_references", "Operations"),
		CapabilityUIRoute("publish", "/mdm/publish", "PublishReadinessConsole", "mdm:publish", "Operations"),
		CapabilityUIRoute("analytics", "/mdm/analytics", "MDMAnalytics", "mdm:view_analytics", "Intelligence"),
		CapabilityUIRoute("audit", "/mdm/audit", "MDMAuditTimeline", "mdm:view_audit", "Governance"),
		CapabilityUIRoute("adapters", "/mdm/adapters", "MDMAdapterHealth", "mdm:admin", "Administration"),
		CapabilityUIRoute("settings", "/mdm/settings", "MDMSettings", "mdm:admin", "Administration")
	]
	return {
		"shell": "apg_python",
		"view_module": "views.py",
		"api_prefix": "/mdm/api/v1",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "static/"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable MDM capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "mdm",
		"display_name": "Master Data Management",
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
	"""Convenience wrapper for default MDM rule evaluation."""
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
