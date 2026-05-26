"""
Executable capability contract for APG Computer Vision.

CVSN is a first-class APG capability: it exposes tenant-scoped
configuration, deterministic policy rules, UI surfaces, and visual theme
tokens that composition, admin, and test tooling can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityConfiguration:
	"""Tenant-scoped CVSN configuration defaults and schema."""

	defaults: dict[str, Any] = field(default_factory=lambda: {
		"tenant_id": "default",
		"processing": {
			"max_file_size_mb": 50,
			"max_batch_size": 100,
			"default_async_threshold_files": 10,
			"allowed_image_types": ["image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"],
			"allowed_document_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff"],
			"allowed_video_types": ["video/mp4", "video/avi", "video/mov", "video/mkv"]
		},
		"ocr": {
			"default_language": "eng",
			"default_engine": "tesseract",
			"enhance_image": True,
			"extract_tables": False,
			"extract_forms": False
		},
		"detection": {
			"default_object_model": "yolov8n.pt",
			"default_confidence_threshold": 0.5,
			"default_iou_threshold": 0.4,
			"max_detections": 100
		},
		"safety": {
			"smoke_fire_alerting": True,
			"osha_zone_monitoring": True,
			"people_counting": True,
			"barcode_qr_tracking": True,
			"severity_alert_threshold": "high"
		},
		"privacy": {
			"facial_recognition_enabled": False,
			"biometric_consent_required": True,
			"biometric_anonymization_required": True,
			"default_retention_days": 30
		},
		"ui": {
			"enable_dashboard": True,
			"enable_document_processing": True,
			"enable_video_analytics": True,
			"enable_factory_safety": True,
			"enable_model_management": True
		},
		"theme": {
			"default_theme": "cvsn_industrial",
			"allow_tenant_overrides": True
		}
	})
	schema: dict[str, Any] = field(default_factory=lambda: {
		"type": "object",
		"required": ["tenant_id", "processing", "ocr", "detection", "safety", "privacy", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
			"processing": {"type": "object"},
			"ocr": {"type": "object"},
			"detection": {"type": "object"},
			"safety": {"type": "object"},
			"privacy": {"type": "object"},
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
	"""Simple CVSN policy rule definition."""

	name: str
	description: str
	condition: dict[str, Any]
	effect: dict[str, Any]


class CapabilityRuleEngine:
	"""Deterministic CVSN rule engine for policy and workflow decisions."""

	def __init__(self, rules: list[CapabilityRule] | None = None):
		self.rules = rules or default_rules()

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate all matching rules against a vision processing context."""
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
	"""UI route exposed by CVSN."""

	name: str
	path: str
	component: str
	permission: str
	nav_group: str


@dataclass(frozen=True)
class CapabilityTheme:
	"""Visual theme contract for CVSN UI surfaces."""

	name: str = "cvsn_industrial"
	tokens: dict[str, str] = field(default_factory=lambda: {
		"color.primary": "#255E5C",
		"color.accent": "#C07A21",
		"color.success": "#2E7D32",
		"color.warning": "#A16207",
		"color.danger": "#B42318",
		"surface.canvas": "#F4F7F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1F2933",
		"text.secondary": "#52616B",
		"border.radius": "8px",
		"density": "compact"
	})
	components: dict[str, dict[str, str]] = field(default_factory=lambda: {
		"vision_canvas": {
			"background": "surface.canvas",
			"tool_overlay": "top-right",
			"zoom_controls": "bottom-right"
		},
		"detection_box": {
			"stroke": "color.accent",
			"label_position": "top-left",
			"confidence_variant": "badge"
		},
		"safety_alert": {
			"icon": "shield-alert",
			"variant": "critical",
			"requires_acknowledgement": "true"
		},
		"ocr_region": {
			"stroke": "color.primary",
			"fill": "transparent",
			"text_anchor": "below"
		}
	})


def default_rules() -> list[CapabilityRule]:
	"""Default CVSN rules available to every tenant."""
	return [
		CapabilityRule(
			name="require_tenant_isolation",
			description="All vision jobs must carry a tenant identifier.",
			condition={"tenant_id_missing": True},
			effect={
				"decision": "deny",
				"reason": "tenant_id_required",
				"required_action": "attach_tenant_context"
			}
		),
		CapabilityRule(
			name="biometric_processing_requires_controls",
			description="Facial recognition requires consent and anonymization controls.",
			condition={"processing_type": "facial_recognition", "consent_recorded": False},
			effect={
				"decision": "deny",
				"reason": "biometric_consent_required",
				"required_action": "record_biometric_consent"
			}
		),
		CapabilityRule(
			name="biometric_retention_requires_limit",
			description="Biometric workloads must declare bounded retention.",
			condition={"processing_type": "facial_recognition", "retention_days_gt": 30},
			effect={
				"decision": "deny",
				"reason": "biometric_retention_exceeds_default",
				"required_action": "lower_retention_or_approve_exception"
			}
		),
		CapabilityRule(
			name="factory_hazard_requires_alerting",
			description="High-severity smoke, fire, and OSHA findings require alerting.",
			condition={"domain": "factory_safety", "severity_in": ["high", "critical"], "alerting_enabled": False},
			effect={
				"decision": "deny",
				"reason": "factory_safety_alerting_required",
				"required_action": "enable_alerting"
			}
		),
		CapabilityRule(
			name="large_batch_requires_async_queue",
			description="Large batches must use asynchronous queue execution.",
			condition={"batch_size_gt": 10, "async_queue_enabled": False},
			effect={
				"decision": "require_review",
				"reason": "large_batch_requires_async_queue",
				"required_action": "enable_async_queue"
			}
		)
	]


def ui_manifest() -> dict[str, Any]:
	"""Return CVSN UI surface manifest."""
	routes = [
		CapabilityUIRoute("dashboard", "/cvsn/dashboard", "ComputerVisionDashboard", "cv:read", "Operations"),
		CapabilityUIRoute("documents", "/cvsn/documents", "DocumentProcessingWorkbench", "cv:ocr", "Process"),
		CapabilityUIRoute("images", "/cvsn/images", "ImageAnalysisWorkbench", "cv:object_detection", "Process"),
		CapabilityUIRoute("video", "/cvsn/video", "VideoAnalyticsWorkbench", "cv:video_analysis", "Process"),
		CapabilityUIRoute("quality", "/cvsn/quality", "QualityInspectionWorkbench", "cv:quality_control", "Factory"),
		CapabilityUIRoute("safety", "/cvsn/safety", "FactorySafetyConsole", "cv:analytics", "Factory"),
		CapabilityUIRoute("models", "/cvsn/models", "VisionModelManagement", "cv:model_management", "Administration"),
		CapabilityUIRoute("rules", "/cvsn/rules", "VisionRuleWorkbench", "cv:admin", "Governance"),
		CapabilityUIRoute("settings", "/cvsn/settings", "VisionSettings", "cv:admin", "Administration")
	]
	return {
		"shell": "fastapi_flask_appbuilder",
		"frontend_bundle": "views.py",
		"routes": [route.__dict__ for route in routes],
		"template_roots": ["templates/", "views.py"],
		"requires_theme": True
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CVSN capability contract."""
	config = CapabilityConfiguration()
	theme = CapabilityTheme()
	return {
		"capability": "cvsn",
		"display_name": "Computer Vision & Visual Intelligence",
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
	"""Convenience wrapper for default CVSN rule evaluation."""
	return CapabilityRuleEngine().evaluate(context)


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_gt"):
			field_name = key[:-3]
			if not context.get(field_name, 0) > expected:
				return False
		elif key.endswith("_in"):
			field_name = key[:-3]
			if context.get(field_name) not in expected:
				return False
		elif key.endswith("_missing"):
			field_name = key[:-8]
			if (field_name in context and context.get(field_name) not in {None, ""}) == expected:
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
