"""Executable capability contract for APG Computer Vision."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"processing": {
		"max_file_size_mb": 50,
		"max_batch_size": 100,
		"async_threshold_files": 10,
		"allowed_image_types": ["image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"],
		"allowed_document_types": ["application/pdf", "image/jpeg", "image/png", "image/tiff"],
		"allowed_video_types": ["video/mp4", "video/avi", "video/quicktime", "video/x-matroska"],
		"asset_hash_required": True,
	},
	"vision_tasks": {
		"enabled": [
			"ocr",
			"object_detection",
			"image_classification",
			"quality_inspection",
			"factory_safety",
			"video_analytics",
			"visual_similarity",
			"barcode_qr",
			"facial_analysis",
			"content_moderation",
		],
		"minimum_confidence_score": 0.70,
		"human_review_for_low_confidence": True,
	},
	"ocr": {
		"default_language": "eng",
		"extract_tables_supported": True,
		"extract_forms_supported": True,
		"minimum_text_confidence": 0.72,
	},
	"detection": {
		"default_object_model": "yolov8n",
		"default_confidence_threshold": 0.50,
		"default_iou_threshold": 0.40,
		"max_detections": 100,
	},
	"video": {
		"max_clip_seconds": 900,
		"sampling_policy_required": True,
		"streaming_supported": True,
	},
	"quality": {
		"inspection_plan_required": True,
		"defect_taxonomy_required": True,
		"critical_defect_requires_alert": True,
		"golden_sample_required": True,
	},
	"safety": {
		"smoke_fire_alerting": True,
		"osha_zone_monitoring": True,
		"people_counting": True,
		"severity_alert_threshold": "high",
		"incident_acknowledgement_required": True,
	},
	"privacy": {
		"facial_recognition_enabled": False,
		"biometric_consent_required": True,
		"biometric_anonymization_required": True,
		"default_retention_days": 30,
		"content_moderation_policy_required": True,
	},
	"model_registry": {
		"mlcm_link_required": True,
		"evaluation_required": True,
		"release_approval_required": True,
		"model_card_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"audit_processing": True,
		"cross_tenant_processing_allowed": False,
		"operator_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
		"quality_metrics_required": True,
	},
	"adapters": {
		"generated_app_runtime": "cvsn_runtime.CvsnService",
		"production_runtime": "service.CVProcessingService",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"model_lifecycle": "mlcm",
		"configuration": "conf",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"object_storage": "stor",
		"search_index": "srch",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_asset_workbench": True,
		"enable_document_processing": True,
		"enable_image_analysis": True,
		"enable_video_analytics": True,
		"enable_quality_inspection": True,
		"enable_safety_console": True,
		"enable_similarity_search": True,
		"enable_model_registry": True,
		"enable_review_console": True,
		"enable_audit_timeline": True,
		"enable_governance": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "cvsn_industrial", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"processing",
		"vision_tasks",
		"ocr",
		"detection",
		"video",
		"quality",
		"safety",
		"privacy",
		"model_registry",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"processing",
		"vision_tasks",
		"ocr",
		"detection",
		"video",
		"quality",
		"safety",
		"privacy",
		"model_registry",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All vision operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "asset_requires_source", "description": "Vision assets require a source reference.", "condition": {"operation": "ingest_asset", "source_ref_present": False}, "effect": {"decision": "deny", "reason": "asset_source_required", "required_action": "attach_source_reference"}},
	{"name": "asset_requires_supported_mime_type", "description": "Vision assets must use supported MIME types.", "condition": {"operation": "ingest_asset", "mime_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_mime_type", "required_action": "choose_supported_media_type"}},
	{"name": "asset_size_within_limit", "description": "Vision assets must fit the configured file-size limit.", "condition": {"operation": "ingest_asset", "file_size_mb_gt": 50}, "effect": {"decision": "deny", "reason": "asset_too_large", "required_action": "compress_or_split_asset"}},
	{"name": "asset_hash_required", "description": "Vision assets require digest evidence.", "condition": {"operation": "ingest_asset", "asset_hash_present": False}, "effect": {"decision": "deny", "reason": "asset_hash_required", "required_action": "record_asset_hash"}},
	{"name": "task_must_be_enabled", "description": "Vision processing tasks must be enabled.", "condition": {"operation": "run_job", "task_enabled": False}, "effect": {"decision": "deny", "reason": "vision_task_not_enabled", "required_action": "enable_task_or_change_pipeline"}},
	{"name": "operator_required", "description": "Vision jobs require an accountable operator.", "condition": {"operation": "run_job", "operator_present": False}, "effect": {"decision": "deny", "reason": "operator_required", "required_action": "attach_operator"}},
	{"name": "document_task_requires_document_asset", "description": "OCR jobs require document or image assets.", "condition": {"processing_type": "ocr", "asset_kind_in": ["video"]}, "effect": {"decision": "deny", "reason": "ocr_requires_document_or_image", "required_action": "choose_document_asset"}},
	{"name": "video_task_requires_video_asset", "description": "Video analytics jobs require video assets.", "condition": {"processing_type": "video_analytics", "asset_kind_ne": "video"}, "effect": {"decision": "deny", "reason": "video_asset_required", "required_action": "choose_video_asset"}},
	{"name": "quality_requires_inspection_plan", "description": "Quality inspection requires an inspection plan.", "condition": {"processing_type": "quality_inspection", "inspection_plan_attached": False}, "effect": {"decision": "deny", "reason": "inspection_plan_required", "required_action": "attach_inspection_plan"}},
	{"name": "quality_requires_defect_taxonomy", "description": "Quality inspection requires defect taxonomy.", "condition": {"processing_type": "quality_inspection", "defect_taxonomy_attached": False}, "effect": {"decision": "deny", "reason": "defect_taxonomy_required", "required_action": "attach_defect_taxonomy"}},
	{"name": "quality_critical_defect_requires_alert", "description": "Critical visual defects require alert routing.", "condition": {"processing_type": "quality_inspection", "critical_defect_detected": True, "alerting_enabled": False}, "effect": {"decision": "deny", "reason": "critical_defect_alert_required", "required_action": "enable_quality_alerts"}},
	{"name": "factory_hazard_requires_alerting", "description": "High-severity factory hazards require alerting.", "condition": {"processing_type": "factory_safety", "severity_in": ["high", "critical"], "alerting_enabled": False}, "effect": {"decision": "deny", "reason": "factory_safety_alerting_required", "required_action": "enable_safety_alerts"}},
	{"name": "factory_hazard_requires_acknowledgement", "description": "Critical factory hazards require acknowledgement.", "condition": {"processing_type": "factory_safety", "severity": "critical", "incident_acknowledged": False}, "effect": {"decision": "require_review", "reason": "incident_acknowledgement_required", "required_action": "acknowledge_incident"}},
	{"name": "facial_analysis_requires_consent", "description": "Facial analysis requires biometric consent.", "condition": {"processing_type": "facial_analysis", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "biometric_consent_required", "required_action": "record_biometric_consent"}},
	{"name": "facial_analysis_requires_anonymization", "description": "Facial analysis requires anonymization controls.", "condition": {"processing_type": "facial_analysis", "anonymization_enabled": False}, "effect": {"decision": "deny", "reason": "biometric_anonymization_required", "required_action": "enable_biometric_anonymization"}},
	{"name": "biometric_retention_requires_limit", "description": "Biometric workloads must declare bounded retention.", "condition": {"processing_type": "facial_analysis", "retention_days_gt": 30}, "effect": {"decision": "deny", "reason": "biometric_retention_exceeds_default", "required_action": "lower_retention_or_approve_exception"}},
	{"name": "content_moderation_requires_policy", "description": "Content moderation requires a policy reference.", "condition": {"processing_type": "content_moderation", "moderation_policy_attached": False}, "effect": {"decision": "deny", "reason": "moderation_policy_required", "required_action": "attach_moderation_policy"}},
	{"name": "low_confidence_requires_review", "description": "Low-confidence vision results require human review.", "condition": {"confidence_score_lt": 0.70, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_confidence_review_required", "required_action": "record_human_review"}},
	{"name": "large_batch_requires_async_queue", "description": "Large batches must use asynchronous queue execution.", "condition": {"batch_size_gt": 10, "async_queue_enabled": False}, "effect": {"decision": "require_review", "reason": "large_batch_requires_async_queue", "required_action": "enable_async_queue"}},
	{"name": "batch_size_within_limit", "description": "Vision batches must fit the configured batch limit.", "condition": {"batch_size_gt": 100}, "effect": {"decision": "deny", "reason": "batch_size_exceeds_limit", "required_action": "split_batch"}},
	{"name": "video_clip_within_limit", "description": "Video clips must fit configured duration limits.", "condition": {"processing_type": "video_analytics", "clip_seconds_gt": 900}, "effect": {"decision": "deny", "reason": "video_clip_too_long", "required_action": "split_video_clip"}},
	{"name": "video_requires_sampling_policy", "description": "Video analytics requires sampling policy evidence.", "condition": {"processing_type": "video_analytics", "sampling_policy_attached": False}, "effect": {"decision": "deny", "reason": "sampling_policy_required", "required_action": "attach_sampling_policy"}},
	{"name": "model_requires_mlcm_link", "description": "Vision model registrations require MLCM linkage.", "condition": {"operation": "register_model", "mlcm_model_ref_present": False}, "effect": {"decision": "deny", "reason": "mlcm_model_ref_required", "required_action": "link_mlcm_model"}},
	{"name": "model_requires_card", "description": "Vision model registrations require model-card evidence.", "condition": {"operation": "register_model", "model_card_present": False}, "effect": {"decision": "deny", "reason": "model_card_required", "required_action": "attach_model_card"}},
	{"name": "model_release_requires_evaluation", "description": "Vision model release requires evaluation evidence.", "condition": {"operation": "release_model", "evaluation_recorded": False}, "effect": {"decision": "deny", "reason": "model_evaluation_required", "required_action": "record_model_evaluation"}},
	{"name": "model_release_requires_approval", "description": "Vision model release requires approval.", "condition": {"operation": "release_model", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "model_release_approval_required", "required_action": "record_release_approval"}},
	{"name": "cross_tenant_processing_denied", "description": "Cross-tenant vision processing is denied by default.", "condition": {"cross_tenant_processing": True}, "effect": {"decision": "deny", "reason": "cross_tenant_processing_denied", "required_action": "use_tenant_scoped_asset"}},
	{"name": "state_change_requires_audit", "description": "Vision processing state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "vision_events_require_bytewax", "description": "Vision event streams must use Bytewax.", "condition": {"operation": "configure_vision_events", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/cvsn/dashboard", "component": "ComputerVisionDashboard", "permission": "cv:read", "nav_group": "Operations"},
	{"name": "assets", "path": "/cvsn/assets", "component": "VisionAssetWorkbench", "permission": "cv:write", "nav_group": "Process"},
	{"name": "documents", "path": "/cvsn/documents", "component": "DocumentProcessingWorkbench", "permission": "cv:ocr", "nav_group": "Process"},
	{"name": "images", "path": "/cvsn/images", "component": "ImageAnalysisWorkbench", "permission": "cv:object_detection", "nav_group": "Process"},
	{"name": "video", "path": "/cvsn/video", "component": "VideoAnalyticsWorkbench", "permission": "cv:video_analysis", "nav_group": "Process"},
	{"name": "quality", "path": "/cvsn/quality", "component": "QualityInspectionWorkbench", "permission": "cv:quality_control", "nav_group": "Factory"},
	{"name": "safety", "path": "/cvsn/safety", "component": "FactorySafetyConsole", "permission": "cv:analytics", "nav_group": "Factory"},
	{"name": "similarity", "path": "/cvsn/similarity", "component": "VisualSimilaritySearch", "permission": "cv:object_detection", "nav_group": "Search"},
	{"name": "review", "path": "/cvsn/review", "component": "VisionReviewConsole", "permission": "cv:review", "nav_group": "Governance"},
	{"name": "models", "path": "/cvsn/models", "component": "VisionModelRegistry", "permission": "cv:model_management", "nav_group": "Administration"},
	{"name": "governance", "path": "/cvsn/governance", "component": "VisionGovernance", "permission": "cv:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/cvsn/audit", "component": "VisionAuditTimeline", "permission": "cv:reports", "nav_group": "Governance"},
	{"name": "settings", "path": "/cvsn/settings", "component": "VisionSettings", "permission": "cv:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "cvsn_industrial",
	"tokens": {
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
		"density": "compact",
	},
	"components": {
		"vision_canvas": {"background": "surface.canvas", "tool_overlay": "top-right", "zoom_controls": "bottom-right"},
		"detection_box": {"stroke": "color.accent", "label_position": "top-left", "confidence_variant": "badge"},
		"safety_alert": {"icon": "shield-alert", "variant": "critical", "requires_acknowledgement": "true"},
		"ocr_region": {"stroke": "color.primary", "fill": "transparent", "text_anchor": "below"},
		"quality_defect_marker": {"stroke": "color.danger", "severity_style": "defect-chip"},
		"video_timeline": {"visual": "frame-strip", "alert_style": "event-marker"},
		"model_card": {"visual": "evaluation-summary", "status_style": "release-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CVSN capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "cvsn",
		"display_name": "Computer Vision",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "view_models.py",
			"api_prefix": "/cvsn/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default CVSN governance rules."""
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
		if key.endswith("_lt"):
			if key[:-3] not in context or not context[key[:-3]] < expected:
				return False
		elif key.endswith("_gt"):
			if key[:-3] not in context or not context[key[:-3]] > expected:
				return False
		elif key.endswith("_in"):
			if key[:-3] not in context or context[key[:-3]] not in expected:
				return False
		elif key.endswith("_ne"):
			if key[:-3] not in context or context[key[:-3]] == expected:
				return False
		elif key not in context or context[key] != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
