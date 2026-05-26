"""Executable capability contract for APG Audio Processing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"transcription": {
		"default_language": "auto",
		"speaker_diarization_enabled": True,
		"minimum_confidence": 0.78,
		"real_time_streaming_enabled": True
	},
	"synthesis": {
		"voice_model_policy_required": True,
		"voice_cloning_consent_required": True,
		"max_text_length": 10000,
		"watermark_synthetic_audio": True
	},
	"analysis": {
		"sentiment_analysis_enabled": True,
		"topic_detection_enabled": True,
		"content_classification_enabled": True,
		"quality_assessment_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_audio_processing": True,
		"recording_consent_required": True,
		"retention_policy_required": True
	},
	"ui": {
		"enable_audio_dashboard": True,
		"enable_transcription_console": True,
		"enable_synthesis_studio": True,
		"enable_analysis_workbench": True
	},
	"theme": {
		"default_theme": "audp_audio_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "transcription", "synthesis", "analysis", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["transcription", "synthesis", "analysis", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All audio operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "recording_consent_required", "description": "Audio processing requires recording consent.", "condition": {"operation": "process_recording", "recording_consent_recorded": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "record_audio_consent"}},
	{"name": "voice_cloning_requires_consent", "description": "Voice cloning requires voice-owner consent.", "condition": {"operation": "clone_voice", "voice_owner_consent_recorded": False}, "effect": {"decision": "deny", "reason": "voice_owner_consent_required", "required_action": "record_voice_owner_consent"}},
	{"name": "synthetic_audio_requires_watermark", "description": "Synthetic audio output requires watermarking.", "condition": {"synthetic_audio_requested": True, "watermark_applied": False}, "effect": {"decision": "deny", "reason": "synthetic_audio_watermark_required", "required_action": "apply_audio_watermark"}},
	{"name": "audio_model_requires_policy", "description": "Audio model use requires an attached policy.", "condition": {"model_invocation": True, "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "low_transcription_confidence_requires_review", "description": "Low-confidence transcripts require review.", "condition": {"transcription_confidence_lt": 0.78, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "transcription_review_required", "required_action": "review_transcript"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/audp/dashboard", "component": "AUPDDashboard", "permission": "audp:view", "nav_group": "Overview"},
	{"name": "transcription", "path": "/audp/transcription", "component": "TranscriptionConsole", "permission": "audp:transcribe", "nav_group": "Processing"},
	{"name": "synthesis", "path": "/audp/synthesis", "component": "SynthesisStudio", "permission": "audp:synthesize", "nav_group": "Processing"},
	{"name": "analysis", "path": "/audp/analysis", "component": "AudioAnalysis", "permission": "audp:analyze", "nav_group": "Analysis"},
	{"name": "sessions", "path": "/audp/sessions", "component": "AudioSessions", "permission": "audp:view", "nav_group": "Runtime"},
	{"name": "models", "path": "/audp/models", "component": "AudioModelRegistry", "permission": "audp:manage_models", "nav_group": "Models"},
	{"name": "quality", "path": "/audp/quality", "component": "AudioQuality", "permission": "audp:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/audp/settings", "component": "AUDPSettings", "permission": "audp:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "audp_audio_intelligence",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#9F7AEA",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"waveform_viewer": {"icon": "audio-lines", "status_indicator": "quality-pill", "risk_style": "consent-band"},
		"transcript_panel": {"visual": "speaker-transcript", "highlight": "confidence-chip"},
		"synthesis_studio": {"visual": "voice-control", "status_style": "watermark-chip"},
		"analysis_grid": {"visual": "audio-metrics", "status_style": "topic-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable AUDP capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "audp",
		"display_name": "Audio Processing",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/audp/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default AUDP governance rules."""
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
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
