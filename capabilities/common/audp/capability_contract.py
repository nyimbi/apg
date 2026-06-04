"""Executable capability contract for APG Audio Processing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_AUDIO_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AUDIO_AGENT_ROLES = ["transcript_reviewer", "synthesis_reviewer", "quality_analyst", "consent_auditor", "workflow_operator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"transcription": {
		"default_language": "auto",
		"speaker_diarization_enabled": True,
		"minimum_confidence": 0.78,
		"real_time_streaming_enabled": True,
		"human_review_required_below_confidence": True
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
	"audio_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_AUDIO_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_AUDIO_AGENT_ROLES
	},
	"governance": {
		"require_tenant_context": True,
		"audit_audio_processing": True,
		"recording_consent_required": True,
		"retention_policy_required": True,
		"state_change_reason_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax"
	},
	"observability": {
		"audit_required": True,
		"quality_metrics_required": True,
		"latency_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax"
	},
	"adapters": {
		"generated_app_runtime": "audio_runtime.AudpService",
		"api_helpers": "api_helpers.py",
		"view_models": "view_models.py",
		"event_stream": "bytewax",
		"ai_core": "aicr",
		"nlp": "nlpc",
		"model_lifecycle": "mlcm",
		"audit_sink": "audl",
		"notification": "ntfy",
		"collaboration": "colb",
		"cache": "cach"
	},
	"ui": {
		"enable_audio_dashboard": True,
		"enable_transcription_console": True,
		"enable_synthesis_studio": True,
		"enable_analysis_workbench": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True
	},
	"theme": {
		"default_theme": "audp_audio_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "transcription", "synthesis", "analysis", "audio_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["transcription", "synthesis", "analysis", "audio_agents", "governance", "observability", "adapters", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All audio operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "recording_consent_required", "description": "Audio processing requires recording consent.", "condition": {"operation": "process_recording", "recording_consent_recorded": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "record_audio_consent"}},
	{"name": "voice_cloning_requires_consent", "description": "Voice cloning requires voice-owner consent.", "condition": {"operation": "clone_voice", "voice_owner_consent_recorded": False}, "effect": {"decision": "deny", "reason": "voice_owner_consent_required", "required_action": "record_voice_owner_consent"}},
	{"name": "synthetic_audio_requires_watermark", "description": "Synthetic audio output requires watermarking.", "condition": {"synthetic_audio_requested": True, "watermark_applied": False}, "effect": {"decision": "deny", "reason": "synthetic_audio_watermark_required", "required_action": "apply_audio_watermark"}},
	{"name": "synthetic_audio_requires_release_review", "description": "Synthetic audio requires explicit release review before completion.", "condition": {"synthetic_audio_requested": True, "synthetic_release_reviewed": False}, "effect": {"decision": "require_review", "reason": "synthetic_audio_release_review_required", "required_action": "review_synthetic_audio"}},
	{"name": "audio_model_requires_policy", "description": "Audio model use requires an attached policy.", "condition": {"model_invocation": True, "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "low_transcription_confidence_requires_review", "description": "Low-confidence transcripts require review.", "condition": {"transcription_confidence_lt": 0.78, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "transcription_review_required", "required_action": "review_transcript"}},
	{"name": "audio_retention_policy_required", "description": "Audio jobs require retention policy evidence.", "condition": {"audio_job_requested": True, "retention_policy_present": False}, "effect": {"decision": "deny", "reason": "audio_retention_policy_required", "required_action": "attach_audio_retention_policy"}},
	{"name": "audio_agent_requires_registration", "description": "AI audio agents must be registered.", "condition": {"audio_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "audio_agent_registration_required", "required_action": "register_audio_agent"}},
	{"name": "audio_agent_runtime_supported", "description": "AI audio agents must use a supported runtime.", "condition": {"audio_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "audio_agent_runtime_not_supported", "required_action": "choose_supported_audio_agent_runtime"}},
	{"name": "audio_agent_requires_scope", "description": "AI audio agents require explicit scope.", "condition": {"audio_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "audio_agent_scope_required", "required_action": "set_audio_agent_scope"}},
	{"name": "audio_agent_requires_disclosure", "description": "AI audio-agent contributions require disclosure.", "condition": {"audio_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "audio_agent_disclosure_required", "required_action": "disclose_audio_agent"}},
	{"name": "audio_state_change_requires_reason", "description": "Audio job state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "audio_state_change_reason_required", "required_action": "record_audio_state_change_reason"}},
	{"name": "audio_state_change_requires_audit", "description": "Audio job state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audio_audit_event_required", "required_action": "record_audio_audit_event"}},
	{"name": "cross_tenant_audio_access_denied", "description": "Audio records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_audio_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_audio_mutation_requires_bytewax", "description": "Batch audio mutations must use Bytewax event streams.", "condition": {"operation": "batch_audio_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "write_requires_policy", "description": "Audio write operations require an explicit authorization policy.", "condition": {"operation_type": "write", "write_policy_present": False}, "effect": {"decision": "deny", "reason": "audp_write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "privilege_escalation_denied", "description": "Audio operators cannot self-grant elevated permissions.", "condition": {"operation": "assign_audp_permission", "target_tier_exceeds_actor_tier": True}, "effect": {"decision": "deny", "reason": "privilege_escalation_prevented", "required_action": "route_to_higher_authority_approver"}},
	{"name": "audio_delete_requires_approval", "description": "Audio asset deletion requires explicit approval.", "condition": {"operation": "delete_audio_asset", "delete_approved": False}, "effect": {"decision": "deny", "reason": "audio_delete_approval_required", "required_action": "record_audio_delete_approval"}},
	{"name": "pii_in_audio_requires_redaction_policy", "description": "Audio assets containing PII require a redaction policy before processing.", "condition": {"audio_contains_pii": True, "redaction_policy_attached": False}, "effect": {"decision": "deny", "reason": "audio_pii_redaction_policy_required", "required_action": "attach_audio_pii_redaction_policy"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/audp/dashboard", "component": "AUPDDashboard", "permission": "audp:view", "nav_group": "Overview"},
	{"name": "transcription", "path": "/audp/transcription", "component": "TranscriptionConsole", "permission": "audp:transcribe", "nav_group": "Processing"},
	{"name": "synthesis", "path": "/audp/synthesis", "component": "SynthesisStudio", "permission": "audp:synthesize", "nav_group": "Processing"},
	{"name": "analysis", "path": "/audp/analysis", "component": "AudioAnalysis", "permission": "audp:analyze", "nav_group": "Analysis"},
	{"name": "sessions", "path": "/audp/sessions", "component": "AudioSessions", "permission": "audp:view", "nav_group": "Runtime"},
	{"name": "models", "path": "/audp/models", "component": "AudioModelRegistry", "permission": "audp:manage_models", "nav_group": "Models"},
	{"name": "consents", "path": "/audp/consents", "component": "AudioConsentCenter", "permission": "audp:govern", "nav_group": "Governance"},
	{"name": "reviews", "path": "/audp/reviews", "component": "AudioReviewQueue", "permission": "audp:review", "nav_group": "Governance"},
	{"name": "quality", "path": "/audp/quality", "component": "AudioQuality", "permission": "audp:view", "nav_group": "Governance"},
	{"name": "agents", "path": "/audp/agents", "component": "AudioAgentPanel", "permission": "audp:govern", "nav_group": "Agents"},
	{"name": "audit", "path": "/audp/audit", "component": "AudioAuditTrail", "permission": "audp:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/audp/analytics", "component": "AudioAnalytics", "permission": "audp:view", "nav_group": "Operations"},
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
		"analysis_grid": {"visual": "audio-metrics", "status_style": "topic-chip"},
		"consent_banner": {"icon": "badge-check", "status_style": "consent-chip"},
		"review_queue": {"icon": "list-checks", "highlight": "confidence-review-pill"},
		"synthetic_watermark": {"icon": "waves", "status_style": "watermark-chip"},
		"agent_panel": {"icon": "bot", "status_style": "scope-chip"},
		"audit_timeline": {"icon": "list-todo", "status_style": "governance-chip"}
	}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.audp.lifecycle",
	"state": ["consents", "model_policies", "jobs", "transcript_reviews", "synthesis_reviews", "audio_agents", "governance_events"],
	"events": [
		"audio_consent_recorded",
		"audio_model_policy_attached",
		"transcription_requested",
		"transcript_review_requested",
		"transcript_review_decided",
		"synthesis_requested",
		"synthesis_review_requested",
		"synthesis_review_decided",
		"voice_clone_requested",
		"analysis_requested",
		"audio_agent_registered",
		"audio_job_state_changed"
	],
	"batch_mutation_guardrail": "batch_audio_mutation_requires_bytewax"
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
		"provides": ["audio_transcription", "voice_synthesis", "audio_analysis", "speaker_diarization", "audio_enhancement", "audio_consent_governance", "audio_review_governance", "audio_agents"],
		"requires": ["aicr", "nlpc", "mlcm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/audp/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING)
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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
