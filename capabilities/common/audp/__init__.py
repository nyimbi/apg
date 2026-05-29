"""APG Audio Processing (AUDP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "audp"
__capability_name__ = "Audio Processing"
__apg_dependencies__ = ["aicr", "nlpc", "mlcm"]

capability_metadata: dict[str, Any] = {
	"name": "audp",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware audio transcription, synthesis, analysis, enhancement, speaker diarization, and voice-model governance",
	"category": "specialized_ai_analytics",
	"subcategory": "audio_processing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["audio_transcription", "voice_synthesis", "audio_analysis", "speaker_diarization", "audio_enhancement", "audio_consent_governance", "audio_review_governance"],
	"permissions": ["audp:view", "audp:transcribe", "audp:synthesize", "audp:analyze", "audp:manage_models", "audp:govern", "audp:review", "audp:admin"]
}

__capability_code__ = "AUDIO_PROCESSING"
__composition_keywords__ = ["processes_audio", "transcription_enabled", "voice_synthesis_capable", "audio_analysis_aware", "real_time_audio"]


def register_capability() -> dict[str, Any]:
	"""Register AUDP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "audp",
		"aliases": ["audio_processing", "audio_intelligence", "speech_processing"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["colb", "ntfy", "cach", "audl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"audio_transcription": "Transcribe tenant-scoped audio with language, speaker, and confidence metadata",
			"voice_synthesis": "Generate governed speech output from approved text and voice models",
			"audio_analysis": "Analyze sentiment, topics, quality, content class, and acoustic signals",
			"speaker_diarization": "Identify and segment speakers with consent and retention controls",
			"audio_consent_governance": "Record and enforce recording and voice-owner consent evidence",
			"audio_review_governance": "Require human review for low-confidence transcripts and governed synthetic audio",
			"capability_rules": "Evaluate deterministic audio-processing governance rules",
			"visual_theming": "Apply audio-intelligence theme tokens and components"
		},
		"endpoints": {
			"transcription": "/audp/api/v1/transcription",
			"synthesis": "/audp/api/v1/synthesis",
			"analysis": "/audp/api/v1/analysis",
			"sessions": "/audp/api/v1/sessions",
			"models": "/audp/api/v1/models",
			"consents": "/audp/api/v1/consents",
			"reviews": "/audp/api/v1/reviews",
			"governance_events": "/audp/api/v1/governance-events"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get AUDP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["composition_keywords"] = __composition_keywords__
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__capability_code__", "__apg_dependencies__", "__composition_keywords__"]
