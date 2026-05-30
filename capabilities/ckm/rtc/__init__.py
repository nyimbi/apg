"""APG CKM Real-Time Collaboration capability package."""

from __future__ import annotations

from .capability_contract import (
	SUPPORTED_RTC_AGENT_ROLES,
	SUPPORTED_RTC_AGENT_RUNTIMES,
	SUPPORTED_RTC_MODES,
	SUPPORTED_RTC_PROTOCOLS,
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)
from .lifecycle import (
	RtcAgent,
	RtcDecision,
	RtcLifecycleService,
	RtcMessage,
	RtcParticipant,
	RtcSession,
)


__version__ = "1.0.0"
__author__ = "Datacraft"

APG_CAPABILITY_INFO = {
	"id": "ckm_rtc",
	"name": "Real-Time Collaboration",
	"version": __version__,
	"description": "Tenant-scoped collaboration sessions, presence, messaging, media guardrails, decisions, and AI-agent review for generated APG applications.",
	"category": "ckm",
	"provides": get_capability_contract()["provides"],
	"requires": get_capability_contract()["requires"],
	"supported_modes": SUPPORTED_RTC_MODES,
	"supported_protocols": SUPPORTED_RTC_PROTOCOLS,
	"supported_agent_runtimes": SUPPORTED_RTC_AGENT_RUNTIMES,
	"streaming": streaming_manifest(),
}

__all__ = [
	"APG_CAPABILITY_INFO",
	"RtcAgent",
	"RtcDecision",
	"RtcLifecycleService",
	"RtcMessage",
	"RtcParticipant",
	"RtcSession",
	"SUPPORTED_RTC_AGENT_ROLES",
	"SUPPORTED_RTC_AGENT_RUNTIMES",
	"SUPPORTED_RTC_MODES",
	"SUPPORTED_RTC_PROTOCOLS",
	"evaluate_capability_rules",
	"get_capability_contract",
	"streaming_manifest",
]
