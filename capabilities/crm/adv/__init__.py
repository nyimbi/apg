"""APG advanced CRM analytics capability package."""

from __future__ import annotations

from .capability_contract import (
	CRM_EVENT_STREAM,
	SUPPORTED_CRM_AGENT_ROLES,
	SUPPORTED_CRM_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import AdvancedCRMService, CRMService


CAPABILITY_ID = "crm_adv"
CAPABILITY_NAME = "Advanced CRM Analytics"
CAPABILITY_VERSION = "2.1.0"


def register_capability() -> dict[str, object]:
	contract = get_capability_contract()
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": contract["provides"],
		"requires": contract["requires"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


__all__ = [
	"AdvancedCRMService",
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"CRM_EVENT_STREAM",
	"CRMService",
	"SUPPORTED_CRM_AGENT_ROLES",
	"SUPPORTED_CRM_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
