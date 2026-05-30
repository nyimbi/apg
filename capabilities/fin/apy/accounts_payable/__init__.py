"""APG accounts payable capability package."""

from __future__ import annotations

from .capability_contract import (
	AP_EVENT_STREAM,
	SUPPORTED_AP_AGENT_ROLES,
	SUPPORTED_AP_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import APService, AccountsPayableService


CAPABILITY_ID = "apy_accounts_payable"
CAPABILITY_NAME = "Accounts Payable"
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
	"APService",
	"AP_EVENT_STREAM",
	"AccountsPayableService",
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"SUPPORTED_AP_AGENT_ROLES",
	"SUPPORTED_AP_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
