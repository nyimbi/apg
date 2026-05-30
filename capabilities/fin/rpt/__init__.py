"""APG financial reporting capability package."""

from __future__ import annotations

from .capability_contract import (
	RPT_EVENT_STREAM,
	SUPPORTED_RPT_AGENT_ROLES,
	SUPPORTED_RPT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import FinancialReportingService, RPTService


CAPABILITY_ID = "fin_rpt"
CAPABILITY_NAME = "Financial Reporting"
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
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"FinancialReportingService",
	"RPTService",
	"RPT_EVENT_STREAM",
	"SUPPORTED_RPT_AGENT_ROLES",
	"SUPPORTED_RPT_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
