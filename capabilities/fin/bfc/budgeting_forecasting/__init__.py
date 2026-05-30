"""APG budgeting and forecasting capability package."""

from __future__ import annotations

from .capability_contract import (
	BFC_EVENT_STREAM,
	SUPPORTED_BFC_AGENT_ROLES,
	SUPPORTED_BFC_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import BFCService, BudgetingForecastingService


CAPABILITY_ID = "bfc_budgeting_forecasting"
CAPABILITY_NAME = "Budgeting and Forecasting"
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
	"BFCService",
	"BFC_EVENT_STREAM",
	"BudgetingForecastingService",
	"CAPABILITY_ID",
	"CAPABILITY_NAME",
	"CAPABILITY_VERSION",
	"SUPPORTED_BFC_AGENT_ROLES",
	"SUPPORTED_BFC_AGENT_RUNTIMES",
	"evaluate_capability_rules",
	"event_stream_name",
	"get_capability_contract",
	"register_capability",
	"streaming_manifest",
]
