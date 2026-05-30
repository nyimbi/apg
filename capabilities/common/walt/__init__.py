"""APG Wallet and Payment Core (WALT) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	SUPPORTED_WALT_AGENT_ROLES,
	SUPPORTED_WALT_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .service import WaltService
from .wallet_runtime import WaltAgentRecord

__version__ = "1.0.0"
__capability_id__ = "walt"
__capability_name__ = "Wallet and Payment Core"
__apg_dependencies__ = ["encr", "auth", "comp", "audl", "wflo"]

capability_metadata: dict[str, Any] = {
	"name": "walt",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware wallet ledgers, payment instruments, transaction policies, settlement, reconciliation, and financial controls",
	"category": "specialized_ai_analytics",
	"subcategory": "wallet_payment_core",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["wallet_ledger", "payment_instruments", "transaction_authorization", "settlement", "reconciliation", "walt_agents"],
	"permissions": ["walt:view", "walt:manage_wallets", "walt:authorize", "walt:settle", "walt:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register WALT with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "walt",
		"aliases": ["wallet", "payment_core", "payments"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ntfy", "conn", "anom"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"wallet_ledger": "Maintain tenant-scoped wallet balances, holds, journals, and ledger events",
			"payment_instruments": "Tokenize, encrypt, verify, and govern payment instruments",
			"transaction_authorization": "Authorize transactions with limits, risk, compliance, and MFA controls",
			"settlement": "Track settlement batches, reconciliation, exceptions, and reversals",
			"capability_rules": "Evaluate deterministic wallet and payment rules",
			"visual_theming": "Apply wallet-operations theme tokens and components",
			"walt_agents": "Govern payment, risk, settlement, reconciliation, and instrument review agents"
		},
		"endpoints": {
			"wallets": "/walt/api/v1/wallets",
			"transactions": "/walt/api/v1/transactions",
			"instruments": "/walt/api/v1/instruments",
			"settlement": "/walt/api/v1/settlement",
			"reconciliation": "/walt/api/v1/reconciliation",
			"agents": "/walt/api/v1/agents",
			"policy": "/walt/api/v1/policy"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get WALT capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = [
	"SUPPORTED_WALT_AGENT_ROLES",
	"SUPPORTED_WALT_AGENT_RUNTIMES",
	"WaltAgentRecord",
	"WaltService",
	"capability_metadata",
	"event_stream_name",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_capability_info",
	"register_capability",
	"streaming_manifest",
	"__version__",
	"__capability_id__",
	"__capability_name__",
	"__apg_dependencies__",
]
