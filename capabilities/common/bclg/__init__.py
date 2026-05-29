"""APG Blockchain Ledger Services capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import BclgService

__version__ = "1.0.0"
__capability_id__ = "bclg"
__capability_name__ = "Blockchain Ledger Services"
__apg_dependencies__ = ["encr", "keym", "comp"]

capability_metadata: dict[str, Any] = {
	"name": "bclg",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant ledger networks, signed transactions, smart contract governance, key custody, and compliance auditability",
	"category": "advanced_infrastructure",
	"subcategory": "distributed_ledger",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["ledger_registry", "transaction_signing", "smart_contract_governance", "key_custody", "ledger_audit"],
	"permissions": ["bclg:view", "bclg:transact", "bclg:review_transactions", "bclg:manage_ledgers", "bclg:manage_contracts", "bclg:review_contracts", "bclg:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register BCLG with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "bclg",
		"aliases": ["blockchain", "ledger", "distributed-ledger"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "anom", "secu", "walt"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"ledger_registry": "Register ledger networks, owners, consensus profiles, and chain policy",
			"transaction_signing": "Sign, submit, and audit ledger transactions through governed key custody",
			"transaction_reviews": "Require independent review evidence before high-value transaction commit",
			"smart_contract_governance": "Review, approve, deploy, and version smart contract artifacts",
			"contract_deployment_reviews": "Require independent approval evidence before smart contract deployment",
			"key_custody": "Bind ledger operations to APG key management and encryption policy",
			"capability_rules": "Evaluate deterministic distributed-ledger governance rules",
			"visual_theming": "Apply blockchain ledger theme tokens and components"
		},
		"endpoints": {"ledgers": "/bclg/api/v1/ledgers", "transactions": "/bclg/api/v1/transactions", "transaction_reviews": "/bclg/api/v1/transactions/reviews", "contracts": "/bclg/api/v1/contracts", "contract_reviews": "/bclg/api/v1/contracts/reviews", "keys": "/bclg/api/v1/keys", "audit": "/bclg/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get BCLG capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["BclgService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
