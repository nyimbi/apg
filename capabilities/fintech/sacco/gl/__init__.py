"""SACCO General Ledger — APG capability metadata and composition registration.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

CAPABILITY_ID = "fintech_sacco_gl"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DESCRIPTION = (
	"SASRA-compliant SACCO General Ledger implementing the ICPAK chart of accounts "
	"standard with full double-entry accounting, period management, financial reporting, "
	"and subsidiary ledger reconciliation."
)

APG_CAPABILITY_METADATA: dict = {
	"id": CAPABILITY_ID,
	"version": CAPABILITY_VERSION,
	"name": "SACCO General Ledger",
	"description": CAPABILITY_DESCRIPTION,
	"category": "fintech",
	"sub_category": "sacco",
	"domain": "accounting",
	"standards": ["ICPAK", "SASRA", "IFRS"],
	"dependencies": [
		"fintech_sacco_lnd",
		"fintech_sacco_mem",
		"fintech_sacco_dep",
	],
	"composable_with": [
		"fintech_sacco_lnd",
		"fintech_sacco_dep",
		"fintech_sacco_div",
		"fintech_treasury",
		"fintech_compliance",
		"grc_audit",
	],
	"api_prefix": "/api/fintech/sacco/gl",
	"status": "active",
}

try:
	from capabilities.composition import register_capability
	register_capability(APG_CAPABILITY_METADATA)
except Exception:
	pass

from .api import bp as blueprint  # noqa: E402

__all__ = ["blueprint", "APG_CAPABILITY_METADATA", "CAPABILITY_ID"]
