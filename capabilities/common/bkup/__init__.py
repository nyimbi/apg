"""APG Backup and Restore (BKUP) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "bkup"
__capability_name__ = "Backup and Restore"
__apg_dependencies__ = ["encr", "conf", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "bkup",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware backup plans, snapshots, restore testing, retention, encryption, and continuity governance",
	"category": "infrastructure_operations",
	"subcategory": "backup_restore",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["backup_plans", "snapshots", "restore_testing", "retention_policy", "continuity_reporting"],
	"permissions": ["bkup:view", "bkup:manage_plans", "bkup:run_backup", "bkup:restore", "bkup:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register BKUP with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "bkup",
		"aliases": ["backup", "restore", "business_continuity"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["schd", "moni", "comp", "depl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"backup_plans": "Define tenant-scoped backup plans, sources, schedules, retention, and ownership",
			"snapshots": "Create encrypted snapshots with integrity and lineage metadata",
			"restore_testing": "Run restore drills, point-in-time validation, and recovery reports",
			"retention_policy": "Enforce retention, legal hold, deletion, and compliance policies",
			"capability_rules": "Evaluate deterministic backup and restore rules",
			"visual_theming": "Apply continuity-operations theme tokens and components"
		},
		"endpoints": {"plans": "/bkup/api/v1/plans", "snapshots": "/bkup/api/v1/snapshots", "restores": "/bkup/api/v1/restores", "retention": "/bkup/api/v1/retention", "reports": "/bkup/api/v1/reports"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get BKUP capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
