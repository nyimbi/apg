"""APG Edge Computing (EDGE) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import EdgeService

__version__ = "1.0.0"
__capability_id__ = "edge"
__capability_name__ = "Edge Computing"
__capability_code__ = "EDGE_COMPUTING"
__apg_dependencies__ = ["dist", "cach", "conf"]
__composition_keywords__ = ["requires_edge_computing", "integrates_with_edge_computing", "uses_edge_computing"]

capability_metadata: dict[str, Any] = {
	"name": "edge",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware edge nodes, workloads, synchronization, offline execution, and edge deployment governance",
	"category": "infrastructure_operations",
	"subcategory": "edge_computing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["edge_nodes", "edge_workloads", "offline_execution", "edge_sync", "edge_deployment"],
	"permissions": ["edge:view", "edge:manage_nodes", "edge:deploy_workloads", "edge:sync", "edge:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register EDGE with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "edge",
		"aliases": ["edge_computing", "edge_runtime", "edge_deployment"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["iotd", "cicd", "moni", "geos"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"edge_nodes": "Register and monitor tenant-scoped edge nodes, fleets, and locations",
			"edge_workloads": "Deploy workloads, models, caches, and agents to edge runtimes",
			"offline_execution": "Run offline-first workflows with sync and conflict policy",
			"edge_sync": "Synchronize state, events, and artifacts between edge and core",
			"capability_rules": "Evaluate deterministic edge-computing rules",
			"visual_theming": "Apply edge-operations theme tokens and components"
		},
		"endpoints": {"nodes": "/edge/api/v1/nodes", "workloads": "/edge/api/v1/workloads", "sync": "/edge/api/v1/sync", "fleets": "/edge/api/v1/fleets", "deployments": "/edge/api/v1/deployments"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get EDGE capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["composition_keywords"] = __composition_keywords__
	info["contract"] = get_capability_contract()
	return info


__all__ = ["EdgeService", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__capability_code__", "__apg_dependencies__", "__composition_keywords__"]
