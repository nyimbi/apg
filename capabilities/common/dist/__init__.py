"""APG Distributed Computing (DIST) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import DistService

__version__ = "1.0.0"
__capability_id__ = "dist"
__capability_name__ = "Distributed Computing"
__capability_code__ = "DISTRIBUTED_COMPUTING"
__apg_dependencies__ = ["mqeb", "moni", "conf"]
__composition_keywords__ = ["requires_distributed_computing", "integrates_with_distributed_computing", "uses_distributed_computing"]

capability_metadata: dict[str, Any] = {
	"name": "dist",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware distributed jobs, worker pools, partitions, coordination, scaling, and distributed execution governance",
	"category": "infrastructure_operations",
	"subcategory": "distributed_computing",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["distributed_jobs", "worker_pools", "partitioned_execution", "coordination", "distributed_scaling", "compute_agents"],
	"permissions": ["dist:view", "dist:submit_jobs", "dist:manage_workers", "dist:scale", "dist:audit", "dist:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register DIST with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "dist",
		"aliases": ["distributed_computing", "parallel_processing", "worker_grid"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["cach", "logt", "edge", "schd", "bytewax", "audl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"distributed_jobs": "Submit tenant-scoped distributed jobs with partitions and retry policy",
			"worker_pools": "Manage worker pools, health, capacity, and queue assignments",
			"partitioned_execution": "Coordinate shards, fanout, aggregation, and idempotency",
			"distributed_scaling": "Scale workloads with monitoring and quota controls",
			"compute_agents": "Register governed AI compute agents with runtime, role, scope, disclosure, and audit",
			"capability_rules": "Evaluate deterministic distributed-computing rules",
			"visual_theming": "Apply distributed-compute theme tokens and components"
		},
		"endpoints": {"jobs": "/dist/api/v1/jobs", "workers": "/dist/api/v1/workers", "partitions": "/dist/api/v1/partitions", "queues": "/dist/api/v1/queues", "scaling": "/dist/api/v1/scaling", "agents": "/dist/api/v1/agents", "audit": "/dist/api/v1/audit"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get DIST capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["composition_keywords"] = __composition_keywords__
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "DistService", "__version__", "__capability_id__", "__capability_name__", "__capability_code__", "__apg_dependencies__", "__composition_keywords__"]
