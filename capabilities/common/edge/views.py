"""Dependency-light view models for the APG EDGE capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import EdgeService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return EDGE UI route contracts."""
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: EdgeService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return dashboard data for node, workload, deployment, and sync posture."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"title": "Edge Operations",
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"recent_audit_events": service.list_audit_events(tenant_id)[-10:],
	}


def node_manager_model(service: EdgeService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return node and fleet state for the EDGE node manager."""
	return {
		"tenant_id": tenant_id,
		"nodes": service.list_nodes(tenant_id),
		"fleets": service.list_fleets(tenant_id),
		"route": "/edge/nodes",
		"permissions": ["edge:view", "edge:manage_nodes"],
	}


def workload_console_model(service: EdgeService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return workload and deployment state for the EDGE workload console."""
	return {
		"tenant_id": tenant_id,
		"workloads": service.list_workloads(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"route": "/edge/workloads",
		"permissions": ["edge:view", "edge:deploy_workloads"],
	}


def sync_monitor_model(service: EdgeService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return synchronization sessions needing replay, conflict handling, or review."""
	sessions = service.list_sync_sessions(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sync_sessions": sessions,
		"review_required": [item for item in sessions if item["review_required"]],
		"conflicts": [item for item in sessions if item["conflicts"]],
		"route": "/edge/sync",
		"permissions": ["edge:view", "edge:sync"],
	}


def edge_agent_model(service: EdgeService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return edge-agent state and governance route metadata."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"edge_agents": service.list_edge_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["edge_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["edge_agents"]["allowed_roles"],
		"route": "/edge/agents",
		"permissions": ["edge:view", "edge:govern"],
	}
