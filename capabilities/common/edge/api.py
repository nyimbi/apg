"""API helpers for the APG EDGE capability."""

from __future__ import annotations

from typing import Any

from .service import EdgeService


SERVICE = EdgeService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"node_count": summary["node_count"],
		"workload_count": summary["workload_count"],
		"deployment_count": summary["deployment_count"],
		"sync_session_count": len(SERVICE.list_sync_sessions(tenant_id)),
	}


def register_node(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_node(
		node_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		node_type=str(payload.get("node_type") or "compute"),
		location=dict(payload.get("location") or {}),
		location_policy=str(payload.get("location_policy") or "default"),
		attested=bool(payload.get("attested")),
		health_status=str(payload.get("health_status") or "healthy"),
		secure_transport=bool(payload.get("secure_transport", True)),
		capacity=dict(payload.get("capacity") or {}),
		capabilities=list(payload.get("capabilities") or []),
	)


def create_fleet(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_fleet(
		fleet_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload["owner"]),
		policy_version=str(payload.get("policy_version") or "v1"),
		node_ids=list(payload.get("node_ids") or []),
	)


def register_workload(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_workload(
		workload_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		version=str(payload.get("version") or "1.0.0"),
		owner=str(payload["owner"]),
		artifact_payload=payload.get("artifact") or {},
		artifact_signed=bool(payload.get("artifact_signed")),
		deployment_policy=str(payload.get("deployment_policy") or ""),
		resource_quota=dict(payload.get("resource_quota") or {}),
		offline_mode_enabled=bool(payload.get("offline_mode_enabled", True)),
	)


def deploy_workload(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_workload(
		deployment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		workload_id=str(payload["workload_id"]),
		node_id=str(payload["node_id"]),
		deployed_by=str(payload["deployed_by"]),
		runtime_mode=str(payload.get("runtime_mode") or "online"),
	)


def sync_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.sync_state(
		sync_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		node_id=str(payload["node_id"]),
		workload_id=str(payload["workload_id"]),
		conflict_policy=str(payload.get("conflict_policy") or ""),
		cache_policy=str(payload.get("cache_policy") or ""),
		offline_hours=int(payload.get("offline_hours") or 0),
		secure_transport=bool(payload.get("secure_transport", True)),
		event_count=int(payload.get("event_count") or 0),
		conflicts=list(payload.get("conflicts") or []),
		reviewed_by=payload.get("reviewed_by"),
	)


def review_offline_window(sync_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
	return SERVICE.review_offline_window(sync_id, tenant_id, reviewer)


def edge_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"dashboard": SERVICE.dashboard_summary(tenant_id),
		"nodes": SERVICE.list_nodes(tenant_id),
		"fleets": SERVICE.list_fleets(tenant_id),
		"workloads": SERVICE.list_workloads(tenant_id),
		"deployments": SERVICE.list_deployments(tenant_id),
		"sync_sessions": SERVICE.list_sync_sessions(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}


# Compatibility helpers for older generated package probes.
def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "healthy"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)
