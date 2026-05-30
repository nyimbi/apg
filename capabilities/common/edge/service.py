"""Tenant-aware EDGE runtime service."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	SUPPORTED_EDGE_AGENT_ROLES,
	SUPPORTED_EDGE_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .edge_engine import artifact_digest, capacity_fits, resource_pressure, stable_digest, sync_status
from .models import EdgeAgent, EdgeAuditEvent, EdgeDeployment, EdgeFleet, EdgeNode, EdgeSyncSession, EdgeWorkload, utc_now


class EdgeService:
	"""Dependency-light edge node, workload, deployment, and sync service."""

	def __init__(self) -> None:
		self._nodes: dict[str, EdgeNode] = {}
		self._fleets: dict[str, EdgeFleet] = {}
		self._workloads: dict[str, EdgeWorkload] = {}
		self._deployments: dict[str, EdgeDeployment] = {}
		self._sync_sessions: dict[str, EdgeSyncSession] = {}
		self._agents: dict[str, EdgeAgent] = {}
		self._audit_events: list[EdgeAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_node(
		self,
		node_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		node_type: str,
		location: dict[str, Any],
		location_policy: str,
		attested: bool,
		health_status: str = "healthy",
		secure_transport: bool = True,
		capacity: dict[str, float] | None = None,
		capabilities: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_node",
			"node_owner_present": bool(owner),
			"node_attested": attested,
			"location_policy_present": bool(location_policy),
			"edge_connection": True,
			"secure_transport": secure_transport,
		})
		self._raise_if_denied(result)
		if DEFAULT_CONFIGURATION["nodes"]["node_owner_required"] and not owner:
			raise PermissionError("node_owner_required")
		if DEFAULT_CONFIGURATION["nodes"]["location_policy_required"] and not location_policy:
			raise PermissionError("location_policy_required")
		if DEFAULT_CONFIGURATION["nodes"]["health_check_required"] and not health_status:
			raise PermissionError("health_check_required")
		node = EdgeNode(
			id=node_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			node_type=node_type,
			location=dict(location),
			location_policy=location_policy,
			attested=attested,
			health_status=health_status,
			secure_transport=secure_transport,
			capacity=dict(capacity or {"cpu": 4, "memory": 8192, "storage": 128}),
			capabilities=list(capabilities or []),
		)
		self._nodes[node_id] = node
		self._record_audit(tenant_id, "node_registered", node_id, owner, node.to_dict())
		return node.to_dict()

	def create_fleet(
		self,
		fleet_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		policy_version: str,
		node_ids: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			result = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "create_fleet",
				"fleet_owner_present": False,
				"policy_version_present": bool(policy_version),
			})
			self._raise_if_denied(result)
		if not policy_version:
			result = self.evaluate({
				"tenant_context_present": bool(tenant_id),
				"operation": "create_fleet",
				"fleet_owner_present": bool(owner),
				"policy_version_present": False,
			})
			self._raise_if_denied(result)
		fleet = EdgeFleet(
			id=fleet_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			policy_version=policy_version,
		)
		self._fleets[fleet_id] = fleet
		for node_id in node_ids or []:
			self.attach_node_to_fleet(node_id, fleet_id, tenant_id)
		self._record_audit(tenant_id, "fleet_created", fleet_id, owner, fleet.to_dict())
		return fleet.to_dict()

	def attach_node_to_fleet(self, node_id: str, fleet_id: str, tenant_id: str) -> dict[str, Any]:
		node = self._require_node(node_id, tenant_id)
		fleet = self._require_fleet(fleet_id, tenant_id)
		if node_id not in fleet.node_ids:
			fleet.node_ids.append(node_id)
			fleet.updated_at = utc_now()
		node.fleet_id = fleet_id
		node.updated_at = utc_now()
		self._record_audit(tenant_id, "node_attached_to_fleet", node_id, fleet.owner, {"fleet_id": fleet_id})
		return fleet.to_dict()

	def register_workload(
		self,
		workload_id: str,
		tenant_id: str,
		name: str,
		version: str,
		owner: str,
		artifact_payload: dict[str, Any] | str,
		artifact_signed: bool,
		deployment_policy: str,
		resource_quota: dict[str, float],
		offline_mode_enabled: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_workload",
			"workload_owner_present": bool(owner),
			"artifact_signed": artifact_signed,
			"resource_quota_present": bool(resource_quota),
		})
		self._raise_if_denied(result)
		if DEFAULT_CONFIGURATION["workloads"]["deployment_policy_required"] and not deployment_policy:
			raise PermissionError("deployment_policy_required")
		workload = EdgeWorkload(
			id=workload_id,
			tenant_id=tenant_id,
			name=name,
			version=version,
			owner=owner,
			artifact_digest=artifact_digest(name, version, artifact_payload),
			artifact_signed=artifact_signed,
			deployment_policy=deployment_policy,
			resource_quota=dict(resource_quota),
			offline_mode_enabled=offline_mode_enabled,
		)
		self._workloads[workload_id] = workload
		self._record_audit(tenant_id, "workload_registered", workload_id, owner, workload.to_dict())
		return workload.to_dict()

	def deploy_workload(
		self,
		deployment_id: str,
		tenant_id: str,
		workload_id: str,
		node_id: str,
		deployed_by: str,
		runtime_mode: str = "online",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		workload = self._require_workload(workload_id, tenant_id)
		node = self._require_node(node_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_workload",
			"artifact_signed": workload.artifact_signed,
			"edge_connection": True,
			"secure_transport": node.secure_transport,
		})
		self._raise_if_denied(result)
		if node.health_status != "healthy":
			raise PermissionError("edge_node_not_healthy")
		if not capacity_fits(node.capacity, workload.resource_quota, node.current_load):
			raise PermissionError("resource_quota_exceeds_node_capacity")
		for resource, requested in workload.resource_quota.items():
			node.current_load[resource] = float(node.current_load.get(resource, 0)) + float(requested)
		node.updated_at = utc_now()
		deployment = EdgeDeployment(
			id=deployment_id,
			tenant_id=tenant_id,
			workload_id=workload_id,
			node_id=node_id,
			deployed_by=deployed_by,
			runtime_mode=runtime_mode,
			resource_reservation=dict(workload.resource_quota),
		)
		self._deployments[deployment_id] = deployment
		self._record_audit(tenant_id, "workload_deployed", deployment_id, deployed_by, deployment.to_dict())
		return deployment.to_dict()

	def sync_state(
		self,
		sync_id: str,
		tenant_id: str,
		node_id: str,
		workload_id: str,
		conflict_policy: str,
		cache_policy: str,
		offline_hours: int = 0,
		secure_transport: bool = True,
		event_count: int = 0,
		conflicts: list[str] | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_node(node_id, tenant_id)
		workload = self._require_workload(workload_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "sync_state",
			"conflict_policy_attached": bool(conflict_policy),
			"cache_policy_attached": bool(cache_policy),
			"edge_connection": True,
			"secure_transport": secure_transport,
			"offline_hours": offline_hours,
			"offline_review_recorded": bool(reviewed_by),
		})
		self._raise_if_denied(result)
		if not workload.offline_mode_enabled and offline_hours > 0:
			raise PermissionError("workload_offline_mode_disabled")
		review_required = result["decision"] == "require_review"
		session = EdgeSyncSession(
			id=sync_id,
			tenant_id=tenant_id,
			node_id=node_id,
			workload_id=workload_id,
			conflict_policy=conflict_policy,
			cache_policy=cache_policy,
			offline_hours=offline_hours,
			secure_transport=secure_transport,
			event_count=event_count,
			conflicts=list(conflicts or []),
			review_required=review_required,
			reviewed_by=reviewed_by,
			status=sync_status(offline_hours, list(conflicts or []), review_required),
		)
		self._sync_sessions[sync_id] = session
		self._record_audit(tenant_id, "sync_completed", sync_id, reviewed_by or "edge-sync", session.to_dict())
		return session.to_dict()

	def review_offline_window(self, sync_id: str, tenant_id: str, reviewer: str) -> dict[str, Any]:
		session = self._require_sync_session(sync_id, tenant_id)
		session.review_required = False
		session.reviewed_by = reviewer
		session.status = sync_status(session.offline_hours, session.conflicts, False)
		self._record_audit(tenant_id, "offline_window_reviewed", sync_id, reviewer, session.to_dict())
		return session.to_dict()

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		nodes = self.list_nodes(tenant_id)
		deployments = self.list_deployments(tenant_id)
		sync_sessions = self.list_sync_sessions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"node_count": len(nodes),
			"fleet_count": len(self.list_fleets(tenant_id)),
			"workload_count": len(self.list_workloads(tenant_id)),
			"deployment_count": len(deployments),
			"healthy_node_count": sum(1 for node in nodes if node["health_status"] == "healthy"),
			"review_required_sync_count": sum(1 for item in sync_sessions if item["review_required"]),
			"conflict_pending_sync_count": sum(1 for item in sync_sessions if item["status"] == "conflict_pending"),
			"edge_agent_count": len(self.list_edge_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def node_pressure(self, node_id: str, tenant_id: str) -> dict[str, Any]:
		node = self._require_node(node_id, tenant_id)
		return {"node_id": node_id, "tenant_id": tenant_id, "pressure": resource_pressure(node.capacity, node.current_load)}

	def list_nodes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._nodes, tenant_id)

	def list_fleets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._fleets, tenant_id)

	def list_workloads(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._workloads, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._deployments, tenant_id)

	def list_sync_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._sync_sessions, tenant_id)

	def register_edge_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"edge_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_EDGE_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_EDGE_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		edge_agent = EdgeAgent(
			id=agent_id or f"edge-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[edge_agent.id] = edge_agent
		self._record_audit(tenant_id, "edge_agent_registered", edge_agent.id, name, edge_agent.to_dict())
		return edge_agent.to_dict()

	def list_edge_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._agents, tenant_id)

	def validate_batch_edge_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_edge_mutation",
			"event_stream": event_stream,
		})

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.to_dict() for event in events]

	# Backward-compatible aliases for older generated helpers.
	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = metadata or {}
		return self.register_node(
			node_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "system"),
			node_type=str(metadata.get("node_type") or "compute"),
			location=dict(metadata.get("location") or {"site": "unknown"}),
			location_policy=str(metadata.get("location_policy") or "default"),
			attested=bool(metadata.get("attested", True)),
			health_status=status,
			secure_transport=bool(metadata.get("secure_transport", True)),
			capacity=dict(metadata.get("capacity") or {"cpu": 4, "memory": 8192, "storage": 128}),
			capabilities=list(metadata.get("capabilities") or []),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_nodes(tenant_id)

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_node(self, node_id: str, tenant_id: str) -> EdgeNode:
		node = self._nodes.get(node_id)
		if node is None or node.tenant_id != tenant_id:
			raise KeyError(f"unknown_edge_node:{node_id}")
		return node

	def _require_fleet(self, fleet_id: str, tenant_id: str) -> EdgeFleet:
		fleet = self._fleets.get(fleet_id)
		if fleet is None or fleet.tenant_id != tenant_id:
			raise KeyError(f"unknown_edge_fleet:{fleet_id}")
		return fleet

	def _require_workload(self, workload_id: str, tenant_id: str) -> EdgeWorkload:
		workload = self._workloads.get(workload_id)
		if workload is None or workload.tenant_id != tenant_id:
			raise KeyError(f"unknown_edge_workload:{workload_id}")
		return workload

	def _require_sync_session(self, sync_id: str, tenant_id: str) -> EdgeSyncSession:
		session = self._sync_sessions.get(sync_id)
		if session is None or session.tenant_id != tenant_id:
			raise KeyError(f"unknown_edge_sync_session:{sync_id}")
		return session

	def _record_audit(self, tenant_id: str, action: str, resource_id: str, actor: str, metadata: dict[str, Any]) -> None:
		payload = {
			"tenant_id": tenant_id,
			"action": action,
			"resource_id": resource_id,
			"actor": actor,
			"metadata": metadata,
		}
		self._audit_events.append(EdgeAuditEvent(
			id=f"aud-{len(self._audit_events) + 1:06d}",
			tenant_id=tenant_id,
			action=action,
			resource_id=resource_id,
			actor=actor,
			digest=stable_digest(payload),
			metadata=dict(metadata),
		))

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "edge_policy_blocked") for action in result["actions"]))

	def _list_for_tenant(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")
