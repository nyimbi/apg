"""Tenant-aware EDGE runtime service — expanded to 42+ methods."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from datetime import datetime, timezone
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
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EdgeComputingService:
	"""
	Edge node registry, workload deployment, state sync, auto-scaling,
	failover, bandwidth optimisation, and analytics service.

	Adapter/store pattern — no external dependencies.
	"""

	def __init__(self) -> None:
		self._nodes: dict[str, EdgeNode] = {}
		self._fleets: dict[str, EdgeFleet] = {}
		self._workloads: dict[str, EdgeWorkload] = {}
		self._deployments: dict[str, EdgeDeployment] = {}
		self._sync_sessions: dict[str, EdgeSyncSession] = {}
		self._agents: dict[str, EdgeAgent] = {}
		self._audit_events: list[EdgeAuditEvent] = []
		self._scaling_events: list[dict[str, Any]] = []
		self._failover_events: list[dict[str, Any]] = []
		self._bandwidth_policies: dict[str, dict[str, Any]] = {}
		self._offload_requests: dict[str, dict[str, Any]] = {}
		# new stores
		self._inference_results: dict[str, dict[str, Any]] = {}
		self._edge_caches: dict[str, dict[str, Any]] = {}
		self._firmware_updates: dict[str, dict[str, Any]] = {}
		self._latency_samples: dict[str, list[dict[str, Any]]] = {}
		self._federated_aggregations: dict[str, dict[str, Any]] = {}
		self._security_events: dict[str, list[dict[str, Any]]] = {}
		self._sovereignty_checks: dict[str, dict[str, Any]] = {}
		self._workload_schedules: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Node management
	# ------------------------------------------------------------------

	def register_edge_node(
		self,
		node_id: str,
		location: dict[str, Any],
		capabilities: list[str],
		network_type: str,
		tenant_id: str = "default",
		name: str = "",
		owner: str = "system",
		node_type: str = "compute",
		location_policy: str = "default",
		attested: bool = True,
		health_status: str = "healthy",
		secure_transport: bool = True,
		capacity: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Register an edge node with location, capability, and network metadata."""
		self._require_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "register_node", "node_owner_present": bool(owner), "node_attested": attested, "location_policy_present": bool(location_policy), "edge_connection": True, "secure_transport": secure_transport})
		self._raise_if_denied(result)
		if DEFAULT_CONFIGURATION["nodes"]["node_owner_required"] and not owner:
			raise PermissionError("node_owner_required")
		if DEFAULT_CONFIGURATION["nodes"]["location_policy_required"] and not location_policy:
			raise PermissionError("location_policy_required")
		full_location = {**location, "network_type": network_type}
		node = EdgeNode(id=node_id, tenant_id=tenant_id, name=name or node_id, owner=owner, node_type=node_type, location=full_location, location_policy=location_policy, attested=attested, health_status=health_status, secure_transport=secure_transport, capacity=dict(capacity or {"cpu": 4.0, "memory": 8192.0, "storage": 128.0}), capabilities=list(capabilities))
		self._nodes[node_id] = node
		self._record_audit(tenant_id, "node_registered", node_id, owner, node.to_dict())
		return node.to_dict()

	def node_health_monitor(self, node_id: str, tenant_id: str = "default", probe_checks: list[str] | None = None) -> dict[str, Any]:
		"""Monitor and report health of an edge node via synthetic probes."""
		node = self._require_node(node_id, tenant_id)
		probes = probe_checks or ["connectivity", "resource_utilisation", "latency", "security"]
		probe_results: dict[str, str] = {}
		pressure = resource_pressure(node.capacity, node.current_load)
		for probe in probes:
			if probe == "resource_utilisation":
				probe_results[probe] = "warning" if pressure.get("overall", 0) > 0.8 else "pass"
			elif probe == "security":
				probe_results[probe] = "pass" if node.secure_transport and node.attested else "fail"
			elif probe == "connectivity":
				probe_results[probe] = "fail" if node.health_status == "offline" else "pass"
			else:
				probe_results[probe] = "pass" if node.health_status == "healthy" else "warning"
		overall = "healthy" if all(v == "pass" for v in probe_results.values()) else "degraded" if any(v == "warning" for v in probe_results.values()) else "unhealthy"
		if overall == "unhealthy" and node.health_status == "healthy":
			node.health_status = "degraded"
			node.updated_at = utc_now()
		record = {"node_id": node_id, "tenant_id": tenant_id, "health_status": node.health_status, "probe_results": probe_results, "overall": overall, "resource_pressure": pressure, "checked_at": _ts()}
		self._record_audit(tenant_id, "node_health_monitored", node_id, "system", record)
		return record

	# ------------------------------------------------------------------
	# Workload deployment
	# ------------------------------------------------------------------

	def deploy_workload(
		self,
		workload_id: str,
		target_nodes: list[str],
		constraints: dict[str, Any],
		tenant_id: str = "default",
		name: str = "",
		version: str = "1.0.0",
		owner: str = "system",
		artifact_payload: dict[str, Any] | None = None,
		artifact_signed: bool = True,
		deployment_policy: str = "default",
		resource_quota: dict[str, float] | None = None,
		offline_mode_enabled: bool = True,
		deployed_by: str = "system",
	) -> dict[str, Any]:
		"""Register a workload and deploy it to one or more target nodes."""
		self._require_tenant(tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "deploy_workload", "workload_owner_present": bool(owner), "artifact_signed": artifact_signed, "resource_quota_present": bool(resource_quota)})
		self._raise_if_denied(result)
		if DEFAULT_CONFIGURATION["workloads"]["deployment_policy_required"] and not deployment_policy:
			raise PermissionError("deployment_policy_required")
		quota = resource_quota or {"cpu": 1.0, "memory": 512.0}
		payload = artifact_payload or {}
		workload = EdgeWorkload(id=workload_id, tenant_id=tenant_id, name=name or workload_id, version=version, owner=owner, artifact_digest=artifact_digest(name or workload_id, version, payload), artifact_signed=artifact_signed, deployment_policy=deployment_policy, resource_quota=quota, offline_mode_enabled=offline_mode_enabled)
		self._workloads[workload_id] = workload
		required_caps = set(constraints.get("required_capabilities", []))
		deployment_results: list[dict[str, Any]] = []
		for node_id in target_nodes:
			try:
				node = self._require_node(node_id, tenant_id)
			except KeyError:
				deployment_results.append({"node_id": node_id, "status": "error", "reason": "node_not_found"})
				continue
			if required_caps and not required_caps.issubset(set(node.capabilities)):
				deployment_results.append({"node_id": node_id, "status": "skipped", "reason": "capability_mismatch"})
				continue
			if node.health_status != "healthy":
				deployment_results.append({"node_id": node_id, "status": "skipped", "reason": "node_not_healthy"})
				continue
			if not capacity_fits(node.capacity, quota, node.current_load):
				deployment_results.append({"node_id": node_id, "status": "skipped", "reason": "insufficient_capacity"})
				continue
			dep_id = self._gen_id("dep", workload_id, node_id)
			deployment = EdgeDeployment(id=dep_id, tenant_id=tenant_id, workload_id=workload_id, node_id=node_id, deployed_by=deployed_by, runtime_mode="online", resource_reservation=dict(quota))
			self._deployments[dep_id] = deployment
			for resource, amount in quota.items():
				node.current_load[resource] = float(node.current_load.get(resource, 0)) + float(amount)
			node.updated_at = utc_now()
			deployment_results.append({"node_id": node_id, "deployment_id": dep_id, "status": "deployed"})
		self._record_audit(tenant_id, "workload_deployed", workload_id, deployed_by, {"target_nodes": target_nodes, "deployments": len([d for d in deployment_results if d["status"] == "deployed"])})
		return {**workload.to_dict(), "deployment_results": deployment_results}

	def workload_status(self, workload_id: str, tenant_id: str = "default") -> dict[str, Any]:
		workload = self._require_workload(workload_id, tenant_id)
		deployments = [d.to_dict() for d in self._deployments.values() if d.tenant_id == tenant_id and d.workload_id == workload_id]
		active = [d for d in deployments if d.get("status") not in {"terminated", "failed"}]
		return {**workload.to_dict(), "deployment_count": len(deployments), "active_deployment_count": len(active), "deployments": deployments}

	# ------------------------------------------------------------------
	# Computation offload
	# ------------------------------------------------------------------

	def offload_computation(self, request_id: str, payload: dict[str, Any], latency_requirement_ms: int, tenant_id: str = "default", preferred_node_id: str | None = None, fallback_to_cloud: bool = True) -> dict[str, Any]:
		"""Offload a computation request to the lowest-latency available node."""
		self._require_tenant(tenant_id)
		tenant_nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.health_status == "healthy"]
		def _score(node: EdgeNode) -> float:
			p = resource_pressure(node.capacity, node.current_load)
			return p.get("overall", 1.0)
		if preferred_node_id:
			candidates = [n for n in tenant_nodes if n.id == preferred_node_id] + [n for n in tenant_nodes if n.id != preferred_node_id]
		else:
			candidates = sorted(tenant_nodes, key=_score)
		selected_node: EdgeNode | None = None
		for candidate in candidates:
			if max(1, int(10 + _score(candidate) * 50)) <= latency_requirement_ms:
				selected_node = candidate
				break
		routing = "cloud" if selected_node is None and fallback_to_cloud else ("edge" if selected_node else "rejected")
		if routing == "rejected":
			raise PermissionError("no_node_meets_latency_requirement")
		record = {"request_id": request_id, "tenant_id": tenant_id, "payload_size_bytes": len(str(payload)), "latency_requirement_ms": latency_requirement_ms, "selected_node_id": selected_node.id if selected_node else None, "routing": routing, "estimated_latency_ms": max(1, int(10 + _score(selected_node) * 50)) if selected_node else None, "offloaded_at": _ts()}
		self._offload_requests[request_id] = record
		self._record_audit(tenant_id, "computation_offloaded", request_id, "system", record)
		return record

	# ------------------------------------------------------------------
	# Cloud sync
	# ------------------------------------------------------------------

	def edge_to_cloud_sync(self, node_id: str, data_type: str, data: dict[str, Any], tenant_id: str = "default", conflict_policy: str = "last_write_wins", cache_policy: str = "write_through", reviewed_by: str | None = None) -> dict[str, Any]:
		"""Initiate edge-to-cloud data sync for a node."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		workload_id = None
		for dep in self._deployments.values():
			if dep.tenant_id == tenant_id and dep.node_id == node_id:
				workload_id = dep.workload_id
				break
		if not workload_id:
			workload_id = f"implicit-{node_id}"
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "sync_state", "conflict_policy_attached": bool(conflict_policy), "cache_policy_attached": bool(cache_policy), "edge_connection": True, "secure_transport": node.secure_transport, "offline_hours": 0, "offline_review_recorded": True})
		self._raise_if_denied(result)
		sync_id = self._gen_id("sync", node_id, data_type, str(len(self._sync_sessions)))
		session = EdgeSyncSession(id=sync_id, tenant_id=tenant_id, node_id=node_id, workload_id=workload_id, conflict_policy=conflict_policy, cache_policy=cache_policy, offline_hours=0, secure_transport=node.secure_transport, event_count=len(data), conflicts=[], review_required=False, reviewed_by=reviewed_by, status=sync_status(0, [], False))
		self._sync_sessions[sync_id] = session
		self._record_audit(tenant_id, "edge_to_cloud_synced", sync_id, reviewed_by or "edge-sync", {"node_id": node_id, "data_type": data_type, "event_count": len(data)})
		return {**session.to_dict(), "data_type": data_type, "records_synced": len(data)}

	# ------------------------------------------------------------------
	# Auto-scaling
	# ------------------------------------------------------------------

	def auto_scaling(self, workload_id: str, metric: str, threshold: float, tenant_id: str = "default", scale_direction: str = "auto", cooldown_seconds: int = 60) -> dict[str, Any]:
		"""Evaluate and record an auto-scaling decision for a workload."""
		workload = self._require_workload(workload_id, tenant_id)
		active_deployments = [d for d in self._deployments.values() if d.tenant_id == tenant_id and d.workload_id == workload_id]
		metric_values: list[float] = []
		for dep in active_deployments:
			node = self._nodes.get(dep.node_id)
			if node:
				cap = node.capacity.get(metric, 1.0)
				load = node.current_load.get(metric, 0.0)
				metric_values.append(load / cap if cap > 0 else 0.0)
		avg_metric = sum(metric_values) / len(metric_values) if metric_values else 0.0
		if scale_direction == "auto":
			decision = "scale_out" if avg_metric > threshold else ("scale_in" if avg_metric < threshold * 0.5 else "no_op")
		else:
			decision = scale_direction
		target_replicas = (len(active_deployments) + 1 if decision == "scale_out" else max(1, len(active_deployments) - 1) if decision == "scale_in" else len(active_deployments))
		record = {"workload_id": workload_id, "tenant_id": tenant_id, "metric": metric, "threshold": threshold, "average_metric_value": round(avg_metric, 4), "current_replicas": len(active_deployments), "target_replicas": target_replicas, "decision": decision, "cooldown_seconds": cooldown_seconds, "evaluated_at": _ts()}
		self._scaling_events.append(record)
		self._record_audit(tenant_id, "auto_scaling_evaluated", workload_id, "autoscaler", record)
		return record

	# ------------------------------------------------------------------
	# Failover
	# ------------------------------------------------------------------

	def failover(self, node_id: str, failover_target: str, tenant_id: str = "default", triggered_by: str = "system", reason: str = "node_failure") -> dict[str, Any]:
		"""Trigger failover from a failed/degraded node to a target node."""
		source_node = self._require_node(node_id, tenant_id)
		target_node = self._require_node(failover_target, tenant_id)
		if target_node.health_status != "healthy":
			raise PermissionError("failover_target_not_healthy")
		source_node.health_status = "failed"
		source_node.updated_at = utc_now()
		migrated: list[str] = []
		for dep in list(self._deployments.values()):
			if dep.tenant_id == tenant_id and dep.node_id == node_id:
				workload = self._workloads.get(dep.workload_id)
				if not workload:
					continue
				if not capacity_fits(target_node.capacity, workload.resource_quota, target_node.current_load):
					continue
				new_dep_id = self._gen_id("dep", dep.workload_id, failover_target, "failover")
				new_dep = EdgeDeployment(id=new_dep_id, tenant_id=tenant_id, workload_id=dep.workload_id, node_id=failover_target, deployed_by=triggered_by, runtime_mode=dep.runtime_mode, resource_reservation=dict(workload.resource_quota))
				self._deployments[new_dep_id] = new_dep
				for resource, amount in workload.resource_quota.items():
					target_node.current_load[resource] = float(target_node.current_load.get(resource, 0)) + float(amount)
				target_node.updated_at = utc_now()
				migrated.append(dep.workload_id)
		record = {"source_node_id": node_id, "failover_target_id": failover_target, "tenant_id": tenant_id, "reason": reason, "triggered_by": triggered_by, "workloads_migrated": migrated, "migration_count": len(migrated), "failover_at": _ts()}
		self._failover_events.append(record)
		self._record_audit(tenant_id, "failover_executed", node_id, triggered_by, record)
		return record

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	def edge_analytics(self, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Aggregate edge computing analytics for a tenant."""
		nodes = self.list_nodes(tenant_id)
		workloads = self.list_workloads(tenant_id)
		deployments = self.list_deployments(tenant_id)
		syncs = self.list_sync_sessions(tenant_id)
		period_scaling = [e for e in self._scaling_events if e["tenant_id"] == tenant_id]
		period_failovers = [e for e in self._failover_events if e["tenant_id"] == tenant_id]
		healthy_nodes = [n for n in nodes if n["health_status"] == "healthy"]
		avg_pressure = 0.0
		if nodes:
			pressures = [resource_pressure(self._nodes[n["id"]].capacity, self._nodes[n["id"]].current_load).get("overall", 0.0) for n in nodes if n["id"] in self._nodes]
			avg_pressure = round(sum(pressures) / len(pressures), 4) if pressures else 0.0
		scale_out = sum(1 for e in period_scaling if e["decision"] == "scale_out")
		scale_in = sum(1 for e in period_scaling if e["decision"] == "scale_in")
		return {
			"tenant_id": tenant_id,
			"period": period,
			"node_count": len(nodes),
			"healthy_node_count": len(healthy_nodes),
			"workload_count": len(workloads),
			"deployment_count": len(deployments),
			"sync_session_count": len(syncs),
			"conflict_sync_count": sum(1 for s in syncs if s.get("status") == "conflict_pending"),
			"scaling_event_count": len(period_scaling),
			"scale_out_count": scale_out,
			"scale_in_count": scale_in,
			"failover_count": len(period_failovers),
			"average_node_pressure": avg_pressure,
			"offload_request_count": sum(1 for r in self._offload_requests.values() if r["tenant_id"] == tenant_id),
			"inference_count": len([r for r in self._inference_results.values() if r["tenant_id"] == tenant_id]),
			"firmware_update_count": len([f for f in self._firmware_updates.values() if f["tenant_id"] == tenant_id]),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# Bandwidth optimisation
	# ------------------------------------------------------------------

	def bandwidth_optimisation(self, node_id: str, policy: dict[str, Any], tenant_id: str = "default", applied_by: str = "system") -> dict[str, Any]:
		"""Apply a bandwidth optimisation policy to an edge node."""
		node = self._require_node(node_id, tenant_id)
		validated_policy = {"compression": bool(policy.get("compression", False)), "deduplication": bool(policy.get("deduplication", False)), "priority_queuing": bool(policy.get("priority_queuing", True)), "max_mbps": float(policy.get("max_mbps", 100.0)), "qos_class": str(policy.get("qos_class", "best_effort"))}
		savings_pct = 0.0
		if validated_policy["compression"]:
			savings_pct += 20.0
		if validated_policy["deduplication"]:
			savings_pct += 15.0
		if validated_policy["priority_queuing"]:
			savings_pct += 5.0
		key = f"{tenant_id}:{node_id}"
		self._bandwidth_policies[key] = {"node_id": node_id, "tenant_id": tenant_id, "policy": validated_policy, "estimated_savings_pct": round(min(savings_pct, 60.0), 2), "applied_by": applied_by, "applied_at": _ts()}
		self._record_audit(tenant_id, "bandwidth_policy_applied", node_id, applied_by, self._bandwidth_policies[key])
		return self._bandwidth_policies[key]

	# ------------------------------------------------------------------
	# Fleet management
	# ------------------------------------------------------------------

	def create_fleet(self, fleet_id: str, tenant_id: str, name: str, owner: str, policy_version: str, node_ids: list[str] | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			result = self.evaluate({"tenant_context_present": bool(tenant_id), "fleet_owner_present": False, "policy_version_present": bool(policy_version)})
			self._raise_if_denied(result)
		if not policy_version:
			result = self.evaluate({"tenant_context_present": bool(tenant_id), "fleet_owner_present": bool(owner), "policy_version_present": False})
			self._raise_if_denied(result)
		fleet = EdgeFleet(id=fleet_id, tenant_id=tenant_id, name=name, owner=owner, policy_version=policy_version)
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

	# ------------------------------------------------------------------
	# Sync / review
	# ------------------------------------------------------------------

	def sync_state(self, sync_id: str, tenant_id: str, node_id: str, workload_id: str, conflict_policy: str, cache_policy: str, offline_hours: int = 0, secure_transport: bool = True, event_count: int = 0, conflicts: list[str] | None = None, reviewed_by: str | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_node(node_id, tenant_id)
		workload = self._require_workload(workload_id, tenant_id)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "sync_state", "conflict_policy_attached": bool(conflict_policy), "cache_policy_attached": bool(cache_policy), "edge_connection": True, "secure_transport": secure_transport, "offline_hours": offline_hours, "offline_review_recorded": bool(reviewed_by)})
		self._raise_if_denied(result)
		if not workload.offline_mode_enabled and offline_hours > 0:
			raise PermissionError("workload_offline_mode_disabled")
		review_required = result["decision"] == "require_review"
		session = EdgeSyncSession(id=sync_id, tenant_id=tenant_id, node_id=node_id, workload_id=workload_id, conflict_policy=conflict_policy, cache_policy=cache_policy, offline_hours=offline_hours, secure_transport=secure_transport, event_count=event_count, conflicts=list(conflicts or []), review_required=review_required, reviewed_by=reviewed_by, status=sync_status(offline_hours, list(conflicts or []), review_required))
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

	# ------------------------------------------------------------------
	# NEW: workload_schedule
	# ------------------------------------------------------------------

	def workload_schedule(
		self,
		tenant_id: str,
		workload_id: str,
		schedule_id: str,
		cron_expression: str,
		target_node_ids: list[str],
		constraints: dict[str, Any] | None = None,
		scheduled_by: str = "system",
	) -> dict[str, Any]:
		"""Schedule a workload to run on a cron expression across target nodes."""
		self._require_tenant(tenant_id)
		self._require_workload(workload_id, tenant_id)
		if not cron_expression:
			raise ValueError("cron_expression required")
		if not target_node_ids:
			raise ValueError("target_node_ids required")
		key = f"{tenant_id}:{schedule_id}"
		if key in self._workload_schedules:
			raise ValueError("schedule_already_exists")
		record = {"schedule_id": schedule_id, "tenant_id": tenant_id, "workload_id": workload_id, "cron_expression": cron_expression, "target_node_ids": target_node_ids, "constraints": dict(constraints or {}), "status": "active", "scheduled_by": scheduled_by, "created_at": _ts()}
		self._workload_schedules[key] = record
		self._record_audit(tenant_id, "workload_scheduled", workload_id, scheduled_by, record)
		return record

	# ------------------------------------------------------------------
	# NEW: edge_cache
	# ------------------------------------------------------------------

	def edge_cache(
		self,
		tenant_id: str,
		node_id: str,
		cache_key: str,
		data: dict[str, Any],
		ttl_seconds: int = 300,
		cached_by: str = "system",
	) -> dict[str, Any]:
		"""Cache data at the edge node to reduce round-trips to cloud."""
		self._require_tenant(tenant_id)
		self._require_node(node_id, tenant_id)
		if not cache_key:
			raise ValueError("cache_key required")
		if ttl_seconds < 1:
			raise ValueError("ttl_seconds must be positive")
		key = f"{tenant_id}:{node_id}:{cache_key}"
		record = {"node_id": node_id, "tenant_id": tenant_id, "cache_key": cache_key, "data_size_bytes": len(str(data)), "ttl_seconds": ttl_seconds, "status": "cached", "cached_by": cached_by, "cached_at": _ts()}
		self._edge_caches[key] = record
		self._record_audit(tenant_id, "edge_cache_written", node_id, cached_by, record)
		return record

	def edge_cache_invalidate(
		self,
		tenant_id: str,
		node_id: str,
		cache_key: str,
		invalidated_by: str = "system",
	) -> dict[str, Any]:
		"""Invalidate a cached entry at an edge node."""
		self._require_tenant(tenant_id)
		key = f"{tenant_id}:{node_id}:{cache_key}"
		record = self._edge_caches.get(key)
		if record is None:
			raise KeyError(f"cache_entry_not_found:{cache_key}")
		record["status"] = "invalidated"
		record["invalidated_by"] = invalidated_by
		record["invalidated_at"] = _ts()
		self._record_audit(tenant_id, "edge_cache_invalidated", node_id, invalidated_by, record)
		return record

	# ------------------------------------------------------------------
	# NEW: local_inference
	# ------------------------------------------------------------------

	def local_inference(
		self,
		tenant_id: str,
		node_id: str,
		request_id: str,
		model_name: str,
		input_data: dict[str, Any],
		inference_type: str = "classification",
		executed_by: str = "system",
	) -> dict[str, Any]:
		"""Execute an ML inference request locally at the edge node."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		if not model_name:
			raise ValueError("model_name required")
		if not input_data:
			raise ValueError("input_data required")
		# Synthetic inference result — production would call Ollama / ONNX runtime
		input_hash = hashlib.sha256(str(sorted(input_data.items())).encode()).hexdigest()[:16]
		result = {
			"request_id": request_id,
			"tenant_id": tenant_id,
			"node_id": node_id,
			"model_name": model_name,
			"inference_type": inference_type,
			"input_hash": input_hash,
			"output": {"label": "class_A", "confidence": 0.87, "latency_ms": 12},
			"executed_by": executed_by,
			"executed_at": _ts(),
		}
		self._inference_results[f"{tenant_id}:{request_id}"] = result
		self._record_audit(tenant_id, "local_inference_executed", node_id, executed_by, result)
		return result

	# ------------------------------------------------------------------
	# NEW: offline_mode
	# ------------------------------------------------------------------

	def offline_mode(
		self,
		tenant_id: str,
		node_id: str,
		enabled: bool,
		max_offline_hours: int = 72,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Enable or disable offline operation mode for a node's workloads."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		# Update all workloads deployed on this node
		updated_workloads: list[str] = []
		for dep in self._deployments.values():
			if dep.tenant_id == tenant_id and dep.node_id == node_id:
				workload = self._workloads.get(dep.workload_id)
				if workload:
					workload.offline_mode_enabled = enabled
					updated_workloads.append(workload.id)
		record = {"node_id": node_id, "tenant_id": tenant_id, "offline_mode_enabled": enabled, "max_offline_hours": max_offline_hours, "updated_workloads": updated_workloads, "configured_by": actor, "configured_at": _ts()}
		self._record_audit(tenant_id, "offline_mode_configured", node_id, actor, record)
		return record

	# ------------------------------------------------------------------
	# NEW: sync_when_connected
	# ------------------------------------------------------------------

	def sync_when_connected(
		self,
		tenant_id: str,
		node_id: str,
		pending_data: dict[str, Any],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Queue data for synchronisation when the node reconnects to the network."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		if node.health_status not in {"offline", "degraded"}:
			# Node is online — sync immediately
			return self.edge_to_cloud_sync(node_id=node_id, data_type="queued", data=pending_data, tenant_id=tenant_id, reviewed_by=actor)
		# Queue for later
		queue_key = f"{tenant_id}:{node_id}:queued_sync"
		record = {"node_id": node_id, "tenant_id": tenant_id, "queued_records": len(pending_data), "status": "queued_for_sync", "queued_by": actor, "queued_at": _ts()}
		self._edge_caches[queue_key] = record
		self._record_audit(tenant_id, "sync_queued_when_offline", node_id, actor, record)
		return record

	# ------------------------------------------------------------------
	# NEW: bandwidth_optimise
	# ------------------------------------------------------------------

	def bandwidth_optimise(
		self,
		tenant_id: str,
		node_id: str,
		compression: bool = True,
		deduplication: bool = True,
		max_mbps: float = 50.0,
		applied_by: str = "system",
	) -> dict[str, Any]:
		"""Shortcut to apply bandwidth optimisation with common settings."""
		return self.bandwidth_optimisation(
			node_id=node_id,
			policy={"compression": compression, "deduplication": deduplication, "priority_queuing": True, "max_mbps": max_mbps, "qos_class": "premium"},
			tenant_id=tenant_id,
			applied_by=applied_by,
		)

	# ------------------------------------------------------------------
	# NEW: power_aware_compute
	# ------------------------------------------------------------------

	def power_aware_compute(
		self,
		tenant_id: str,
		node_id: str,
		power_budget_watts: float,
		workload_priority: dict[str, int],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Schedule workloads according to a power budget and workload priorities."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		if power_budget_watts <= 0:
			raise ValueError("power_budget_watts must be positive")
		sorted_workloads = sorted(workload_priority.items(), key=lambda x: x[1], reverse=True)
		# Synthetic allocation — production would query node power telemetry
		allocated: list[dict[str, Any]] = []
		remaining_power = power_budget_watts
		for wid, priority in sorted_workloads:
			est_power = 10.0 + priority * 2.0  # synthetic estimate
			if remaining_power >= est_power:
				allocated.append({"workload_id": wid, "priority": priority, "allocated_watts": est_power, "status": "scheduled"})
				remaining_power -= est_power
			else:
				allocated.append({"workload_id": wid, "priority": priority, "allocated_watts": 0, "status": "deferred"})
		record = {"node_id": node_id, "tenant_id": tenant_id, "power_budget_watts": power_budget_watts, "remaining_watts": round(remaining_power, 2), "scheduled_count": len([a for a in allocated if a["status"] == "scheduled"]), "deferred_count": len([a for a in allocated if a["status"] == "deferred"]), "allocation": allocated, "configured_by": actor, "configured_at": _ts()}
		self._record_audit(tenant_id, "power_aware_compute_configured", node_id, actor, record)
		return record

	# ------------------------------------------------------------------
	# NEW: federated_aggregate
	# ------------------------------------------------------------------

	def federated_aggregate(
		self,
		tenant_id: str,
		aggregation_id: str,
		node_ids: list[str],
		aggregation_fn: str,
		metric: str,
		aggregated_by: str = "system",
	) -> dict[str, Any]:
		"""Aggregate a metric across multiple edge nodes without centralising raw data."""
		self._require_tenant(tenant_id)
		if not node_ids:
			raise ValueError("node_ids required")
		valid_fns = {"mean", "sum", "min", "max", "count"}
		if aggregation_fn not in valid_fns:
			raise ValueError(f"aggregation_fn must be one of: {valid_fns}")
		values: list[float] = []
		for nid in node_ids:
			node = self._nodes.get(nid)
			if node and node.tenant_id == tenant_id:
				load = node.current_load.get(metric, 0.0)
				cap = node.capacity.get(metric, 1.0)
				values.append(load / cap if cap > 0 else 0.0)
		if not values:
			raise PermissionError("no_valid_nodes_for_aggregation")
		result_value: float
		if aggregation_fn == "mean":
			result_value = statistics.mean(values)
		elif aggregation_fn == "sum":
			result_value = sum(values)
		elif aggregation_fn == "min":
			result_value = min(values)
		elif aggregation_fn == "max":
			result_value = max(values)
		else:  # count
			result_value = float(len(values))
		record = {"aggregation_id": aggregation_id, "tenant_id": tenant_id, "node_ids": node_ids, "metric": metric, "aggregation_fn": aggregation_fn, "result": round(result_value, 6), "node_count": len(values), "aggregated_by": aggregated_by, "aggregated_at": _ts()}
		self._federated_aggregations[f"{tenant_id}:{aggregation_id}"] = record
		self._record_audit(tenant_id, "federated_aggregate_computed", aggregation_id, aggregated_by, record)
		return record

	# ------------------------------------------------------------------
	# NEW: edge_health
	# ------------------------------------------------------------------

	def edge_health(self, tenant_id: str) -> dict[str, Any]:
		"""Return fleet-wide health summary for all edge nodes in a tenant."""
		self._require_tenant(tenant_id)
		nodes = self.list_nodes(tenant_id)
		healthy = [n for n in nodes if n["health_status"] == "healthy"]
		degraded = [n for n in nodes if n["health_status"] == "degraded"]
		failed = [n for n in nodes if n["health_status"] == "failed"]
		pressures = [resource_pressure(self._nodes[n["id"]].capacity, self._nodes[n["id"]].current_load).get("overall", 0.0) for n in nodes if n["id"] in self._nodes]
		avg_pressure = round(statistics.mean(pressures), 4) if pressures else 0.0
		return {"tenant_id": tenant_id, "total_nodes": len(nodes), "healthy_count": len(healthy), "degraded_count": len(degraded), "failed_count": len(failed), "health_ratio": round(len(healthy) / len(nodes), 4) if nodes else 0.0, "average_pressure": avg_pressure, "checked_at": _ts()}

	# ------------------------------------------------------------------
	# NEW: firmware_update
	# ------------------------------------------------------------------

	def firmware_update(
		self,
		tenant_id: str,
		node_id: str,
		firmware_version: str,
		firmware_hash: str,
		rollback_version: str,
		initiated_by: str = "system",
		staged: bool = True,
	) -> dict[str, Any]:
		"""Initiate a firmware update for an edge node."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		if not firmware_version:
			raise ValueError("firmware_version required")
		if not firmware_hash:
			raise PermissionError("firmware_hash required for integrity verification")
		key = f"{tenant_id}:{node_id}"
		record = {"node_id": node_id, "tenant_id": tenant_id, "firmware_version": firmware_version, "firmware_hash": firmware_hash, "rollback_version": rollback_version, "status": "staged" if staged else "applying", "initiated_by": initiated_by, "initiated_at": _ts()}
		self._firmware_updates[key] = record
		self._record_audit(tenant_id, "firmware_update_initiated", node_id, initiated_by, record)
		return record

	# ------------------------------------------------------------------
	# NEW: locality_routing
	# ------------------------------------------------------------------

	def locality_routing(
		self,
		tenant_id: str,
		request_id: str,
		client_location: dict[str, Any],
		required_capability: str | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Route a request to the nearest edge node satisfying capability requirements."""
		self._require_tenant(tenant_id)
		tenant_nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.health_status == "healthy"]
		if required_capability:
			tenant_nodes = [n for n in tenant_nodes if required_capability in n.capabilities]
		if not tenant_nodes:
			raise PermissionError("no_suitable_nodes_for_locality_routing")
		# Score by pressure; in production would include geo-distance
		def _score(node: EdgeNode) -> float:
			return resource_pressure(node.capacity, node.current_load).get("overall", 1.0)
		selected = sorted(tenant_nodes, key=_score)[0]
		record = {"request_id": request_id, "tenant_id": tenant_id, "client_location": client_location, "required_capability": required_capability, "selected_node_id": selected.id, "selected_node_location": selected.location, "pressure_score": round(_score(selected), 4), "routed_by": actor, "routed_at": _ts()}
		self._record_audit(tenant_id, "locality_routing_resolved", selected.id, actor, record)
		return record

	# ------------------------------------------------------------------
	# NEW: latency_monitor
	# ------------------------------------------------------------------

	def latency_monitor(
		self,
		tenant_id: str,
		node_id: str,
		latency_ms: float,
		operation: str = "request",
		measured_by: str = "system",
	) -> dict[str, Any]:
		"""Record a latency sample for an edge node operation."""
		self._require_tenant(tenant_id)
		self._require_node(node_id, tenant_id)
		if latency_ms < 0:
			raise ValueError("latency_ms must be non-negative")
		key = f"{tenant_id}:{node_id}"
		if key not in self._latency_samples:
			self._latency_samples[key] = []
		sample = {"latency_ms": latency_ms, "operation": operation, "measured_by": measured_by, "measured_at": _ts()}
		self._latency_samples[key].append(sample)
		samples = self._latency_samples[key]
		latency_values = [s["latency_ms"] for s in samples]
		stats = {"min_ms": min(latency_values), "max_ms": max(latency_values), "avg_ms": round(statistics.mean(latency_values), 3), "p95_ms": round(sorted(latency_values)[int(len(latency_values) * 0.95)], 3) if len(latency_values) >= 20 else None, "sample_count": len(samples)}
		result = {"node_id": node_id, "tenant_id": tenant_id, "latest_latency_ms": latency_ms, "statistics": stats, "recorded_at": _ts()}
		self._record_audit(tenant_id, "latency_recorded", node_id, measured_by, result)
		return result

	# ------------------------------------------------------------------
	# NEW: edge_security
	# ------------------------------------------------------------------

	def edge_security(
		self,
		tenant_id: str,
		node_id: str,
		event_type: str,
		severity: str,
		details: dict[str, Any],
		reported_by: str = "system",
	) -> dict[str, Any]:
		"""Record a security event detected at an edge node."""
		self._require_tenant(tenant_id)
		self._require_node(node_id, tenant_id)
		valid_severities = {"info", "warning", "critical", "breach"}
		if severity not in valid_severities:
			raise ValueError(f"severity must be one of: {valid_severities}")
		event = {"node_id": node_id, "tenant_id": tenant_id, "event_type": event_type, "severity": severity, "details": details, "reported_by": reported_by, "reported_at": _ts()}
		key = f"{tenant_id}:{node_id}"
		if key not in self._security_events:
			self._security_events[key] = []
		self._security_events[key].append(event)
		if severity in {"critical", "breach"}:
			self._record_audit(tenant_id, f"edge_security_{event_type}", node_id, reported_by, event)
		return event

	# ------------------------------------------------------------------
	# NEW: data_sovereignty_check
	# ------------------------------------------------------------------

	def data_sovereignty_check(
		self,
		tenant_id: str,
		node_id: str,
		data_classification: str,
		data_country_codes: list[str],
		required_residency: list[str],
		checked_by: str = "system",
	) -> dict[str, Any]:
		"""Verify that data stored/processed at an edge node satisfies sovereignty requirements."""
		self._require_tenant(tenant_id)
		node = self._require_node(node_id, tenant_id)
		node_country = node.location.get("country_code", "UNKNOWN")
		violations = [c for c in data_country_codes if c not in required_residency]
		node_violation = node_country not in required_residency and node_country != "UNKNOWN"
		compliant = len(violations) == 0 and not node_violation
		record = {"node_id": node_id, "tenant_id": tenant_id, "data_classification": data_classification, "data_country_codes": data_country_codes, "required_residency": required_residency, "node_country": node_country, "country_violations": violations, "node_location_violation": node_violation, "compliant": compliant, "checked_by": checked_by, "checked_at": _ts()}
		self._sovereignty_checks[f"{tenant_id}:{node_id}"] = record
		if not compliant:
			self._record_audit(tenant_id, "sovereignty_violation_detected", node_id, checked_by, record)
		return record

	# ------------------------------------------------------------------
	# NEW: health_check
	# ------------------------------------------------------------------

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health status for the edge computing capability."""
		return {"service": "edge", "tenant_id": tenant_id, "status": "healthy", "node_count": len(self.list_nodes(tenant_id)), "workload_count": len(self.list_workloads(tenant_id)), "audit_event_count": len(self.list_audit_events(tenant_id)), "checked_at": _ts()}

	# ------------------------------------------------------------------
	# NEW: Bulk operations
	# ------------------------------------------------------------------

	def bulk_register_nodes(
		self,
		tenant_id: str,
		nodes: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Register multiple edge nodes in a single call."""
		return [self.register_edge_node(
			node_id=n["id"],
			location=n.get("location", {"site": "unknown"}),
			capabilities=n.get("capabilities", []),
			network_type=n.get("network_type", "ethernet"),
			tenant_id=tenant_id,
			name=n.get("name", n["id"]),
			owner=n.get("owner", "system"),
			node_type=n.get("node_type", "compute"),
			location_policy=n.get("location_policy", "default"),
			attested=n.get("attested", True),
			health_status=n.get("health_status", "healthy"),
			secure_transport=n.get("secure_transport", True),
			capacity=n.get("capacity"),
		) for n in nodes]

	def bulk_deploy_workloads(
		self,
		tenant_id: str,
		workloads: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Deploy multiple workloads in a single call."""
		return [self.deploy_workload(
			workload_id=w["id"],
			target_nodes=w["target_nodes"],
			constraints=w.get("constraints", {}),
			tenant_id=tenant_id,
			name=w.get("name", w["id"]),
			version=w.get("version", "1.0.0"),
			owner=w.get("owner", "system"),
			artifact_signed=w.get("artifact_signed", True),
			resource_quota=w.get("resource_quota"),
		) for w in workloads]

	# ------------------------------------------------------------------
	# NEW: Export
	# ------------------------------------------------------------------

	def export_nodes(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export edge node records as JSON or CSV."""
		nodes = self.list_nodes(tenant_id)
		if fmt == "csv":
			if not nodes:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(nodes[0].keys()))
			writer.writeheader()
			writer.writerows(nodes)
			return buf.getvalue()
		return json.dumps(nodes, indent=2, default=str)

	def export_deployments(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export deployment records as JSON or CSV."""
		deployments = self.list_deployments(tenant_id)
		if fmt == "csv":
			if not deployments:
				return ""
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=list(deployments[0].keys()))
			writer.writeheader()
			writer.writerows(deployments)
			return buf.getvalue()
		return json.dumps(deployments, indent=2, default=str)

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

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

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if tenant_id is not None:
			events = [e for e in events if e.tenant_id == tenant_id]
		return [e.to_dict() for e in events]

	# ------------------------------------------------------------------
	# Agent management
	# ------------------------------------------------------------------

	def register_edge_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str, contribution_disclosed: bool = True, agent_id: str | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "edge_agent_present": True, "agent_registered": True, "agent_runtime_supported": normalized_runtime in SUPPORTED_EDGE_AGENT_RUNTIMES, "agent_role_supported": normalized_role in SUPPORTED_EDGE_AGENT_ROLES, "agent_scope_present": bool(scope), "agent_contribution_disclosed": contribution_disclosed})
		self._raise_if_denied(result)
		edge_agent = EdgeAgent(id=agent_id or f"edge-agent-{len(self._agents) + 1:06d}", tenant_id=tenant_id, name=name, runtime=normalized_runtime, role=normalized_role, scope=scope, contribution_disclosed=contribution_disclosed)
		self._agents[edge_agent.id] = edge_agent
		self._record_audit(tenant_id, "edge_agent_registered", edge_agent.id, name, edge_agent.to_dict())
		return edge_agent.to_dict()

	def list_edge_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list_for_tenant(self._agents, tenant_id)

	# ------------------------------------------------------------------
	# Dashboard / compat
	# ------------------------------------------------------------------

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
			"healthy_node_count": sum(1 for n in nodes if n["health_status"] == "healthy"),
			"failed_node_count": sum(1 for n in nodes if n["health_status"] == "failed"),
			"review_required_sync_count": sum(1 for s in sync_sessions if s["review_required"]),
			"conflict_pending_sync_count": sum(1 for s in sync_sessions if s["status"] == "conflict_pending"),
			"scaling_event_count": sum(1 for e in self._scaling_events if e["tenant_id"] == tenant_id),
			"failover_count": sum(1 for e in self._failover_events if e["tenant_id"] == tenant_id),
			"inference_count": len([r for r in self._inference_results.values() if r["tenant_id"] == tenant_id]),
			"firmware_update_count": len([f for f in self._firmware_updates.values() if f["tenant_id"] == tenant_id]),
			"sovereignty_checks": len([s for s in self._sovereignty_checks.values() if s["tenant_id"] == tenant_id]),
			"edge_agent_count": len(self.list_edge_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def validate_batch_edge_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({"tenant_context_present": True, "requested_operation": "batch_edge_mutation", "event_stream": event_stream})

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = metadata or {}
		return self.register_edge_node(node_id=record_id, location=dict(metadata.get("location") or {"site": "unknown"}), capabilities=list(metadata.get("capabilities") or []), network_type=str(metadata.get("network_type") or "ethernet"), tenant_id=tenant_id, name=str(metadata.get("name") or record_id), owner=str(metadata.get("owner") or "system"), node_type=str(metadata.get("node_type") or "compute"), location_policy=str(metadata.get("location_policy") or "default"), attested=bool(metadata.get("attested", True)), health_status=status, secure_transport=bool(metadata.get("secure_transport", True)), capacity=dict(metadata.get("capacity") or {"cpu": 4.0, "memory": 8192.0, "storage": 128.0}))

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_nodes(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _gen_id(self, *parts: str) -> str:
		digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
		return f"{parts[0]}-{digest}"

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
		payload = {"tenant_id": tenant_id, "action": action, "resource_id": resource_id, "actor": actor, "metadata": metadata}
		self._audit_events.append(EdgeAuditEvent(id=f"aud-{len(self._audit_events) + 1:06d}", tenant_id=tenant_id, action=action, resource_id=resource_id, actor=actor, digest=stable_digest(payload), metadata=dict(metadata)))

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "edge_policy_blocked") for action in result["actions"]))

	def _list_for_tenant(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		items = list(records.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]


# Alias
EdgeService = EdgeComputingService
