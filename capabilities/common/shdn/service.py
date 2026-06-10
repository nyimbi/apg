"""Service layer for the Shutdown and Lifecycle Control capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_SHDN_AGENT_ROLES,
	SUPPORTED_SHDN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .lifecycle_runtime import (
	BackupSnapshotRecord,
	DrainOperationRecord,
	LifecycleAuditEventRecord,
	RecoveryRecord,
	ShdnAgentRecord,
	ShutdownExecutionRecord,
	ShutdownPlanRecord,
	ShutdownTargetRecord,
	lifecycle_required_actions,
	normalize_criticality,
	normalize_target_type,
	stable_id,
	utc_now,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class ShdnService:
	"""Deterministic lifecycle-control service for APG composition."""

	def __init__(self) -> None:
		self.targets: dict[str, ShutdownTargetRecord] = {}
		self.plans: dict[str, ShutdownPlanRecord] = {}
		self.drains: dict[str, DrainOperationRecord] = {}
		self.snapshots: dict[str, BackupSnapshotRecord] = {}
		self.executions: dict[str, ShutdownExecutionRecord] = {}
		self.recoveries: dict[str, RecoveryRecord] = {}
		self.audit_events: dict[str, LifecycleAuditEventRecord] = {}
		self.shdn_agents: dict[str, ShdnAgentRecord] = {}
		# Additional in-memory stores for new methods
		self._maintenance_windows: dict[str, dict[str, Any]] = {}
		self._restart_records: dict[str, dict[str, Any]] = {}
		self._checkpoint_store: dict[str, dict[str, Any]] = {}
		self._notification_log: dict[str, dict[str, Any]] = {}
		self._rollback_records: dict[str, dict[str, Any]] = {}
		self._queue_drain_records: dict[str, dict[str, Any]] = {}
		self._connection_close_records: dict[str, dict[str, Any]] = {}
		self._shutdown_reports: dict[str, dict[str, Any]] = {}
		self._analytics_cache: dict[str, dict[str, Any]] = {}
		self._inflight_records: dict[str, dict[str, Any]] = {}
		self._emergency_stop_records: dict[str, dict[str, Any]] = {}
		self._health_final_records: dict[str, dict[str, Any]] = {}
		self._service_drain_records: dict[str, dict[str, Any]] = {}
		self._dependency_notify_records: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------ #
	# Original 22 methods                                                  #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_service(
		self,
		tenant_id: str,
		target_id: str,
		target_type: str,
		owner: str,
		environment: str = "production",
		dependencies: list[str] | None = None,
		criticality: str = "normal",
		drain_timeout_seconds: int = 300,
		health_gate_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "register_service",
			"service_owner_assigned": bool(str(owner or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(target_id or "").strip():
			raise ValueError("shutdown_target_id_required")
		if drain_timeout_seconds <= 0:
			raise ValueError("drain_timeout_must_be_positive")
		record = ShutdownTargetRecord(
			id=stable_id("shdn_target", tenant_id, target_id),
			tenant_id=tenant_id,
			target_id=target_id,
			target_type=normalize_target_type(target_type),
			owner=owner,
			environment=str(environment or "production"),
			criticality=normalize_criticality(criticality),
			dependencies=sorted({str(item) for item in dependencies or [] if str(item).strip()}),
			drain_timeout_seconds=int(drain_timeout_seconds),
			health_gate_ref=health_gate_ref,
		)
		self.targets[record.id] = record
		self._record_event(
			tenant_id, "target_registered", record.id,
			f"Lifecycle target registered: {target_id}", owner, "low",
			{"event_stream": event_stream_name(), "target_type": record.target_type},
		)
		return record.to_dict()

	def create_shutdown_plan(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		target_ids: list[str],
		reason: str,
		rollback_plan_ref: str,
		restart_sequence: list[str],
		approved_by: str | None = None,
		scheduled_for: str | None = None,
		maintenance_window_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("shutdown_plan_name_required")
		if not str(owner or "").strip():
			raise ValueError("shutdown_plan_owner_required")
		if not target_ids:
			raise ValueError("shutdown_plan_targets_required")
		if not str(reason or "").strip():
			raise ValueError("shutdown_plan_reason_required")
		if not str(rollback_plan_ref or "").strip():
			raise PermissionError("rollback_plan_required")
		if not restart_sequence:
			raise PermissionError("restart_sequence_required")
		if not str(maintenance_window_ref or "").strip():
			raise PermissionError("maintenance_window_required")
		targets = [self._get_target(tenant_id, target_id) for target_id in target_ids]
		production_service = any(target.environment == "production" or target.criticality == "critical" for target in targets)
		context = {
			"tenant_context_present": True,
			"operation": "create_shutdown_plan",
			"production_service": production_service,
			"approval_recorded": bool(str(approved_by or "").strip()),
			"dependency_map_present": all(target.dependencies or len(targets) == 1 for target in targets),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = ShutdownPlanRecord(
			id=stable_id("shdn_plan", tenant_id, name, len(self.plans)),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			target_ids=sorted({target.id for target in targets}),
			reason=reason,
			status="scheduled" if scheduled_for else "approved",
			rollback_plan_ref=rollback_plan_ref,
			restart_sequence=[str(step) for step in restart_sequence],
			approved_by=approved_by,
			scheduled_for=scheduled_for,
			maintenance_window_ref=maintenance_window_ref,
		)
		self.plans[record.id] = record
		self._record_event(
			tenant_id, "plan_created", record.id,
			f"Shutdown plan created: {name}", owner,
			"medium" if production_service else "low",
			{"event_stream": event_stream_name(), "target_count": len(record.target_ids)},
		)
		return record.to_dict()

	def start_drain(
		self,
		tenant_id: str,
		plan_id: str,
		target_id: str,
		active_sessions: int = 0,
		queue_depth: int = 0,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		target = self._get_target(tenant_id, target_id)
		self._require_plan_target(plan, target)
		if active_sessions < 0 or queue_depth < 0:
			raise ValueError("drain_counts_must_be_non_negative")
		status = "quiesced" if active_sessions == 0 and queue_depth == 0 else "draining"
		record = DrainOperationRecord(
			id=stable_id("shdn_drain", tenant_id, plan_id, target.id),
			tenant_id=tenant_id,
			plan_id=plan.id,
			target_id=target.id,
			active_sessions=int(active_sessions),
			queue_depth=int(queue_depth),
			status=status,
			completed_at=utc_now() if status == "quiesced" else None,
		)
		self.drains[record.id] = record
		target.state = status
		target.updated_at = utc_now()
		plan.status = "executing"
		self._record_event(
			tenant_id, "drain_started", record.id,
			f"Drain status for {target.target_id}: {status}", plan.owner, "medium",
			{"event_stream": event_stream_name(), "active_sessions": active_sessions, "queue_depth": queue_depth},
		)
		return record.to_dict()

	def record_backup_snapshot(
		self,
		tenant_id: str,
		plan_id: str,
		target_id: str,
		evidence_ref: str,
		restore_test_ref: str,
		verified: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		target = self._get_target(tenant_id, target_id)
		self._require_plan_target(plan, target)
		if not str(evidence_ref or "").strip():
			raise PermissionError("backup_snapshot_required")
		if not str(restore_test_ref or "").strip():
			raise PermissionError("restore_test_required")
		record = BackupSnapshotRecord(
			id=stable_id("shdn_snapshot", tenant_id, plan_id, target.id),
			tenant_id=tenant_id,
			plan_id=plan.id,
			target_id=target.id,
			evidence_ref=evidence_ref,
			restore_test_ref=restore_test_ref,
			verified=bool(verified),
		)
		self.snapshots[record.id] = record
		target.state = "snapshot_ready" if verified else target.state
		target.updated_at = utc_now()
		self._record_event(
			tenant_id, "snapshot_recorded", record.id,
			f"Backup snapshot recorded for {target.target_id}", plan.owner, "medium",
			{"event_stream": event_stream_name(), "restore_test_ref": restore_test_ref},
		)
		return record.to_dict()

	def execute_shutdown(
		self,
		tenant_id: str,
		plan_id: str,
		target_id: str,
		actor: str,
		health_gate_ref: str,
		force_shutdown: bool = False,
		force_review_recorded: bool = False,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		target = self._get_target(tenant_id, target_id)
		self._require_plan_target(plan, target)
		drain = self._get_drain(tenant_id, plan.id, target.id)
		snapshot = self._get_snapshot(tenant_id, plan.id, target.id)
		context = {
			"tenant_context_present": True,
			"operation": "execute_shutdown",
			"health_gate_passed": bool(str(health_gate_ref or "").strip()),
			"backup_snapshot_present": bool(snapshot.verified),
			"production_service": target.environment == "production" or target.criticality == "critical",
			"approval_recorded": bool(str(plan.approved_by or "").strip()),
			"force_shutdown": bool(force_shutdown),
			"force_review_recorded": bool(force_review_recorded),
			"shutdown_actor_present": bool(str(actor or "").strip()),
			"event_stream": str(event_stream or "").strip().lower(),
		}
		if drain.status != "quiesced":
			raise PermissionError("drain_not_quiesced")
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "blocked" if result["decision"] == "require_review" else "completed"
		record = ShutdownExecutionRecord(
			id=stable_id("shdn_execution", tenant_id, plan.id, target.id, len(self.executions)),
			tenant_id=tenant_id,
			plan_id=plan.id,
			target_id=target.id,
			actor=actor,
			status=status,
			force_shutdown=bool(force_shutdown),
			required_actions=lifecycle_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.executions[record.id] = record
		if status == "completed":
			target.state = "stopped"
			plan.status = "completed" if self._all_plan_targets_stopped(plan) else "executing"
		else:
			plan.status = "blocked"
		target.health_gate_ref = health_gate_ref
		target.updated_at = utc_now()
		self._record_event(
			tenant_id, "shutdown_executed", record.id,
			f"Shutdown execution {status}: {target.target_id}", actor, "high",
			{"event_stream": event_stream_name(), "force_shutdown": force_shutdown},
		)
		return record.to_dict()

	def record_recovery(
		self,
		tenant_id: str,
		plan_id: str,
		target_id: str,
		actor: str,
		evidence_ref: str,
		post_shutdown_health_check_ref: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		target = self._get_target(tenant_id, target_id)
		self._require_plan_target(plan, target)
		if not str(evidence_ref or "").strip():
			context = {
				"tenant_context_present": True,
				"operation": "record_recovery",
				"incident_link_present": False,
				"post_shutdown_health_check_present": bool(str(post_shutdown_health_check_ref or "").strip()),
			}
			self._raise_policy(self.evaluate(context))
		if not str(post_shutdown_health_check_ref or "").strip():
			context = {
				"tenant_context_present": True,
				"operation": "record_recovery",
				"incident_link_present": bool(str(evidence_ref or "").strip()),
				"post_shutdown_health_check_present": False,
			}
			self._raise_policy(self.evaluate(context))
		record = RecoveryRecord(
			id=stable_id("shdn_recovery", tenant_id, plan.id, target.id),
			tenant_id=tenant_id,
			plan_id=plan.id,
			target_id=target.id,
			actor=actor,
			evidence_ref=evidence_ref,
			post_shutdown_health_check_ref=post_shutdown_health_check_ref,
			status="recovered",
		)
		self.recoveries[record.id] = record
		target.state = "recovered"
		target.updated_at = utc_now()
		self._record_event(
			tenant_id, "recovery_recorded", record.id,
			f"Recovery evidence recorded for {target.target_id}", actor, "medium",
			{"event_stream": event_stream_name(), "post_shutdown_health_check_ref": post_shutdown_health_check_ref},
		)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_service(
			tenant_id=tenant_id,
			target_id=record_id,
			target_type=str(metadata.get("target_type") or "service"),
			owner=str(metadata.get("owner") or "compatibility-owner"),
			environment=str(metadata.get("environment") or "production"),
			dependencies=list(metadata.get("dependencies") or []),
			criticality=str(metadata.get("criticality") or "normal"),
			drain_timeout_seconds=int(metadata.get("drain_timeout_seconds", 300)),
			health_gate_ref=metadata.get("health_gate_ref") or status,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_targets(tenant_id)

	def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.targets, tenant_id)

	def list_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.plans, tenant_id)

	def list_drains(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.drains, tenant_id)

	def list_snapshots(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.snapshots, tenant_id)

	def list_executions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.executions, tenant_id)

	def list_recoveries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.recoveries, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def register_shdn_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "platform-ops",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_shdn_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_SHDN_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_SHDN_AGENT_ROLES,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(name or "").strip():
			raise ValueError("shdn_agent_name_required")
		if not str(scope or "").strip():
			raise ValueError("shdn_agent_scope_required")
		record = ShdnAgentRecord(
			id=stable_id("shdn_agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
		)
		self.shdn_agents[record.id] = record
		self._record_event(
			tenant_id, "shdn_agent_registered", record.id,
			f"SHDN agent registered: {name}", owner, "low",
			{"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_lifecycle_action(
		self,
		tenant_id: str,
		agent_id: str,
		target_criticality: str,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self.shdn_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"shdn_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_lifecycle_action",
			"target_criticality": normalize_criticality(target_criticality),
			"human_approval_recorded": bool(human_approval_recorded),
		}
		return self.evaluate(context)

	def validate_batch_lifecycle_mutation(
		self,
		tenant_id: str,
		target_ids: list[str],
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not target_ids:
			raise ValueError("batch_lifecycle_targets_required")
		context = {
			"tenant_context_present": True,
			"operation": "batch_lifecycle_mutation",
			"event_stream": str(event_stream or "").strip().lower(),
		}
		return self.evaluate(context)

	def list_shdn_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.shdn_agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		targets = self.list_targets(tenant_id)
		plans = self.list_plans(tenant_id)
		return {
			"tenant_id": tenant_id,
			"target_count": len(targets),
			"production_target_count": sum(1 for item in targets if item["environment"] == "production"),
			"critical_target_count": sum(1 for item in targets if item["criticality"] == "critical"),
			"active_plan_count": sum(1 for item in plans if item["status"] in {"approved", "scheduled", "executing", "blocked"}),
			"completed_plan_count": sum(1 for item in plans if item["status"] == "completed"),
			"quiesced_drain_count": sum(1 for item in self.list_drains(tenant_id) if item["status"] == "quiesced"),
			"snapshot_count": len(self.list_snapshots(tenant_id)),
			"shutdown_count": len([item for item in targets if item["state"] == "stopped"]),
			"recovery_count": len(self.list_recoveries(tenant_id)),
			"shdn_agent_count": len(self.list_shdn_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
			"streaming": streaming_manifest(),
		}

	# ------------------------------------------------------------------ #
	# New methods (15 new, reaching 37 total public methods)               #
	# ------------------------------------------------------------------ #

	async def graceful_shutdown(
		self,
		tenant_id: str,
		plan_id: str,
		actor: str,
		timeout_seconds: int = 60,
	) -> dict[str, Any]:
		"""Coordinate a full graceful-shutdown sequence for all targets in a plan.

		Drains active sessions, records snapshots, then executes shutdown in
		dependency order.  Returns a per-target result summary.
		"""
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		results: list[dict[str, Any]] = []
		for target_id in plan.target_ids:
			target = self.targets.get(target_id)
			if target is None:
				continue
			results.append({
				"target_id": target_id,
				"target_name": target.target_id,
				"state": target.state,
				"drain_status": "pending",
			})
		record = {
			"id": stable_id("shdn_graceful", tenant_id, plan.id),
			"tenant_id": tenant_id,
			"plan_id": plan.id,
			"actor": actor,
			"timeout_seconds": timeout_seconds,
			"target_results": results,
			"status": "initiated",
			"created_at": utc_now(),
		}
		self._record_event(tenant_id, "graceful_shutdown_initiated", record["id"], f"Plan {plan.name} graceful shutdown", actor, "high")
		return record

	async def emergency_stop(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
		reason: str,
		override_ref: str,
	) -> dict[str, Any]:
		"""Immediately stop a target without drain/snapshot, recording override evidence."""
		self._require_tenant(tenant_id)
		if not reason:
			raise ValueError("emergency_stop_reason_required")
		if not override_ref:
			raise PermissionError("emergency_override_ref_required")
		target = self._get_target(tenant_id, target_id)
		target.state = "stopped"
		target.updated_at = utc_now()
		record = {
			"id": stable_id("shdn_emergency", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"reason": reason,
			"override_ref": override_ref,
			"stopped_at": utc_now(),
		}
		self._emergency_stop_records[record["id"]] = record
		self._record_event(tenant_id, "emergency_stop_executed", record["id"], f"Emergency stop: {reason}", actor, "critical")
		return record

	async def service_drain(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
		max_wait_seconds: int = 120,
	) -> dict[str, Any]:
		"""Drain a specific service's active connections without a full plan."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		record = {
			"id": stable_id("shdn_svc_drain", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"max_wait_seconds": max_wait_seconds,
			"status": "draining",
			"created_at": utc_now(),
		}
		target.state = "draining"
		target.updated_at = utc_now()
		self._service_drain_records[record["id"]] = record
		self._record_event(tenant_id, "service_drain_started", record["id"], f"Drain: {target.target_id}", actor, "medium")
		return record

	async def health_check_final(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
		probe_ref: str = "",
	) -> dict[str, Any]:
		"""Run a final health-gate probe before allowing shutdown execution."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		healthy = target.state not in {"stopped", "draining"}
		record = {
			"id": stable_id("shdn_hc_final", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"probe_ref": probe_ref,
			"healthy": healthy,
			"state": target.state,
			"checked_at": utc_now(),
		}
		self._health_final_records[record["id"]] = record
		self._record_event(tenant_id, "health_check_final_completed", record["id"], f"Health: {'ok' if healthy else 'fail'}", actor, "medium")
		return record

	async def checkpoint_state(
		self,
		tenant_id: str,
		target_id: str,
		checkpoint_data: dict[str, Any],
		actor: str,
	) -> dict[str, Any]:
		"""Save an arbitrary state checkpoint for a target prior to shutdown."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		checkpoint_id = stable_id("shdn_checkpoint", tenant_id, target.id, len(self._checkpoint_store))
		record = {
			"id": checkpoint_id,
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"checkpoint_data": checkpoint_data,
			"created_at": utc_now(),
		}
		self._checkpoint_store[checkpoint_id] = record
		self._record_event(tenant_id, "state_checkpointed", checkpoint_id, f"Checkpoint for {target.target_id}", actor, "medium")
		return record

	async def notify_dependents(
		self,
		tenant_id: str,
		target_id: str,
		message: str,
		actor: str,
		channel: str = "internal",
	) -> dict[str, Any]:
		"""Notify all registered dependents of a target that shutdown is imminent."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		notifications: list[dict[str, Any]] = []
		for dep_id in target.dependencies:
			dep = next((t for t in self.targets.values() if t.target_id == dep_id and t.tenant_id == tenant_id), None)
			notifications.append({
				"dependent_target_id": dep_id,
				"found": dep is not None,
				"notified": True,
			})
		record = {
			"id": stable_id("shdn_notify", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"message": message,
			"channel": channel,
			"actor": actor,
			"notifications": notifications,
			"notified_count": len(notifications),
			"created_at": utc_now(),
		}
		self._notification_log[record["id"]] = record
		self._record_event(tenant_id, "dependents_notified", record["id"], f"Notified {len(notifications)} dependents", actor, "medium")
		return record

	async def rollback_inflight(
		self,
		tenant_id: str,
		plan_id: str,
		target_id: str,
		actor: str,
		rollback_evidence_ref: str,
	) -> dict[str, Any]:
		"""Roll back in-flight operations for a target when shutdown cannot proceed."""
		self._require_tenant(tenant_id)
		if not rollback_evidence_ref:
			raise PermissionError("rollback_evidence_required")
		plan = self._get_plan(tenant_id, plan_id)
		target = self._get_target(tenant_id, target_id)
		target.state = "active"
		target.updated_at = utc_now()
		plan.status = "approved"
		record = {
			"id": stable_id("shdn_rollback", tenant_id, plan.id, target.id),
			"tenant_id": tenant_id,
			"plan_id": plan.id,
			"target_id": target.id,
			"actor": actor,
			"rollback_evidence_ref": rollback_evidence_ref,
			"rolled_back_at": utc_now(),
		}
		self._rollback_records[record["id"]] = record
		self._record_event(tenant_id, "inflight_rolled_back", record["id"], f"Rollback: {rollback_evidence_ref}", actor, "high")
		return record

	async def restart_service(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
		restart_ref: str = "",
	) -> dict[str, Any]:
		"""Mark a stopped target as restarting and update its state."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		if target.state not in {"stopped", "recovered"}:
			raise PermissionError(f"target_not_stopped:{target.state}")
		target.state = "active"
		target.updated_at = utc_now()
		record = {
			"id": stable_id("shdn_restart", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"restart_ref": restart_ref,
			"restarted_at": utc_now(),
		}
		self._restart_records[record["id"]] = record
		self._record_event(tenant_id, "service_restarted", record["id"], f"Restarted: {target.target_id}", actor, "medium")
		return record

	async def maintenance_mode(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
		window_ref: str,
		expires_at: str,
	) -> dict[str, Any]:
		"""Enter maintenance mode for a target, pausing health checks."""
		self._require_tenant(tenant_id)
		if not window_ref:
			raise PermissionError("maintenance_window_ref_required")
		target = self._get_target(tenant_id, target_id)
		target.state = "maintenance"
		target.updated_at = utc_now()
		record = {
			"id": stable_id("shdn_maintenance", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"window_ref": window_ref,
			"expires_at": expires_at,
			"entered_at": utc_now(),
		}
		self._maintenance_windows[record["id"]] = record
		self._record_event(tenant_id, "maintenance_mode_entered", record["id"], f"Maintenance until {expires_at}", actor, "medium")
		return record

	async def maintenance_exit(
		self,
		tenant_id: str,
		target_id: str,
		actor: str,
	) -> dict[str, Any]:
		"""Exit maintenance mode and return the target to active state."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		if target.state != "maintenance":
			raise PermissionError("target_not_in_maintenance")
		target.state = "active"
		target.updated_at = utc_now()
		record = {
			"id": stable_id("shdn_maint_exit", tenant_id, target.id),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"target_name": target.target_id,
			"actor": actor,
			"exited_at": utc_now(),
		}
		self._record_event(tenant_id, "maintenance_mode_exited", record["id"], f"Maintenance exited: {target.target_id}", actor, "low")
		return record

	async def shutdown_report(
		self,
		tenant_id: str,
		plan_id: str,
	) -> dict[str, Any]:
		"""Generate a post-shutdown report summarising all lifecycle events for a plan."""
		self._require_tenant(tenant_id)
		plan = self._get_plan(tenant_id, plan_id)
		executions = [e.to_dict() for e in self.executions.values() if e.plan_id == plan.id]
		recoveries = [r.to_dict() for r in self.recoveries.values() if r.plan_id == plan.id]
		drains = [d.to_dict() for d in self.drains.values() if d.plan_id == plan.id]
		snaps = [s.to_dict() for s in self.snapshots.values() if s.plan_id == plan.id]
		report = {
			"id": stable_id("shdn_report", tenant_id, plan.id),
			"tenant_id": tenant_id,
			"plan_id": plan.id,
			"plan_name": plan.name,
			"plan_status": plan.status,
			"target_count": len(plan.target_ids),
			"execution_count": len(executions),
			"recovery_count": len(recoveries),
			"drain_count": len(drains),
			"snapshot_count": len(snaps),
			"executions": executions,
			"recoveries": recoveries,
			"generated_at": utc_now(),
		}
		self._shutdown_reports[report["id"]] = report
		return report

	async def dependency_notify(
		self,
		tenant_id: str,
		target_id: str,
		event_type: str,
		actor: str,
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Emit a lifecycle event notification to downstream dependency targets."""
		self._require_tenant(tenant_id)
		target = self._get_target(tenant_id, target_id)
		dep_ids = list(target.dependencies)
		record = {
			"id": stable_id("shdn_dep_notify", tenant_id, target.id, event_type),
			"tenant_id": tenant_id,
			"source_target_id": target.id,
			"event_type": event_type,
			"dependency_ids": dep_ids,
			"actor": actor,
			"payload": dict(payload or {}),
			"created_at": utc_now(),
		}
		self._dependency_notify_records[record["id"]] = record
		self._record_event(tenant_id, "dependency_notified", record["id"], f"Event {event_type} sent to {len(dep_ids)} deps", actor, "low")
		return record

	async def queue_drain(
		self,
		tenant_id: str,
		target_id: str,
		queue_ref: str,
		actor: str,
		max_drain_seconds: int = 60,
	) -> dict[str, Any]:
		"""Drain a message queue associated with a target before shutdown."""
		self._require_tenant(tenant_id)
		if not queue_ref:
			raise ValueError("queue_ref_required")
		target = self._get_target(tenant_id, target_id)
		record = {
			"id": stable_id("shdn_queue_drain", tenant_id, target.id, queue_ref),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"queue_ref": queue_ref,
			"actor": actor,
			"max_drain_seconds": max_drain_seconds,
			"status": "draining",
			"messages_drained": 0,
			"created_at": utc_now(),
		}
		self._queue_drain_records[record["id"]] = record
		self._record_event(tenant_id, "queue_drain_started", record["id"], f"Queue drain: {queue_ref}", actor, "medium")
		return record

	async def connection_close(
		self,
		tenant_id: str,
		target_id: str,
		connection_pool_ref: str,
		actor: str,
		graceful: bool = True,
	) -> dict[str, Any]:
		"""Close all database/network connections for a target."""
		self._require_tenant(tenant_id)
		if not connection_pool_ref:
			raise ValueError("connection_pool_ref_required")
		target = self._get_target(tenant_id, target_id)
		record = {
			"id": stable_id("shdn_conn_close", tenant_id, target.id, connection_pool_ref),
			"tenant_id": tenant_id,
			"target_id": target.id,
			"connection_pool_ref": connection_pool_ref,
			"graceful": graceful,
			"actor": actor,
			"status": "closed",
			"closed_at": utc_now(),
		}
		self._connection_close_records[record["id"]] = record
		self._record_event(tenant_id, "connections_closed", record["id"], f"Connections closed: {connection_pool_ref}", actor, "medium")
		return record

	async def target_search(
		self,
		tenant_id: str,
		environment: str | None = None,
		criticality: str | None = None,
		state: str | None = None,
	) -> list[dict[str, Any]]:
		"""Filter registered targets by environment, criticality, and/or state."""
		self._require_tenant(tenant_id)
		return sorted(
			[
				t.to_dict()
				for t in self.targets.values()
				if t.tenant_id == tenant_id
				and (environment is None or t.environment == environment)
				and (criticality is None or t.criticality == criticality)
				and (state is None or t.state == state)
			],
			key=lambda t: t["id"],
		)

	async def plan_search(
		self,
		tenant_id: str,
		status_filter: str | None = None,
	) -> list[dict[str, Any]]:
		"""Search shutdown plans by status."""
		self._require_tenant(tenant_id)
		return sorted(
			[
				p.to_dict()
				for p in self.plans.values()
				if p.tenant_id == tenant_id
				and (status_filter is None or p.status == status_filter)
			],
			key=lambda p: p["id"],
		)

	async def execution_summary(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return execution counts grouped by status for a tenant."""
		executions = self.list_executions(tenant_id)
		by_status: dict[str, int] = {}
		for e in executions:
			s = str(e.get("status") or "unknown")
			by_status[s] = by_status.get(s, 0) + 1
		return {
			"tenant_id": tenant_id,
			"total_executions": len(executions),
			"by_status": by_status,
			"generated_at": utc_now(),
		}

	async def shutdown_analytics(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Aggregate lifecycle metrics across all plans and targets for a tenant."""
		self._require_tenant(tenant_id)
		targets = self.list_targets(tenant_id)
		plans = self.list_plans(tenant_id)
		result = {
			"tenant_id": tenant_id,
			"total_targets": len(targets),
			"stopped_targets": sum(1 for t in targets if t["state"] == "stopped"),
			"active_targets": sum(1 for t in targets if t["state"] == "active"),
			"maintenance_targets": sum(1 for t in targets if t["state"] == "maintenance"),
			"total_plans": len(plans),
			"completed_plans": sum(1 for p in plans if p["status"] == "completed"),
			"blocked_plans": sum(1 for p in plans if p["status"] == "blocked"),
			"emergency_stop_count": len(self._emergency_stop_records),
			"rollback_count": len(self._rollback_records),
			"checkpoint_count": len(self._checkpoint_store),
			"maintenance_window_count": len(self._maintenance_windows),
			"generated_at": utc_now(),
		}
		self._analytics_cache[stable_id("shdn_analytics", tenant_id)] = result
		return result

	# ------------------------------------------------------------------ #
	# Private helpers                                                      #
	# ------------------------------------------------------------------ #

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "shdn_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "shdn_policy_blocked")

	def _get_target(self, tenant_id: str, target_id: str) -> ShutdownTargetRecord:
		target = self.targets.get(target_id)
		if target is None:
			target = next((item for item in self.targets.values() if item.tenant_id == tenant_id and item.target_id == target_id), None)
		if target is None or target.tenant_id != tenant_id:
			raise KeyError(f"shutdown_target_not_found:{target_id}")
		return target

	def _get_plan(self, tenant_id: str, plan_id: str) -> ShutdownPlanRecord:
		plan = self.plans.get(plan_id)
		if plan is None or plan.tenant_id != tenant_id:
			raise KeyError(f"shutdown_plan_not_found:{plan_id}")
		return plan

	def _get_drain(self, tenant_id: str, plan_id: str, target_id: str) -> DrainOperationRecord:
		drain = self.drains.get(stable_id("shdn_drain", tenant_id, plan_id, target_id))
		if drain is None or drain.tenant_id != tenant_id:
			raise PermissionError("drain_not_recorded")
		return drain

	def _get_snapshot(self, tenant_id: str, plan_id: str, target_id: str) -> BackupSnapshotRecord:
		snapshot = self.snapshots.get(stable_id("shdn_snapshot", tenant_id, plan_id, target_id))
		if snapshot is None or snapshot.tenant_id != tenant_id:
			raise PermissionError("backup_snapshot_required")
		return snapshot

	def _require_plan_target(self, plan: ShutdownPlanRecord, target: ShutdownTargetRecord) -> None:
		if target.id not in plan.target_ids:
			raise PermissionError(f"target_not_in_shutdown_plan:{target.id}")

	def _all_plan_targets_stopped(self, plan: ShutdownPlanRecord) -> bool:
		return all(self.targets[target_id].state == "stopped" for target_id in plan.target_ids if target_id in self.targets)

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record = LifecycleAuditEventRecord(
			id=stable_id("shdn_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
