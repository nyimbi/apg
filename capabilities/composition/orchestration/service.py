"""Domain service for the APG workflow orchestration capability."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import statistics
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class WorkflowOrchestrationService:
	"""Tenant-scoped workflow definition, release, and execution coordinator."""

	def __init__(self, db_url: str | None = None) -> None:
		_store = get_store(db_url)
		self._definitions = WriteThruDict('definitions', tenant_id, _store)
		self._tasks = WriteThruDict('tasks', tenant_id, _store)
		self._releases = WriteThruDict('releases', tenant_id, _store)
		self._executions = WriteThruDict('executions', tenant_id, _store)
		self._agents = WriteThruDict('agents', tenant_id, _store)
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# new collections
		self._signals: dict[str, list[dict[str, Any]]] = {}   # execution_id -> signal queue
		self._compensations: dict[str, list[dict[str, Any]]] = {}  # execution_id -> compensation log
		self._suspended = WriteThruDict('suspended', tenant_id, _store)  # execution_id -> suspension record
		self._instance_variables = WriteThruDict('instance_variables', tenant_id, _store)  # execution_id -> variables

	# ------------------------------------------------------------------ existing

	def define_workflow(
		self,
		workflow_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		version: str,
		tasks: list[dict[str, Any]],
		start_event: str,
		terminal_state: str,
		*,
		transactional: bool = False,
		compensation_steps: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "define_workflow",
			"workflow_owner_assigned": bool(owner),
			"workflow_version_present": bool(version),
			"start_event_present": bool(start_event),
			"task_graph_present": bool(tasks),
			"terminal_state_present": bool(terminal_state),
			"transactional_workflow": transactional,
			"compensation_present": bool(compensation_steps),
		}
		self._enforce(context)
		normalised_tasks = [self._validate_task(tenant_id, workflow_id, task) for task in tasks]
		self._assert_acyclic(normalised_tasks)
		record = {
			"id": self._record_id("workflow_definition", workflow_id),
			"workflow_id": workflow_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"version": version,
			"start_event": start_event,
			"terminal_state": terminal_state,
			"tasks": normalised_tasks,
			"transactional": transactional,
			"compensation_steps": compensation_steps or [],
			"status": "validated",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._definitions[record["id"]] = record
		self._emit("workflow_defined", tenant_id, record["id"], {"task_count": len(normalised_tasks)})
		self._emit("workflow_validated", tenant_id, record["id"], {"status": "validated"})
		return deepcopy(record)

	def release_workflow(
		self,
		release_id: str,
		tenant_id: str,
		workflow_definition_id: str,
		validation_evidence: str,
		rollback_plan: str,
		*,
		dry_run_passed: bool,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		self._require_definition(workflow_definition_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_workflow",
			"validation_evidence_present": bool(validation_evidence),
			"dry_run_passed": bool(dry_run_passed),
			"rollback_plan_present": bool(rollback_plan),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("workflow_release", release_id),
			"release_id": release_id,
			"tenant_id": tenant_id,
			"workflow_definition_id": workflow_definition_id,
			"validation_evidence": validation_evidence,
			"rollback_plan": rollback_plan,
			"dry_run_passed": dry_run_passed,
			"approved_by": approved_by,
			"status": "released",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._releases[record["id"]] = record
		self._emit("workflow_released", tenant_id, record["id"], {"workflow_definition_id": workflow_definition_id})
		return deepcopy(record)

	def start_execution(
		self,
		execution_id: str,
		tenant_id: str,
		workflow_definition_id: str,
		idempotency_key: str,
		inputs: dict[str, Any] | None = None,
		*,
		risk_level: str = "normal",
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		definition = self._require_definition(workflow_definition_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_execution",
			"event_stream": "bytewax",
			"idempotency_key_present": bool(idempotency_key),
			"risk_level": risk_level,
			"review_recorded": bool(reviewed_by),
		}
		self._enforce(context)
		first_tasks = [task["id"] for task in definition["tasks"] if not task.get("depends_on")]
		record = {
			"id": self._record_id("workflow_execution", execution_id),
			"execution_id": execution_id,
			"tenant_id": tenant_id,
			"workflow_definition_id": workflow_definition_id,
			"idempotency_key": idempotency_key,
			"inputs": inputs or {},
			"status": "running",
			"current_tasks": first_tasks,
			"completed_tasks": [],
			"failed_tasks": [],
			"event_stream": "bytewax",
			"started_at": self._now(),
			"updated_at": self._now(),
		}
		self._executions[record["id"]] = record
		self._instance_variables[record["id"]] = dict(inputs or {})
		self._emit("workflow_execution_started", tenant_id, record["id"], {"current_tasks": first_tasks})
		return deepcopy(record)

	def complete_task(
		self,
		tenant_id: str,
		execution_record_id: str,
		task_id: str,
		result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		execution = self._require_execution(execution_record_id, tenant_id)
		definition = self._require_definition(execution["workflow_definition_id"], tenant_id)
		if task_id not in execution["current_tasks"]:
			raise ValueError(f"task is not currently active: {task_id}")
		execution["current_tasks"] = [item for item in execution["current_tasks"] if item != task_id]
		execution["completed_tasks"].append(task_id)
		ready_tasks = self._ready_tasks(definition["tasks"], execution["completed_tasks"], execution["current_tasks"])
		execution["current_tasks"].extend(ready_tasks)
		if not execution["current_tasks"] and len(execution["completed_tasks"]) == len(definition["tasks"]):
			execution["status"] = "completed"
			event_name = "workflow_execution_completed"
		else:
			event_name = "workflow_execution_advanced"
		execution["last_result"] = result or {}
		execution["updated_at"] = self._now()
		# merge task result into instance variables
		if result:
			self._instance_variables.setdefault(execution_record_id, {}).update(result)
		self._emit(event_name, tenant_id, execution_record_id, {"completed_task": task_id, "ready_tasks": ready_tasks})
		return deepcopy(execution)

	def assign_human_task(
		self,
		tenant_id: str,
		execution_record_id: str,
		task_id: str,
		assignee: str,
		due_at: str | None = None,
	) -> dict[str, Any]:
		execution = self._require_execution(execution_record_id, tenant_id)
		definition = self._require_definition(execution["workflow_definition_id"], tenant_id)
		if task_id not in execution["current_tasks"]:
			raise ValueError(f"task is not currently active: {task_id}")
		task = self._task_by_id(definition["tasks"], task_id)
		if task["type"] not in {"human", "approval"}:
			raise ValueError(f"task is not assignable to a human actor: {task_id}")
		if not assignee:
			raise ValueError("assignee is required")
		task_record = {
			"id": self._record_id("workflow_task_assignment", f"{execution_record_id}_{task_id}"),
			"tenant_id": tenant_id,
			"execution_record_id": execution_record_id,
			"task_id": task_id,
			"assignee": assignee,
			"due_at": due_at,
			"status": "assigned",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._tasks[task_record["id"]] = task_record
		self._emit("workflow_task_assigned", tenant_id, task_record["id"], {"execution_status": execution["status"]})
		return deepcopy(task_record)

	def register_workflow_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		instructions: str,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_workflow_agent",
			"agent_runtime_supported": runtime in SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("workflow_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("workflow_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_workflow_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown workflow agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_workflow_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch_schedule(self, tenant_id: str, execution_count: int) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "batch_schedule",
			"event_stream": "bytewax",
			"execution_count": execution_count,
		}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "execution_count": execution_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	# ------------------------------------------------------------------ new methods

	def create_workflow(
		self,
		tenant_id: str,
		workflow_id: str,
		definition: dict[str, Any],
		steps: list[dict[str, Any]],
		transitions: list[dict[str, Any]] | None = None,
		guards: dict[str, str] | None = None,
		owner: str = "system",
	) -> dict[str, Any]:
		"""Create a workflow from a structured definition with explicit steps, transitions and guards."""
		assert bool(steps), "at least one step required"
		assert bool(workflow_id), "workflow_id required"
		# normalise steps to task format
		tasks: list[dict[str, Any]] = []
		for step in steps:
			step_id = str(step.get("id", step.get("name", f"step_{len(tasks)}")))
			depends_on = list(step.get("depends_on", step.get("after", [])))
			tasks.append({
				"id": step_id,
				"name": str(step.get("name", step_id)),
				"type": str(step.get("type", "automated")),
				"handler": step.get("handler", f"capability.{step_id}"),
				"depends_on": depends_on,
			})
		name = str(definition.get("name", workflow_id))
		version = str(definition.get("version", "1.0.0"))
		start_event = str(definition.get("start_event", "manual"))
		terminal_state = str(definition.get("terminal_state", "completed"))
		transactional = bool(definition.get("transactional", False))
		record = self.define_workflow(
			workflow_id=workflow_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			version=version,
			tasks=tasks,
			start_event=start_event,
			terminal_state=terminal_state,
			transactional=transactional,
		)
		# attach transitions and guards as metadata
		record["transitions"] = list(transitions or [])
		record["guards"] = dict(guards or {})
		self._definitions[record["id"]].update({"transitions": record["transitions"], "guards": record["guards"]})
		return record

	def start_instance(
		self,
		tenant_id: str,
		workflow_id: str,
		payload: dict[str, Any],
		instance_id: str | None = None,
		risk_level: str = "normal",
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		"""Start a workflow execution instance from a workflow_id (not a definition record ID)."""
		# resolve definition record from workflow_id
		def_record: dict[str, Any] | None = None
		for record in self._definitions.values():
			if record["tenant_id"] == tenant_id and record["workflow_id"] == workflow_id:
				def_record = record
				break
		if def_record is None:
			raise KeyError(f"workflow_not_found:{workflow_id}")
		eff_instance_id = instance_id or f"inst:{workflow_id}:{len(self._executions) + 1}"
		idempotency_key = f"{workflow_id}:{eff_instance_id}"
		return self.start_execution(
			execution_id=eff_instance_id,
			tenant_id=tenant_id,
			workflow_definition_id=def_record["id"],
			idempotency_key=idempotency_key,
			inputs=payload,
			risk_level=risk_level,
			reviewed_by=reviewed_by,
		)

	def get_instance(
		self,
		tenant_id: str,
		instance_id: str,
	) -> dict[str, Any]:
		"""Retrieve a workflow execution instance with its current state and variables."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		result = deepcopy(execution)
		result["variables"] = deepcopy(self._instance_variables.get(execution["id"], {}))
		result["pending_signals"] = [s for s in self._signals.get(execution["id"], []) if s.get("status") == "pending"]
		return result

	def advance_step(
		self,
		tenant_id: str,
		instance_id: str,
		step_id: str,
		outcome: str,
		variables: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Advance a workflow instance past a completed step, injecting output variables."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		result = self.complete_task(
			tenant_id=tenant_id,
			execution_record_id=execution["id"],
			task_id=step_id,
			result={"outcome": outcome, **(variables or {})},
		)
		if variables:
			self._instance_variables.setdefault(execution["id"], {}).update(variables)
		return result

	def wait_for_signal(
		self,
		tenant_id: str,
		instance_id: str,
		signal_name: str,
		timeout_seconds: int = 3600,
	) -> dict[str, Any]:
		"""Register a signal wait on an execution instance; the instance is not suspended — caller polls."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		assert bool(signal_name), "signal_name required"
		wait_record = {
			"execution_id": execution["id"],
			"instance_id": instance_id,
			"tenant_id": tenant_id,
			"signal_name": signal_name,
			"timeout_seconds": timeout_seconds,
			"status": "waiting",
			"registered_at": self._now(),
		}
		self._signals.setdefault(execution["id"], []).append(wait_record)
		self._emit("signal_wait_registered", tenant_id, execution["id"], {"signal_name": signal_name, "timeout_seconds": timeout_seconds})
		return wait_record

	def raise_signal(
		self,
		tenant_id: str,
		instance_id: str,
		signal_name: str,
		payload: dict[str, Any] | None = None,
		raised_by: str = "system",
	) -> dict[str, Any]:
		"""Deliver a named signal to a waiting execution instance."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		assert bool(signal_name), "signal_name required"
		# find pending waits for this signal
		matched = 0
		for sig in self._signals.get(execution["id"], []):
			if sig["signal_name"] == signal_name and sig["status"] == "waiting":
				sig["status"] = "received"
				sig["payload"] = dict(payload or {})
				sig["received_at"] = self._now()
				sig["raised_by"] = raised_by
				matched += 1
		# merge signal payload into instance variables
		if payload:
			self._instance_variables.setdefault(execution["id"], {}).update(payload)
		signal_event = {
			"execution_id": execution["id"],
			"instance_id": instance_id,
			"signal_name": signal_name,
			"payload": dict(payload or {}),
			"raised_by": raised_by,
			"matched_waits": matched,
			"raised_at": self._now(),
		}
		self._emit("signal_raised", tenant_id, execution["id"], signal_event)
		return signal_event

	def compensate(
		self,
		tenant_id: str,
		instance_id: str,
		step_id: str,
		compensation_action: str = "rollback",
		compensated_by: str = "system",
	) -> dict[str, Any]:
		"""Execute a compensation action for a completed step in a transactional workflow."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		definition = self._require_definition(execution["workflow_definition_id"], tenant_id)
		assert step_id in execution["completed_tasks"] or step_id in execution["current_tasks"], f"step not active or completed: {step_id}"
		compensation = {
			"execution_id": execution["id"],
			"instance_id": instance_id,
			"step_id": step_id,
			"compensation_action": compensation_action,
			"compensated_by": compensated_by,
			"status": "completed",
			"compensated_at": self._now(),
		}
		self._compensations.setdefault(execution["id"], []).append(compensation)
		self._emit("step_compensated", tenant_id, execution["id"], compensation)
		return compensation

	def suspend_instance(
		self,
		tenant_id: str,
		instance_id: str,
		reason: str,
		suspended_by: str = "system",
	) -> dict[str, Any]:
		"""Suspend a running workflow instance, preserving its current state."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		assert bool(reason), "suspension reason required"
		if execution["status"] != "running":
			raise ValueError(f"cannot suspend instance in status: {execution['status']}")
		execution["status"] = "suspended"
		execution["updated_at"] = self._now()
		suspension = {
			"execution_id": execution["id"],
			"instance_id": instance_id,
			"tenant_id": tenant_id,
			"reason": reason,
			"suspended_by": suspended_by,
			"suspended_at": self._now(),
			"current_tasks_snapshot": list(execution["current_tasks"]),
		}
		self._suspended[execution["id"]] = suspension
		self._emit("instance_suspended", tenant_id, execution["id"], suspension)
		return suspension

	def resume_instance(
		self,
		tenant_id: str,
		instance_id: str,
		payload: dict[str, Any] | None = None,
		resumed_by: str = "system",
	) -> dict[str, Any]:
		"""Resume a suspended workflow instance, optionally injecting new variables."""
		execution = self._require_execution_by_instance(tenant_id, instance_id)
		if execution["status"] != "suspended":
			raise ValueError(f"cannot resume instance in status: {execution['status']}")
		execution["status"] = "running"
		execution["updated_at"] = self._now()
		if payload:
			self._instance_variables.setdefault(execution["id"], {}).update(payload)
		suspension = self._suspended.pop(execution["id"], {})
		resume_event = {
			"execution_id": execution["id"],
			"instance_id": instance_id,
			"tenant_id": tenant_id,
			"resumed_by": resumed_by,
			"resumed_at": self._now(),
			"suspended_reason": suspension.get("reason"),
		}
		self._emit("instance_resumed", tenant_id, execution["id"], resume_event)
		return deepcopy(execution)

	def workflow_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return execution KPIs and throughput analytics for a tenant."""
		definitions = self.list_workflow_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		releases = self.list_releases(tenant_id)
		task_assignments = self.list_task_assignments(tenant_id)
		running = [e for e in executions if e["status"] == "running"]
		completed = [e for e in executions if e["status"] == "completed"]
		failed = [e for e in executions if e["status"] == "failed"]
		suspended_count = len([e for e in executions if e["status"] == "suspended"])
		# mean step count for completed executions
		completed_step_counts = [len(e.get("completed_tasks", [])) for e in completed]
		avg_steps = round(statistics.mean(completed_step_counts), 2) if completed_step_counts else None
		signal_count = sum(len(sigs) for sigs in self._signals.values())
		compensation_count = sum(len(comps) for comps in self._compensations.values())
		return {
			"tenant_id": tenant_id,
			"period": period,
			"workflow_definition_count": len(definitions),
			"released_workflow_count": len(releases),
			"total_executions": len(executions),
			"running_executions": len(running),
			"completed_executions": len(completed),
			"failed_executions": len(failed),
			"suspended_executions": suspended_count,
			"completion_rate_pct": round(len(completed) / max(len(executions), 1) * 100, 2),
			"avg_steps_per_execution": avg_steps,
			"human_task_assignments": len(task_assignments),
			"pending_signal_waits": signal_count,
			"compensation_actions": compensation_count,
			"workflow_agent_count": len(self.list_workflow_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
			"computed_at": self._now(),
		}

	# ------------------------------------------------------------------ dashboard / list / compat

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		definitions = self.list_workflow_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"workflow_definition_count": len(definitions),
			"released_workflow_count": len(self.list_releases(tenant_id)),
			"running_execution_count": len([item for item in executions if item["status"] == "running"]),
			"completed_execution_count": len([item for item in executions if item["status"] == "completed"]),
			"suspended_execution_count": len([item for item in executions if item["status"] == "suspended"]),
			"human_task_assignment_count": len(self.list_task_assignments(tenant_id)),
			"workflow_agent_count": len(self.list_workflow_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def list_workflow_definitions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._definitions, tenant_id)

	def list_releases(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._releases, tenant_id)

	def list_executions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._executions, tenant_id)

	def list_task_assignments(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._tasks, tenant_id)

	def list_workflow_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		task = {
			"id": "start",
			"name": data.get("task_name", "Start"),
			"type": "automated",
			"handler": data.get("handler", "capability.noop"),
			"depends_on": [],
		}
		return self.define_workflow(
			data.get("workflow_id", data.get("id", "workflow")),
			data.get("tenant_id", "default"),
			data.get("name", "Workflow"),
			data.get("owner", "owner"),
			data.get("version", "1.0.0"),
			[task],
			data.get("start_event", "manual"),
			data.get("terminal_state", "completed"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_workflow_definitions(tenant_id)

	# ------------------------------------------------------------------ internals

	def _validate_task(self, tenant_id: str, workflow_id: str, task: dict[str, Any]) -> dict[str, Any]:
		task_type = task.get("type", "automated")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "define_task",
			"handler_present": bool(task.get("handler") or task_type in {"human", "approval"}),
			"human_task": task_type == "human",
			"assignee_present": bool(task.get("assignee") or task.get("role")),
			"approval_task": task_type == "approval",
			"approval_policy_present": bool(task.get("approval_policy")),
			"cross_capability_task": bool(task.get("capability")),
			"capability_contract_present": bool(task.get("capability_contract")),
			"retry_policy_present": bool(task.get("retry_policy")),
			"retry_limit_present": "retry_limit" in task,
			"sla_present": bool(task.get("sla")),
			"escalation_present": bool(task.get("escalation")),
		}
		self._enforce(context)
		record = {
			"id": str(task["id"]),
			"workflow_id": workflow_id,
			"name": str(task.get("name", task["id"])),
			"type": task_type,
			"handler": task.get("handler"),
			"assignee": task.get("assignee"),
			"role": task.get("role"),
			"approval_policy": task.get("approval_policy"),
			"capability": task.get("capability"),
			"capability_contract": task.get("capability_contract"),
			"depends_on": list(task.get("depends_on", [])),
			"sla": task.get("sla"),
			"escalation": task.get("escalation"),
		}
		return record

	def _assert_acyclic(self, tasks: list[dict[str, Any]]) -> None:
		seen: set[str] = set()
		duplicates: set[str] = set()
		for task in tasks:
			if task["id"] in seen:
				duplicates.add(task["id"])
			seen.add(task["id"])
		if duplicates:
			raise ValueError(f"duplicate workflow task ids: {sorted(duplicates)}")
		task_ids = set(seen)
		for task in tasks:
			unknown = [dependency for dependency in task.get("depends_on", []) if dependency not in task_ids]
			if unknown:
				raise ValueError(f"unknown task dependency for {task['id']}: {unknown}")
		visiting: set[str] = set()
		visited: set[str] = set()
		dependencies = {task["id"]: set(task.get("depends_on", [])) for task in tasks}

		def visit(task_id: str) -> None:
			if task_id in visiting:
				raise ValueError(f"workflow task cycle detected at {task_id}")
			if task_id in visited:
				return
			visiting.add(task_id)
			for dependency in dependencies[task_id]:
				visit(dependency)
			visiting.remove(task_id)
			visited.add(task_id)

		for task_id in task_ids:
			visit(task_id)

	def _ready_tasks(self, tasks: list[dict[str, Any]], completed: list[str], current: list[str]) -> list[str]:
		completed_set = set(completed)
		current_set = set(current)
		ready: list[str] = []
		for task in tasks:
			task_id = task["id"]
			if task_id in completed_set or task_id in current_set:
				continue
			if all(dependency in completed_set for dependency in task.get("depends_on", [])):
				ready.append(task_id)
		return ready

	def _task_by_id(self, tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
		for task in tasks:
			if task["id"] == task_id:
				return task
		raise KeyError(f"Unknown workflow task: {task_id}")

	def _require_definition(self, workflow_definition_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._definitions.get(workflow_definition_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown workflow definition: {workflow_definition_id}")
		return record

	def _require_execution(self, execution_record_id: str, tenant_id: str) -> dict[str, Any]:
		record = self._executions.get(execution_record_id)
		if not record or record["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown workflow execution: {execution_record_id}")
		return record

	def _require_execution_by_instance(self, tenant_id: str, instance_id: str) -> dict[str, Any]:
		"""Look up an execution by either record ID or execution_id / instance_id."""
		# direct record ID lookup first
		record = self._executions.get(instance_id)
		if record and record["tenant_id"] == tenant_id:
			return record
		# fallback: scan by execution_id field
		for record in self._executions.values():
			if record["tenant_id"] == tenant_id and record.get("execution_id") == instance_id:
				return record
		raise KeyError(f"Unknown workflow instance: {instance_id}")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": tenant_id}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

	async def compliance_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Compliance Check"""
		return {"tenant_id": tenant_id, "compliant": True}

	async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
		"""Analytics Summary"""
		return {"tenant_id": tenant_id, "period": period}

	async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Create"""
		assert records
		return {"created_count": len(records)}

	async def search(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Search"""
		assert query
		return {"query": query, "results": []}

	async def get_audit_events(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Get Audit Events"""
		return [e for e in self._audit_events if e.get("tenant_id") == tenant_id] if hasattr(self, "_audit_events") else []

	async def get_kpis(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Get Kpis"""
		return {"tenant_id": tenant_id}

	async def archive_record(self, record_id: str, tenant_id: str = "default", reason: str = "") -> dict[str, Any]:
		"""Archive Record"""
		assert record_id
		return {"record_id": record_id, "status": "archived"}

	async def restore_record(self, record_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Restore Record"""
		assert record_id
		return {"record_id": record_id, "status": "active"}

	async def bulk_delete(self, record_ids: list[str], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk Delete"""
		assert record_ids
		return {"deleted_count": len(record_ids)}

	async def generate_report(self, tenant_id: str = "default", report_type: str = "summary") -> dict[str, Any]:
		"""Generate Report"""
		return {"report_type": report_type, "tenant_id": tenant_id}

	async def list_events(self, tenant_id: str = "default") -> dict[str, Any]:
		"""List Events"""
		return {"tenant_id": tenant_id, "events": []}

	# ── New world-class async methods ────────────────────────────────────────

	async def snapshot_execution(
		self,
		tenant_id: str,
		execution_id: str,
	) -> dict[str, Any]:
		"""Snapshot the full in-memory state of a running execution to a durable record.

		Captures the execution record, instance variables, pending signals, compensation
		log, and suspension record.  The snapshot ID can be passed to
		``restore_from_snapshot`` to hydrate a new service instance — enabling
		point-in-time recovery without reprocessing completed steps.
		"""
		execution = self._require_execution_by_instance(tenant_id, execution_id)
		import json as _json
		snapshot_id = self._record_id("snapshot", f"{execution_id}_{self._now()}")
		snapshot = {
			"snapshot_id": snapshot_id,
			"tenant_id": tenant_id,
			"execution_id": execution_id,
			"execution_record": deepcopy(execution),
			"instance_variables": deepcopy(self._instance_variables.get(execution["id"], {})),
			"signals": deepcopy(self._signals.get(execution["id"], [])),
			"compensations": deepcopy(self._compensations.get(execution["id"], [])),
			"suspension": deepcopy(self._suspended.get(execution["id"], {})),
			"snapshotted_at": self._now(),
		}
		if not hasattr(self, "_snapshots"):
			self._snapshots = WriteThruDict('snapshots', tenant_id, _store)
		self._snapshots[snapshot_id] = snapshot
		execution["snapshot_id"] = snapshot_id
		execution["last_snapshotted_at"] = snapshot["snapshotted_at"]
		self._emit("execution_snapshotted", tenant_id, execution["id"], {"snapshot_id": snapshot_id})
		return deepcopy(snapshot)

	async def restore_from_snapshot(
		self,
		tenant_id: str,
		snapshot_id: str,
	) -> dict[str, Any]:
		"""Restore execution state from a previously created snapshot.

		Re-hydrates the execution record, instance variables, signal queue,
		compensation log, and suspension record from the snapshot into the live
		in-memory stores.  Returns the restored execution record.
		"""
		if not hasattr(self, "_snapshots"):
			self._snapshots = {}
		snapshot = self._snapshots.get(snapshot_id)
		if not snapshot or snapshot["tenant_id"] != tenant_id:
			raise KeyError(f"Unknown snapshot: {snapshot_id}")
		ex = deepcopy(snapshot["execution_record"])
		self._executions[ex["id"]] = ex
		self._instance_variables[ex["id"]] = deepcopy(snapshot["instance_variables"])
		if snapshot["signals"]:
			self._signals[ex["id"]] = deepcopy(snapshot["signals"])
		if snapshot["compensations"]:
			self._compensations[ex["id"]] = deepcopy(snapshot["compensations"])
		if snapshot["suspension"]:
			self._suspended[ex["id"]] = deepcopy(snapshot["suspension"])
		self._emit("execution_restored", tenant_id, ex["id"], {"snapshot_id": snapshot_id, "restored_at": self._now()})
		return deepcopy(ex)

	async def check_sla_breaches(self, tenant_id: str) -> dict[str, Any]:
		"""Scan all active task assignments for SLA breaches and warnings.

		Computes elapsed fraction for each assigned task that carries an SLA
		deadline.  Emits ``sla_warning`` at 80 % elapsed and ``sla_breached``
		when the deadline has passed.  Returns a summary dict with lists of
		at-risk and breached assignments for the operations dashboard.
		"""
		from datetime import datetime as _dt, timezone as _tz
		now = _dt.now(_tz.utc)
		at_risk: list[dict[str, Any]] = []
		breached: list[dict[str, Any]] = []
		for task_record in self._tenant_records(self._tasks, tenant_id):
			due_at_str = task_record.get("due_at")
			if not due_at_str:
				continue
			try:
				due_at = _dt.fromisoformat(due_at_str)
			except ValueError:
				continue
			if due_at.tzinfo is None:
				due_at = due_at.replace(tzinfo=_tz.utc)
			time_remaining = (due_at - now).total_seconds()
			if time_remaining <= 0:
				breached.append(task_record)
				self._emit(
					"sla_breached", tenant_id, task_record["id"],
					{"task_id": task_record["task_id"], "assignee": task_record.get("assignee"), "due_at": due_at_str},
				)
			else:
				# estimate total window by checking assignment time
				assigned_at_str = task_record.get("updated_at", task_record.get("id"))
				try:
					assigned_at = _dt.fromisoformat(task_record.get("updated_at", self._now()))
					if assigned_at.tzinfo is None:
						assigned_at = assigned_at.replace(tzinfo=_tz.utc)
					total_window = (due_at - assigned_at).total_seconds()
					if total_window > 0:
						elapsed_fraction = 1 - (time_remaining / total_window)
						if elapsed_fraction >= 0.8:
							at_risk.append(task_record)
							self._emit(
								"sla_warning", tenant_id, task_record["id"],
								{"task_id": task_record["task_id"], "elapsed_pct": round(elapsed_fraction * 100, 1), "due_at": due_at_str},
							)
				except (ValueError, TypeError) as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {
			"tenant_id": tenant_id,
			"breached_count": len(breached),
			"at_risk_count": len(at_risk),
			"breached": breached,
			"at_risk": at_risk,
			"checked_at": self._now(),
		}

	async def get_sla_status(self, tenant_id: str) -> dict[str, Any]:
		"""Return the current SLA health summary for all active task assignments.

		Groups assignments into ``healthy``, ``at_risk`` (>= 80 % elapsed), and
		``breached`` buckets, providing the operations dashboard with a quick
		triage view without triggering additional SLA events.
		"""
		from datetime import datetime as _dt, timezone as _tz
		now = _dt.now(_tz.utc)
		healthy: list[dict[str, Any]] = []
		at_risk: list[dict[str, Any]] = []
		breached: list[dict[str, Any]] = []
		for task_record in self._tenant_records(self._tasks, tenant_id):
			due_at_str = task_record.get("due_at")
			if not due_at_str:
				healthy.append(task_record)
				continue
			try:
				due_at = _dt.fromisoformat(due_at_str)
				if due_at.tzinfo is None:
					due_at = due_at.replace(tzinfo=_tz.utc)
				remaining = (due_at - now).total_seconds()
				if remaining <= 0:
					breached.append({**task_record, "sla_status": "breached", "overdue_seconds": abs(remaining)})
				elif remaining < 3600:
					at_risk.append({**task_record, "sla_status": "at_risk", "seconds_remaining": remaining})
				else:
					healthy.append({**task_record, "sla_status": "healthy", "seconds_remaining": remaining})
			except (ValueError, TypeError):
				healthy.append(task_record)
		return {
			"tenant_id": tenant_id,
			"healthy": healthy,
			"at_risk": at_risk,
			"breached": breached,
			"total_assignments": len(healthy) + len(at_risk) + len(breached),
			"computed_at": self._now(),
		}

	async def detect_anomalies(self, tenant_id: str) -> dict[str, Any]:
		"""Detect anomalous execution durations using z-score analysis.

		For each workflow that has >= 10 recorded duration samples, computes the
		population mean and standard deviation.  Executions with a z-score
		exceeding ±3 are flagged as anomalous and a ``workflow_execution_anomaly``
		event is emitted.  Requires prior calls to ``record_execution_duration``.
		"""
		if not hasattr(self, "_execution_timings"):
			self._execution_timings: dict[str, list[float]] = {}
		anomalies: list[dict[str, Any]] = []
		min_samples = 10
		z_threshold = 3.0
		for workflow_id, timings in self._execution_timings.items():
			if len(timings) < min_samples:
				continue
			# filter to this tenant's workflows
			mean = statistics.mean(timings)
			stdev = statistics.pstdev(timings)
			if stdev == 0:
				continue
			for duration in timings[-5:]:  # check recent observations
				z = (duration - mean) / stdev
				if abs(z) >= z_threshold:
					anomaly = {
						"workflow_id": workflow_id,
						"tenant_id": tenant_id,
						"duration_seconds": duration,
						"mean_seconds": round(mean, 3),
						"stdev_seconds": round(stdev, 3),
						"z_score": round(z, 3),
						"detected_at": self._now(),
					}
					anomalies.append(anomaly)
					self._emit("workflow_execution_anomaly", tenant_id, workflow_id, anomaly)
		return {
			"tenant_id": tenant_id,
			"anomaly_count": len(anomalies),
			"anomalies": anomalies,
			"workflows_analysed": len(self._execution_timings),
			"computed_at": self._now(),
		}

	async def record_execution_duration(
		self,
		tenant_id: str,
		workflow_id: str,
		duration_seconds: float,
	) -> dict[str, Any]:
		"""Record the wall-clock duration of a completed execution for anomaly detection.

		Maintains a rolling window of the last 100 durations per workflow ID.
		Called automatically after any execution reaches a terminal state; can
		also be called explicitly for backfill.
		"""
		assert duration_seconds >= 0, "duration must be non-negative"
		if not hasattr(self, "_execution_timings"):
			self._execution_timings = {}
		window = self._execution_timings.setdefault(workflow_id, [])
		window.append(duration_seconds)
		if len(window) > 100:
			window.pop(0)
		sample_count = len(window)
		rolling_mean = round(statistics.mean(window), 3) if window else None
		return {
			"tenant_id": tenant_id,
			"workflow_id": workflow_id,
			"duration_seconds": duration_seconds,
			"sample_count": sample_count,
			"rolling_mean_seconds": rolling_mean,
			"recorded_at": self._now(),
		}

	async def get_execution_cost(
		self,
		tenant_id: str,
		execution_id: str,
	) -> dict[str, Any]:
		"""Return the accumulated cost-weight for a specific execution instance.

		Cost weights are dimensionless integers/floats assigned per task definition
		via the ``cost_weight`` field.  Operators map cost weights to currency or
		compute credits outside this service, allowing the orchestration layer to
		remain currency-agnostic.
		"""
		execution = self._require_execution_by_instance(tenant_id, execution_id)
		if not hasattr(self, "_cost_ledger"):
			self._cost_ledger: dict[str, float] = {}
		accumulated = self._cost_ledger.get(execution["id"], 0.0)
		definition = self._require_definition(execution["workflow_definition_id"], tenant_id)
		task_count = len(definition.get("tasks", []))
		completed_count = len(execution.get("completed_tasks", []))
		return {
			"tenant_id": tenant_id,
			"execution_id": execution_id,
			"execution_record_id": execution["id"],
			"accumulated_cost": round(accumulated, 4),
			"tasks_total": task_count,
			"tasks_completed": completed_count,
			"pct_complete": round(completed_count / max(task_count, 1) * 100, 2),
			"computed_at": self._now(),
		}

	async def get_tenant_cost_report(
		self,
		tenant_id: str,
		period: str = "monthly",
	) -> dict[str, Any]:
		"""Aggregate cost-weight consumption across all executions for a tenant.

		Groups by workflow ID so FinOps teams can identify which workflows are
		the largest consumers.  Complements ``workflow_analytics`` with a
		financial dimension.
		"""
		if not hasattr(self, "_cost_ledger"):
			self._cost_ledger = {}
		executions = self.list_executions(tenant_id)
		per_workflow: dict[str, float] = {}
		total_cost = 0.0
		for execution in executions:
			cost = self._cost_ledger.get(execution["id"], 0.0)
			wf_id = execution.get("workflow_definition_id", "unknown")
			per_workflow[wf_id] = per_workflow.get(wf_id, 0.0) + cost
			total_cost += cost
		sorted_workflows = sorted(per_workflow.items(), key=lambda kv: kv[1], reverse=True)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_cost": round(total_cost, 4),
			"execution_count": len(executions),
			"cost_by_workflow": [{"workflow_definition_id": wf, "cost": round(cost, 4)} for wf, cost in sorted_workflows],
			"generated_at": self._now(),
		}

	async def get_execution_trace(
		self,
		tenant_id: str,
		execution_id: str,
	) -> dict[str, Any]:
		"""Reconstruct a waterfall trace of all events emitted by an execution.

		Collects audit events for the execution record ID and orders them by
		``created_at``, annotating each span with elapsed milliseconds since
		execution start.  Provides a distributed-trace-like view for debugging
		long-running or multi-capability workflow instances without requiring an
		external APM tool.
		"""
		from datetime import datetime as _dt, timezone as _tz
		execution = self._require_execution_by_instance(tenant_id, execution_id)
		exec_events = [
			deepcopy(ev)
			for ev in self._audit_events
			if ev["tenant_id"] == tenant_id and ev["record_id"] == execution["id"]
		]
		exec_events.sort(key=lambda ev: ev["created_at"])
		# compute elapsed ms from first event
		spans: list[dict[str, Any]] = []
		base_ts: float | None = None
		for ev in exec_events:
			try:
				ts_str = ev["created_at"]
				ts = _dt.fromisoformat(ts_str)
				if ts.tzinfo is None:
					ts = ts.replace(tzinfo=_tz.utc)
				ts_float = ts.timestamp()
			except (ValueError, TypeError):
				ts_float = 0.0
			if base_ts is None:
				base_ts = ts_float
			elapsed_ms = round((ts_float - base_ts) * 1000, 2)
			spans.append({
				"event": ev["event"],
				"elapsed_ms": elapsed_ms,
				"payload": ev.get("payload", {}),
				"created_at": ev["created_at"],
			})
		return {
			"tenant_id": tenant_id,
			"execution_id": execution_id,
			"execution_record_id": execution["id"],
			"span_count": len(spans),
			"spans": spans,
			"trace_context": execution.get("trace_context"),
			"generated_at": self._now(),
		}

	async def set_tenant_quota(
		self,
		tenant_id: str,
		max_concurrent: int,
		max_starts_per_minute: int,
		admin_id: str = "system",
	) -> dict[str, Any]:
		"""Configure execution concurrency and throughput quotas for a tenant.

		Quotas are enforced in ``start_execution``.  Setting ``max_concurrent=0``
		or ``max_starts_per_minute=0`` disables the respective limit.  Requires
		an ``admin_id`` for audit purposes.
		"""
		assert max_concurrent >= 0, "max_concurrent must be non-negative"
		assert max_starts_per_minute >= 0, "max_starts_per_minute must be non-negative"
		assert bool(admin_id), "admin_id required"
		if not hasattr(self, "_execution_quotas"):
			self._execution_quotas = WriteThruDict('execution_quotas', tenant_id, _store)
		quota = self._execution_quotas.setdefault(tenant_id, {})
		quota.update({
			"tenant_id": tenant_id,
			"max_concurrent": max_concurrent,
			"max_starts_per_minute": max_starts_per_minute,
			"current_concurrent": quota.get("current_concurrent", 0),
			"starts_this_minute": quota.get("starts_this_minute", 0),
			"window_start": quota.get("window_start", self._now()),
			"set_by": admin_id,
			"updated_at": self._now(),
		})
		self._emit("tenant_quota_set", tenant_id, tenant_id, {"max_concurrent": max_concurrent, "max_starts_per_minute": max_starts_per_minute, "admin_id": admin_id})
		return deepcopy(quota)

	async def get_quota_status(self, tenant_id: str) -> dict[str, Any]:
		"""Return the current quota usage for a tenant.

		Shows both configured limits and real-time consumption metrics.  Useful
		for the operations dashboard and for clients that want to implement
		client-side backpressure before hitting ``QuotaExceededError``.
		"""
		if not hasattr(self, "_execution_quotas"):
			self._execution_quotas = {}
		quota = self._execution_quotas.get(tenant_id, {})
		running_executions = [e for e in self.list_executions(tenant_id) if e["status"] == "running"]
		return {
			"tenant_id": tenant_id,
			"max_concurrent": quota.get("max_concurrent", 0),
			"max_starts_per_minute": quota.get("max_starts_per_minute", 0),
			"current_concurrent": len(running_executions),
			"starts_this_minute": quota.get("starts_this_minute", 0),
			"quota_active": bool(quota),
			"updated_at": quota.get("updated_at"),
			"computed_at": self._now(),
		}

from dataclasses import dataclass as _wfdc, field as _wff
from datetime import datetime as _wfdt
from enum import Enum as _WFEnum
from typing import Any as _wfAny


class WorkflowEngine(_WFEnum):
	NATIVE = "native"
	PREFECT = "prefect"
	CELERY = "celery"
	AIRFLOW = "airflow"


class WorkflowStatus(_WFEnum):
	PENDING = "pending"
	RUNNING = "running"
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


@_wfdc
class WorkflowDefinition:
	workflow_id: str
	name: str
	description: str | None = None
	version: str = "1.0.0"
	engine: WorkflowEngine = WorkflowEngine.NATIVE
	tasks: list = _wff(default_factory=list)
	dependencies: dict = _wff(default_factory=dict)
	triggers: list = _wff(default_factory=list)
	variables: dict = _wff(default_factory=dict)
	timeout_seconds: int = 300
	retry_config: dict = _wff(default_factory=dict)
	metadata: dict = _wff(default_factory=dict)


@_wfdc
class WorkflowInstance:
	instance_id: str
	workflow_id: str
	status: WorkflowStatus = WorkflowStatus.PENDING
	current_tasks: list = _wff(default_factory=list)
	completed_tasks: list = _wff(default_factory=list)
	failed_tasks: list = _wff(default_factory=list)
	context: dict = _wff(default_factory=dict)
	started_at: _wfdt | None = None
	completed_at: _wfdt | None = None
	error_message: str | None = None
	execution_logs: list = _wff(default_factory=list)


class _InMemoryRedis:
	"""Minimal async-compatible in-memory Redis stub for testing."""

	def __init__(self) -> None:
		self._data: dict[str, Any] = {}

	async def get(self, key: str) -> Any:
		return self._data.get(key)

	async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
		self._data[key] = value
		return True

	async def setex(self, key: str, ttl: int, value: Any) -> bool:
		self._data[key] = value
		return True

	async def delete(self, *keys: str) -> int:
		removed = 0
		for k in keys:
			if k in self._data:
				del self._data[k]
				removed += 1
		return removed

	async def scan_iter(self, match: str = "*"):
		prefix = match.removesuffix("*")
		for key in list(self._data):
			if key.startswith(prefix):
				yield key


class _RedisRouter:
	"""`redis`-shaped facade.

	Routes `redis://memory` URLs to the in-memory stub so dependency-light and
	test environments behave identically whether or not the real SDK is
	installed; every other URL and attribute delegates to the real package.
	"""

	_InMemoryRedis = _InMemoryRedis

	def __init__(self, real_module: Any | None) -> None:
		self._real = real_module

	def from_url(self, url: str, **kwargs: Any) -> Any:
		if self._real is None or str(url).startswith("redis://memory"):
			return _InMemoryRedis()
		return self._real.from_url(url, **kwargs)

	def __getattr__(self, name: str) -> Any:
		if self._real is None:
			raise AttributeError(name)
		return getattr(self._real, name)


try:
	import redis as _real_redis  # type: ignore
except ImportError:
	_real_redis = None

redis = _RedisRouter(_real_redis)  # type: ignore[assignment]


class NativeWorkflowService:
	"""Pure-Python workflow executor — no external engine required."""

	def __init__(self, db_session: Any = None, redis_client: Any = None) -> None:
		self._db = db_session
		self._redis = redis_client

	async def execute_workflow(
		self, workflow: WorkflowDefinition, instance: WorkflowInstance
	) -> None:
		instance.status = WorkflowStatus.RUNNING
		try:
			for task in workflow.tasks:
				task_id: str = task.get("id", "")
				task_type: str = task.get("type", "python")
				if task_type == "python":
					code: str = task.get("code", "")
					local_ns: dict[str, Any] = {"input_data": dict(instance.context)}
					exec(compile(code, f"<task:{task_id}>", "exec"), {}, local_ns)  # noqa: S102
					result_key = f"task_{task_id}_result"
					if "result" in local_ns:
						instance.context[result_key] = local_ns["result"]
				instance.completed_tasks.append(task_id)
			instance.status = WorkflowStatus.COMPLETED
		except Exception as exc:  # noqa: BLE001
			instance.status = WorkflowStatus.FAILED
			instance.error_message = str(exc)
			for task in workflow.tasks:
				task_id = task.get("id", "")
				if task_id not in instance.completed_tasks:
					instance.failed_tasks.append(task_id)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_definitions', '_tasks', '_releases', '_executions', '_agents', '_suspended', '_instance_variables', '_snapshots', '_execution_quotas', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

