"""Domain service for the APG workflow orchestration capability."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

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

	def __init__(self) -> None:
		self._definitions: dict[str, dict[str, Any]] = {}
		self._tasks: dict[str, dict[str, Any]] = {}
		self._releases: dict[str, dict[str, Any]] = {}
		self._executions: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		definitions = self.list_workflow_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		return {
			"tenant_id": tenant_id,
			"workflow_definition_count": len(definitions),
			"released_workflow_count": len(self.list_releases(tenant_id)),
			"running_execution_count": len([item for item in executions if item["status"] == "running"]),
			"completed_execution_count": len([item for item in executions if item["status"] == "completed"]),
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
