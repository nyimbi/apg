"""Service layer for the Workflow Orchestration capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .workflow_runtime import (
	WorkflowApprovalRecord,
	WorkflowAuditEventRecord,
	WorkflowDefinitionRecord,
	WorkflowEventRecord,
	WorkflowExecutionRecord,
	WorkflowStepRecord,
	WorkflowTaskRecord,
	normalize_step_type,
	stable_id,
	utc_now,
	workflow_required_actions,
)


class WfloService:
	"""Deterministic workflow definition, execution, task, and approval service."""

	def __init__(self) -> None:
		self.definitions: dict[str, WorkflowDefinitionRecord] = {}
		self.executions: dict[str, WorkflowExecutionRecord] = {}
		self.tasks: dict[str, WorkflowTaskRecord] = {}
		self.approvals: dict[str, WorkflowApprovalRecord] = {}
		self.events: dict[str, WorkflowEventRecord] = {}
		self.audit_events: dict[str, WorkflowAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_workflow_definition(
		self,
		tenant_id: str,
		name: str,
		owner_ref: str,
		steps: list[dict[str, Any]],
		trigger_type: str = "manual",
		trigger_policy_ref: str = "",
		retry_policy_ref: str = "",
		compensation_ref: str = "",
		expected_runtime_minutes: int = 60,
		runtime_review_recorded: bool = False,
		version: int = 1,
		actor: str = "system",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("workflow_name_required")
		if not steps:
			raise ValueError("workflow_steps_required")
		normalized_steps = self._normalize_steps(tenant_id, name, steps)
		ai_step_present = any(step["step_type"] == "ai" for step in normalized_steps)
		context = {
			"tenant_context_present": True,
			"operation": "create_workflow",
			"workflow_owner_assigned": bool(str(owner_ref or "").strip()),
			"external_trigger": str(trigger_type or "").strip().lower() == "external",
			"trigger_policy_attached": bool(str(trigger_policy_ref or "").strip()),
			"ai_step_present": ai_step_present,
			"ai_policy_attached": all(bool(str(step.get("ai_policy_ref") or "").strip()) for step in normalized_steps if step["step_type"] == "ai"),
			"expected_runtime_minutes": int(expected_runtime_minutes),
			"runtime_review_recorded": bool(runtime_review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(retry_policy_ref or "").strip():
			raise PermissionError("retry_policy_required")
		status = "review_required" if result["decision"] == "require_review" else "draft"
		record = WorkflowDefinitionRecord(
			id=stable_id("wflo_definition", tenant_id, name, version),
			tenant_id=tenant_id,
			name=name,
			owner_ref=owner_ref,
			version=int(version),
			steps=normalized_steps,
			trigger_type=str(trigger_type or "manual").strip().lower(),
			trigger_policy_ref=trigger_policy_ref,
			retry_policy_ref=retry_policy_ref,
			compensation_ref=compensation_ref,
			expected_runtime_minutes=int(expected_runtime_minutes),
			status=status,
			required_actions=workflow_required_actions(result),
			matched_rules=list(result["matched_rules"]),
		)
		self.definitions[record.id] = record
		self._record_audit(tenant_id, "workflow_created", record.id, f"Workflow definition {status}: {name}", actor)
		return record.to_dict()

	def publish_workflow(
		self,
		tenant_id: str,
		definition_id: str,
		approval_ref: str,
		published_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		definition = self._get_definition(tenant_id, definition_id)
		context = {
			"tenant_context_present": True,
			"operation": "publish_workflow",
			"approval_recorded": bool(str(approval_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		definition.status = "published"
		definition.published_at = utc_now()
		definition.published_by = published_by
		self._record_audit(tenant_id, "workflow_published", definition.id, f"Workflow published: {definition.name}", published_by)
		return definition.to_dict()

	def start_execution(
		self,
		tenant_id: str,
		definition_id: str,
		correlation_id: str,
		started_by: str,
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		definition = self._get_definition(tenant_id, definition_id)
		if definition.status != "published":
			raise PermissionError(f"workflow_not_published:{definition.status}")
		if not str(correlation_id or "").strip():
			raise ValueError("correlation_id_required")
		record = WorkflowExecutionRecord(
			id=stable_id("wflo_execution", tenant_id, definition.id, correlation_id),
			tenant_id=tenant_id,
			definition_id=definition.id,
			correlation_id=correlation_id,
			started_by=started_by,
			current_step=definition.steps[0]["id"] if definition.steps else None,
			payload=dict(payload or {}),
		)
		self.executions[record.id] = record
		self.emit_event(tenant_id, record.id, "workflow_started", {"definition_id": definition.id, "correlation_id": correlation_id})
		self._record_audit(tenant_id, "execution_started", record.id, f"Workflow execution started: {definition.name}", started_by)
		return record.to_dict()

	def create_task(
		self,
		tenant_id: str,
		execution_id: str,
		step_id: str,
		title: str,
		assignee_ref: str,
		due_at: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		if not str(title or "").strip():
			raise ValueError("task_title_required")
		if not str(assignee_ref or "").strip():
			raise PermissionError("task_assignee_required")
		record = WorkflowTaskRecord(
			id=stable_id("wflo_task", tenant_id, execution.id, step_id, len(self.tasks)),
			tenant_id=tenant_id,
			execution_id=execution.id,
			step_id=step_id,
			title=title,
			assignee_ref=assignee_ref,
			due_at=due_at,
		)
		self.tasks[record.id] = record
		self.emit_event(tenant_id, execution.id, "task_created", {"task_id": record.id, "step_id": step_id})
		return record.to_dict()

	def complete_task(self, tenant_id: str, task_id: str, completed_by: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		task = self._get_task(tenant_id, task_id)
		if task.status == "completed":
			return task.to_dict()
		task.status = "completed"
		task.completed_at = utc_now()
		task.completed_by = completed_by
		self.emit_event(tenant_id, task.execution_id, "task_completed", {"task_id": task.id, "completed_by": completed_by})
		return task.to_dict()

	def request_approval(
		self,
		tenant_id: str,
		execution_id: str,
		subject_ref: str,
		approver_ref: str,
		reason: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		if not str(approver_ref or "").strip():
			raise PermissionError("approval_approver_required")
		record = WorkflowApprovalRecord(
			id=stable_id("wflo_approval", tenant_id, execution.id, subject_ref, len(self.approvals)),
			tenant_id=tenant_id,
			execution_id=execution.id,
			subject_ref=subject_ref,
			approver_ref=approver_ref,
			reason=reason,
		)
		self.approvals[record.id] = record
		execution.status = "waiting_approval"
		self.emit_event(tenant_id, execution.id, "approval_requested", {"approval_id": record.id})
		return record.to_dict()

	def record_approval(
		self,
		tenant_id: str,
		approval_id: str,
		decision: str,
		decision_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		approval = self._get_approval(tenant_id, approval_id)
		decision_value = str(decision or "").strip().lower()
		if decision_value not in {"approved", "rejected", "delegated"}:
			raise ValueError(f"unsupported_approval_decision:{decision}")
		approval.status = decision_value
		approval.decided_at = utc_now()
		approval.decision_by = decision_by
		execution = self._get_execution(tenant_id, approval.execution_id)
		execution.status = "running" if decision_value == "approved" else "failed"
		self.emit_event(tenant_id, execution.id, f"approval_{decision_value}", {"approval_id": approval.id})
		return approval.to_dict()

	def complete_execution(self, tenant_id: str, execution_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		if any(task.status != "completed" for task in self._tasks_for_execution(tenant_id, execution.id)):
			raise PermissionError("open_tasks_block_completion")
		if any(approval.status == "pending" for approval in self._approvals_for_execution(tenant_id, execution.id)):
			raise PermissionError("pending_approvals_block_completion")
		execution.status = "completed"
		execution.completed_at = utc_now()
		self.emit_event(tenant_id, execution.id, "workflow_completed", {"actor": actor})
		self._record_audit(tenant_id, "execution_completed", execution.id, "Workflow execution completed", actor)
		return execution.to_dict()

	def emit_event(
		self,
		tenant_id: str,
		execution_id: str,
		event_type: str,
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		record = WorkflowEventRecord(
			id=stable_id("wflo_event", tenant_id, execution.id, event_type, len(self.events)),
			tenant_id=tenant_id,
			execution_id=execution.id,
			event_type=event_type,
			payload=dict(payload or {}),
		)
		self.events[record.id] = record
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.create_workflow_definition(
			tenant_id=tenant_id,
			name=record_id,
			owner_ref=str(metadata.get("owner_ref") or "compatibility-owner"),
			steps=list(metadata.get("steps") or [{"name": "compatibility_task", "step_type": "human"}]),
			trigger_type=str(metadata.get("trigger_type") or "manual"),
			trigger_policy_ref=str(metadata.get("trigger_policy_ref") or ""),
			retry_policy_ref=str(metadata.get("retry_policy_ref") or "retry://compatibility"),
			compensation_ref=str(metadata.get("compensation_ref") or ""),
			expected_runtime_minutes=int(metadata.get("expected_runtime_minutes", 60)),
			runtime_review_recorded=bool(metadata.get("runtime_review_recorded", True)),
			actor=str(metadata.get("actor") or "compatibility"),
		)
		if status == "published":
			record = self.publish_workflow(tenant_id, record["id"], "approval://compatibility", str(metadata.get("actor") or "compatibility"))
		elif status != record["status"]:
			definition = self._get_definition(tenant_id, record["id"])
			definition.status = status
			record = definition.to_dict()
		return record

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_definitions(tenant_id)

	def list_definitions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.definitions, tenant_id)

	def list_executions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.executions, tenant_id)

	def list_tasks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.tasks, tenant_id)

	def list_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.approvals, tenant_id)

	def list_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.events, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		definitions = self.list_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		tasks = self.list_tasks(tenant_id)
		approvals = self.list_approvals(tenant_id)
		return {
			"tenant_id": tenant_id,
			"definition_count": len(definitions),
			"published_definition_count": sum(1 for item in definitions if item["status"] == "published"),
			"review_required_definition_count": sum(1 for item in definitions if item["status"] == "review_required"),
			"execution_count": len(executions),
			"running_execution_count": sum(1 for item in executions if item["status"] == "running"),
			"completed_execution_count": sum(1 for item in executions if item["status"] == "completed"),
			"open_task_count": sum(1 for item in tasks if item["status"] in {"open", "claimed"}),
			"pending_approval_count": sum(1 for item in approvals if item["status"] == "pending"),
			"event_count": len(self.list_events(tenant_id)),
			"recent_events": self.list_events(tenant_id)[-5:],
		}

	def _normalize_steps(self, tenant_id: str, workflow_name: str, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
		normalized: list[dict[str, Any]] = []
		for index, step in enumerate(steps):
			step_type = normalize_step_type(str(step.get("step_type") or step.get("type") or "human"))
			record = WorkflowStepRecord(
				id=str(step.get("id") or stable_id("wflo_step", tenant_id, workflow_name, index, step.get("name", "step"))),
				name=str(step.get("name") or f"step_{index + 1}"),
				step_type=step_type,
				assignee_ref=str(step.get("assignee_ref") or ""),
				sla_minutes=int(step.get("sla_minutes", 1440)),
				requires_approval=bool(step.get("requires_approval", step_type == "approval")),
				ai_policy_ref=str(step.get("ai_policy_ref") or ""),
			)
			normalized.append(record.to_dict())
		return normalized

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "workflow_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "workflow_policy_blocked")

	def _get_definition(self, tenant_id: str, definition_id: str) -> WorkflowDefinitionRecord:
		definition = self.definitions.get(definition_id)
		if definition is None:
			definition = next((item for item in self.definitions.values() if item.tenant_id == tenant_id and item.name == definition_id), None)
		if definition is None or definition.tenant_id != tenant_id:
			raise KeyError(f"workflow_definition_not_found:{definition_id}")
		return definition

	def _get_execution(self, tenant_id: str, execution_id: str) -> WorkflowExecutionRecord:
		execution = self.executions.get(execution_id)
		if execution is None or execution.tenant_id != tenant_id:
			raise KeyError(f"workflow_execution_not_found:{execution_id}")
		return execution

	def _get_task(self, tenant_id: str, task_id: str) -> WorkflowTaskRecord:
		task = self.tasks.get(task_id)
		if task is None or task.tenant_id != tenant_id:
			raise KeyError(f"workflow_task_not_found:{task_id}")
		return task

	def _get_approval(self, tenant_id: str, approval_id: str) -> WorkflowApprovalRecord:
		approval = self.approvals.get(approval_id)
		if approval is None or approval.tenant_id != tenant_id:
			raise KeyError(f"workflow_approval_not_found:{approval_id}")
		return approval

	def _tasks_for_execution(self, tenant_id: str, execution_id: str) -> list[WorkflowTaskRecord]:
		return [
			task
			for task in self.tasks.values()
			if task.tenant_id == tenant_id and task.execution_id == execution_id
		]

	def _approvals_for_execution(self, tenant_id: str, execution_id: str) -> list[WorkflowApprovalRecord]:
		return [
			approval
			for approval in self.approvals.values()
			if approval.tenant_id == tenant_id and approval.execution_id == execution_id
		]

	def _record_audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		record = WorkflowAuditEventRecord(
			id=stable_id("wflo_audit", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
