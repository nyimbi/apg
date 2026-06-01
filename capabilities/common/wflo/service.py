"""Service layer for the Workflow Orchestration capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	DEFAULT_CONFIGURATION,
	PRIVILEGED_WFLO_AGENT_ROLES,
	SUPPORTED_WFLO_AGENT_ROLES,
	SUPPORTED_WFLO_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .workflow_runtime import (
	WorkflowApprovalRecord,
	WorkflowAuditEventRecord,
	WorkflowAgentRecord,
	WorkflowDefinitionRecord,
	WorkflowEventRecord,
	WorkflowExecutionRecord,
	WorkflowStepRecord,
	WorkflowTaskRecord,
	WfloLifecycleBatchRecord,
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
		self.agents: dict[str, WorkflowAgentRecord] = {}
		self.lifecycle_batches: dict[str, WfloLifecycleBatchRecord] = {}
		self.audit_events: dict[str, WorkflowAuditEventRecord] = {}
		self._agent_runtimes = {self._normalize_token(value) for value in SUPPORTED_WFLO_AGENT_RUNTIMES}
		self._agent_roles = {self._normalize_token(value) for value in SUPPORTED_WFLO_AGENT_ROLES}
		self._privileged_agent_roles = {self._normalize_token(value) for value in PRIVILEGED_WFLO_AGENT_ROLES}
		self._lifecycle_operations = {self._normalize_token(value) for value in DEFAULT_CONFIGURATION["streaming"]["required_operations"]}

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
		normalized_steps = self._normalize_steps(tenant_id, name, steps)
		step_ids = [step["id"] for step in normalized_steps]
		ai_step_present = any(step["step_type"] == "ai" for step in normalized_steps)
		automation_step_present = any(step["step_type"] == "automation" for step in normalized_steps)
		event_step_present = any(step["step_type"] == "event" for step in normalized_steps)
		context = {
			"tenant_context_present": True,
			"operation": "create_workflow",
			"workflow_owner_assigned": bool(str(owner_ref or "").strip()),
			"workflow_name_present": bool(str(name or "").strip()),
			"step_count": len(normalized_steps),
			"workflow_size_review_recorded": len(normalized_steps) <= DEFAULT_CONFIGURATION["definitions"]["max_steps_per_workflow"],
			"duplicate_step_ids_present": len(step_ids) != len(set(step_ids)),
			"retry_policy_attached": bool(str(retry_policy_ref or "").strip()),
			"external_trigger": str(trigger_type or "").strip().lower() == "external",
			"trigger_policy_attached": bool(str(trigger_policy_ref or "").strip()),
			"ai_step_present": ai_step_present,
			"ai_policy_attached": all(bool(str(step.get("ai_policy_ref") or "").strip()) for step in normalized_steps if step["step_type"] == "ai"),
			"automation_step_present": automation_step_present,
			"automation_policy_attached": all(bool(str(step.get("automation_policy_ref") or "").strip()) for step in normalized_steps if step["step_type"] == "automation"),
			"event_step_present": event_step_present,
			"event_policy_attached": all(bool(str(step.get("event_policy_ref") or "").strip()) for step in normalized_steps if step["step_type"] == "event"),
			"expected_runtime_minutes": int(expected_runtime_minutes),
			"runtime_review_recorded": bool(runtime_review_recorded),
			"state_change_requested": True,
			"audit_event_recorded": True,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
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
			decision=result["decision"],
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result, runtime_review_recorded),
		)
		self.definitions[record.id] = record
		self._record_audit(tenant_id, "workflow_created", record.id, f"Workflow definition {status}: {name}", actor, policy_result=result)
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
			"state_change_requested": True,
			"audit_event_recorded": True,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		definition.status = "published"
		definition.published_at = utc_now()
		definition.published_by = published_by
		definition.decision = "allow"
		definition.matched_rules = []
		definition.review_reasons = []
		definition.audit_evidence = {"required_actions": [], "reasons": [], "review_recorded": True}
		self._record_audit(tenant_id, "workflow_published", definition.id, f"Workflow published: {definition.name}", published_by, policy_result=result)
		return definition.to_dict()

	def retire_workflow(
		self,
		tenant_id: str,
		definition_id: str,
		approval_ref: str,
		retired_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		definition = self._get_definition(tenant_id, definition_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "retire_workflow",
			"approval_recorded": bool(str(approval_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		definition.status = "retired"
		self._record_audit(tenant_id, "workflow_retired", definition.id, f"Workflow retired: {definition.name}", retired_by, policy_result=result)
		return definition.to_dict()

	def start_execution(
		self,
		tenant_id: str,
		definition_id: str,
		correlation_id: str,
		started_by: str,
		payload: dict[str, Any] | None = None,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		definition = self._get_definition(tenant_id, definition_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "start_execution",
			"definition_published": definition.status == "published",
			"correlation_id_present": bool(str(correlation_id or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = WorkflowExecutionRecord(
			id=stable_id("wflo_execution", tenant_id, definition.id, correlation_id),
			tenant_id=tenant_id,
			definition_id=definition.id,
			correlation_id=correlation_id,
			started_by=started_by,
			current_step=definition.steps[0]["id"] if definition.steps else None,
			payload=dict(payload or {}),
			event_stream=event_stream,
			compensation_status="available" if definition.compensation_ref else "not_required",
		)
		self.executions[record.id] = record
		self.emit_event(tenant_id, record.id, "workflow_started", {"definition_id": definition.id, "correlation_id": correlation_id}, event_stream=event_stream)
		self._record_audit(tenant_id, "execution_started", record.id, f"Workflow execution started: {definition.name}", started_by, policy_result=result)
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
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "create_task",
			"task_assignee_present": bool(str(assignee_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
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

	def claim_task(self, tenant_id: str, task_id: str, claimed_by: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		task = self._get_task(tenant_id, task_id)
		if task.status == "completed":
			raise PermissionError("task_already_completed")
		if not str(claimed_by or "").strip():
			raise PermissionError("task_claim_actor_required")
		task.status = "claimed"
		task.claimed_by = claimed_by
		task.claimed_at = utc_now()
		self.emit_event(tenant_id, task.execution_id, "task_claimed", {"task_id": task.id, "claimed_by": claimed_by})
		return task.to_dict()

	def complete_task(self, tenant_id: str, task_id: str, completed_by: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		task = self._get_task(tenant_id, task_id)
		if task.status == "completed":
			return task.to_dict()
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "complete_task",
			"task_claimed": bool(task.claimed_by),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		task.status = "completed"
		task.completed_at = utc_now()
		task.completed_by = completed_by
		self.emit_event(tenant_id, task.execution_id, "task_completed", {"task_id": task.id, "completed_by": completed_by})
		return task.to_dict()

	def escalate_task(self, tenant_id: str, task_id: str, escalated_by: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		task = self._get_task(tenant_id, task_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "escalate_task",
			"escalation_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		task.status = "escalated"
		task.escalated_at = utc_now()
		task.escalation_reason = reason
		self.emit_event(tenant_id, task.execution_id, "task_escalated", {"task_id": task.id, "escalated_by": escalated_by, "reason": reason})
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
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "request_approval",
			"approver_present": bool(str(approver_ref or "").strip()),
			"approval_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = WorkflowApprovalRecord(
			id=stable_id("wflo_approval", tenant_id, execution.id, subject_ref, len(self.approvals)),
			tenant_id=tenant_id,
			execution_id=execution.id,
			subject_ref=subject_ref,
			approver_ref=approver_ref,
			reason=reason,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result),
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
		decision_evidence_ref: str = "",
		delegated_to: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		approval = self._get_approval(tenant_id, approval_id)
		decision_value = str(decision or "").strip().lower()
		if decision_value not in {"approved", "rejected", "delegated"}:
			raise ValueError(f"unsupported_approval_decision:{decision}")
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "record_approval",
			"decision_evidence_present": bool(str(decision_evidence_ref or "").strip()),
			"approval_delegated": decision_value == "delegated",
			"delegate_present": bool(str(delegated_to or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		approval.status = decision_value
		approval.decided_at = utc_now()
		approval.decision_by = decision_by
		approval.decision_evidence_ref = decision_evidence_ref
		approval.delegated_to = delegated_to
		approval.decision = result["decision"]
		approval.matched_rules = list(result["matched_rules"])
		approval.review_reasons = self._review_reasons(result)
		approval.audit_evidence = self._audit_evidence(result, True)
		execution = self._get_execution(tenant_id, approval.execution_id)
		execution.status = "running" if decision_value in {"approved", "delegated"} else "failed"
		self.emit_event(tenant_id, execution.id, f"approval_{decision_value}", {"approval_id": approval.id})
		return approval.to_dict()

	def complete_execution(self, tenant_id: str, execution_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "complete_execution",
			"open_tasks_present": any(task.status != "completed" for task in self._tasks_for_execution(tenant_id, execution.id)),
			"pending_approvals_present": any(approval.status == "pending" for approval in self._approvals_for_execution(tenant_id, execution.id)),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		execution.status = "completed"
		execution.completed_at = utc_now()
		self.emit_event(tenant_id, execution.id, "workflow_completed", {"actor": actor})
		self._record_audit(tenant_id, "execution_completed", execution.id, "Workflow execution completed", actor)
		return execution.to_dict()

	def cancel_execution(self, tenant_id: str, execution_id: str, actor: str, reason: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "change_execution_state",
			"state_change_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		execution.status = "cancelled"
		execution.cancel_reason = reason
		execution.completed_at = utc_now()
		self.emit_event(tenant_id, execution.id, "workflow_cancelled", {"actor": actor, "reason": reason})
		self._record_audit(tenant_id, "execution_cancelled", execution.id, "Workflow execution cancelled", actor, severity="medium")
		return execution.to_dict()

	def fail_execution(
		self,
		tenant_id: str,
		execution_id: str,
		actor: str,
		reason: str,
		compensation_requested: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "change_execution_state",
			"state_change_reason_present": bool(str(reason or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		execution.status = "failed"
		execution.failure_reason = reason
		execution.completed_at = utc_now()
		if compensation_requested:
			execution.compensation_status = "requested"
		self.emit_event(tenant_id, execution.id, "workflow_failed", {"actor": actor, "reason": reason, "compensation_requested": compensation_requested})
		self._record_audit(tenant_id, "execution_failed", execution.id, "Workflow execution failed", actor, severity="high")
		return execution.to_dict()

	def run_compensation(self, tenant_id: str, execution_id: str, actor: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		definition = self._get_definition(tenant_id, execution.definition_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "run_compensation",
			"compensation_plan_present": bool(str(definition.compensation_ref or "").strip()),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		execution.compensation_status = "completed"
		self.emit_event(tenant_id, execution.id, "compensation_completed", {"actor": actor, "compensation_ref": definition.compensation_ref})
		self._record_audit(tenant_id, "compensation_completed", execution.id, "Workflow compensation completed", actor, severity="medium")
		return execution.to_dict()

	def emit_event(
		self,
		tenant_id: str,
		execution_id: str,
		event_type: str,
		payload: dict[str, Any] | None = None,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "emit_event",
			"event_stream": event_stream,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = WorkflowEventRecord(
			id=stable_id("wflo_event", tenant_id, execution.id, event_type, len(self.events)),
			tenant_id=tenant_id,
			execution_id=execution.id,
			event_type=event_type,
			payload=dict(payload or {}),
		)
		self.events[record.id] = record
		return record.to_dict()

	def register_workflow_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope_ref: str,
		registered_by: str,
		contribution_disclosed: bool,
		owner_ref: str = "",
		purpose: str = "",
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		owner_value = str(owner_ref or "").strip()
		purpose_value = str(purpose or "").strip()
		approval_recorded = self._coerce_bool(human_approval_required)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_workflow_agent",
			"agent_id_present": bool(str(agent_id or "").strip()),
			"agent_name_present": bool(str(name or "").strip()),
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"agent_scope_present": bool(str(scope_ref or "").strip()),
			"agent_owner_present": bool(owner_value),
			"agent_purpose_present": bool(purpose_value),
			"agent_contribution_disclosed": self._coerce_bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": approval_recorded,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = WorkflowAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope_ref=scope_ref,
			registered_by=registered_by,
			contribution_disclosed=self._coerce_bool(contribution_disclosed),
			owner_ref=owner_value,
			purpose=purpose_value,
			human_approval_required=approval_recorded,
			status="pending_review" if result["decision"] == "require_review" else "active",
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result, approval_recorded),
		)
		self.agents[self._tenant_key(tenant_id, agent_id)] = record
		self._record_audit(tenant_id, "workflow_agent_registered", agent_id, f"Workflow agent registered: {name}", registered_by, policy_result=result)
		return record.to_dict()

	def validate_batch_mutation(self, event_stream: str) -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": True, "operation": "batch_workflow_mutation", "event_stream": event_stream})
		if result["decision"] == "deny":
			self._raise_policy(result)
		return result

	def validate_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "workflow_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		operation_value = self._normalize_token(operation)
		result = self.evaluate({
			"tenant_context_present": True,
			"operation": "validate_wflo_lifecycle_batch",
			"event_stream": self._normalize_token(event_stream),
			"mutation_count": int(mutation_count),
			"lifecycle_operation_supported": operation_value in self._lifecycle_operations,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		record_id = batch_id or stable_id("wflo_lifecycle_batch", tenant_id, operation_value, len(self.lifecycle_batches))
		record = WfloLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=self._normalize_token(event_stream),
			operation=operation_value,
			mutation_count=int(mutation_count),
			status="denied" if result["decision"] == "deny" else "accepted" if result["decision"] == "allow" else "review_required",
			matched_rules=list(result["matched_rules"]),
			required_actions=workflow_required_actions(result),
			decision=result["decision"],
			review_reasons=self._review_reasons(result),
			audit_evidence=self._audit_evidence(result),
		)
		self.lifecycle_batches[self._tenant_key(tenant_id, record_id)] = record
		self._record_audit(tenant_id, "wflo_lifecycle_batch_validated", record.id, f"WFLO lifecycle batch {record.status}: {operation_value}", "wflo", policy_result=result)
		if result["decision"] == "deny":
			self._raise_policy(result)
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

	def list_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.lifecycle_batches, tenant_id)

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [
			item
			for item in (
				self.list_definitions(tenant_id)
				+ self.list_agents(tenant_id)
				+ self.list_lifecycle_batches(tenant_id)
			)
			if item.get("status") == "review_required" or item.get("status") == "pending_review"
		]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		definitions = self.list_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		tasks = self.list_tasks(tenant_id)
		approvals = self.list_approvals(tenant_id)
		pending_reviews = self.list_pending_reviews(tenant_id)
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
			"agent_count": len(self.list_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_agents(tenant_id) if item["status"] == "pending_review"),
			"pending_review_count": len(pending_reviews),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] not in {"accepted", "review_required"}),
			"cancelled_execution_count": sum(1 for item in executions if item["status"] == "cancelled"),
			"failed_execution_count": sum(1 for item in executions if item["status"] == "failed"),
			"event_count": len(self.list_events(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
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
				automation_policy_ref=str(step.get("automation_policy_ref") or ""),
				event_policy_ref=str(step.get("event_policy_ref") or ""),
				compensation_ref=str(step.get("compensation_ref") or ""),
			)
			normalized.append(record.to_dict())
		return normalized

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _tenant_key(self, tenant_id: str, record_id: str) -> str:
		return f"{str(tenant_id or '').strip()}::{str(record_id or '').strip()}"

	def _normalize_token(self, value: object) -> str:
		return str(value or "").strip().lower()

	def _coerce_bool(self, value: object) -> bool:
		if isinstance(value, bool):
			return value
		if value is None:
			return False
		if isinstance(value, str):
			return value.strip().lower() in {"1", "true", "yes", "y", "on"}
		return bool(value)

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
		policy_result: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		policy_result = policy_result or {"decision": "allow", "matched_rules": [], "actions": []}
		record = WorkflowAuditEventRecord(
			id=stable_id("wflo_audit", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			policy_decision=policy_result["decision"],
			matched_rules=list(policy_result["matched_rules"]),
			review_reasons=self._review_reasons(policy_result),
			audit_evidence=self._audit_evidence(policy_result),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _review_reasons(self, result: dict[str, Any]) -> list[str]:
		if result["decision"] == "allow":
			return []
		return [
			action.get("reason", "workflow_policy_blocked")
			for action in result.get("actions", [])
		]

	def _audit_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": workflow_required_actions(result),
			"reasons": [
				action.get("reason", "workflow_policy_blocked")
				for action in result.get("actions", [])
			],
			"review_recorded": bool(review_recorded),
		}

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
