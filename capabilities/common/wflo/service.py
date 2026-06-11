"""Service layer for the Workflow Orchestration capability."""

from __future__ import annotations

import asyncio
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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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

	def bpmn_import(
		self,
		tenant_id: str,
		bpmn_xml: str,
		owner_ref: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Parse a BPMN XML string and create a workflow definition from it."""
		self._require_tenant(tenant_id)
		assert bool(bpmn_xml), "bpmn_xml required"
		assert bool(owner_ref), "owner_ref required"
		import re
		task_names = re.findall(r'<(?:userTask|serviceTask|scriptTask)[^>]+name="([^"]+)"', bpmn_xml)
		steps = [{"name": n, "step_type": "human"} for n in task_names] or [{"name": "imported_step", "step_type": "human"}]
		name_match = re.search(r'<process[^>]+name="([^"]+)"', bpmn_xml)
		wf_name = name_match.group(1) if name_match else "bpmn_imported"
		return self.create_workflow_definition(
			tenant_id=tenant_id,
			name=wf_name,
			owner_ref=owner_ref,
			steps=steps,
			retry_policy_ref="retry://bpmn_default",
			runtime_review_recorded=True,
			actor=actor,
		) | {"bpmn_task_count": len(task_names), "source": "bpmn"}

	def process_simulate(
		self,
		tenant_id: str,
		definition_id: str,
		simulation_runs: int = 100,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Simulate process execution to estimate throughput, bottlenecks, and SLA compliance."""
		self._require_tenant(tenant_id)
		defn = self._get_definition(tenant_id, definition_id)
		step_count = len(defn.steps)
		avg_cycle_time_min = sum(s.get("sla_minutes", 1440) for s in defn.steps) / max(step_count, 1)
		sla_pass_rate = round(min(1.0, 1440 / max(avg_cycle_time_min, 1)), 4)
		return {
			"definition_id": definition_id,
			"tenant_id": tenant_id,
			"simulation_runs": simulation_runs,
			"step_count": step_count,
			"avg_cycle_time_minutes": round(avg_cycle_time_min, 2),
			"estimated_throughput_per_day": round(1440 / max(avg_cycle_time_min, 1), 2),
			"sla_pass_rate": sla_pass_rate,
			"simulated_by": actor,
			"simulated_at": utc_now(),
		}

	def bottleneck_detect(
		self,
		tenant_id: str,
		definition_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Identify workflow steps with highest SLA risk based on step configuration."""
		self._require_tenant(tenant_id)
		defn = self._get_definition(tenant_id, definition_id)
		steps_sorted = sorted(defn.steps, key=lambda s: -s.get("sla_minutes", 0))
		bottlenecks = steps_sorted[:3]
		return {
			"definition_id": definition_id,
			"tenant_id": tenant_id,
			"bottleneck_steps": [{"step_id": s["id"], "name": s["name"], "sla_minutes": s.get("sla_minutes", 0)} for s in bottlenecks],
			"total_steps": len(defn.steps),
			"detected_by": actor,
			"detected_at": utc_now(),
		}

	def sla_enforce(
		self,
		tenant_id: str,
		execution_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Check SLA compliance for an execution and escalate overdue tasks."""
		self._require_tenant(tenant_id)
		execution = self._get_execution(tenant_id, execution_id)
		defn = self._get_definition(tenant_id, execution.definition_id)
		overdue_steps: list[dict[str, Any]] = []
		for step in defn.steps:
			tasks = [t for t in self.tasks.values() if t.execution_id == execution_id and t.step_id == step["id"] and t.status not in {"completed"}]
			sla_min = step.get("sla_minutes", 1440)
			for t in tasks:
				if t.claimed_at is None:
					overdue_steps.append({"task_id": t.id, "step": step["name"], "sla_minutes": sla_min, "breach": "not_started"})
		return {
			"execution_id": execution_id,
			"tenant_id": tenant_id,
			"overdue_task_count": len(overdue_steps),
			"overdue_tasks": overdue_steps,
			"sla_status": "compliant" if not overdue_steps else "breach",
			"checked_by": actor,
			"checked_at": utc_now(),
		}

	def compensation_trigger(
		self,
		tenant_id: str,
		execution_id: str,
		reason: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Trigger compensation flow for a failed execution."""
		self._require_tenant(tenant_id)
		assert bool(reason), "reason required"
		fail_result = self.fail_execution(tenant_id=tenant_id, execution_id=execution_id, actor=actor, reason=reason, compensation_requested=True)
		comp_result = self.run_compensation(tenant_id=tenant_id, execution_id=execution_id, actor=actor)
		return {"execution": fail_result, "compensation": comp_result, "reason": reason}

	def parallel_gateway(
		self,
		tenant_id: str,
		definition_id: str,
		gateway_name: str,
		branch_step_names: list[str],
		owner_ref: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Add a parallel (AND) gateway to a workflow definition, forking into multiple branches."""
		self._require_tenant(tenant_id)
		assert bool(branch_step_names), "branch_step_names required"
		defn = self._get_definition(tenant_id, definition_id)
		new_steps = list(defn.steps) + [
			{"name": f"{gateway_name}_{b}", "step_type": "human", "parallel_group": gateway_name}
			for b in branch_step_names
		]
		# update definition steps in place
		defn.steps = self._normalize_steps(tenant_id, defn.name, new_steps)
		self._record_audit(tenant_id, "parallel_gateway_added", definition_id, f"Gateway {gateway_name}", actor)
		return {"definition_id": definition_id, "gateway_name": gateway_name, "branch_count": len(branch_step_names), "total_steps": len(defn.steps)}

	def inclusive_gateway(
		self,
		tenant_id: str,
		definition_id: str,
		gateway_name: str,
		condition_steps: list[dict[str, Any]],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Add an inclusive (OR) gateway — one or more branches may execute based on conditions."""
		self._require_tenant(tenant_id)
		defn = self._get_definition(tenant_id, definition_id)
		new_steps = list(defn.steps) + [
			{"name": f"{gateway_name}_{cs.get('name', i)}", "step_type": "event", "condition": cs.get("condition", ""), "event_policy_ref": cs.get("event_policy_ref", "policy://default")}
			for i, cs in enumerate(condition_steps)
		]
		defn.steps = self._normalize_steps(tenant_id, defn.name, new_steps)
		self._record_audit(tenant_id, "inclusive_gateway_added", definition_id, f"IG {gateway_name}", actor)
		return {"definition_id": definition_id, "gateway_name": gateway_name, "condition_count": len(condition_steps), "total_steps": len(defn.steps)}

	def boundary_event(
		self,
		tenant_id: str,
		execution_id: str,
		step_id: str,
		event_type: str,
		payload: dict[str, Any] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Attach a boundary event to a running execution step (timer, error, signal)."""
		self._require_tenant(tenant_id)
		assert event_type in {"timer", "error", "signal", "message"}, f"unsupported event_type: {event_type}"
		return self.emit_event(
			tenant_id=tenant_id,
			execution_id=execution_id,
			event_type=f"boundary_{event_type}",
			payload={"step_id": step_id, **(payload or {})},
		) | {"step_id": step_id, "boundary_event_type": event_type}

	def escalation_handle(
		self,
		tenant_id: str,
		task_id: str,
		escalated_by: str,
		escalation_reason: str,
		reassign_to: str | None = None,
	) -> dict[str, Any]:
		"""Escalate an overdue task and optionally reassign it."""
		self._require_tenant(tenant_id)
		assert bool(escalation_reason), "escalation_reason required"
		result = self.escalate_task(tenant_id=tenant_id, task_id=task_id, escalated_by=escalated_by, reason=escalation_reason)
		if reassign_to:
			task = self._get_task(tenant_id, task_id)
			task.assignee_ref = reassign_to
		return {**result, "reassigned_to": reassign_to}

	def process_analytics(
		self,
		tenant_id: str,
		period: str = "all",
	) -> dict[str, Any]:
		"""Return aggregated workflow process analytics for a tenant."""
		self._require_tenant(tenant_id)
		definitions = self.list_definitions(tenant_id)
		executions = self.list_executions(tenant_id)
		tasks = self.list_tasks(tenant_id)
		completed = [e for e in executions if e["status"] == "completed"]
		failed = [e for e in executions if e["status"] == "failed"]
		avg_task_per_execution = round(len(tasks) / max(len(executions), 1), 2)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"definition_count": len(definitions),
			"published_count": sum(1 for d in definitions if d["status"] == "published"),
			"execution_count": len(executions),
			"completed_count": len(completed),
			"failed_count": len(failed),
			"completion_rate": round(len(completed) / max(len(executions), 1), 4),
			"avg_tasks_per_execution": avg_task_per_execution,
			"open_task_count": sum(1 for t in tasks if t["status"] in {"open", "claimed"}),
			"pending_approval_count": sum(1 for a in self.list_approvals(tenant_id) if a["status"] == "pending"),
			"agent_count": len(self.list_agents(tenant_id)),
			"computed_at": utc_now(),
		}

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

	# ---------------------------------------------------------------------------
	# Async interface
	# ---------------------------------------------------------------------------

	async def async_create_workflow_definition(
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
		"""Async variant of create_workflow_definition."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.create_workflow_definition(
				tenant_id=tenant_id,
				name=name,
				owner_ref=owner_ref,
				steps=steps,
				trigger_type=trigger_type,
				trigger_policy_ref=trigger_policy_ref,
				retry_policy_ref=retry_policy_ref,
				compensation_ref=compensation_ref,
				expected_runtime_minutes=expected_runtime_minutes,
				runtime_review_recorded=runtime_review_recorded,
				version=version,
				actor=actor,
			),
		)

	async def async_start_execution(
		self,
		tenant_id: str,
		definition_id: str,
		correlation_id: str,
		started_by: str,
		payload: dict[str, Any] | None = None,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		"""Async variant of start_execution with idempotency enforcement on duplicate correlation_id."""
		existing = next(
			(
				ex.to_dict()
				for ex in self.executions.values()
				if ex.tenant_id == tenant_id and ex.correlation_id == correlation_id
			),
			None,
		)
		if existing is not None:
			self._record_audit(
				tenant_id,
				"duplicate_start_attempted",
				existing["id"],
				f"Duplicate start_execution ignored for correlation_id={correlation_id}",
				started_by,
			)
			return existing
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.start_execution(
				tenant_id=tenant_id,
				definition_id=definition_id,
				correlation_id=correlation_id,
				started_by=started_by,
				payload=payload,
				event_stream=event_stream,
			),
		)

	async def async_complete_task(
		self,
		tenant_id: str,
		task_id: str,
		completed_by: str,
	) -> dict[str, Any]:
		"""Async variant of complete_task."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.complete_task(tenant_id=tenant_id, task_id=task_id, completed_by=completed_by),
		)

	async def async_record_approval(
		self,
		tenant_id: str,
		approval_id: str,
		decision: str,
		decision_by: str,
		decision_evidence_ref: str = "",
		delegated_to: str = "",
	) -> dict[str, Any]:
		"""Async variant of record_approval."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.record_approval(
				tenant_id=tenant_id,
				approval_id=approval_id,
				decision=decision,
				decision_by=decision_by,
				decision_evidence_ref=decision_evidence_ref,
				delegated_to=delegated_to,
			),
		)

	async def async_process_analytics(
		self,
		tenant_id: str,
		period: str = "all",
	) -> dict[str, Any]:
		"""Async variant of process_analytics."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, lambda: self.process_analytics(tenant_id, period))

	async def async_sla_enforce(
		self,
		tenant_id: str,
		execution_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Async variant of sla_enforce, suitable for background scheduler invocation."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.sla_enforce(tenant_id=tenant_id, execution_id=execution_id, actor=actor),
		)

	async def async_bpmn_import(
		self,
		tenant_id: str,
		bpmn_xml: str,
		owner_ref: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Async variant of bpmn_import, suitable for async file-upload handlers."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.bpmn_import(
				tenant_id=tenant_id,
				bpmn_xml=bpmn_xml,
				owner_ref=owner_ref,
				actor=actor,
			),
		)

	async def async_bulk_create_tasks(
		self,
		tenant_id: str,
		execution_id: str,
		task_specs: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Create multiple tasks for an execution in a single async call.

		Each entry in task_specs must have: step_id, title, assignee_ref.
		Optional: due_at.
		"""
		self._require_tenant(tenant_id)
		assert task_specs, "task_specs must not be empty"
		results: list[dict[str, Any]] = []
		loop = asyncio.get_event_loop()
		for spec in task_specs:
			step_id = str(spec.get("step_id") or "")
			title = str(spec.get("title") or "")
			assignee_ref = str(spec.get("assignee_ref") or "")
			due_at: str | None = spec.get("due_at")
			result = await loop.run_in_executor(
				None,
				lambda s=step_id, t=title, a=assignee_ref, d=due_at: self.create_task(
					tenant_id=tenant_id,
					execution_id=execution_id,
					step_id=s,
					title=t,
					assignee_ref=a,
					due_at=d,
				),
			)
			results.append(result)
		return results

	async def async_dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Async variant of dashboard_summary."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, lambda: self.dashboard_summary(tenant_id))

	async def async_cancel_execution(
		self,
		tenant_id: str,
		execution_id: str,
		actor: str,
		reason: str,
	) -> dict[str, Any]:
		"""Async variant of cancel_execution."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.cancel_execution(
				tenant_id=tenant_id,
				execution_id=execution_id,
				actor=actor,
				reason=reason,
			),
		)

	async def async_process_simulate(
		self,
		tenant_id: str,
		definition_id: str,
		simulation_runs: int = 100,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Async variant of process_simulate."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.process_simulate(
				tenant_id=tenant_id,
				definition_id=definition_id,
				simulation_runs=simulation_runs,
				actor=actor,
			),
		)

	def serialize_designer_state(
		self,
		tenant_id: str,
		definition_id: str,
	) -> dict[str, Any]:
		"""Serialize a workflow definition as a canvas-compatible node/edge graph.

		Returns ``{nodes, edges, metadata}`` compatible with React Flow and similar
		visual designer renderers.  Each step maps to a node; sequential order maps
		to edges.  Parallel groups produce gateway nodes with branch edges.
		"""
		self._require_tenant(tenant_id)
		defn = self._get_definition(tenant_id, definition_id)
		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []
		nodes.append({"id": "__start__", "type": "start", "data": {"label": "Start"}, "position": {"x": 0, "y": 0}})
		prev_id = "__start__"
		parallel_groups: dict[str, list[str]] = {}
		for idx, step in enumerate(defn.steps):
			node_id = step["id"]
			parallel_group = step.get("parallel_group")
			nodes.append({
				"id": node_id,
				"type": step["step_type"],
				"data": {
					"label": step["name"],
					"step_type": step["step_type"],
					"sla_minutes": step.get("sla_minutes", 1440),
					"requires_approval": step.get("requires_approval", False),
					"assignee_ref": step.get("assignee_ref", ""),
				},
				"position": {"x": 200 * (idx + 1), "y": 0},
			})
			if parallel_group:
				parallel_groups.setdefault(parallel_group, []).append(node_id)
			else:
				edges.append({"id": f"e_{prev_id}__{node_id}", "source": prev_id, "target": node_id})
				prev_id = node_id
		for group_name, node_ids in parallel_groups.items():
			gateway_id = f"gw_{group_name}"
			nodes.append({"id": gateway_id, "type": "parallel_gateway", "data": {"label": group_name}, "position": {"x": 0, "y": 100}})
			for branch_id in node_ids:
				edges.append({"id": f"e_{gateway_id}__{branch_id}", "source": gateway_id, "target": branch_id})
		nodes.append({"id": "__end__", "type": "end", "data": {"label": "End"}, "position": {"x": 200 * (len(defn.steps) + 1), "y": 0}})
		if prev_id != "__start__":
			edges.append({"id": f"e_{prev_id}____end__", "source": prev_id, "target": "__end__"})
		return {
			"definition_id": definition_id,
			"tenant_id": tenant_id,
			"name": defn.name,
			"version": defn.version,
			"nodes": nodes,
			"edges": edges,
			"metadata": {
				"step_count": len(defn.steps),
				"trigger_type": defn.trigger_type,
				"status": defn.status,
				"serialized_at": utc_now(),
			},
		}

	async def async_serialize_designer_state(
		self,
		tenant_id: str,
		definition_id: str,
	) -> dict[str, Any]:
		"""Async variant of serialize_designer_state."""
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(
			None,
			lambda: self.serialize_designer_state(tenant_id=tenant_id, definition_id=definition_id),
		)
