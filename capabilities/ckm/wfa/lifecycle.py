"""Dependency-light lifecycle surface for the CKM WFA capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .capability_contract import (
	SUPPORTED_TASK_TYPES,
	SUPPORTED_WFA_AGENT_ROLES,
	SUPPORTED_WFA_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


@dataclass
class WfaProcess:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	version: str
	trigger: str
	variable_schema: dict[str, Any]
	status: str = "draft"
	approved: bool = False
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner_id": self.owner_id,
			"version": self.version,
			"trigger": self.trigger,
			"variable_schema": dict(self.variable_schema),
			"status": self.status,
			"approved": self.approved,
			"created_at": self.created_at,
		}


@dataclass
class WfaProcessInstance:
	id: str
	tenant_id: str
	process_id: str
	initiated_by: str
	context: dict[str, Any]
	status: str = "running"
	correlation_key: str | None = None
	started_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"process_id": self.process_id,
			"initiated_by": self.initiated_by,
			"context": dict(self.context),
			"status": self.status,
			"correlation_key": self.correlation_key,
			"started_at": self.started_at,
		}


@dataclass
class WfaTask:
	id: str
	tenant_id: str
	instance_id: str
	name: str
	task_type: str
	assignee_id: str | None = None
	queue_id: str | None = None
	due_at: str | None = None
	status: str = "open"
	completion_evidence: dict[str, Any] | None = None
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"instance_id": self.instance_id,
			"name": self.name,
			"task_type": self.task_type,
			"assignee_id": self.assignee_id,
			"queue_id": self.queue_id,
			"due_at": self.due_at,
			"status": self.status,
			"completion_evidence": dict(self.completion_evidence or {}),
			"created_at": self.created_at,
		}


@dataclass
class WfaApproval:
	id: str
	tenant_id: str
	task_id: str
	reviewer_id: str
	requester_id: str
	decision: str
	reason: str
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"task_id": self.task_id,
			"reviewer_id": self.reviewer_id,
			"requester_id": self.requester_id,
			"decision": self.decision,
			"reason": self.reason,
			"created_at": self.created_at,
		}


@dataclass
class WfaAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
		}


class WfaLifecycleService:
	"""In-package workflow lifecycle engine for generated APG applications."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._processes: dict[str, WfaProcess] = {}
		self._instances: dict[str, WfaProcessInstance] = {}
		self._tasks: dict[str, WfaTask] = {}
		self._approvals: dict[str, WfaApproval] = {}
		self._agents: dict[str, WfaAgent] = {}
		self._exceptions: list[dict[str, Any]] = []
		self._audit_events: list[dict[str, Any]] = []

	def describe(self) -> dict[str, Any]:
		return get_capability_contract(self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_process(
		self,
		process_id: str,
		name: str,
		owner_id: str,
		version: str,
		variable_schema: dict[str, Any],
		trigger: str = "manual",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "create_definition",
			"owner_present": bool(owner_id),
			"version_present": bool(version),
		})
		self._raise_on_deny(result)
		process = WfaProcess(
			id=process_id,
			tenant_id=self.tenant_id,
			name=name,
			owner_id=owner_id,
			version=version,
			trigger=trigger,
			variable_schema=dict(variable_schema),
		)
		self._processes[process_id] = process
		self._record_audit("workflow_definition_created", process.to_dict())
		return process.to_dict()

	def activate_process(self, process_id: str, approval_recorded: bool, reviewer_id: str) -> dict[str, Any]:
		process = self._processes[process_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "activate_definition",
			"approval_recorded": approval_recorded,
		})
		self._raise_on_deny(result)
		process.status = "active"
		process.approved = True
		self._record_audit("workflow_definition_activated", {
			"process_id": process_id,
			"reviewer_id": reviewer_id,
			"status": process.status,
		})
		return process.to_dict()

	def start_instance(
		self,
		instance_id: str,
		process_id: str,
		initiated_by: str,
		context: dict[str, Any],
		correlation_key: str | None = None,
	) -> dict[str, Any]:
		process = self._processes[process_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "start_instance",
			"definition_active": process.status == "active",
			"initiator_present": bool(initiated_by),
		})
		self._raise_on_deny(result)
		instance = WfaProcessInstance(
			id=instance_id,
			tenant_id=self.tenant_id,
			process_id=process_id,
			initiated_by=initiated_by,
			context=dict(context),
			correlation_key=correlation_key,
		)
		self._instances[instance_id] = instance
		self._record_audit("workflow_instance_started", instance.to_dict())
		return instance.to_dict()

	def create_task(
		self,
		task_id: str,
		instance_id: str,
		name: str,
		task_type: str = "human",
		assignee_id: str | None = None,
		queue_id: str | None = None,
		due_at: str | None = None,
		sla_tracked: bool = False,
	) -> dict[str, Any]:
		if task_type not in SUPPORTED_TASK_TYPES:
			raise ValueError(f"Unsupported workflow task type: {task_type}")
		self._instances[instance_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "create_task",
			"task_type": task_type,
			"assignee_present": bool(assignee_id or queue_id),
			"sla_tracked": sla_tracked,
			"due_at_present": bool(due_at),
		})
		self._raise_on_deny(result)
		task = WfaTask(
			id=task_id,
			tenant_id=self.tenant_id,
			instance_id=instance_id,
			name=name,
			task_type=task_type,
			assignee_id=assignee_id,
			queue_id=queue_id,
			due_at=due_at,
		)
		self._tasks[task_id] = task
		self._record_audit("workflow_task_created", task.to_dict())
		return task.to_dict()

	def complete_task(
		self,
		task_id: str,
		completed_by: str,
		completion_evidence: dict[str, Any] | None,
	) -> dict[str, Any]:
		task = self._tasks[task_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "complete_task",
			"completion_evidence_present": bool(completion_evidence),
		})
		self._raise_on_deny(result)
		task.status = "complete"
		task.completion_evidence = dict(completion_evidence or {})
		self._record_audit("workflow_task_completed", {
			"task_id": task_id,
			"completed_by": completed_by,
			"completion_evidence": dict(completion_evidence or {}),
		})
		return task.to_dict()

	def record_approval(
		self,
		task_id: str,
		reviewer_id: str,
		requester_id: str,
		decision: str,
		reason: str,
	) -> dict[str, Any]:
		self._tasks[task_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "record_approval",
			"decision": decision,
			"reviewer_same_as_requester": reviewer_id == requester_id,
			"decision_reason_present": bool(reason),
		})
		self._raise_on_deny(result)
		approval = WfaApproval(
			id=f"approval-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			task_id=task_id,
			reviewer_id=reviewer_id,
			requester_id=requester_id,
			decision=decision,
			reason=reason,
		)
		self._approvals[approval.id] = approval
		self._record_audit("workflow_task_approval_recorded", approval.to_dict())
		return approval.to_dict()

	def record_exception(
		self,
		instance_id: str,
		code: str,
		severity: str,
		details: dict[str, Any],
		owner_id: str,
	) -> dict[str, Any]:
		self._instances[instance_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "raise_exception",
			"exception_owner_present": bool(owner_id),
		})
		self._raise_on_deny(result)
		exception = {
			"id": f"exception-{uuid4().hex[:12]}",
			"tenant_id": self.tenant_id,
			"instance_id": instance_id,
			"code": code,
			"severity": severity,
			"details": dict(details),
			"owner_id": owner_id,
			"created_at": _utc_now(),
		}
		self._exceptions.append(exception)
		self._record_audit("workflow_exception_raised", exception)
		return dict(exception)

	def register_wfa_agent(
		self,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"wfa_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_WFA_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_WFA_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_on_deny(result)
		agent = WfaAgent(
			id=agent_id or f"wfa-agent-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[agent.id] = agent
		self._record_audit("workflow_agent_registered", agent.to_dict())
		return agent.to_dict()

	def validate_batch_wfa_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"requested_operation": "batch_workflow_mutation",
			"event_stream": event_stream,
		})

	def dashboard_summary(self) -> dict[str, Any]:
		return {
			"tenant_id": self.tenant_id,
			"process_count": len(self._processes),
			"active_process_count": sum(1 for process in self._processes.values() if process.status == "active"),
			"instance_count": len(self._instances),
			"running_instance_count": sum(1 for instance in self._instances.values() if instance.status == "running"),
			"task_count": len(self._tasks),
			"open_task_count": sum(1 for task in self._tasks.values() if task.status == "open"),
			"approval_count": len(self._approvals),
			"exception_count": len(self._exceptions),
			"wfa_agent_count": len(self._agents),
			"audit_event_count": len(self._audit_events),
			"streaming": self.describe()["streaming"],
		}

	def _raise_on_deny(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reason = result["actions"][0].get("reason", "workflow_policy_denied")
			raise PermissionError(reason)

	def _record_audit(self, event_type: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": f"audit-{uuid4().hex[:12]}",
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"payload": dict(payload),
			"created_at": _utc_now(),
		})


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()
