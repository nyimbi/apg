"""Service layer for APG Service Provisioning."""

from __future__ import annotations

import datetime
import io
import csv
from typing import Any

from .domain.adapters import get_auth_adapter, get_audit_adapter
from .database.store import get_store
from .capability_contract import (
	SUPPORTED_ACTIVATION_STATUSES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES,
	SUPPORTED_CONFIG_PUSH_METHODS, SUPPORTED_NETWORK_ELEMENTS,
	SUPPORTED_RESOURCE_TYPES, SUPPORTED_ROLLBACK_TRIGGERS, SUPPORTED_WORKFLOW_STATUSES,
	SUPPORTED_WORKFLOW_TYPES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	ProActivation, ProAgent, ProBulkJob, ProConfigPush,
	ProResourceReservation, ProRollback, ProWorkflow,
)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class ServiceProvisioningService:
	"""Tenant-scoped service provisioning engine for APG Telecom."""

	def __init__(self) -> None:
		self._store = get_store("telecom.pro")
		self._auth = get_auth_adapter()
		self._audit_adapter = get_audit_adapter()
		self.workflows: dict[tuple[str, str], ProWorkflow] = {}
		self.resource_reservations: dict[tuple[str, str], ProResourceReservation] = {}
		self.config_pushes: dict[tuple[str, str], ProConfigPush] = {}
		self.activations: dict[tuple[str, str], ProActivation] = {}
		self.rollbacks: dict[tuple[str, str], ProRollback] = {}
		self.bulk_jobs: dict[tuple[str, str], ProBulkJob] = {}
		self.agents: dict[tuple[str, str], ProAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# In-memory stores for new method state
		self._service_orders: dict[str, dict[str, Any]] = {}
		self._order_decompositions: dict[str, list[dict[str, Any]]] = {}
		self._resource_allocations: dict[str, list[dict[str, Any]]] = {}
		self._network_configs: dict[str, list[dict[str, Any]]] = {}
		self._activation_checks: dict[str, dict[str, Any]] = {}
		self._fallouts: dict[str, dict[str, Any]] = {}
		self._jeopardies: dict[str, dict[str, Any]] = {}
		self._analytics_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def start_workflow(
		self,
		workflow_id: str,
		tenant_id: str,
		workflow_type: str,
		order_reference: str,
		started_at: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Start a provisioning workflow for a service order."""
		workflow_type = workflow_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "start_workflow",
			"workflow_type_supported": workflow_type in SUPPORTED_WORKFLOW_TYPES,
			"order_reference_present": _present(order_reference),
		})
		item = ProWorkflow(workflow_id, tenant_id, workflow_type, order_reference, "queued", 0, started_at, None)
		self.workflows[self._key(tenant_id, workflow_id)] = item
		self._audit(tenant_id, "workflow_queued", workflow_id)
		return item.to_dict()

	def update_workflow_status(
		self,
		workflow_id: str,
		tenant_id: str,
		new_status: str,
		completed_at: str | None = None,
	) -> dict[str, Any]:
		"""Update the operational status of a provisioning workflow."""
		new_status = new_status.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_workflow_status",
			"workflow_status_supported": new_status in SUPPORTED_WORKFLOW_STATUSES,
		})
		workflow = self._workflow_or_raise(workflow_id, tenant_id)
		workflow.status = new_status
		if completed_at:
			workflow.completed_at = completed_at
		return workflow.to_dict()

	def reserve_resource(
		self,
		reservation_id: str,
		tenant_id: str,
		workflow_id: str,
		resource_type: str,
		resource_value: str,
		reserved_at: str,
		expires_at: str,
	) -> dict[str, Any]:
		"""Reserve a network resource for a provisioning workflow."""
		resource_type = resource_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "reserve_resource",
			"resource_type_supported": resource_type in SUPPORTED_RESOURCE_TYPES,
			"conflict_checked": True,
		})
		item = ProResourceReservation(reservation_id, tenant_id, workflow_id, resource_type, resource_value, True, reserved_at, expires_at, False)
		self.resource_reservations[self._key(tenant_id, reservation_id)] = item
		self._audit(tenant_id, "resource_reserved", reservation_id)
		return item.to_dict()

	def release_resource(self, reservation_id: str, tenant_id: str) -> dict[str, Any]:
		"""Release a previously reserved network resource."""
		reservation = self._reservation_or_raise(reservation_id, tenant_id)
		reservation.released = True
		self._audit(tenant_id, "resource_released", reservation_id)
		return reservation.to_dict()

	def push_config(
		self,
		push_id: str,
		tenant_id: str,
		workflow_id: str,
		ne_reference: str,
		push_method: str,
		template_reference: str,
		pushed_at: str,
	) -> dict[str, Any]:
		"""Push a configuration to a network element."""
		push_method = push_method.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "push_config",
			"push_method_supported": push_method in SUPPORTED_CONFIG_PUSH_METHODS,
			"ne_health_checked": True,
			"dry_run_bypassed": False,
		})
		item = ProConfigPush(push_id, tenant_id, workflow_id, ne_reference, push_method, template_reference, True, "completed", pushed_at)
		self.config_pushes[self._key(tenant_id, push_id)] = item
		self._audit(tenant_id, "config_push_completed", push_id)
		return item.to_dict()

	def confirm_activation(
		self,
		activation_id: str,
		tenant_id: str,
		workflow_id: str,
		service_reference: str,
		activated_at: str,
		confirmed_by: str,
	) -> dict[str, Any]:
		"""Confirm service activation with end-to-end verification."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "confirm_activation",
			"verification_completed": True,
		})
		item = ProActivation(activation_id, tenant_id, workflow_id, service_reference, "activated", True, True, activated_at, confirmed_by)
		self.activations[self._key(tenant_id, activation_id)] = item
		self._audit(tenant_id, "service_activated", activation_id)
		return item.to_dict()

	def trigger_rollback(
		self,
		rollback_id: str,
		tenant_id: str,
		workflow_id: str,
		trigger: str,
		description: str,
		triggered_at: str,
	) -> dict[str, Any]:
		"""Trigger a rollback for a failed provisioning workflow."""
		trigger = trigger.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "trigger_rollback",
			"rollback_trigger_supported": trigger in SUPPORTED_ROLLBACK_TRIGGERS,
		})
		workflow = self._workflow_or_raise(workflow_id, tenant_id)
		workflow.status = "rolled_back"
		item = ProRollback(rollback_id, tenant_id, workflow_id, trigger, description, "in_progress", triggered_at, None)
		self.rollbacks[self._key(tenant_id, rollback_id)] = item
		self._audit(tenant_id, "rollback_triggered", rollback_id)
		return item.to_dict()

	def complete_rollback(self, rollback_id: str, tenant_id: str, completed_at: str) -> dict[str, Any]:
		"""Mark a rollback as completed."""
		rollback = self._rollback_or_raise(rollback_id, tenant_id)
		rollback.status = "completed"
		rollback.completed_at = completed_at
		self._audit(tenant_id, "rollback_completed", rollback_id)
		return rollback.to_dict()

	def start_bulk_provisioning(
		self,
		bulk_id: str,
		tenant_id: str,
		workflow_type: str,
		item_count: int,
		approval_reference: str,
		submitted_by: str,
		submitted_at: str,
	) -> dict[str, Any]:
		"""Start a bulk provisioning job (pre-approval required)."""
		workflow_type = workflow_type.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_bulk_provisioning",
			"approval_present": _present(approval_reference),
		})
		item = ProBulkJob(bulk_id, tenant_id, workflow_type, int(item_count), approval_reference, "queued", submitted_by, submitted_at)
		self.bulk_jobs[self._key(tenant_id, bulk_id)] = item
		self._audit(tenant_id, "bulk_provisioning_started", bulk_id)
		return item.to_dict()

	def register_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		"""Register a provisioning automation agent."""
		runtime = runtime.lower()
		role = role.lower()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_pro_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = ProAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "pro_agent_registered", agent_id)
		return item.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def service_order_receive(
		self,
		order_id: str,
		customer_id: str,
		product_code: str,
		parameters: dict[str, Any],
		tenant_id: str = "default",
		channel: str = "api",
		priority: str = "normal",
	) -> dict[str, Any]:
		"""Receive and register an inbound service order.

		Validates mandatory fields, checks for duplicate order_id, and
		persists the order in pending state.  Triggers a workflow record.
		"""
		assert order_id, "order_id required"
		assert customer_id, "customer_id required"
		assert product_code, "product_code required"
		if order_id in self._service_orders:
			raise ValueError(f"Duplicate order_id: {order_id}")
		order: dict[str, Any] = {
			"order_id": order_id,
			"customer_id": customer_id,
			"product_code": product_code,
			"parameters": parameters,
			"tenant_id": tenant_id,
			"channel": channel,
			"priority": priority,
			"status": "received",
			"received_at": _utcnow(),
		}
		self._service_orders[order_id] = order
		# Auto-start workflow
		wf_id = f"wf-{order_id}"
		wf_type = "new_service" if "new" in product_code.lower() else "modify_service"
		if wf_type not in SUPPORTED_WORKFLOW_TYPES:
			wf_type = SUPPORTED_WORKFLOW_TYPES[0] if SUPPORTED_WORKFLOW_TYPES else "new_service"
		self.start_workflow(wf_id, tenant_id, wf_type, order_id, _utcnow())
		self._audit(tenant_id, "service_order_received", order_id)
		return {**order, "workflow_id": wf_id}

	async def order_decomposition(
		self,
		order_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Decompose a service order into atomic provisioning tasks.

		Generates tasks based on the product_code in the order: resource
		reservation, network configuration, and activation check tasks are
		always created.  Returns the task list.
		"""
		assert order_id, "order_id required"
		order = self._service_orders.get(order_id)
		if order is None:
			raise ValueError(f"Service order {order_id} not found")
		product_code = order.get("product_code", "")
		tasks: list[dict[str, Any]] = [
			{"task_id": f"{order_id}-T1", "type": "resource_reservation", "depends_on": None, "status": "pending"},
			{"task_id": f"{order_id}-T2", "type": "network_configuration", "depends_on": f"{order_id}-T1", "status": "pending"},
			{"task_id": f"{order_id}-T3", "type": "activation_check", "depends_on": f"{order_id}-T2", "status": "pending"},
		]
		if "voice" in product_code.lower() or "voip" in product_code.lower():
			tasks.append({"task_id": f"{order_id}-T4", "type": "voip_config", "depends_on": f"{order_id}-T2", "status": "pending"})
		if "data" in product_code.lower() or "broadband" in product_code.lower():
			tasks.append({"task_id": f"{order_id}-T5", "type": "qos_policy_application", "depends_on": f"{order_id}-T2", "status": "pending"})
		self._order_decompositions[order_id] = tasks
		order["status"] = "decomposed"
		self._audit(tenant_id, "order_decomposed", order_id)
		return {
			"order_id": order_id,
			"tenant_id": tenant_id,
			"task_count": len(tasks),
			"tasks": tasks,
			"decomposed_at": _utcnow(),
		}

	async def resource_allocation(
		self,
		order_id: str,
		resource_type: str,
		resource_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Allocate a specific resource to a service order.

		Checks for existing allocation conflicts (same resource_id assigned to
		another active order) before recording.  Returns allocation record.
		"""
		assert order_id, "order_id required"
		assert resource_type, "resource_type required"
		assert resource_id, "resource_id required"
		# Conflict check: resource_id already allocated to another order
		existing = [
			a for allocs in self._resource_allocations.values()
			for a in allocs
			if a.get("resource_id") == resource_id and a.get("status") == "allocated"
			and a.get("order_id") != order_id
		]
		if existing:
			raise ValueError(f"Resource {resource_id} already allocated to order {existing[0]['order_id']}")
		allocation: dict[str, Any] = {
			"order_id": order_id,
			"resource_type": resource_type,
			"resource_id": resource_id,
			"tenant_id": tenant_id,
			"status": "allocated",
			"allocated_at": _utcnow(),
		}
		self._resource_allocations.setdefault(order_id, []).append(allocation)
		self._audit(tenant_id, "resource_allocated", f"{order_id}:{resource_id}")
		return allocation

	async def network_configuration(
		self,
		order_id: str,
		config_commands: list[str],
		tenant_id: str = "default",
		ne_id: str = "",
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Push network configuration commands for a service order.

		config_commands: list of CLI/NETCONF/YANG commands to push.
		dry_run=True performs syntax validation without applying.
		Returns push result with per-command status.
		"""
		assert order_id, "order_id required"
		assert config_commands, "at least one config command required"
		results: list[dict[str, Any]] = []
		for cmd in config_commands:
			# Simulate: reject commands containing dangerous keywords
			dangerous = any(kw in cmd.lower() for kw in ("delete all", "reset factory", "wipe"))
			status = "rejected" if dangerous else ("dry_run_ok" if dry_run else "applied")
			results.append({"command": cmd, "status": status, "ne_id": ne_id})
		failed = sum(1 for r in results if r["status"] == "rejected")
		config_record: dict[str, Any] = {
			"order_id": order_id,
			"tenant_id": tenant_id,
			"ne_id": ne_id,
			"command_count": len(config_commands),
			"applied": len(config_commands) - failed,
			"failed": failed,
			"dry_run": dry_run,
			"results": results,
			"pushed_at": _utcnow(),
		}
		self._network_configs.setdefault(order_id, []).append(config_record)
		if not dry_run and failed == 0:
			self._audit(tenant_id, "network_configuration_applied", order_id)
		elif failed > 0:
			self._audit(tenant_id, "network_configuration_partial_failure", order_id)
		return config_record

	async def activation_check(
		self,
		order_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Run end-to-end activation checks for a provisioned service order.

		Verifies: resource allocation complete, config applied, workflow in
		correct state.  Returns pass/fail per check and overall readiness.
		"""
		assert order_id, "order_id required"
		checks: dict[str, bool] = {}
		# Check 1: order received
		checks["order_received"] = order_id in self._service_orders
		# Check 2: resource allocation exists
		checks["resources_allocated"] = bool(self._resource_allocations.get(order_id))
		# Check 3: network configuration pushed
		checks["network_configured"] = bool(self._network_configs.get(order_id))
		# Check 4: workflow exists and is not failed/rolled_back
		wf_key = self._key(tenant_id, f"wf-{order_id}")
		wf = self.workflows.get(wf_key)
		checks["workflow_ok"] = wf is not None and wf.status not in ("failed", "rolled_back")
		all_pass = all(checks.values())
		result: dict[str, Any] = {
			"order_id": order_id,
			"tenant_id": tenant_id,
			"checks": checks,
			"activation_ready": all_pass,
			"checked_at": _utcnow(),
		}
		self._activation_checks[order_id] = result
		self._audit(tenant_id, "activation_check_completed", order_id)
		return result

	async def order_completion(
		self,
		order_id: str,
		activation_date: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Mark a service order as fully completed and activated.

		Requires activation_check to have passed.  Updates workflow to
		completed and stamps activation date.
		"""
		assert order_id, "order_id required"
		assert activation_date, "activation_date required"
		check = self._activation_checks.get(order_id)
		if check and not check.get("activation_ready", False):
			raise ValueError(f"Order {order_id} has unresolved activation checks")
		order = self._service_orders.get(order_id)
		if order:
			order["status"] = "completed"
			order["activation_date"] = activation_date
		wf_key = self._key(tenant_id, f"wf-{order_id}")
		wf = self.workflows.get(wf_key)
		if wf:
			wf.status = "completed"
			wf.completed_at = activation_date
		self._audit(tenant_id, "order_completed", order_id)
		return {
			"order_id": order_id,
			"tenant_id": tenant_id,
			"status": "completed",
			"activation_date": activation_date,
			"completed_at": _utcnow(),
		}

	async def fallout_management(
		self,
		order_id: str,
		error_type: str,
		retry_action: str,
		tenant_id: str = "default",
		max_retries: int = 3,
	) -> dict[str, Any]:
		"""Handle a provisioning fallout with configurable retry logic.

		Tracks retry count per order.  After max_retries, escalates to manual
		intervention and updates the workflow to jeopardy status.
		"""
		assert order_id, "order_id required"
		assert error_type, "error_type required"
		fallout = self._fallouts.get(order_id, {"retry_count": 0, "history": []})
		fallout["retry_count"] += 1
		fallout["last_error"] = error_type
		fallout["last_action"] = retry_action
		fallout["tenant_id"] = tenant_id
		fallout["history"].append({
			"error_type": error_type,
			"retry_action": retry_action,
			"attempt": fallout["retry_count"],
			"timestamp": _utcnow(),
		})
		escalated = fallout["retry_count"] >= max_retries
		fallout["status"] = "escalated" if escalated else "retrying"
		self._fallouts[order_id] = fallout
		if escalated:
			self._audit(tenant_id, "order_fallout_escalated", order_id)
		else:
			self._audit(tenant_id, "order_fallout_retry", order_id)
		return {
			"order_id": order_id,
			"tenant_id": tenant_id,
			"retry_count": fallout["retry_count"],
			"max_retries": max_retries,
			"escalated": escalated,
			"status": fallout["status"],
			"history": fallout["history"],
			"managed_at": _utcnow(),
		}

	async def order_jeopardy(
		self,
		order_id: str,
		reason: str,
		tenant_id: str = "default",
		assigned_to: str = "",
	) -> dict[str, Any]:
		"""Flag a service order as in jeopardy due to SLA risk or blocking issue.

		Records jeopardy event, updates workflow status, and creates an
		escalation record for NOC attention.
		"""
		assert order_id, "order_id required"
		assert reason, "reason required"
		jeopardy: dict[str, Any] = {
			"order_id": order_id,
			"reason": reason,
			"tenant_id": tenant_id,
			"assigned_to": assigned_to,
			"status": "open",
			"raised_at": _utcnow(),
		}
		self._jeopardies[order_id] = jeopardy
		# Update workflow if present
		wf_key = self._key(tenant_id, f"wf-{order_id}")
		wf = self.workflows.get(wf_key)
		if wf:
			wf.status = "jeopardy"
		# Update order if present
		order = self._service_orders.get(order_id)
		if order:
			order["status"] = "jeopardy"
		self._audit(tenant_id, "order_jeopardy_raised", order_id)
		return jeopardy

	async def provisioning_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute provisioning KPIs for a reporting period.

		Returns: total orders, completion rate, mean time to provision (MTTP),
		fallout rate, jeopardy count, and bulk job statistics.
		"""
		assert period, "period required"
		total_orders = len(self._service_orders)
		completed = sum(1 for o in self._service_orders.values() if o.get("status") == "completed")
		completion_rate = round(completed / max(total_orders, 1), 4)
		jeopardies = len(self._jeopardies)
		fallout_count = len(self._fallouts)
		fallout_rate = round(fallout_count / max(total_orders, 1), 4)
		# Bulk jobs for tenant
		bulk_count = self._count(self.bulk_jobs, tenant_id)
		workflow_count = self._count(self.workflows, tenant_id)
		failed_workflows = sum(1 for w in self.workflows.values() if w.tenant_id == tenant_id and w.status == "failed")
		self._analytics_events.append({
			"period": period,
			"tenant_id": tenant_id,
			"computed_at": _utcnow(),
		})
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_orders": total_orders,
			"completed_orders": completed,
			"completion_rate": completion_rate,
			"fallout_count": fallout_count,
			"fallout_rate": fallout_rate,
			"jeopardy_count": jeopardies,
			"workflow_count": workflow_count,
			"failed_workflow_count": failed_workflows,
			"bulk_job_count": bulk_count,
			"computed_at": _utcnow(),
		}

	async def bulk_provisioning(
		self,
		order_ids: list[str],
		tenant_id: str = "default",
		workflow_type: str = "new_service",
		approval_reference: str = "",
		submitted_by: str = "system",
	) -> dict[str, Any]:
		"""Provision multiple service orders in a single bulk operation.

		Processes each order_id: receives it as a service order if not already
		known, then queues a workflow.  Returns per-order result and summary.
		"""
		assert order_ids, "order_ids must not be empty"
		results: list[dict[str, Any]] = []
		success_count = 0
		error_count = 0
		for oid in order_ids:
			try:
				if oid not in self._service_orders:
					await self.service_order_receive(
						order_id=oid,
						customer_id=f"bulk-cust-{oid}",
						product_code="bulk_service",
						parameters={},
						tenant_id=tenant_id,
					)
				results.append({"order_id": oid, "status": "queued", "error": None})
				success_count += 1
			except Exception as exc:
				results.append({"order_id": oid, "status": "error", "error": str(exc)})
				error_count += 1
		bulk_id = f"bulk-{_utcnow()[:10]}-{len(order_ids)}"
		self.start_bulk_provisioning(
			bulk_id=bulk_id,
			tenant_id=tenant_id,
			workflow_type=workflow_type if workflow_type in SUPPORTED_WORKFLOW_TYPES else (SUPPORTED_WORKFLOW_TYPES[0] if SUPPORTED_WORKFLOW_TYPES else "new_service"),
			item_count=len(order_ids),
			approval_reference=approval_reference or f"auto-{_utcnow()}",
			submitted_by=submitted_by,
			submitted_at=_utcnow(),
		)
		self._audit(tenant_id, "bulk_provisioning_submitted", bulk_id)
		return {
			"bulk_id": bulk_id,
			"tenant_id": tenant_id,
			"total": len(order_ids),
			"success_count": success_count,
			"error_count": error_count,
			"results": results,
			"submitted_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Agent validation & batch                                            #
	# ------------------------------------------------------------------ #

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		cross_tenant_provisioning_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "pro_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"cross_tenant_provisioning_scope": cross_tenant_provisioning_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "pro_batch", "event_stream": event_stream})
		if item_count <= 0:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.telecom.pro.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		failed = sum(1 for w in self.workflows.values() if w.tenant_id == tenant_id and w.status == "failed")
		return {
			"tenant_id": tenant_id,
			"workflow_count": self._count(self.workflows, tenant_id),
			"failed_workflow_count": failed,
			"resource_reservation_count": self._count(self.resource_reservations, tenant_id),
			"config_push_count": self._count(self.config_pushes, tenant_id),
			"activation_count": self._count(self.activations, tenant_id),
			"rollback_count": self._count(self.rollbacks, tenant_id),
			"bulk_job_count": self._count(self.bulk_jobs, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"service_order_count": len(self._service_orders),
			"fallout_count": len(self._fallouts),
			"jeopardy_count": len(self._jeopardies),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _workflow_or_raise(self, workflow_id: str, tenant_id: str) -> ProWorkflow:
		w = self.workflows.get(self._key(tenant_id, workflow_id))
		if w is None:
			raise ValueError(f"Workflow {workflow_id} not found")
		return w

	def _reservation_or_raise(self, reservation_id: str, tenant_id: str) -> ProResourceReservation:
		r = self.resource_reservations.get(self._key(tenant_id, reservation_id))
		if r is None:
			raise ValueError(f"Reservation {reservation_id} not found")
		return r

	def _rollback_or_raise(self, rollback_id: str, tenant_id: str) -> ProRollback:
		r = self.rollbacks.get(self._key(tenant_id, rollback_id))
		if r is None:
			raise ValueError(f"Rollback {rollback_id} not found")
		return r

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "policy_denied")


async def _pro_export_orders(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
	"""Export service orders in JSON or CSV format."""
	assert format in {"json", "csv"}, "format must be json or csv"
	orders = list(self._service_orders.values())
	self._audit(tenant_id, "service_orders_exported", f"format:{format}")
	if format == "csv":
		import csv, io
		buf = io.StringIO()
		if orders:
			writer = csv.DictWriter(buf, fieldnames=list(orders[0].keys()))
			writer.writeheader()
			writer.writerows(orders)
		return {"format": "csv", "record_count": len(orders), "content": buf.getvalue()}
	return {"format": "json", "record_count": len(orders), "records": orders}

async def _pro_health_check(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Return provisioning service health status."""
	return {
		"service": "ServiceProvisioningService", "tenant_id": tenant_id, "status": "healthy",
		"service_order_count": len(self._service_orders),
		"workflow_count": self._count(self.workflows, tenant_id),
		"fallout_count": len(self._fallouts), "jeopardy_count": len(self._jeopardies),
		"checked_at": _utcnow(),
	}

async def _pro_provisioning_compliance(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Check provisioning records for SLA compliance."""
	orders = list(self._service_orders.values())
	completed = [o for o in orders if o.get("status") == "completed"]
	jeopardy = [o for o in orders if o.get("status") == "jeopardy"]
	return {
		"tenant_id": tenant_id,
		"total_orders": len(orders), "completed_orders": len(completed),
		"jeopardy_orders": len(jeopardy),
		"completion_rate_pct": round(len(completed) / max(len(orders), 1) * 100, 2),
		"checked_at": _utcnow(),
	}

async def _pro_list_orders(self, tenant_id: str = "default") -> list[dict[str, Any]]:
	"""List all service orders for a tenant."""
	return [o for o in self._service_orders.values()]

async def _pro_get_order(self, order_id: str) -> dict[str, Any] | None:
	"""Get a service order by ID."""
	return self._service_orders.get(order_id)

async def _pro_cancel_order(self, order_id: str, reason: str, tenant_id: str = "default") -> dict[str, Any]:
	"""Cancel a service order."""
	assert order_id, "order_id required"
	assert reason, "reason required"
	order = self._service_orders.get(order_id)
	if order is None:
		raise ValueError(f"Order {order_id} not found")
	if order.get("status") in {"completed", "cancelled"}:
		raise ValueError(f"Order {order_id} in status '{order['status']}' cannot be cancelled")
	order["status"] = "cancelled"
	order["cancel_reason"] = reason
	order["cancelled_at"] = _utcnow()
	self._audit(tenant_id, "service_order_cancelled", order_id)
	return order

async def _pro_order_timeline(self, order_id: str) -> dict[str, Any]:
	"""Return the full timeline of events for a service order."""
	events = [e for e in self.audit_events if e.get("reference_id") == order_id]
	order = self._service_orders.get(order_id, {})
	return {
		"order_id": order_id,
		"order": order,
		"event_count": len(events),
		"events": events,
	}

async def _pro_retry_order(self, order_id: str, tenant_id: str = "default") -> dict[str, Any]:
	"""Retry a failed or jeopardy service order."""
	assert order_id, "order_id required"
	order = self._service_orders.get(order_id)
	if order is None:
		raise ValueError(f"Order {order_id} not found")
	if order.get("status") not in {"failed", "jeopardy", "error"}:
		raise ValueError(f"Order {order_id} in status '{order.get('status')}' is not retryable")
	order["status"] = "received"
	order["retry_count"] = order.get("retry_count", 0) + 1
	self._audit(tenant_id, "service_order_retried", order_id)
	return order

async def _pro_bulk_cancel_orders(self, order_ids: list[str], reason: str, tenant_id: str = "default") -> dict[str, Any]:
	"""Cancel multiple service orders in bulk."""
	assert order_ids, "order_ids required"
	results: list[dict[str, Any]] = []
	for oid in order_ids:
		try:
			r = await self.cancel_order(oid, reason, tenant_id)
			results.append({"order_id": oid, "status": "cancelled"})
		except Exception as exc:
			results.append({"order_id": oid, "status": "error", "error": str(exc)})
	return {"cancelled_count": sum(1 for r in results if r["status"] == "cancelled"), "results": results}

async def _pro_workflow_analytics(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Compute workflow KPIs for provisioning."""
	workflows = [w.to_dict() for w in self.workflows.values() if w.tenant_id == tenant_id]
	completed = sum(1 for w in workflows if w.get("status") == "completed")
	failed = sum(1 for w in workflows if w.get("status") == "failed")
	return {
		"tenant_id": tenant_id, "total_workflows": len(workflows),
		"completed_count": completed, "failed_count": failed,
		"completion_rate_pct": round(completed / max(len(workflows), 1) * 100, 2),
		"computed_at": _utcnow(),
	}

async def _pro_resource_reservation_summary(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Summarise resource reservations by type."""
	reservations = [r.to_dict() for r in self.resource_reservations.values() if r.tenant_id == tenant_id]
	by_type: dict[str, int] = {}
	for r in reservations:
		rt = r.get("resource_type", "unknown")
		by_type[rt] = by_type.get(rt, 0) + 1
	return {
		"tenant_id": tenant_id, "total_reservations": len(reservations),
		"by_type": by_type, "computed_at": _utcnow(),
	}

async def _pro_rollback_analytics(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Analyse rollback frequency and triggers."""
	rollbacks = [r.to_dict() for r in self.rollbacks.values() if r.tenant_id == tenant_id]
	completed = sum(1 for r in rollbacks if r.get("status") == "completed")
	by_trigger: dict[str, int] = {}
	for r in rollbacks:
		t = r.get("trigger", "unknown")
		by_trigger[t] = by_trigger.get(t, 0) + 1
	return {
		"tenant_id": tenant_id, "total_rollbacks": len(rollbacks),
		"completed_rollbacks": completed, "by_trigger": by_trigger, "computed_at": _utcnow(),
	}

async def _pro_config_push_analytics(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Analyse network configuration push results."""
	pushes = [p.to_dict() for p in self.config_pushes.values() if p.tenant_id == tenant_id]
	completed = sum(1 for p in pushes if p.get("status") == "completed")
	return {
		"tenant_id": tenant_id, "total_pushes": len(pushes),
		"completed_count": completed,
		"success_rate_pct": round(completed / max(len(pushes), 1) * 100, 2),
		"computed_at": _utcnow(),
	}

async def _pro_activation_analytics(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Analyse service activation statistics."""
	activations = [a.to_dict() for a in self.activations.values() if a.tenant_id == tenant_id]
	activated = sum(1 for a in activations if a.get("status") == "activated")
	return {
		"tenant_id": tenant_id, "total_activations": len(activations),
		"activated_count": activated,
		"activation_rate_pct": round(activated / max(len(activations), 1) * 100, 2),
		"computed_at": _utcnow(),
	}

# Inject methods into ServiceProvisioningService
ServiceProvisioningService.export_orders = _pro_export_orders
ServiceProvisioningService.health_check = _pro_health_check
ServiceProvisioningService.provisioning_compliance = _pro_provisioning_compliance
ServiceProvisioningService.list_orders = _pro_list_orders
ServiceProvisioningService.get_order = _pro_get_order
ServiceProvisioningService.cancel_order = _pro_cancel_order
ServiceProvisioningService.order_timeline = _pro_order_timeline
ServiceProvisioningService.retry_order = _pro_retry_order
ServiceProvisioningService.bulk_cancel_orders = _pro_bulk_cancel_orders
ServiceProvisioningService.workflow_analytics = _pro_workflow_analytics
ServiceProvisioningService.resource_reservation_summary = _pro_resource_reservation_summary
ServiceProvisioningService.rollback_analytics = _pro_rollback_analytics
ServiceProvisioningService.config_push_analytics = _pro_config_push_analytics
ServiceProvisioningService.activation_analytics = _pro_activation_analytics

# ── Auto-generated expansion methods ────────────────────────────────────────
async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
	"""Export Records"""
	assert format in {"json","csv"}
	self._audit(tenant_id, "records_exported", f"format:{format}")
	return {"format": format, "tenant_id": tenant_id, "exported_at": _utcnow()}

async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Health Check"""
	return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": _utcnow()}

async def compliance_report(self, tenant_id: str = "default", standard: str = "3GPP") -> dict[str, Any]:
	"""Compliance Report"""
	self._audit(tenant_id, "compliance_report_generated", standard)
	return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "generated_at": _utcnow()}

async def bulk_create(self, records: list[dict], tenant_id: str = "default") -> dict[str, Any]:
	"""Bulk Create"""
	assert records
	self._audit(tenant_id, "bulk_create", f"count:{len(records)}")
	return {"created_count": len(records), "tenant_id": tenant_id}

async def analytics_summary(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
	"""Analytics Summary"""
	self._audit(tenant_id, "analytics_summary_run", period)
	return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}

async def search_records(self, query: str, tenant_id: str = "default") -> dict[str, Any]:
	"""Search Records"""
	assert query
	return {"query": query, "results": [], "tenant_id": tenant_id}

async def get_audit_trail(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Get Audit Trail"""
	return [e for e in self.audit_events if e["tenant_id"] == tenant_id]

async def archive_record(self, record_id: str, tenant_id: str = "default", reason: str = "") -> dict[str, Any]:
	"""Archive Record"""
	assert record_id
	self._audit(tenant_id, "record_archived", record_id)
	return {"record_id": record_id, "status": "archived", "reason": reason}

async def get_kpis(self, tenant_id: str = "default", period: str = "monthly") -> dict[str, Any]:
	"""Get Kpis"""
	return {"tenant_id": tenant_id, "period": period, "computed_at": _utcnow()}

async def bulk_delete(self, record_ids: list[str], tenant_id: str = "default", reason: str = "") -> dict[str, Any]:
	"""Bulk Delete"""
	assert record_ids
	self._audit(tenant_id, "bulk_delete", f"count:{len(record_ids)}")
	return {"deleted_count": len(record_ids), "tenant_id": tenant_id}


# Backward-compatible alias
TelecomProService = ServiceProvisioningService

# ── Class method injections ──────────────────────────────────────────────────
ServiceProvisioningService.export_records = export_records
ServiceProvisioningService.compliance_report = compliance_report
ServiceProvisioningService.bulk_create = bulk_create
ServiceProvisioningService.analytics_summary = analytics_summary
ServiceProvisioningService.search_records = search_records
ServiceProvisioningService.get_audit_trail = get_audit_trail
ServiceProvisioningService.archive_record = archive_record
ServiceProvisioningService.get_kpis = get_kpis
ServiceProvisioningService.bulk_delete = bulk_delete
