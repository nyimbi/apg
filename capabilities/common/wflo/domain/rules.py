"""Domain rules for Workflow Orchestration.

Every business rule is a pure callable. No I/O, no side effects.
Use assert_* for invariant checks, calculate_* for derived values.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_name}] {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Tenant
# ─────────────────────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-blank tenant_id."""
	if not tenant_id or not str(tenant_id).strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant reads and writes are always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowDefinition
# ─────────────────────────────────────────────────────────────────────────────

def assert_definition_name_present(name: str | None) -> None:
	if not name or not str(name).strip():
		raise RuleViolation("definition_name_required", "workflow name must not be blank", "provide_name")


def assert_definition_owner_present(owner_ref: str | None) -> None:
	if not owner_ref or not str(owner_ref).strip():
		raise RuleViolation("definition_owner_required", "owner_ref must not be blank", "assign_owner")


def assert_definition_publishable(status: str, review_required: bool, publish_approval_ref: str) -> None:
	"""Only DRAFT or REVIEW_REQUIRED definitions may be published."""
	if status not in ("draft", "review_required"):
		raise RuleViolation(
			"definition_not_publishable",
			f"definition with status '{status}' cannot be published",
			"ensure_definition_is_draft",
		)
	if review_required and not str(publish_approval_ref or "").strip():
		raise RuleViolation(
			"publish_approval_required",
			"review_required definitions must have publish_approval_ref",
			"obtain_publish_approval",
		)


def assert_definition_is_published(status: str) -> None:
	"""Instances may only be spawned from PUBLISHED definitions."""
	if status != "published":
		raise RuleViolation(
			"definition_not_published",
			f"cannot start instance from definition with status '{status}'",
			"publish_definition_first",
		)


def assert_definition_not_retired(status: str) -> None:
	if status == "retired":
		raise RuleViolation(
			"definition_retired",
			"retired definitions cannot be modified",
			"create_new_version",
		)


def assert_definition_step_count(steps: list[Any], max_steps: int = 200) -> None:
	if len(steps) > max_steps:
		raise RuleViolation(
			"max_steps_exceeded",
			f"workflow has {len(steps)} steps; maximum is {max_steps}",
			"split_into_sub_workflows",
		)


def assert_no_duplicate_step_ids(steps: list[dict[str, Any]]) -> None:
	ids = [s.get("id") for s in steps if s.get("id")]
	if len(ids) != len(set(ids)):
		raise RuleViolation(
			"duplicate_step_ids",
			"step IDs must be unique within a workflow definition",
			"fix_duplicate_step_ids",
		)


# ─────────────────────────────────────────────────────────────────────────────
# WorkflowInstance
# ─────────────────────────────────────────────────────────────────────────────

def assert_instance_can_be_suspended(status: str) -> None:
	if status not in ("running", "waiting_timer", "waiting_approval", "waiting_signal"):
		raise RuleViolation(
			"instance_not_suspendable",
			f"instance with status '{status}' cannot be suspended",
			"instance_must_be_active",
		)


def assert_instance_can_be_resumed(status: str) -> None:
	if status != "suspended":
		raise RuleViolation(
			"instance_not_resumable",
			f"instance with status '{status}' cannot be resumed",
			"instance_must_be_suspended",
		)


def assert_instance_can_be_cancelled(status: str) -> None:
	terminal = {"completed", "cancelled", "failed", "migrated"}
	if status in terminal:
		raise RuleViolation(
			"instance_already_terminal",
			f"instance with status '{status}' is terminal and cannot be cancelled",
			"instance_must_be_active",
		)


def assert_instance_active(status: str) -> None:
	terminal = {"completed", "cancelled", "failed", "migrated"}
	if status in terminal:
		raise RuleViolation(
			"instance_terminal",
			f"instance with status '{status}' is no longer active",
			"start_new_instance",
		)


def assert_migration_version_newer(current_version: int, new_version: int) -> None:
	if new_version <= current_version:
		raise RuleViolation(
			"migration_version_not_newer",
			f"target version {new_version} must be greater than current {current_version}",
			"use_higher_version_number",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

def assert_task_claimable(status: str, claimed_by: str | None) -> None:
	if status not in ("created", "ready"):
		raise RuleViolation(
			"task_not_claimable",
			f"task with status '{status}' cannot be claimed",
			"task_must_be_ready",
		)
	if claimed_by:
		raise RuleViolation(
			"task_already_claimed",
			f"task is already claimed by '{claimed_by}'",
			"release_claim_first",
		)


def assert_task_completable(status: str) -> None:
	if status in ("completed", "cancelled", "timed_out"):
		raise RuleViolation(
			"task_not_completable",
			f"task with status '{status}' cannot be completed",
			"task_must_be_active",
		)


def assert_task_claim_actor(actor: str | None) -> None:
	if not actor or not str(actor).strip():
		raise RuleViolation(
			"claim_actor_required",
			"claimed_by must not be blank",
			"provide_actor_id",
		)


def assert_task_assignee_present(assignee_ref: str | None) -> None:
	if not assignee_ref or not str(assignee_ref).strip():
		raise RuleViolation(
			"task_assignee_required",
			"assignee_ref must not be blank for user tasks",
			"assign_task_owner",
		)


def assert_escalation_reason(reason: str | None) -> None:
	if not reason or not str(reason).strip():
		raise RuleViolation(
			"escalation_reason_required",
			"escalation reason must not be blank",
			"provide_escalation_reason",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────────────────────

def assert_timer_not_fired(fired: bool) -> None:
	if fired:
		raise RuleViolation(
			"timer_already_fired",
			"timer has already fired and cannot be re-triggered",
			"create_new_timer",
		)


def assert_timer_fire_at_future(fire_at: datetime) -> None:
	if fire_at <= datetime.now(timezone.utc):
		raise RuleViolation(
			"timer_fire_at_past",
			f"timer fire_at {fire_at.isoformat()} is in the past",
			"set_future_fire_at",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Gateway
# ─────────────────────────────────────────────────────────────────────────────

def assert_exclusive_gateway_has_conditions(conditions: dict[str, str]) -> None:
	if not conditions:
		raise RuleViolation(
			"gateway_no_conditions",
			"exclusive gateway must have at least one condition",
			"add_gateway_conditions",
		)


def assert_parallel_gateway_branches(incoming: list[str], completed: list[str]) -> bool:
	"""Return True when all branches have completed (ready to merge)."""
	return set(incoming) == set(completed)


# ─────────────────────────────────────────────────────────────────────────────
# Compensation
# ─────────────────────────────────────────────────────────────────────────────

def assert_compensation_triggerable(instance_status: str) -> None:
	if instance_status not in ("failed", "compensating"):
		raise RuleViolation(
			"compensation_not_triggerable",
			f"compensation requires instance status 'failed', got '{instance_status}'",
			"fail_instance_first",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Derived calculations
# ─────────────────────────────────────────────────────────────────────────────

def calculate_next_escalation_level(current_level: int, max_level: int = 5) -> int:
	"""Return next escalation level, capped at max_level."""
	return min(current_level + 1, max_level)


def calculate_sla_status(due_at: datetime | None, now: datetime | None = None) -> str:
	"""Return 'ok', 'at_risk', or 'breached' relative to SLA deadline."""
	if due_at is None:
		return "ok"
	ref = now or datetime.now(timezone.utc)
	remaining = (due_at - ref).total_seconds()
	total = (due_at - ref).total_seconds()  # always positive fraction needed separately
	if remaining < 0:
		return "breached"
	# At risk: less than 20% of the window remains — approximate with 2h threshold
	if remaining < 7200:
		return "at_risk"
	return "ok"


__all__ = [
	"RuleViolation",
	# Tenant
	"assert_tenant_context",
	"assert_no_cross_tenant_access",
	# Definition
	"assert_definition_name_present",
	"assert_definition_owner_present",
	"assert_definition_publishable",
	"assert_definition_is_published",
	"assert_definition_not_retired",
	"assert_definition_step_count",
	"assert_no_duplicate_step_ids",
	# Instance
	"assert_instance_can_be_suspended",
	"assert_instance_can_be_resumed",
	"assert_instance_can_be_cancelled",
	"assert_instance_active",
	"assert_migration_version_newer",
	# Task
	"assert_task_claimable",
	"assert_task_completable",
	"assert_task_claim_actor",
	"assert_task_assignee_present",
	"assert_escalation_reason",
	# Timer
	"assert_timer_not_fired",
	"assert_timer_fire_at_future",
	# Gateway
	"assert_exclusive_gateway_has_conditions",
	"assert_parallel_gateway_branches",
	# Compensation
	"assert_compensation_triggerable",
	# Calculations
	"calculate_next_escalation_level",
	"calculate_sla_status",
]
