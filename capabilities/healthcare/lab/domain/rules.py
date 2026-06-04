"""Deterministic domain rules for Laboratory Information System.

All business rules are implemented as callable Python functions.
RuleViolation is raised on any violation.
assert_* functions enforce preconditions.
calculate_* functions compute derived values.

© 2025 Datacraft — nyimbi@gmail.com
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any


# ── Exception ──────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ── Tenant / access rules ──────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant context."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied regardless of actor role."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource owned by '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(operation_type: str, policy_attached: bool) -> None:
	"""All write operations require an attached, evaluated policy."""
	if operation_type == "write" and not policy_attached:
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


# ── Order rules ────────────────────────────────────────────────────────────────

def assert_order_cancellable(status: str) -> None:
	"""Orders can only be cancelled when not already in a terminal state."""
	terminal = {"cancelled", "reported"}
	if status in terminal:
		raise RuleViolation(
			"order_not_cancellable",
			f"order with status '{status}' cannot be cancelled",
			"review_order_status",
		)


def assert_specimen_collectable(order_status: str) -> None:
	"""Specimens may not be collected for cancelled or completed orders."""
	if order_status == "cancelled":
		raise RuleViolation(
			"cancelled_order_not_collectable",
			"cannot collect specimen for a cancelled order",
			"reorder_test",
		)
	if order_status == "reported":
		raise RuleViolation(
			"reported_order_not_collectable",
			"cannot collect specimen for an already-reported order",
			"review_order_status",
		)


def assert_order_status_supported(status: str, supported: list[str]) -> None:
	"""Order status must be one of the supported lifecycle states."""
	if status not in supported:
		raise RuleViolation(
			"order_status_not_supported",
			f"order status '{status}' is not supported; valid: {supported}",
			"select_supported_order_status",
		)


def assert_test_category_supported(category: str, supported: list[str]) -> None:
	"""Test category must belong to the capability's supported set."""
	if category not in supported:
		raise RuleViolation(
			"test_category_not_supported",
			f"test category '{category}' is not supported; valid: {supported}",
			"select_supported_test_category",
		)


def assert_collection_priority_supported(priority: str, supported: list[str]) -> None:
	"""Collection priority must be a supported value."""
	if priority not in supported:
		raise RuleViolation(
			"collection_priority_not_supported",
			f"priority '{priority}' is not supported; valid: {supported}",
			"select_supported_collection_priority",
		)


# ── Specimen rules ─────────────────────────────────────────────────────────────

def assert_specimen_type_supported(specimen_type: str, supported: list[str]) -> None:
	"""Specimen type must be in the configured supported list."""
	if specimen_type not in supported:
		raise RuleViolation(
			"specimen_type_not_supported",
			f"specimen type '{specimen_type}' is not supported; valid: {supported}",
			"select_supported_specimen_type",
		)


def assert_rejection_reason_present(reason: str | None) -> None:
	"""Specimen rejection requires a documented reason."""
	if not reason or not reason.strip():
		raise RuleViolation(
			"rejection_reason_required",
			"a rejection reason must be specified when rejecting a specimen",
			"specify_rejection_reason",
		)


def assert_rejection_reason_supported(reason: str, supported: list[str]) -> None:
	"""Rejection reason must be one of the supported, codified reasons."""
	if reason not in supported:
		raise RuleViolation(
			"rejection_reason_not_supported",
			f"rejection reason '{reason}' is not supported; valid: {supported}",
			"select_supported_rejection_reason",
		)


def assert_specimen_not_rejected(status: str) -> None:
	"""Processing cannot continue on a rejected specimen."""
	if status == "rejected":
		raise RuleViolation(
			"rejected_specimen_not_processable",
			"a rejected specimen cannot be processed; recollect if needed",
			"recollect_specimen",
		)


def assert_specimen_not_disposed(status: str) -> None:
	"""No operations are permitted on a disposed specimen."""
	if status == "disposed":
		raise RuleViolation(
			"disposed_specimen_not_modifiable",
			"operations on disposed specimens are prohibited",
			"recollect_specimen",
		)


def assert_specimen_volume_adequate(
	volume_ml: float | None,
	required_ml: float | None,
) -> None:
	"""Collected volume must meet the test's minimum requirement."""
	if volume_ml is None or required_ml is None:
		return  # no volume data; pass through
	if volume_ml < required_ml:
		raise RuleViolation(
			"insufficient_specimen_volume",
			f"collected volume {volume_ml} mL is below required {required_ml} mL",
			"recollect_specimen",
		)


# ── Result rules ───────────────────────────────────────────────────────────────

def assert_specimen_present_for_result(specimen_id: str | None, found: bool) -> None:
	"""A result cannot be entered without a linked, existing specimen."""
	if not specimen_id or not found:
		raise RuleViolation(
			"specimen_required_before_result_entry",
			"a specimen must be collected before entering a result",
			"collect_specimen_first",
		)


def assert_result_status_supported(status: str, supported: list[str]) -> None:
	"""Result status must be a supported lifecycle value."""
	if status not in supported:
		raise RuleViolation(
			"result_status_not_supported",
			f"result status '{status}' is not supported; valid: {supported}",
			"select_supported_result_status",
		)


def assert_critical_value_notification_sent(
	is_critical: bool,
	notification_sent: bool,
) -> None:
	"""A critical value result may not be verified without prior notification."""
	if is_critical and not notification_sent:
		raise RuleViolation(
			"critical_value_notification_required",
			"critical value notification must be sent before result verification",
			"send_critical_value_notification",
		)


def assert_result_validated_for_release(result_status: str) -> None:
	"""Only validated or final results may be released to clinicians."""
	if result_status not in {"validated", "final"}:
		raise RuleViolation(
			"result_must_be_validated_before_release",
			f"result with status '{result_status}' must be validated before release",
			"validate_result_first",
		)


def assert_original_result_present_for_amendment(original_found: bool) -> None:
	"""Result amendments require the original result to exist in the LIS."""
	if not original_found:
		raise RuleViolation(
			"original_result_required_for_amendment",
			"the original result must exist before creating an amendment",
			"reference_original_result",
		)


def assert_critical_value_acknowledgement_present(acknowledged_by: str | None) -> None:
	"""Closing a critical value alert requires a physician acknowledgement."""
	if not acknowledged_by or not acknowledged_by.strip():
		raise RuleViolation(
			"critical_value_acknowledgement_required",
			"a critical value notification must be acknowledged before closing",
			"obtain_critical_value_acknowledgement",
		)


# ── QC rules ───────────────────────────────────────────────────────────────────

def assert_instrument_not_on_qc_hold(qc_status: str) -> None:
	"""Results from an instrument on QC hold may not be verified or released."""
	if qc_status == "qc_hold":
		raise RuleViolation(
			"qc_hold_blocks_result_release",
			"instrument is on QC hold; resolve QC failure before releasing results",
			"resolve_qc_hold_before_releasing_results",
		)


def assert_qc_status_supported(status: str, supported: list[str]) -> None:
	"""QC status must be a supported value."""
	if status not in supported:
		raise RuleViolation(
			"qc_status_not_supported",
			f"QC status '{status}' is not supported; valid: {supported}",
			"select_supported_qc_status",
		)


def assert_instrument_online_for_processing(instrument_status: str) -> None:
	"""Test processing requires the instrument to be online (not offline/maintenance)."""
	if instrument_status not in {"online", "calibrating"}:
		raise RuleViolation(
			"instrument_not_available",
			f"instrument with status '{instrument_status}' is not available for processing",
			"bring_instrument_online",
		)


def assert_qc_lot_not_expired(expiry_date: datetime | None) -> None:
	"""QC material must not be used past its expiry date."""
	if expiry_date is not None and expiry_date < datetime.utcnow():
		raise RuleViolation(
			"qc_material_expired",
			f"QC material expired on {expiry_date.isoformat()}; use fresh lot",
			"replace_qc_material",
		)


# ── Instrument rules ───────────────────────────────────────────────────────────

def assert_instrument_status_supported(status: str, supported: list[str]) -> None:
	"""Instrument status must be a supported lifecycle value."""
	if status not in supported:
		raise RuleViolation(
			"instrument_status_not_supported",
			f"instrument status '{status}' is not supported; valid: {supported}",
			"select_supported_instrument_status",
		)


def assert_calibration_not_overdue(calibration_due_at: datetime | None) -> None:
	"""Instruments with overdue calibration should be flagged before processing."""
	if calibration_due_at is not None and calibration_due_at < datetime.utcnow():
		raise RuleViolation(
			"instrument_calibration_overdue",
			f"instrument calibration was due on {calibration_due_at.isoformat()}",
			"calibrate_instrument_before_use",
		)


# ── Reference range rules ──────────────────────────────────────────────────────

def assert_reference_range_bounds_valid(
	low: float | None,
	high: float | None,
	critical_low: float | None,
	critical_high: float | None,
) -> None:
	"""Reference range bounds must form a logically consistent set."""
	if low is not None and high is not None and high <= low:
		raise RuleViolation(
			"reference_range_high_must_exceed_low",
			f"reference high ({high}) must be greater than low ({low})",
			"correct_reference_range_bounds",
		)
	if critical_low is not None and low is not None and critical_low >= low:
		raise RuleViolation(
			"critical_low_must_be_below_normal_low",
			f"critical_low ({critical_low}) must be below normal low ({low})",
			"correct_critical_low_bound",
		)
	if critical_high is not None and high is not None and critical_high <= high:
		raise RuleViolation(
			"critical_high_must_exceed_normal_high",
			f"critical_high ({critical_high}) must exceed normal high ({high})",
			"correct_critical_high_bound",
		)


# ── Referral rules ─────────────────────────────────────────────────────────────

def assert_referral_specimen_not_disposed(specimen_status: str) -> None:
	"""A disposed specimen cannot be referred to an external lab."""
	if specimen_status == "disposed":
		raise RuleViolation(
			"disposed_specimen_not_referable",
			"a disposed specimen cannot be referred to an external laboratory",
			"recollect_specimen",
		)


def assert_referral_not_duplicate(
	existing_referrals: list[dict[str, Any]],
	specimen_id: str,
	test_code: str,
) -> None:
	"""Prevent duplicate active referrals for the same specimen/test."""
	active = [
		r for r in existing_referrals
		if r.get("specimen_id") == specimen_id
		and r.get("test_code") == test_code
		and r.get("status") not in {"cancelled", "resulted"}
	]
	if active:
		raise RuleViolation(
			"duplicate_referral",
			f"an active referral already exists for specimen '{specimen_id}' / test '{test_code}'",
			"cancel_existing_referral_first",
		)


# ── Agent privilege rules ──────────────────────────────────────────────────────

def assert_agent_privileged_action_approved(
	agent_action: bool,
	privileged_scope: bool,
	human_approval_recorded: bool,
) -> None:
	"""Privileged agent actions require documented human approval."""
	if agent_action and privileged_scope and not human_approval_recorded:
		raise RuleViolation(
			"privileged_agent_action_requires_human_approval",
			"this action is flagged as privileged; human approval must be recorded first",
			"record_human_approval",
		)


# ── TAT / SLA rules ────────────────────────────────────────────────────────────

def assert_stat_tat_not_breached(
	ordered_at: datetime,
	stat_tat_minutes: int,
	warn_only: bool = True,
) -> None:
	"""STAT orders must complete within the configured turnaround time.

	When warn_only=True, logs a warning rather than raising a violation.
	"""
	elapsed = (datetime.utcnow() - ordered_at).total_seconds() / 60
	if elapsed > stat_tat_minutes:
		if not warn_only:
			raise RuleViolation(
				"stat_order_turnaround_exceeded",
				f"STAT order TAT of {elapsed:.0f} min exceeds limit of {stat_tat_minutes} min",
				"escalate_to_lab_supervisor",
			)


def calculate_tat_deadline(
	ordered_at: datetime,
	priority: str,
	routine_tat_minutes: int = 120,
	stat_tat_minutes: int = 60,
	asap_tat_minutes: int = 90,
) -> datetime:
	"""Return the TAT deadline for an order based on priority."""
	mapping = {
		"stat": stat_tat_minutes,
		"asap": asap_tat_minutes,
		"routine": routine_tat_minutes,
		"timed": routine_tat_minutes,
	}
	minutes = mapping.get(priority, routine_tat_minutes)
	return ordered_at + timedelta(minutes=minutes)


# ── Critical value SLA rules ───────────────────────────────────────────────────

def assert_critical_value_sla(
	result_verified_at: datetime,
	notified_at: datetime,
	sla_minutes: int = 60,
) -> None:
	"""Critical value notification must be sent within SLA (default 60 minutes of verification).

	CAP standard: critical value notification within 60 minutes of result
	release, with read-back confirmation.
	"""
	lag_minutes = (notified_at - result_verified_at).total_seconds() / 60
	if lag_minutes > sla_minutes:
		raise RuleViolation(
			"critical_value_sla_breached",
			f"critical value notified {lag_minutes:.0f} min after verification; SLA is {sla_minutes} min",
			"escalate_to_lab_director",
		)


# ── Calculation helpers ────────────────────────────────────────────────────────

def calculate_z_score(measured: float, target: float, sd: float) -> float:
	"""Z-score = (measured - target) / SD.  Returns 0 when SD is zero."""
	if sd == 0:
		return 0.0
	return round((measured - target) / sd, 4)


def calculate_cv_percent(measured: float, sd: float) -> float:
	"""Coefficient of variation as a percentage."""
	if measured == 0:
		return 0.0
	return round((sd / abs(measured)) * 100, 2)


def calculate_delta_percent(current: float, previous: float) -> float:
	"""Absolute percent change between successive results for the same patient."""
	if previous == 0:
		return 100.0 if current != 0 else 0.0
	return round(abs((current - previous) / previous) * 100, 2)


def calculate_rejection_rate_pct(total: int, rejected: int) -> float:
	"""Specimen rejection rate as a percentage."""
	if total == 0:
		return 0.0
	return round((rejected / total) * 100, 2)


def calculate_qc_pass_rate_pct(total: int, passed: int) -> float:
	"""QC run pass rate as a percentage."""
	if total == 0:
		return 0.0
	return round((passed / total) * 100, 2)


def calculate_critical_value_rate_pct(total_results: int, critical_count: int) -> float:
	"""Rate of critical values among all results."""
	if total_results == 0:
		return 0.0
	return round((critical_count / total_results) * 100, 2)


def calculate_on_time_rate_pct(total: int, on_time: int) -> float:
	"""Percentage of orders completed within their TAT target."""
	if total == 0:
		return 0.0
	return round((on_time / total) * 100, 1)


def calculate_percentile(values: list[float], percentile: float) -> float | None:
	"""Compute an arbitrary percentile from a list of values.

	Uses linear interpolation (method='linear').
	Returns None on empty input.
	"""
	if not values:
		return None
	sorted_vals = sorted(values)
	n = len(sorted_vals)
	rank = percentile / 100 * (n - 1)
	lower = int(math.floor(rank))
	upper = int(math.ceil(rank))
	if lower == upper:
		return round(sorted_vals[lower], 2)
	frac = rank - lower
	return round(sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac, 2)
