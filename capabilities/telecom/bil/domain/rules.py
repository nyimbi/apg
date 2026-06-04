"""Deterministic domain rules for Telecom Billing.

Single source of truth for all billing governance decisions.
All functions are pure — no I/O, no side effects.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DISCOUNT_PCT = Decimal("50")
MIN_INVOICE_AMOUNT = Decimal("0")
MAX_CREDIT_LIMIT = Decimal("10_000_000")
MAX_DPD_BEFORE_LEGAL = 90
SUPPORTED_MEDIATION_STATUSES = {
	"raw", "normalised", "rated", "aggregated", "billed",
	"rejected", "held", "duplicate",
}
SUPPORTED_CDR_TYPES = {
	"voice", "sms", "data", "mms", "video_call",
	"roaming", "interconnect", "short_code",
}
SUPPORTED_DUNNING_STEPS = {
	"reminder_1", "reminder_2", "suspension_warning",
	"service_suspended", "legal_notice", "collections", "write_off",
}
SUPPORTED_PAYMENT_METHODS = {
	"bank_transfer", "mobile_money", "credit_card", "debit_card",
	"direct_debit", "cheque", "cash", "voucher", "crypto",
}
SUPPORTED_INVOICE_STATUSES = {
	"draft", "pending_approval", "approved", "sent", "paid",
	"partially_paid", "overdue", "disputed", "cancelled", "written_off",
}
SUPPORTED_SETTLEMENT_STATUSES = {
	"draft", "submitted", "acknowledged", "disputed", "agreed", "paid", "overdue",
}
VALID_ROAMING_ZONES = {
	"domestic", "zone_a", "zone_b", "zone_c", "premium", "global",
}
VALID_DISPUTE_RESOLUTIONS = {"upheld", "rejected", "partial"}


# ---------------------------------------------------------------------------
# Tenant / auth
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a valid tenant context."""
	if not context.get("tenant_id") or not str(context["tenant_id"]).strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all billing operations",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy reference."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant data access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor on tenant '{actor_tenant}' cannot access resource on '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_actor_present(actor_id: str | None) -> None:
	"""All mutations require a non-blank actor_id for audit purposes."""
	if not actor_id or not actor_id.strip():
		raise RuleViolation(
			"actor_required",
			"actor_id must be set for all mutating operations",
			"set_actor_id",
		)


# ---------------------------------------------------------------------------
# CDR rules
# ---------------------------------------------------------------------------

def assert_cdr_source_present(source: str | None) -> None:
	if not source or not source.strip():
		raise RuleViolation(
			"cdr_source_required",
			"CDR must identify the source network element",
			"set_source_field",
		)


def assert_cdr_msisdn_present(msisdn: str | None) -> None:
	if not msisdn or not msisdn.strip():
		raise RuleViolation(
			"cdr_msisdn_required",
			"CDR must contain a valid MSISDN",
			"set_msisdn_field",
		)


def assert_cdr_type_supported(cdr_type: str) -> None:
	if cdr_type.lower() not in SUPPORTED_CDR_TYPES:
		raise RuleViolation(
			"cdr_type_not_supported",
			f"CDR type '{cdr_type}' is not supported; valid: {SUPPORTED_CDR_TYPES}",
			"use_supported_cdr_type",
		)


def assert_cdr_not_duplicate(cdr_id: str, existing_ids: set[str]) -> None:
	if cdr_id in existing_ids:
		raise RuleViolation(
			"duplicate_cdr",
			f"CDR '{cdr_id}' has already been processed",
			"investigate_deduplication",
		)


def assert_cdr_duration_non_negative(duration_seconds: int) -> None:
	if duration_seconds < 0:
		raise RuleViolation(
			"cdr_negative_duration",
			f"CDR duration cannot be negative; got {duration_seconds}",
			"correct_cdr_data",
		)


def assert_cdr_mediation_status_valid(status: str) -> None:
	if status.lower() not in SUPPORTED_MEDIATION_STATUSES:
		raise RuleViolation(
			"cdr_mediation_status_invalid",
			f"Mediation status '{status}' not recognised",
			"use_valid_mediation_status",
		)


# ---------------------------------------------------------------------------
# Rating rules
# ---------------------------------------------------------------------------

def assert_tariff_plan_active(is_active: bool, plan_id: str) -> None:
	if not is_active:
		raise RuleViolation(
			"tariff_plan_inactive",
			f"TariffPlan '{plan_id}' is not active",
			"assign_active_tariff_plan",
		)


def assert_tariff_plan_valid_date(valid_from: datetime, valid_to: datetime | None, event_time: datetime) -> None:
	if event_time < valid_from:
		raise RuleViolation(
			"tariff_not_yet_effective",
			f"TariffPlan effective from {valid_from.date()}, event at {event_time.date()}",
			"use_correct_tariff_period",
		)
	if valid_to and event_time > valid_to:
		raise RuleViolation(
			"tariff_expired",
			f"TariffPlan expired {valid_to.date()}, event at {event_time.date()}",
			"renew_or_replace_tariff_plan",
		)


def assert_rate_non_negative(rate: Decimal, label: str = "rate") -> None:
	if rate < Decimal("0"):
		raise RuleViolation(
			"negative_rate",
			f"{label} cannot be negative; got {rate}",
			"correct_tariff_configuration",
		)


# ---------------------------------------------------------------------------
# Bundle rules
# ---------------------------------------------------------------------------

def assert_bundle_active(status: str, bundle_id: str) -> None:
	if status.lower() not in {"active"}:
		raise RuleViolation(
			"bundle_not_active",
			f"Bundle '{bundle_id}' status is '{status}'; cannot consume from inactive bundle",
			"reactivate_or_purchase_new_bundle",
		)


def assert_bundle_not_expired(valid_to: datetime, now: datetime, bundle_id: str) -> None:
	if now > valid_to:
		raise RuleViolation(
			"bundle_expired",
			f"Bundle '{bundle_id}' expired on {valid_to.date()}",
			"purchase_new_bundle",
		)


def assert_bundle_has_units(remaining: Decimal, bundle_id: str) -> None:
	if remaining <= Decimal("0"):
		raise RuleViolation(
			"bundle_exhausted",
			f"Bundle '{bundle_id}' has no remaining units",
			"purchase_bundle_topup",
		)


# ---------------------------------------------------------------------------
# Invoice rules
# ---------------------------------------------------------------------------

def assert_invoice_in_draft(status: str, invoice_id: str) -> None:
	if status.lower() != "draft":
		raise RuleViolation(
			"invoice_not_draft",
			f"Invoice '{invoice_id}' is in status '{status}'; expected 'draft' to modify",
			"use_adjustment_instead",
		)


def assert_invoice_approvable(status: str, invoice_id: str) -> None:
	if status.lower() not in {"draft", "pending_approval"}:
		raise RuleViolation(
			"invoice_not_approvable",
			f"Invoice '{invoice_id}' in status '{status}' cannot be approved",
			"check_invoice_lifecycle",
		)


def assert_invoice_not_paid(status: str, invoice_id: str) -> None:
	if status.lower() in {"paid", "written_off", "cancelled"}:
		raise RuleViolation(
			"invoice_already_closed",
			f"Invoice '{invoice_id}' in terminal status '{status}' — no further actions",
			"open_new_invoice_or_dispute",
		)


def assert_invoice_amount_positive(amount: Decimal, invoice_id: str) -> None:
	if amount < MIN_INVOICE_AMOUNT:
		raise RuleViolation(
			"invoice_negative_amount",
			f"Invoice '{invoice_id}' total {amount} is negative",
			"correct_line_items",
		)


def assert_approval_reference_present(ref: str | None) -> None:
	if not ref or not ref.strip():
		raise RuleViolation(
			"approval_reference_required",
			"Approval reference is required for this operation",
			"provide_approval_reference",
		)


# ---------------------------------------------------------------------------
# Payment rules
# ---------------------------------------------------------------------------

def assert_payment_method_supported(method: str) -> None:
	if method.lower() not in SUPPORTED_PAYMENT_METHODS:
		raise RuleViolation(
			"payment_method_not_supported",
			f"Payment method '{method}' not supported; valid: {SUPPORTED_PAYMENT_METHODS}",
			"use_supported_payment_method",
		)


def assert_payment_amount_positive(amount: Decimal) -> None:
	if amount <= Decimal("0"):
		raise RuleViolation(
			"payment_amount_must_be_positive",
			f"Payment amount must be > 0; got {amount}",
			"correct_payment_amount",
		)


def assert_payment_not_overpayment(
	amount: Decimal,
	outstanding: Decimal,
	tolerance_pct: Decimal = Decimal("5"),
) -> None:
	"""Warn (not block) on overpayments exceeding 5% of outstanding."""
	if outstanding > Decimal("0") and amount > outstanding * (1 + tolerance_pct / 100):
		raise RuleViolation(
			"payment_overpayment_threshold_exceeded",
			f"Payment {amount} exceeds outstanding {outstanding} by more than {tolerance_pct}%",
			"confirm_overpayment_or_correct_amount",
		)


# ---------------------------------------------------------------------------
# Discount rules
# ---------------------------------------------------------------------------

def assert_discount_pct_within_limit(discount_pct: Decimal) -> None:
	if discount_pct > MAX_DISCOUNT_PCT:
		raise RuleViolation(
			"discount_exceeds_max_allowed",
			f"Discount {discount_pct}% exceeds maximum {MAX_DISCOUNT_PCT}% — requires executive approval",
			"obtain_executive_approval_for_excessive_discount",
		)


def assert_discount_not_expired(valid_to: datetime, now: datetime) -> None:
	if now > valid_to:
		raise RuleViolation(
			"discount_expired",
			f"Discount expired on {valid_to.date()}",
			"renew_discount_agreement",
		)


# ---------------------------------------------------------------------------
# Credit limit rules
# ---------------------------------------------------------------------------

def assert_credit_limit_sane(soft_limit: Decimal, hard_limit: Decimal) -> None:
	if soft_limit >= hard_limit:
		raise RuleViolation(
			"soft_limit_must_be_below_hard_limit",
			f"Soft limit {soft_limit} must be less than hard limit {hard_limit}",
			"correct_credit_limit_configuration",
		)


def assert_within_credit_limit(current_usage: Decimal, hard_limit: Decimal, account_id: str) -> None:
	if current_usage >= hard_limit:
		raise RuleViolation(
			"credit_hard_limit_breached",
			f"Account '{account_id}' usage {current_usage} has reached hard limit {hard_limit}",
			"suspend_services_or_increase_limit",
		)


def assert_credit_limit_not_exceeded(proposed_limit: Decimal) -> None:
	if proposed_limit > MAX_CREDIT_LIMIT:
		raise RuleViolation(
			"credit_limit_exceeds_system_maximum",
			f"Proposed limit {proposed_limit} exceeds system maximum {MAX_CREDIT_LIMIT}",
			"obtain_board_approval_for_limit_increase",
		)


# ---------------------------------------------------------------------------
# Roaming rules
# ---------------------------------------------------------------------------

def assert_roaming_zone_valid(zone: str) -> None:
	if zone.lower() not in VALID_ROAMING_ZONES:
		raise RuleViolation(
			"invalid_roaming_zone",
			f"Roaming zone '{zone}' not recognised; valid: {VALID_ROAMING_ZONES}",
			"use_valid_roaming_zone",
		)


def assert_tap_reference_present(tap_ref: str | None) -> None:
	"""TAP file reference required for wholesale roaming settlement."""
	if not tap_ref or not tap_ref.strip():
		raise RuleViolation(
			"tap_reference_required",
			"TAP/NRTRDE file reference required for roaming settlement",
			"attach_tap_file_reference",
		)


# ---------------------------------------------------------------------------
# Interconnect rules
# ---------------------------------------------------------------------------

def assert_settlement_period_valid(period_start: datetime, period_end: datetime) -> None:
	if period_end <= period_start:
		raise RuleViolation(
			"settlement_period_invalid",
			f"period_end {period_end} must be after period_start {period_start}",
			"correct_settlement_period",
		)


def assert_carrier_id_present(carrier_id: str | None) -> None:
	if not carrier_id or not carrier_id.strip():
		raise RuleViolation(
			"carrier_id_required",
			"carrier_id is required for interconnect operations",
			"set_carrier_id",
		)


# ---------------------------------------------------------------------------
# Dispute rules
# ---------------------------------------------------------------------------

def assert_dispute_amount_valid(disputed_amount: Decimal, invoice_total: Decimal) -> None:
	if disputed_amount > invoice_total:
		raise RuleViolation(
			"disputed_amount_exceeds_invoice",
			f"Disputed amount {disputed_amount} cannot exceed invoice total {invoice_total}",
			"correct_dispute_amount",
		)


def assert_dispute_resolution_valid(resolution: str) -> None:
	if resolution not in VALID_DISPUTE_RESOLUTIONS:
		raise RuleViolation(
			"invalid_dispute_resolution",
			f"Resolution '{resolution}' invalid; must be one of {VALID_DISPUTE_RESOLUTIONS}",
			"use_valid_resolution",
		)


def assert_dispute_open(status: str, dispute_id: str) -> None:
	closed = {"resolved_upheld", "resolved_rejected", "withdrawn"}
	if status in closed:
		raise RuleViolation(
			"dispute_already_closed",
			f"Dispute '{dispute_id}' is in terminal status '{status}'",
			"open_new_dispute_if_required",
		)


# ---------------------------------------------------------------------------
# Dunning rules
# ---------------------------------------------------------------------------

def assert_dunning_step_valid(step: str) -> None:
	if step.lower() not in SUPPORTED_DUNNING_STEPS:
		raise RuleViolation(
			"dunning_step_not_supported",
			f"Dunning step '{step}' not recognised; valid: {SUPPORTED_DUNNING_STEPS}",
			"use_valid_dunning_step",
		)


def assert_dunning_sequence(current_step: str | None, new_step: str) -> None:
	"""Dunning must escalate, never de-escalate."""
	order = [
		"reminder_1", "reminder_2", "suspension_warning",
		"service_suspended", "legal_notice", "collections", "write_off",
	]
	if current_step is None:
		return
	try:
		cur_idx = order.index(current_step)
		new_idx = order.index(new_step)
		if new_idx < cur_idx:
			raise RuleViolation(
				"dunning_de_escalation_not_permitted",
				f"Cannot de-escalate from '{current_step}' to '{new_step}'",
				"use_higher_dunning_step",
			)
	except ValueError:
		pass  # unknown step — let assert_dunning_step_valid handle


# ---------------------------------------------------------------------------
# Revenue assurance rules
# ---------------------------------------------------------------------------

def assert_leakage_pct_acceptable(leakage_pct: Decimal, threshold_pct: Decimal = Decimal("2")) -> None:
	if leakage_pct > threshold_pct:
		raise RuleViolation(
			"revenue_leakage_threshold_breached",
			f"Leakage {leakage_pct}% exceeds threshold {threshold_pct}%",
			"investigate_unrated_cdrs",
		)


def assert_collection_rate_acceptable(
	collection_rate_pct: Decimal,
	min_acceptable: Decimal = Decimal("80"),
) -> None:
	if collection_rate_pct < min_acceptable:
		raise RuleViolation(
			"collection_rate_below_threshold",
			f"Collection rate {collection_rate_pct}% is below minimum {min_acceptable}%",
			"escalate_collections_process",
		)


# ---------------------------------------------------------------------------
# Calculation helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def calculate_dpd(due_date: datetime, as_of: datetime) -> int:
	"""Days past due. 0 if not yet due."""
	delta = (as_of.date() - due_date.date()).days
	return max(0, delta)


def calculate_outstanding_balance(
	total_amount: Decimal,
	paid_amount: Decimal,
	credit_notes: Decimal = Decimal("0"),
) -> Decimal:
	"""Net balance due after payments and credits."""
	return max(Decimal("0"), total_amount - paid_amount - credit_notes)


def calculate_penalty_accrual(
	outstanding: Decimal,
	dpd: int,
	daily_rate_pct: Decimal = Decimal("0.1"),
	cap_pct: Decimal = Decimal("10"),
) -> Decimal:
	"""Accrued penalty capped at cap_pct of outstanding."""
	from decimal import ROUND_HALF_UP
	accrued_pct = Decimal(str(dpd)) * daily_rate_pct
	effective_pct = min(accrued_pct, cap_pct)
	return (outstanding * effective_pct / Decimal("100")).quantize(
		Decimal("0.01"), rounding=ROUND_HALF_UP
	)


def calculate_realtime_spend_headroom(
	current_usage: Decimal,
	hard_limit: Decimal,
	pending_charge: Decimal,
) -> dict[str, Any]:
	"""Returns headroom and whether the pending charge can proceed."""
	headroom = max(Decimal("0"), hard_limit - current_usage)
	can_proceed = pending_charge <= headroom
	return {
		"headroom": headroom,
		"pending_charge": pending_charge,
		"can_proceed": can_proceed,
		"utilisation_pct": (
			(current_usage / hard_limit * 100).quantize(Decimal("0.01"))
			if hard_limit > Decimal("0") else Decimal("0")
		),
	}
