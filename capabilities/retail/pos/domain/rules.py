"""Deterministic business rules for APG Point of Sale.

All governance decisions flow through these functions.
RuleViolation is raised — never silently swallowed.
assert_* functions guard state transitions.
calculate_* functions are side-effect-free computations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(
		self,
		rule_name: str,
		reason: str,
		required_action: str = "",
		context: dict[str, Any] | None = None,
	) -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		self.context = context or {}
		super().__init__(f"[{rule_name}] {reason}")


# ---------------------------------------------------------------------------
# Tenant & auth
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant data access is strictly forbidden."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_permission(actor_permissions: list[str], required: str) -> None:
	"""Actor must hold the required permission."""
	if required not in actor_permissions and "superadmin" not in actor_permissions:
		raise RuleViolation(
			"insufficient_permission",
			f"permission '{required}' required",
			f"grant_{required}_to_actor",
		)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def assert_terminal_not_suspended(terminal_status: str) -> None:
	if terminal_status == "suspended":
		raise RuleViolation(
			"terminal_suspended",
			"terminal is suspended and cannot process transactions",
			"take_terminal_out_of_suspension",
		)


def assert_terminal_online(terminal_status: str, offline_mode: bool = False) -> None:
	"""Terminal must be online unless offline mode is explicitly set."""
	if terminal_status == "offline" and not offline_mode:
		raise RuleViolation(
			"terminal_offline",
			"terminal is offline; set offline_mode=True to proceed without connectivity",
			"connect_terminal_or_enable_offline_mode",
		)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def assert_session_open(session_status: str) -> None:
	if session_status != "open":
		raise RuleViolation(
			"session_not_open",
			f"session status is '{session_status}'; must be 'open'",
			"open_a_session",
		)


def assert_no_duplicate_open_session(existing_open_session_id: str | None) -> None:
	"""A terminal can hold at most one open session at a time."""
	if existing_open_session_id is not None:
		raise RuleViolation(
			"duplicate_open_session",
			f"terminal already has open session '{existing_open_session_id}'",
			"close_existing_session_first",
		)


def assert_opening_float_non_negative(opening_float: float) -> None:
	if opening_float < 0:
		raise RuleViolation(
			"negative_opening_float",
			f"opening float {opening_float} must be ≥ 0",
			"provide_non_negative_float",
		)


def assert_session_has_no_pending_transactions(pending_count: int) -> None:
	"""Session cannot be closed while transactions are pending."""
	if pending_count > 0:
		raise RuleViolation(
			"pending_transactions_on_close",
			f"{pending_count} transaction(s) are still PENDING; complete or void them first",
			"complete_or_void_pending_transactions",
		)


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

def assert_transaction_pending(txn_status: str) -> None:
	if txn_status != "pending":
		raise RuleViolation(
			"transaction_not_pending",
			f"transaction status is '{txn_status}'; must be 'pending'",
			"begin_new_transaction",
		)


def assert_transaction_voidable(txn_status: str) -> None:
	if txn_status not in ("pending", "completed", "authorised"):
		raise RuleViolation(
			"transaction_not_voidable",
			f"cannot void a transaction in status '{txn_status}'",
			"check_transaction_status",
		)


def assert_transaction_refundable(txn_status: str) -> None:
	if txn_status not in ("completed", "partially_refunded"):
		raise RuleViolation(
			"transaction_not_refundable",
			f"transaction status '{txn_status}' is not refundable",
			"ensure_transaction_is_completed",
		)


def assert_void_same_session(txn_session_id: str, void_session_id: str) -> None:
	if txn_session_id != void_session_id:
		raise RuleViolation(
			"cross_session_void_denied",
			"voids must occur within the same session as the original transaction",
			"open_session_that_owns_transaction",
		)


def assert_void_same_terminal(txn_terminal_id: str, void_terminal_id: str) -> None:
	if txn_terminal_id != void_terminal_id:
		raise RuleViolation(
			"cross_terminal_void_denied",
			"void must originate from the originating terminal",
			"use_originating_terminal",
		)


def assert_refund_items_in_original(
	refund_skus: list[str],
	original_skus: list[str],
) -> None:
	extra = set(refund_skus) - set(original_skus)
	if extra:
		raise RuleViolation(
			"refund_items_not_in_original",
			f"SKUs not in original transaction: {extra}",
			"only_refund_purchased_items",
		)


def assert_refund_quantity_valid(
	refund_qty: float,
	original_qty: float,
	already_refunded_qty: float,
	sku: str,
) -> None:
	available = original_qty - already_refunded_qty
	if refund_qty > available:
		raise RuleViolation(
			"refund_quantity_exceeds_original",
			f"SKU '{sku}': refund qty {refund_qty} > available {available}",
			"reduce_refund_quantity",
		)


def assert_transaction_has_items(item_count: int) -> None:
	if item_count == 0:
		raise RuleViolation(
			"empty_transaction",
			"transaction must contain at least one item",
			"add_items_before_completing",
		)


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------

def assert_sufficient_payment(
	amount_tendered: float,
	grand_total: float,
	tolerance: float = 0.005,
) -> None:
	if amount_tendered < grand_total - tolerance:
		raise RuleViolation(
			"insufficient_payment",
			f"tendered {amount_tendered:.2f} < due {grand_total:.2f}",
			"collect_remaining_balance",
		)


def assert_payment_amount_positive(amount: float) -> None:
	if amount <= 0:
		raise RuleViolation(
			"non_positive_payment",
			f"payment amount {amount} must be > 0",
			"provide_positive_amount",
		)


def assert_loyalty_points_sufficient(balance: int, points_to_redeem: int) -> None:
	if balance < points_to_redeem:
		raise RuleViolation(
			"insufficient_loyalty_points",
			f"balance {balance} points < {points_to_redeem} to redeem",
			"reduce_points_redemption",
		)


def assert_loyalty_redemption_within_limit(
	points_value: float,
	grand_total: float,
	max_redemption_pct: float = 0.5,
) -> None:
	"""Points cannot cover more than max_redemption_pct of the transaction."""
	limit = round(grand_total * max_redemption_pct, 2)
	if points_value > limit:
		raise RuleViolation(
			"loyalty_redemption_limit_exceeded",
			f"loyalty redemption value {points_value:.2f} exceeds {max_redemption_pct*100:.0f}% "
			f"of transaction total ({limit:.2f})",
			"reduce_points_redemption",
		)


def assert_floor_limit(amount: float, floor_limit: float) -> None:
	"""Single payment above floor_limit requires supervisor authorisation."""
	if amount > floor_limit:
		raise RuleViolation(
			"floor_limit_exceeded",
			f"payment {amount:.2f} exceeds floor limit {floor_limit:.2f}; supervisor authorisation required",
			"obtain_supervisor_override",
		)


# ---------------------------------------------------------------------------
# Discount
# ---------------------------------------------------------------------------

def assert_discount_percentage_valid(pct: float) -> None:
	if not 0 < pct <= 100:
		raise RuleViolation(
			"invalid_discount_percentage",
			f"discount percentage {pct} must be in (0, 100]",
			"provide_valid_percentage",
		)


def assert_discount_not_expired(valid_until: datetime | None, now: datetime | None = None) -> None:
	if valid_until is None:
		return
	if (now or datetime.utcnow()) > valid_until:
		raise RuleViolation(
			"discount_expired",
			f"discount expired at {valid_until.isoformat()}",
			"use_active_discount",
		)


def assert_discount_active(valid_from: datetime | None, now: datetime | None = None) -> None:
	if valid_from is None:
		return
	if (now or datetime.utcnow()) < valid_from:
		raise RuleViolation(
			"discount_not_yet_active",
			f"discount not valid until {valid_from.isoformat()}",
			"use_active_discount",
		)


def assert_minimum_purchase_met(subtotal: float, min_purchase: float | None) -> None:
	if min_purchase is not None and subtotal < min_purchase:
		raise RuleViolation(
			"minimum_purchase_not_met",
			f"subtotal {subtotal:.2f} < minimum purchase {min_purchase:.2f}",
			"add_more_items_to_qualify",
		)


def assert_coupon_not_exhausted(times_used: int, max_uses: int | None) -> None:
	if max_uses is not None and times_used >= max_uses:
		raise RuleViolation(
			"coupon_exhausted",
			f"coupon used {times_used}/{max_uses} times",
			"use_different_coupon",
		)


def assert_supervisor_present_for_manager_discount(
	discount_type: str,
	supervisor_id: str | None,
) -> None:
	if discount_type in ("manager", "staff") and not supervisor_id:
		raise RuleViolation(
			"supervisor_required_for_manager_discount",
			f"'{discount_type}' discounts require supervisor authorisation",
			"obtain_supervisor_override",
		)


# ---------------------------------------------------------------------------
# Price override
# ---------------------------------------------------------------------------

def assert_price_override_non_negative(override_price: float) -> None:
	if override_price < 0:
		raise RuleViolation(
			"negative_override_price",
			f"override price {override_price} must be ≥ 0",
			"provide_non_negative_price",
		)


def assert_price_override_changed(original: float, override: float) -> None:
	if original == override:
		raise RuleViolation(
			"price_override_unchanged",
			"override price is identical to original price",
			"provide_different_price",
		)


def assert_price_override_supervisor(supervisor_id: str | None) -> None:
	if not supervisor_id:
		raise RuleViolation(
			"supervisor_required_for_price_override",
			"price overrides require supervisor authorisation",
			"obtain_supervisor_override",
		)


# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

def assert_tax_exempt_ref(tax_exempt: bool, tax_exempt_ref: str | None) -> None:
	"""Tax-exempt transactions must carry a reference number."""
	if tax_exempt and not tax_exempt_ref:
		raise RuleViolation(
			"tax_exempt_ref_required",
			"tax_exempt_ref is required when tax_exempt=True",
			"provide_tax_exemption_certificate_number",
		)


# ---------------------------------------------------------------------------
# Offline sync
# ---------------------------------------------------------------------------

def assert_offline_sync_sequence_monotone(last_sequence: int, incoming_sequence: int) -> None:
	if incoming_sequence <= last_sequence:
		raise RuleViolation(
			"sync_sequence_not_monotone",
			f"incoming sync_sequence {incoming_sequence} ≤ last accepted {last_sequence}",
			"resend_with_correct_sequence",
		)


def assert_offline_transaction_count_within_limit(count: int, limit: int = 200) -> None:
	if count > limit:
		raise RuleViolation(
			"offline_transaction_limit_exceeded",
			f"{count} offline transactions exceed limit of {limit} per sync batch",
			"split_into_smaller_batches",
		)


# ---------------------------------------------------------------------------
# Cash management
# ---------------------------------------------------------------------------

def assert_safe_drop_authorised(amount: float, authorised_by: str | None) -> None:
	if amount > 0 and not authorised_by:
		raise RuleViolation(
			"safe_drop_requires_authorisation",
			"safe drops require manager authorisation",
			"provide_authorised_by",
		)


def assert_cash_variance_within_tolerance(variance: float, tolerance: float = 50.0) -> None:
	if abs(variance) > tolerance:
		raise RuleViolation(
			"cash_variance_exceeds_tolerance",
			f"cash variance {variance:.2f} exceeds tolerance ±{tolerance:.2f}",
			"investigate_till_discrepancy",
		)


# ---------------------------------------------------------------------------
# EOD
# ---------------------------------------------------------------------------

def assert_all_sessions_closed_for_eod(open_session_count: int) -> None:
	if open_session_count > 0:
		raise RuleViolation(
			"open_sessions_prevent_eod",
			f"{open_session_count} session(s) are still open; close them before running EOD",
			"close_all_sessions",
		)


def assert_eod_not_already_run(existing_eod_id: str | None, business_date: str) -> None:
	if existing_eod_id is not None:
		raise RuleViolation(
			"eod_already_run",
			f"end-of-day report for {business_date} already exists (id={existing_eod_id})",
			"review_existing_eod_report",
		)


# ---------------------------------------------------------------------------
# Supervisor override
# ---------------------------------------------------------------------------

def assert_supervisor_not_self_approving(cashier_id: str, supervisor_id: str) -> None:
	if cashier_id == supervisor_id:
		raise RuleViolation(
			"self_approval_denied",
			"cashier cannot approve their own override",
			"use_different_supervisor",
		)


def assert_override_not_expired(expires_at: datetime | None) -> None:
	if expires_at and datetime.utcnow() > expires_at:
		raise RuleViolation(
			"override_expired",
			f"supervisor override expired at {expires_at.isoformat()}",
			"request_new_supervisor_override",
		)


# ---------------------------------------------------------------------------
# Promotional race-condition guard
# ---------------------------------------------------------------------------

def assert_promotion_price_stable(
	promotion_id: str,
	price_at_basket_add: float,
	current_price: float,
	tolerance: float = 0.005,
) -> None:
	"""Guard against promotion price changing between basket-add and complete."""
	if abs(price_at_basket_add - current_price) > tolerance:
		raise RuleViolation(
			"promotion_price_race_condition",
			f"promotion '{promotion_id}' price changed from {price_at_basket_add:.4f} "
			f"to {current_price:.4f} during transaction",
			"recheck_promotion_price_with_customer",
		)


# ---------------------------------------------------------------------------
# calculate_* helpers (pure, side-effect-free)
# ---------------------------------------------------------------------------

def calculate_change_due(grand_total: float, amount_tendered: float) -> float:
	return round(max(amount_tendered - grand_total, 0.0), 2)


def calculate_balance_due(grand_total: float, payments_total: float) -> float:
	return round(max(grand_total - payments_total, 0.0), 2)


def calculate_loyalty_points_earned(purchase_amount: float, earn_rate: float = 1.0) -> int:
	return int(purchase_amount * earn_rate)


def calculate_loyalty_redemption_value(points: int, redeem_rate: float = 0.01) -> float:
	return round(points * redeem_rate, 2)


def calculate_vat_from_inclusive(
	inclusive_amount: float,
	vat_rate: float = 0.16,
) -> dict[str, float]:
	vat = round(inclusive_amount * vat_rate / (1 + vat_rate), 4)
	net = round(inclusive_amount - vat, 4)
	return {"net": round(net, 2), "vat": round(vat, 2), "gross": round(inclusive_amount, 2)}


def calculate_discount_amount(subtotal: float, discount_type: str, value: float) -> float:
	if discount_type == "percentage":
		assert 0 < value <= 100
		return round(subtotal * value / 100, 2)
	elif discount_type == "fixed_amount":
		return round(min(value, subtotal), 2)
	return 0.0
