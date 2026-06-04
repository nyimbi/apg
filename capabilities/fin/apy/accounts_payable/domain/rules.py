"""
Accounts Payable — Domain Rules
© 2025 Datacraft. All rights reserved.

Every business rule governing AP lifecycle, encoded as pure callables.
Violations raise RuleViolation with enough context for caller to surface
a meaningful error message without leaking internal state.

Rule naming convention:
  assert_*   — raises RuleViolation if invariant is broken
  check_*    — returns (ok: bool, reason: str)
  calculate_ — pure computation (see calculations.py for numeric formulas)
"""
from __future__ import annotations

from datetime import date, datetime
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
		super().__init__(f"[{rule_name}] {reason}")

	def to_dict(self) -> dict[str, str]:
		return {
			"rule": self.rule_name,
			"reason": self.reason,
			"required_action": self.required_action,
		}


# ---------------------------------------------------------------------------
# Tenant & access control
# ---------------------------------------------------------------------------

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id:
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all AP operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Resources must belong to the actor's tenant."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_actor_present(actor_id: str | None) -> None:
	"""Mutable operations must identify the actor for audit trail."""
	if not actor_id:
		raise RuleViolation(
			"actor_required",
			"actor_id must be provided for all write operations",
			"provide_actor_id",
		)


# ---------------------------------------------------------------------------
# Supplier rules
# ---------------------------------------------------------------------------

def assert_supplier_active(supplier: dict[str, Any]) -> None:
	"""Invoices can only be created against active suppliers."""
	status = supplier.get("status", "")
	if status not in ("active",):
		raise RuleViolation(
			"supplier_not_active",
			f"supplier '{supplier.get('supplier_number')}' has status '{status}' — cannot receive invoices",
			"reactivate_supplier_or_use_different_supplier",
		)


def assert_supplier_not_on_hold(supplier: dict[str, Any]) -> None:
	"""Payment runs must skip suppliers on payment hold."""
	if supplier.get("on_hold", False):
		raise RuleViolation(
			"supplier_on_payment_hold",
			f"supplier '{supplier.get('supplier_number')}' is on payment hold",
			"release_supplier_hold_before_payment",
		)


def assert_supplier_has_bank_account(supplier: dict[str, Any]) -> None:
	"""Electronic payments require at least one verified bank account."""
	bank_accounts = supplier.get("bank_accounts", [])
	active = [b for b in bank_accounts if b.get("is_active") and b.get("verified")]
	if not active:
		raise RuleViolation(
			"no_verified_bank_account",
			f"supplier '{supplier.get('supplier_number')}' has no verified active bank account",
			"verify_supplier_bank_account",
		)


def assert_po_required_when_configured(supplier: dict[str, Any], invoice: dict[str, Any]) -> None:
	"""If supplier is configured to require a PO, invoice must reference one."""
	if supplier.get("po_required") and not invoice.get("po_refs"):
		raise RuleViolation(
			"po_required_for_supplier",
			f"supplier '{supplier.get('supplier_number')}' requires a PO reference on all invoices",
			"attach_po_reference_to_invoice",
		)


def check_supplier_credit_limit(
	supplier: dict[str, Any],
	new_invoice_total: Decimal,
	current_outstanding: Decimal,
) -> tuple[bool, str]:
	"""
	Returns (ok, reason). Does not raise — caller decides whether to block or warn.
	Credit limit in AP context is a payable credit ceiling, not a receivable limit.
	"""
	limit = supplier.get("credit_limit")
	if limit is None:
		return True, ""
	limit_d = Decimal(str(limit))
	projected = current_outstanding + new_invoice_total
	if projected > limit_d:
		return (
			False,
			f"invoice would take outstanding ({projected}) above credit limit ({limit_d})",
		)
	return True, ""


# ---------------------------------------------------------------------------
# Invoice rules
# ---------------------------------------------------------------------------

def assert_invoice_dates_valid(invoice_date: date, due_date: date, received_date: date) -> None:
	"""Due date must be on or after invoice date; received date must be <= today."""
	if due_date < invoice_date:
		raise RuleViolation(
			"due_date_before_invoice_date",
			f"due_date {due_date} is before invoice_date {invoice_date}",
			"correct_due_date",
		)
	if received_date > date.today():
		raise RuleViolation(
			"received_date_in_future",
			f"received_date {received_date} cannot be in the future",
			"correct_received_date",
		)


def assert_invoice_has_lines(lines: list[Any]) -> None:
	if not lines:
		raise RuleViolation(
			"invoice_must_have_lines",
			"invoice must contain at least one line item",
			"add_invoice_lines",
		)


def assert_invoice_not_duplicate(
	duplicate_check_result: dict[str, Any],
	force: bool = False,
) -> None:
	"""
	Block exact duplicates always. Fuzzy matches block unless force=True
	(requires supervisor override with audit trail).
	"""
	exact = duplicate_check_result.get("exact_match_ids", [])
	if exact:
		raise RuleViolation(
			"exact_duplicate_invoice",
			f"invoice is an exact duplicate of: {exact}",
			"cancel_duplicate_or_use_existing",
		)
	fuzzy = duplicate_check_result.get("fuzzy_match_ids", [])
	confidence = duplicate_check_result.get("confidence", 0.0)
	if fuzzy and confidence >= 0.75 and not force:
		raise RuleViolation(
			"probable_duplicate_invoice",
			f"invoice matches existing invoices with {confidence:.0%} confidence: {fuzzy}",
			"review_duplicates_and_resubmit_with_force_flag",
		)


def assert_invoice_can_transition(current_status: str, target_status: str) -> None:
	"""Enforce the AP invoice state machine."""
	_TRANSITIONS: dict[str, set[str]] = {
		"received":      {"validated", "rejected", "cancelled", "disputed", "on_hold"},
		"validated":     {"matched", "approved", "rejected", "on_hold", "disputed"},
		"matched":       {"approved", "disputed", "on_hold", "rejected"},
		"approved":      {"posted", "on_hold", "disputed", "cancelled"},
		"posted":        {"paid", "partially_paid", "disputed", "on_hold"},
		"partially_paid": {"paid", "disputed", "on_hold"},
		"paid":          {"disputed"},   # rare — overpayment / returned payment
		"disputed":      {"approved", "rejected", "cancelled", "on_hold", "validated"},
		"on_hold":       {"validated", "matched", "approved", "disputed", "cancelled", "rejected"},
		"cancelled":     set(),
		"rejected":      {"received"},   # supplier can resubmit
		"duplicate":     {"cancelled"},
	}
	allowed = _TRANSITIONS.get(current_status, set())
	if target_status not in allowed:
		raise RuleViolation(
			"invalid_status_transition",
			f"cannot transition invoice from '{current_status}' to '{target_status}'",
			f"valid_targets_from_{current_status}: {sorted(allowed)}",
		)


def assert_invoice_postable(invoice: dict[str, Any]) -> None:
	"""Invoice must be approved and not on hold before posting to GL."""
	status = invoice.get("status", "")
	if status != "approved":
		raise RuleViolation(
			"invoice_not_approved",
			f"invoice status is '{status}' — must be 'approved' before posting",
			"approve_invoice_first",
		)
	if invoice.get("on_hold"):
		raise RuleViolation(
			"invoice_on_hold",
			"cannot post an invoice that is on hold",
			"release_hold_before_posting",
		)


def assert_invoice_payable(invoice: dict[str, Any]) -> None:
	"""Invoice must be posted and have outstanding balance before payment selection."""
	status = invoice.get("status", "")
	if status not in ("posted", "partially_paid"):
		raise RuleViolation(
			"invoice_not_payable",
			f"invoice status is '{status}' — must be 'posted' or 'partially_paid' for payment",
			"post_invoice_before_payment",
		)
	outstanding = Decimal(str(invoice.get("outstanding", "0")))
	if outstanding <= Decimal("0"):
		raise RuleViolation(
			"invoice_fully_paid",
			"invoice has no outstanding balance",
			"no_action_required",
		)


def assert_self_billing_supplier(supplier: dict[str, Any]) -> None:
	"""Self-billed invoices require the supplier to have self-billing enabled."""
	if not supplier.get("self_billing_enabled"):
		raise RuleViolation(
			"self_billing_not_enabled",
			f"supplier '{supplier.get('supplier_number')}' does not have self-billing enabled",
			"enable_self_billing_on_supplier",
		)


# ---------------------------------------------------------------------------
# Three-way match rules
# ---------------------------------------------------------------------------

def assert_match_references_present(
	po_matching_type: str,
	po_refs: list[str],
	grn_refs: list[str],
) -> None:
	"""2-way needs a PO; 3-way needs both PO and GRN."""
	if po_matching_type == "two_way" and not po_refs:
		raise RuleViolation(
			"po_ref_required_for_2way_match",
			"2-way matching requires at least one PO reference",
			"add_po_reference",
		)
	if po_matching_type == "three_way":
		if not po_refs:
			raise RuleViolation(
				"po_ref_required_for_3way_match",
				"3-way matching requires a PO reference",
				"add_po_reference",
			)
		if not grn_refs:
			raise RuleViolation(
				"grn_ref_required_for_3way_match",
				"3-way matching requires a GRN reference",
				"add_grn_reference",
			)


def assert_match_within_tolerance(
	price_variance_pct: Decimal,
	qty_variance_pct: Decimal,
	price_tolerance_pct: Decimal,
	qty_tolerance_pct: Decimal,
) -> None:
	"""
	Both price and quantity variances must be within configured tolerances.
	Absolute values used — direction of variance is recorded separately.
	"""
	if abs(price_variance_pct) > price_tolerance_pct:
		raise RuleViolation(
			"price_variance_exceeds_tolerance",
			f"price variance {price_variance_pct:.4f}% exceeds tolerance {price_tolerance_pct}%",
			"review_and_dispute_or_request_credit_note",
		)
	if abs(qty_variance_pct) > qty_tolerance_pct:
		raise RuleViolation(
			"qty_variance_exceeds_tolerance",
			f"quantity variance {qty_variance_pct:.4f}% exceeds tolerance {qty_tolerance_pct}%",
			"reconcile_quantities_with_supplier",
		)


# ---------------------------------------------------------------------------
# Payment run rules
# ---------------------------------------------------------------------------

def assert_payment_run_approvable(run: dict[str, Any]) -> None:
	if run.get("status") != "draft":
		raise RuleViolation(
			"payment_run_not_draft",
			f"payment run status is '{run.get('status')}' — only draft runs can be approved",
			"recreate_payment_run",
		)
	if not run.get("invoices_selected"):
		raise RuleViolation(
			"payment_run_empty",
			"payment run has no invoices selected",
			"run_payment_selection_first",
		)


def assert_payment_run_processable(run: dict[str, Any]) -> None:
	if run.get("status") != "approved":
		raise RuleViolation(
			"payment_run_not_approved",
			f"payment run must be in 'approved' status to process, got '{run.get('status')}'",
			"approve_payment_run_first",
		)


def assert_payment_amount_matches(
	allocated_total: Decimal,
	invoice_outstanding: Decimal,
	tolerance: Decimal = Decimal("0.01"),
) -> None:
	"""Allocated amount must not exceed outstanding balance (within rounding tolerance)."""
	if allocated_total > invoice_outstanding + tolerance:
		raise RuleViolation(
			"payment_exceeds_outstanding",
			f"allocated {allocated_total} exceeds outstanding {invoice_outstanding}",
			"reduce_payment_allocation",
		)


def assert_no_duplicate_payment(
	existing_payment_refs: list[str],
	new_ref: str,
) -> None:
	if new_ref in existing_payment_refs:
		raise RuleViolation(
			"duplicate_payment_reference",
			f"payment reference '{new_ref}' already exists",
			"use_unique_payment_reference",
		)


# ---------------------------------------------------------------------------
# Dispute rules
# ---------------------------------------------------------------------------

def assert_dispute_amount_valid(
	disputed_amount: Decimal,
	invoice_total: Decimal,
) -> None:
	if disputed_amount <= Decimal("0"):
		raise RuleViolation(
			"dispute_amount_must_be_positive",
			"disputed amount must be greater than zero",
			"provide_positive_disputed_amount",
		)
	if disputed_amount > invoice_total:
		raise RuleViolation(
			"dispute_amount_exceeds_invoice",
			f"disputed amount {disputed_amount} exceeds invoice total {invoice_total}",
			"reduce_disputed_amount",
		)


def assert_dispute_resolvable(dispute: dict[str, Any]) -> None:
	if dispute.get("status") in ("resolved_accepted", "resolved_rejected", "closed"):
		raise RuleViolation(
			"dispute_already_resolved",
			f"dispute is already in status '{dispute.get('status')}'",
			"no_action_required",
		)


# ---------------------------------------------------------------------------
# Accrual rules
# ---------------------------------------------------------------------------

def assert_accrual_period_format(period: str) -> None:
	"""Period must be YYYY-MM."""
	import re
	if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period):
		raise RuleViolation(
			"invalid_accrual_period",
			f"accounting period '{period}' must be in YYYY-MM format",
			"use_YYYY-MM_format",
		)


def assert_accrual_not_reversed(accrual: dict[str, Any]) -> None:
	if accrual.get("reversed"):
		raise RuleViolation(
			"accrual_already_reversed",
			f"accrual '{accrual.get('accrual_number')}' has already been reversed",
			"no_action_required",
		)


def assert_accrual_posted(accrual: dict[str, Any]) -> None:
	if not accrual.get("posted"):
		raise RuleViolation(
			"accrual_not_posted",
			f"accrual '{accrual.get('accrual_number')}' must be posted before it can be reversed",
			"post_accrual_first",
		)


# ---------------------------------------------------------------------------
# Credit note rules
# ---------------------------------------------------------------------------

def assert_credit_note_not_over_applied(
	credit_amount: Decimal,
	already_applied: Decimal,
	new_application: Decimal,
) -> None:
	remaining = credit_amount - already_applied
	if new_application > remaining:
		raise RuleViolation(
			"credit_note_over_applied",
			f"applying {new_application} would exceed remaining credit balance {remaining}",
			"reduce_application_amount",
		)


# ---------------------------------------------------------------------------
# FX / multi-currency rules
# ---------------------------------------------------------------------------

def assert_exchange_rate_positive(rate: Decimal) -> None:
	if rate <= Decimal("0"):
		raise RuleViolation(
			"invalid_exchange_rate",
			f"exchange rate must be positive, got {rate}",
			"provide_valid_exchange_rate",
		)


def assert_payment_currency_consistent(
	invoice_currency: str,
	payment_currency: str,
	allow_cross_currency: bool = True,
) -> None:
	"""
	Cross-currency payments are allowed (with FX gain/loss booking) unless
	the system is configured to disallow them.
	"""
	if not allow_cross_currency and invoice_currency != payment_currency:
		raise RuleViolation(
			"cross_currency_payment_not_allowed",
			f"invoice currency '{invoice_currency}' does not match payment currency '{payment_currency}'",
			"use_matching_currency_or_enable_cross_currency",
		)


# ---------------------------------------------------------------------------
# Period close rules
# ---------------------------------------------------------------------------

def assert_no_open_invoices_for_period_close(
	open_invoice_count: int,
	period: str,
) -> None:
	if open_invoice_count > 0:
		raise RuleViolation(
			"open_invoices_block_period_close",
			f"period '{period}' has {open_invoice_count} open invoices — resolve before closing",
			"process_or_accrue_open_invoices",
		)


def assert_no_unposted_accruals_for_period_close(
	unposted_count: int,
	period: str,
) -> None:
	if unposted_count > 0:
		raise RuleViolation(
			"unposted_accruals_block_period_close",
			f"period '{period}' has {unposted_count} unposted accruals",
			"post_all_accruals_before_close",
		)


# ---------------------------------------------------------------------------
# Retention rules
# ---------------------------------------------------------------------------

def assert_retention_release_valid(
	release_pct: Decimal,
	already_released_pct: Decimal,
) -> None:
	"""Total released cannot exceed 100% of held retention."""
	total = already_released_pct + release_pct
	if total > Decimal("100"):
		raise RuleViolation(
			"retention_release_exceeds_held",
			f"releasing {release_pct}% when {already_released_pct}% already released would exceed 100%",
			"reduce_release_percentage",
		)


# ---------------------------------------------------------------------------
# Intercompany rules
# ---------------------------------------------------------------------------

def assert_intercompany_entity_valid(
	supplier_entity_id: str | None,
	valid_entity_ids: list[str],
) -> None:
	if supplier_entity_id and supplier_entity_id not in valid_entity_ids:
		raise RuleViolation(
			"invalid_intercompany_entity",
			f"intercompany entity '{supplier_entity_id}' is not a recognised entity",
			"register_entity_before_intercompany_transactions",
		)


# ---------------------------------------------------------------------------
# Approval rules
# ---------------------------------------------------------------------------

def assert_approval_authority(
	approver_id: str,
	invoice_amount: Decimal,
	approval_limits: dict[str, Decimal],
) -> None:
	"""
	approval_limits: dict of approver_id -> max_amount they can approve.
	None value means unlimited authority.
	"""
	limit = approval_limits.get(approver_id)
	if limit is None:
		return  # unlimited authority
	if invoice_amount > limit:
		raise RuleViolation(
			"approval_limit_exceeded",
			f"approver '{approver_id}' can approve up to {limit}, invoice total is {invoice_amount}",
			"escalate_to_higher_authority",
		)


def assert_segregation_of_duties(
	invoice_creator_id: str,
	approver_id: str,
) -> None:
	"""Classic AP control: the person who enters the invoice cannot approve it."""
	if invoice_creator_id == approver_id:
		raise RuleViolation(
			"segregation_of_duties_violation",
			f"user '{approver_id}' cannot approve an invoice they created",
			"assign_different_approver",
		)


__all__ = [
	"RuleViolation",
	# Tenant / access
	"assert_tenant_context",
	"assert_no_cross_tenant_access",
	"assert_actor_present",
	# Supplier
	"assert_supplier_active",
	"assert_supplier_not_on_hold",
	"assert_supplier_has_bank_account",
	"assert_po_required_when_configured",
	"check_supplier_credit_limit",
	# Invoice
	"assert_invoice_dates_valid",
	"assert_invoice_has_lines",
	"assert_invoice_not_duplicate",
	"assert_invoice_can_transition",
	"assert_invoice_postable",
	"assert_invoice_payable",
	"assert_self_billing_supplier",
	# Match
	"assert_match_references_present",
	"assert_match_within_tolerance",
	# Payment run
	"assert_payment_run_approvable",
	"assert_payment_run_processable",
	"assert_payment_amount_matches",
	"assert_no_duplicate_payment",
	# Dispute
	"assert_dispute_amount_valid",
	"assert_dispute_resolvable",
	# Accrual
	"assert_accrual_period_format",
	"assert_accrual_not_reversed",
	"assert_accrual_posted",
	# Credit note
	"assert_credit_note_not_over_applied",
	# FX
	"assert_exchange_rate_positive",
	"assert_payment_currency_consistent",
	# Period close
	"assert_no_open_invoices_for_period_close",
	"assert_no_unposted_accruals_for_period_close",
	# Retention
	"assert_retention_release_valid",
	# Intercompany
	"assert_intercompany_entity_valid",
	# Approval
	"assert_approval_authority",
	"assert_segregation_of_duties",
]
