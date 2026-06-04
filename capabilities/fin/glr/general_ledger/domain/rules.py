"""Deterministic domain rules for Financial Management General Ledger.

Single source of truth for all GL governance decisions.
Every rule is a pure function — no I/O, no side effects.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant / context guards
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Chart of accounts rules
# ---------------------------------------------------------------------------

_VALID_ACCOUNT_TYPES = frozenset({"asset", "liability", "equity", "revenue", "expense", "contra"})
_DEBIT_NORMAL_TYPES = frozenset({"asset", "expense", "contra"})
_CREDIT_NORMAL_TYPES = frozenset({"liability", "equity", "revenue"})


def assert_valid_account_type(account_type: str) -> None:
	"""Account type must be one of the six standard types."""
	if account_type not in _VALID_ACCOUNT_TYPES:
		raise RuleViolation(
			"invalid_account_type",
			f"'{account_type}' is not a valid account type; "
			f"must be one of {sorted(_VALID_ACCOUNT_TYPES)}",
			"use_valid_account_type",
		)


def assert_no_parent_cycle(account_id: str, parent_id: str | None) -> None:
	"""An account cannot be its own parent."""
	if parent_id and parent_id == account_id:
		raise RuleViolation(
			"account_parent_cycle",
			f"account {account_id} cannot reference itself as parent",
			"choose_different_parent",
		)


def assert_posting_account(account: dict[str, Any]) -> None:
	"""Posting is only allowed to accounts with allow_posting=True and status=active."""
	if not account.get("allow_posting", True):
		raise RuleViolation(
			"account_disallows_posting",
			f"account {account.get('code', account.get('id'))} has allow_posting=False",
			"use_postable_account",
		)
	if account.get("status", "active") != "active":
		raise RuleViolation(
			"account_inactive",
			f"account {account.get('code', account.get('id'))} is {account.get('status')}",
			"reactivate_account_or_choose_different",
		)


def assert_account_code_unique(existing_codes: set[str], new_code: str) -> None:
	"""Account code must be unique within a tenant."""
	if new_code in existing_codes:
		raise RuleViolation(
			"duplicate_account_code",
			f"account code '{new_code}' already exists in this tenant",
			"use_unique_account_code",
		)


# ---------------------------------------------------------------------------
# Journal entry rules
# ---------------------------------------------------------------------------

def assert_journal_balanced(total_debit: Decimal, total_credit: Decimal) -> None:
	"""Every journal entry must have sum(debits) == sum(credits) and both > 0."""
	if total_debit != total_credit:
		raise RuleViolation(
			"journal_must_balance",
			f"debits {total_debit} != credits {total_credit}",
			"balance_journal_lines",
		)
	if total_debit <= 0:
		raise RuleViolation(
			"journal_total_must_be_positive",
			"journal must have at least one non-zero line",
			"add_journal_lines",
		)


def assert_minimum_two_lines(lines: list[Any]) -> None:
	"""A valid double-entry journal requires at least two lines."""
	if len(lines) < 2:
		raise RuleViolation(
			"journal_requires_minimum_two_lines",
			f"journal has {len(lines)} line(s); minimum is 2",
			"add_more_journal_lines",
		)


def assert_line_single_sided(debit: Decimal, credit: Decimal) -> None:
	"""Each journal line must be either debit OR credit, never both."""
	if debit > 0 and credit > 0:
		raise RuleViolation(
			"line_cannot_have_both_debit_and_credit",
			f"line has debit={debit} and credit={credit}; only one side allowed",
			"zero_one_side",
		)


def assert_line_non_negative(debit: Decimal, credit: Decimal) -> None:
	"""Journal line amounts must be non-negative."""
	if debit < 0 or credit < 0:
		raise RuleViolation(
			"line_amount_must_be_non_negative",
			f"debit={debit}, credit={credit}; both must be >= 0",
			"use_non_negative_amounts",
		)


def assert_journal_approved(journal: dict[str, Any]) -> None:
	"""A journal must be approved before posting."""
	if not journal.get("approved_by"):
		raise RuleViolation(
			"journal_approval_required",
			f"journal {journal.get('id')} has not been approved",
			"obtain_approval_before_posting",
		)


def assert_segregation_of_duties(prepared_by: str, posted_by: str) -> None:
	"""The person who prepared a journal may not post it (SOD)."""
	if prepared_by and posted_by and prepared_by == posted_by:
		raise RuleViolation(
			"segregation_of_duties_required",
			f"'{prepared_by}' cannot both prepare and post the same journal",
			"different_user_must_post",
		)


def assert_journal_not_already_posted(journal: dict[str, Any]) -> None:
	"""Prevent double-posting."""
	if journal.get("status") == "posted":
		raise RuleViolation(
			"journal_already_posted",
			f"journal {journal.get('id')} is already posted",
			"reverse_if_correction_needed",
		)


def assert_journal_is_posted(journal: dict[str, Any]) -> None:
	"""Reversal can only target posted journals."""
	if journal.get("status") != "posted":
		raise RuleViolation(
			"journal_not_posted",
			f"journal {journal.get('id')} has status={journal.get('status')}; must be posted",
			"post_journal_first",
		)


def assert_reversal_reason(reason: str) -> None:
	"""Reversals must carry a non-empty reason for the audit trail."""
	if not reason or not reason.strip():
		raise RuleViolation(
			"reversal_reason_required",
			"a reason must be provided when reversing a journal",
			"supply_reversal_reason",
		)


# ---------------------------------------------------------------------------
# Period rules
# ---------------------------------------------------------------------------

_OPENABLE_STATUSES = frozenset({"future", "closed"})
_CLOSEABLE_STATUSES = frozenset({"open", "soft_closed"})


def assert_period_open_for_posting(period: dict[str, Any]) -> None:
	"""Journal entries can only be posted to open periods."""
	status = period.get("status", "")
	if status not in {"open"}:
		raise RuleViolation(
			"period_not_open",
			f"period {period.get('period_code', period.get('id'))} has status={status}",
			"open_period_before_posting",
		)


def assert_period_can_be_opened(period: dict[str, Any]) -> None:
	"""A period can only be opened if it is in future or closed status."""
	status = period.get("status", "")
	if status not in _OPENABLE_STATUSES:
		raise RuleViolation(
			"period_cannot_be_opened",
			f"period {period.get('period_code')} is in status={status}; "
			f"can only open from {sorted(_OPENABLE_STATUSES)}",
			"check_period_status",
		)


def assert_period_can_be_closed(period: dict[str, Any]) -> None:
	"""A period can only be closed if it is open or soft-closed."""
	status = period.get("status", "")
	if status not in _CLOSEABLE_STATUSES:
		raise RuleViolation(
			"period_cannot_be_closed",
			f"period {period.get('period_code')} is in status={status}",
			"check_period_status",
		)


def assert_period_is_closed_for_locking(period: dict[str, Any]) -> None:
	"""A period must be closed before it can be locked."""
	if period.get("status") != "closed":
		raise RuleViolation(
			"period_must_be_closed_before_locking",
			f"period {period.get('period_code')} status={period.get('status')}",
			"close_period_first",
		)


def assert_period_not_locked(period: dict[str, Any]) -> None:
	"""Locked periods cannot be modified."""
	if period.get("status") == "locked":
		raise RuleViolation(
			"period_is_locked",
			f"period {period.get('period_code')} is locked; no modifications allowed",
			"contact_finance_admin",
		)


def assert_reopen_authorised(reason: str, authorised_by: str) -> None:
	"""Re-opening a closed period requires both a reason and an authoriser."""
	if not reason or not reason.strip():
		raise RuleViolation(
			"reopen_reason_required",
			"a reason is required to reopen a closed period",
			"provide_reopen_reason",
		)
	if not authorised_by or not authorised_by.strip():
		raise RuleViolation(
			"reopen_authorisation_required",
			"an authorising user is required to reopen a closed period",
			"obtain_cfo_authorisation",
		)


def assert_date_in_open_period(
	journal_date: str,
	periods: list[dict[str, Any]],
	tenant_id: str,
) -> dict[str, Any]:
	"""Return the open period that covers journal_date; raise if none exists."""
	for p in periods:
		if (
			p.get("tenant_id") == tenant_id
			and p.get("status") == "open"
			and p.get("period_start", "") <= journal_date <= p.get("period_end", "")
		):
			return p
	raise RuleViolation(
		"no_open_period_for_date",
		f"no open period covers journal_date={journal_date}",
		"open_appropriate_period",
	)


# ---------------------------------------------------------------------------
# Reconciliation rules
# ---------------------------------------------------------------------------

def assert_reconciliation_submitted(reconciliation: dict[str, Any]) -> None:
	"""Reconciliation must be submitted before it can be approved."""
	if reconciliation.get("status") != "submitted":
		raise RuleViolation(
			"reconciliation_not_submitted",
			f"reconciliation {reconciliation.get('id')} status={reconciliation.get('status')}",
			"submit_reconciliation_first",
		)


def assert_reconciliation_open(reconciliation: dict[str, Any]) -> None:
	"""Items can only be added to open reconciliations."""
	if reconciliation.get("status") not in {"open", "submitted"}:
		raise RuleViolation(
			"reconciliation_not_open",
			f"reconciliation {reconciliation.get('id')} is {reconciliation.get('status')}",
			"create_new_reconciliation",
		)


# ---------------------------------------------------------------------------
# Budget rules
# ---------------------------------------------------------------------------

def assert_budget_amount_positive(amount: Decimal) -> None:
	"""Budget amounts must be non-negative (zero is allowed for placeholders)."""
	if amount < 0:
		raise RuleViolation(
			"budget_amount_must_be_non_negative",
			f"budget amount {amount} is negative",
			"use_non_negative_budget_amount",
		)


# ---------------------------------------------------------------------------
# Currency rules
# ---------------------------------------------------------------------------

def assert_exchange_rate_positive(rate: Decimal) -> None:
	"""Exchange rates must be strictly positive."""
	if rate <= 0:
		raise RuleViolation(
			"exchange_rate_must_be_positive",
			f"exchange rate {rate} must be > 0",
			"supply_valid_exchange_rate",
		)


# ---------------------------------------------------------------------------
# Year-end rules
# ---------------------------------------------------------------------------

def assert_retained_earnings_account_exists(
	accounts: list[dict[str, Any]],
	account_code: str,
) -> dict[str, Any]:
	"""Retained earnings account must exist, be active, and be an equity account."""
	for acct in accounts:
		if acct.get("code") == account_code or acct.get("account_code") == account_code:
			if acct.get("account_type") != "equity":
				raise RuleViolation(
					"retained_earnings_must_be_equity",
					f"account {account_code} is type={acct.get('account_type')}; must be equity",
					"use_equity_type_account",
				)
			if acct.get("status", "active") != "active":
				raise RuleViolation(
					"retained_earnings_account_inactive",
					f"retained earnings account {account_code} is inactive",
					"reactivate_account",
				)
			return acct
	raise RuleViolation(
		"retained_earnings_account_not_found",
		f"no account with code {account_code} found",
		"create_retained_earnings_account",
	)


def assert_fiscal_year_not_already_closed(
	fiscal_years: dict[str, Any],
	tenant_id: str,
	fiscal_year: int,
) -> None:
	"""A fiscal year can only be closed once."""
	for fy in fiscal_years.values():
		if (
			fy.get("tenant_id") == tenant_id
			and fy.get("fiscal_year") == fiscal_year
			and fy.get("status") == "closed"
		):
			raise RuleViolation(
				"fiscal_year_already_closed",
				f"fiscal year {fiscal_year} has already been closed",
				"contact_finance_admin_to_reopen",
			)


# ---------------------------------------------------------------------------
# IAS 8 / prior-year adjustment rules
# ---------------------------------------------------------------------------

def assert_adjustment_reason(reason: str) -> None:
	"""IAS 8 corrections require a substantive reason."""
	if not reason or len(reason.strip()) < 5:
		raise RuleViolation(
			"adjustment_reason_required",
			"prior-year adjustment reason must be at least 5 characters",
			"supply_ias8_reason",
		)


# ---------------------------------------------------------------------------
# Intercompany rules
# ---------------------------------------------------------------------------

def assert_intercompany_amount_positive(amount: Decimal) -> None:
	if amount <= 0:
		raise RuleViolation(
			"intercompany_amount_must_be_positive",
			f"intercompany amount {amount} must be > 0",
			"supply_positive_amount",
		)


def assert_intercompany_account_mapping(mapping: dict[str, str]) -> None:
	"""Both entity_account and counterpart_account must be present."""
	if not mapping.get("entity_account"):
		raise RuleViolation(
			"intercompany_entity_account_required",
			"account_mapping must include 'entity_account'",
			"supply_entity_account",
		)
	if not mapping.get("counterpart_account"):
		raise RuleViolation(
			"intercompany_counterpart_account_required",
			"account_mapping must include 'counterpart_account'",
			"supply_counterpart_account",
		)


# ---------------------------------------------------------------------------
# Utility calculations (domain layer)
# ---------------------------------------------------------------------------

def calculate_normal_balance(account_type: str) -> str:
	"""Derive the normal balance side from account type."""
	assert_valid_account_type(account_type)
	return "debit" if account_type in _DEBIT_NORMAL_TYPES else "credit"


def calculate_net_balance(
	opening: Decimal,
	debits: Decimal,
	credits: Decimal,
	normal_balance: str,
) -> Decimal:
	"""Compute closing balance respecting normal-balance convention."""
	if normal_balance == "debit":
		return opening + debits - credits
	return opening + credits - debits


def calculate_variance(actual: Decimal, budget: Decimal) -> tuple[Decimal, Decimal, str]:
	"""Return (variance_amount, variance_pct, indicator F/A)."""
	from decimal import ROUND_HALF_UP
	_2 = Decimal("0.01")
	var = (actual - budget).quantize(_2, rounding=ROUND_HALF_UP)
	pct = (
		((actual - budget) / abs(budget) * 100).quantize(_2, rounding=ROUND_HALF_UP)
		if budget != 0
		else Decimal("0")
	)
	indicator = "F" if var >= 0 else "A"
	return var, pct, indicator
