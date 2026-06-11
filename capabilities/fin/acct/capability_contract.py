"""Executable capability contract for Bank Account Management (ACCT)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "fin_acct"
CAPABILITY_NAME = "Bank Account Management"
CAPABILITY_VERSION = "1.0.0"
ACCT_EVENT_STREAM = "apg.fin.acct.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_ACCOUNT_TYPES = ["current", "savings", "fixed_deposit", "loan", "overdraft", "escrow"]
SUPPORTED_TRANSACTION_TYPES = [
	"deposit", "withdrawal", "transfer_in", "transfer_out",
	"fee", "interest", "reversal", "adjustment", "bulk_credit",
]
SUPPORTED_CLOSE_REASONS = ["customer_request", "dormant", "regulatory", "fraud", "deceased"]
SUPPORTED_FREEZE_REASONS = ["fraud_investigation", "legal_order", "aml", "kyc_pending", "admin"]
SUPPORTED_SIGNING_AUTHORITIES = ["single", "joint_any", "joint_all"]

DORMANCY_THRESHOLD_DAYS = 180

STREAMING = {
	"account_opened": ACCT_EVENT_STREAM,
	"account_closed": ACCT_EVENT_STREAM,
	"account_frozen": ACCT_EVENT_STREAM,
	"account_unfrozen": ACCT_EVENT_STREAM,
	"account_dormant": ACCT_EVENT_STREAM,
	"account_reactivated": ACCT_EVENT_STREAM,
	"credit_posted": ACCT_EVENT_STREAM,
	"debit_posted": ACCT_EVENT_STREAM,
	"transfer_completed": ACCT_EVENT_STREAM,
	"funds_locked": ACCT_EVENT_STREAM,
	"funds_released": ACCT_EVENT_STREAM,
	"overdraft_limit_set": ACCT_EVENT_STREAM,
	"gl_journal_requested": "apg.fin.glr.lifecycle",
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"dormancy_threshold_days": DORMANCY_THRESHOLD_DAYS,
	"max_overdraft_limit": 1_000_000,
	"min_opening_deposit": 0,
	"iban_country_code": "KE",
	"supported_currencies": SUPPORTED_CURRENCIES,
	"supported_account_types": SUPPORTED_ACCOUNT_TYPES,
	"gl_integration_enabled": True,
	"nats_events_enabled": True,
	"circuit_breaker_threshold": 5,
	"circuit_breaker_timeout_seconds": 60,
	"statement_formats": ["json", "pdf"],
	"bulk_credit_max_items": 5000,
	"lock_max_duration_days": 90,
}


def evaluate_capability_rules(config: dict[str, Any]) -> list[str]:
	"""Return list of rule violations; empty = pass."""
	violations: list[str] = []
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg.update(config)
	if cfg.get("dormancy_threshold_days", 0) < 1:
		violations.append("dormancy_threshold_days must be >= 1")
	if cfg.get("max_overdraft_limit", 0) < 0:
		violations.append("max_overdraft_limit must be >= 0")
	return violations
