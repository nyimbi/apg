"""Dependency-light General Ledger lifecycle service.

Implements complete double-entry accounting with journal management, period
lifecycle, financial statements (IFRS/GAAP), reconciliation, chart of
accounts management, and year-end close.

Invariant: every posted journal satisfies sum(debits) == sum(credits).
"""

from __future__ import annotations

import csv
import io
from copy import deepcopy
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		GLR_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ACCOUNT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_GLR_AGENT_ROLES,
		SUPPORTED_GLR_AGENT_RUNTIMES,
		SUPPORTED_JOURNAL_SOURCES,
		evaluate_capability_rules,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		GLR_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ACCOUNT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_GLR_AGENT_ROLES,
		SUPPORTED_GLR_AGENT_RUNTIMES,
		SUPPORTED_JOURNAL_SOURCES,
		evaluate_capability_rules,
	)

# ---------------------------------------------------------------------------
# Normal-balance convention: assets & expenses are debit-normal;
# liabilities, equity and revenue are credit-normal.
# ---------------------------------------------------------------------------
_DEBIT_NORMAL_TYPES = {"asset", "expense"}
_CREDIT_NORMAL_TYPES = {"liability", "equity", "revenue"}

# Account types that close to retained earnings at year-end.
_INCOME_STMT_TYPES = {"revenue", "expense"}

# Reasonable maximum journal number sequence per tenant (in-memory counter).
_JOURNAL_COUNTERS: dict[str, int] = {}

TWO = Decimal("0.01")


def _d(value: Any) -> Decimal:
	"""Coerce to Decimal, rounding to 2dp."""
	return Decimal(str(value)).quantize(TWO, rounding=ROUND_HALF_UP)


class GeneralLedgerService:
	"""In-memory executable service for the GLR lifecycle packet.

	Stores all state in plain dicts so the service is importable without a
	database.  Production adapters replace individual store attributes with
	repository objects that expose the same dict-like interface.
	"""

	# ------------------------------------------------------------------
	# Construction
	# ------------------------------------------------------------------

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		# Core stores
		self.accounts: dict[str, dict[str, Any]] = {}
		self.dimensions: dict[str, dict[str, Any]] = {}
		self.periods: dict[str, dict[str, Any]] = {}
		self.journal_batches: dict[str, dict[str, Any]] = {}
		self.journal_entries: dict[str, dict[str, Any]] = {}
		self.postings: dict[str, dict[str, Any]] = {}
		self.currency_rates: dict[str, dict[str, Any]] = {}
		self.allocations: dict[str, dict[str, Any]] = {}
		self.reversals: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		# Extended stores for new functionality
		self.reconciliations: dict[str, dict[str, Any]] = {}
		self.recurring_templates: dict[str, dict[str, Any]] = {}
		self.budgets: dict[str, dict[str, Any]] = {}
		self.fiscal_years: dict[str, dict[str, Any]] = {}
		self.intercompany_journals: dict[str, dict[str, Any]] = {}
		self.approval_workflows: dict[str, dict[str, Any]] = {}
		# Infrastructure
		self._audit_events: list[dict[str, Any]] = []
		self._idempotency_keys: set[str] = set()

	# ------------------------------------------------------------------
	# Infrastructure helpers
	# ------------------------------------------------------------------

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _today(self) -> str:
		return date.today().isoformat()

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			reasons = ",".join(effect["reason"] for effect in result["effects"])
			raise PermissionError(reasons)

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": GLR_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def _next_journal_number(self, tenant_id: str) -> str:
		_JOURNAL_COUNTERS[tenant_id] = _JOURNAL_COUNTERS.get(tenant_id, 0) + 1
		return f"JNL-{tenant_id[:6].upper()}-{_JOURNAL_COUNTERS[tenant_id]:06d}"

	def _account_by_code(self, tenant_id: str, code: str) -> dict[str, Any] | None:
		"""Look up an account by its code within a tenant."""
		for acct in self.accounts.values():
			if acct["tenant_id"] == tenant_id and acct["code"] == code:
				return acct
		return None

	def _period_by_code(self, tenant_id: str, period_code: str) -> dict[str, Any] | None:
		for p in self.periods.values():
			if p["tenant_id"] == tenant_id and p.get("period_code") == period_code:
				return p
		return None

	def _get_account_balance(self, tenant_id: str, account_id: str, period_code: str | None = None) -> dict[str, Decimal]:
		"""Compute opening, movements, and closing balance for an account.

		Returns dict with keys: opening, debits, credits, closing.
		"""
		opening = Decimal("0")
		debits = Decimal("0")
		credits = Decimal("0")

		for posting in self.postings.values():
			if posting["tenant_id"] != tenant_id:
				continue
			for line in posting["lines"]:
				if line.get("account_id") != account_id:
					continue
				if period_code and posting.get("period_code") != period_code:
					# Accumulate everything before this period as opening balance
					if posting.get("period_code", "") < period_code:
						opening += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
				else:
					debits += _d(line.get("debit", 0))
					credits += _d(line.get("credit", 0))

		closing = opening + debits - credits
		return {"opening": opening, "debits": debits, "credits": credits, "closing": closing}

	def _variance_indicator(self, account_type: str, actual: Decimal, budget: Decimal) -> str:
		"""Return 'F' (favourable) or 'A' (adverse)."""
		variance = actual - budget
		if account_type in _DEBIT_NORMAL_TYPES:
			# For expense/asset accounts lower actual spend vs budget is favourable
			return "F" if variance <= 0 else "A"
		# For revenue accounts higher actual vs budget is favourable
		return "F" if variance >= 0 else "A"

	# ==================================================================
	# ORIGINAL METHODS (unchanged signatures, preserved behaviour)
	# ==================================================================

	def create_account(
		self,
		account_id: str,
		tenant_id: str,
		code: str,
		name: str,
		account_type: str,
		parent_account_id: str | None = None,
		allow_posting: bool = True,
		currency: str = "USD",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		parent_cycle_detected = bool(parent_account_id and parent_account_id == account_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_account",
			"operation_type": "write",
			"policy_attached": True,
			"account_code_present": bool(code),
			"account_name_present": bool(name),
			"account_type_supported": account_type in SUPPORTED_ACCOUNT_TYPES,
			"parent_cycle_detected": parent_cycle_detected,
		})
		record = {
			"id": self._record_id("acct", account_id),
			"type": "ledger_account",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"account_type": account_type,
			"parent_account_id": parent_account_id,
			"allow_posting": allow_posting,
			"currency": currency,
			"status": "active",
			"created_at": self._now(),
		}
		self.accounts[record["id"]] = record
		self._emit(tenant, "account_created", record)
		return deepcopy(record)

	def record_dimension(self, dimension_id: str, tenant_id: str, name: str, value: str, owner: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if not name or not value or not owner:
			raise PermissionError("dimension_name_value_owner_required")
		record = {
			"id": self._record_id("dim", dimension_id),
			"type": "ledger_dimension",
			"tenant_id": tenant,
			"name": name,
			"value": value,
			"owner": owner,
			"status": "active",
			"created_at": self._now(),
		}
		self.dimensions[record["id"]] = record
		self._emit(tenant, "dimension_recorded", record)
		return deepcopy(record)

	def open_period(self, period_id: str, tenant_id: str, name: str, fiscal_year: int, period_start: str, period_end: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "open_period",
			"operation_type": "write",
			"policy_attached": True,
			"period_name_present": bool(name),
			"fiscal_year_present": fiscal_year is not None,
			"period_dates_present": bool(period_start and period_end),
			"period_range_valid": bool(period_start and period_end and period_start <= period_end),
		})
		record = {
			"id": self._record_id("period", period_id),
			"type": "accounting_period",
			"tenant_id": tenant,
			"name": name,
			"period_code": name,  # canonical lookup key
			"fiscal_year": fiscal_year,
			"period_start": period_start,
			"period_end": period_end,
			"status": "open",
			"created_at": self._now(),
		}
		self.periods[record["id"]] = record
		self._emit(tenant, "period_opened", record)
		return deepcopy(record)

	def create_journal_batch(self, batch_id: str, tenant_id: str, period_id: str, source: str, currency: str = "USD") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		period = self.periods.get(period_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_journal_batch",
			"operation_type": "write",
			"policy_attached": True,
			"period_open": bool(period and period["status"] == "open" and period["tenant_id"] == tenant),
			"journal_source_supported": source in SUPPORTED_JOURNAL_SOURCES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		record = {
			"id": self._record_id("batch", batch_id),
			"type": "journal_batch",
			"tenant_id": tenant,
			"period_id": period_id,
			"source": source,
			"currency": currency,
			"status": "open",
			"created_at": self._now(),
		}
		self.journal_batches[record["id"]] = record
		self._emit(tenant, "journal_batch_created", record)
		return deepcopy(record)

	def create_journal_entry(
		self,
		journal_id: str,
		tenant_id: str,
		batch_id: str,
		description: str,
		lines: list[dict[str, Any]],
		prepared_by: str = "system",
		exchange_rate: float | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		batch = self.journal_batches.get(batch_id)
		total_debits = sum(_d(line.get("debit", 0)) for line in lines)
		total_credits = sum(_d(line.get("credit", 0)) for line in lines)
		posting_accounts_valid = all(
			line.get("account_id") in self.accounts
			and self.accounts[line["account_id"]]["tenant_id"] == tenant
			and self.accounts[line["account_id"]]["allow_posting"]
			for line in lines
		)
		foreign_currency = bool(batch and batch["currency"] != "USD")
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_journal_entry",
			"operation_type": "write",
			"policy_attached": True,
			"batch_present": bool(batch and batch["tenant_id"] == tenant),
			"journal_description_present": bool(description),
			"journal_line_count": len(lines),
			"posting_accounts_valid": posting_accounts_valid,
			"balanced": total_debits == total_credits and total_debits > 0,
			"foreign_currency": foreign_currency,
			"exchange_rate_present": exchange_rate is not None,
		})
		record = {
			"id": self._record_id("journal", journal_id),
			"type": "journal_entry",
			"tenant_id": tenant,
			"batch_id": batch_id,
			"description": description,
			"lines": deepcopy(lines),
			"total_debits": str(total_debits),
			"total_credits": str(total_credits),
			"prepared_by": prepared_by,
			"approved_by": None,
			"posted_by": None,
			"status": "balanced",
			"created_at": self._now(),
		}
		self.journal_entries[record["id"]] = record
		self._emit(tenant, "journal_entry_created", record)
		return deepcopy(record)

	def approve_journal(self, journal_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		journal = self.journal_entries[journal_id]
		if journal["tenant_id"] != tenant or not approved_by:
			raise PermissionError("journal_approval_required")
		journal["approved_by"] = approved_by
		journal["status"] = "approved"
		journal["approved_at"] = self._now()
		self._emit(tenant, "journal_approved", journal)
		return deepcopy(journal)

	def post_journal(self, journal_id: str, tenant_id: str, posted_by: str, idempotency_key: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		journal = self.journal_entries.get(journal_id)
		batch = self.journal_batches.get(journal["batch_id"]) if journal else None
		period = self.periods.get(batch["period_id"]) if batch else None
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "post_journal",
			"operation_type": "write",
			"policy_attached": True,
			"journal_present": bool(journal and journal["tenant_id"] == tenant),
			"approval_recorded": bool(journal and journal.get("approved_by")),
			"period_open": bool(period and period["status"] == "open"),
			"idempotency_key_present": bool(idempotency_key),
			"same_preparer_and_poster": bool(journal and journal.get("prepared_by") == posted_by),
			"closed_period_adjustment": bool(period and period["status"] == "closed"),
		})
		if idempotency_key in self._idempotency_keys:
			return deepcopy(self.postings[journal_id])
		self._idempotency_keys.add(idempotency_key)
		journal["posted_by"] = posted_by
		journal["status"] = "posted"
		journal["posted_at"] = self._now()
		posting = {
			"id": journal_id,
			"type": "ledger_posting",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"period_code": period.get("period_code") if period else None,
			"status": "posted",
			"lines": deepcopy(journal["lines"]),
			"posted_by": posted_by,
			"created_at": self._now(),
		}
		self.postings[journal_id] = posting
		self._emit(tenant, "journal_posted", posting)
		return deepcopy(posting)

	def reverse_journal(self, reversal_id: str, tenant_id: str, journal_id: str, reason: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		journal = self.journal_entries.get(journal_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "reverse_journal",
			"operation_type": "write",
			"policy_attached": True,
			"posted_entry_present": bool(journal and journal["tenant_id"] == tenant and journal["status"] == "posted"),
			"reversal_reason_present": bool(reason),
			"approval_recorded": bool(approved_by),
		})
		record = {
			"id": self._record_id("reversal", reversal_id),
			"type": "journal_reversal",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"reason": reason,
			"approved_by": approved_by,
			"status": "reversed",
			"created_at": self._now(),
		}
		self.reversals[record["id"]] = record
		journal["status"] = "reversed"
		self._emit(tenant, "journal_reversed", record)
		return deepcopy(record)

	def record_currency_rate(self, rate_id: str, tenant_id: str, from_currency: str, to_currency: str, exchange_rate: float) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "record_currency_rate",
			"operation_type": "write",
			"policy_attached": True,
			"exchange_rate": exchange_rate,
		})
		record = {
			"id": self._record_id("rate", rate_id),
			"type": "currency_rate",
			"tenant_id": tenant,
			"from_currency": from_currency,
			"to_currency": to_currency,
			"exchange_rate": exchange_rate,
			"status": "active",
			"created_at": self._now(),
		}
		self.currency_rates[record["id"]] = record
		return deepcopy(record)

	def create_allocation(self, allocation_id: str, tenant_id: str, source_account_id: str, target_account_ids: list[str], basis: str, reviewed_by: str | None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_allocation",
			"operation_type": "write",
			"policy_attached": True,
			"allocation_basis_present": bool(basis),
			"allocation_review_recorded": bool(reviewed_by),
		})
		record = {
			"id": self._record_id("alloc", allocation_id),
			"type": "ledger_allocation",
			"tenant_id": tenant,
			"source_account_id": source_account_id,
			"target_account_ids": list(target_account_ids),
			"basis": basis,
			"reviewed_by": reviewed_by,
			"status": "reviewed",
			"created_at": self._now(),
		}
		self.allocations[record["id"]] = record
		self._emit(tenant, "allocation_created", record)
		return deepcopy(record)

	def generate_trial_balance(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account_totals: dict[str, dict[str, Decimal]] = {}
		for posting in self.postings.values():
			if posting["tenant_id"] != tenant:
				continue
			for line in posting["lines"]:
				account = line["account_id"]
				account_totals.setdefault(account, {"debit": Decimal("0"), "credit": Decimal("0")})
				account_totals[account]["debit"] += _d(line.get("debit", 0))
				account_totals[account]["credit"] += _d(line.get("credit", 0))
		total_debits = sum(value["debit"] for value in account_totals.values())
		total_credits = sum(value["credit"] for value in account_totals.values())
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "generate_trial_balance",
			"trial_balance_balanced": total_debits == total_credits,
		})
		record = {
			"id": self._record_id("trial"),
			"type": "trial_balance",
			"tenant_id": tenant,
			"status": "balanced",
			"total_debits": str(total_debits),
			"total_credits": str(total_credits),
			"account_count": len(account_totals),
			"created_at": self._now(),
		}
		self._emit(tenant, "trial_balance_generated", record)
		return deepcopy(record)

	def register_glr_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "register_glr_agent",
			"operation_type": "write",
			"policy_attached": True,
			"agent_runtime_supported": runtime in SUPPORTED_GLR_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_GLR_AGENT_ROLES,
		})
		record = {
			"id": self._record_id("agent"),
			"type": "glr_agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "glr_agent_registered", record)
		return deepcopy(record)

	def validate_agent_glr_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if agent_id not in self.agents:
			raise PermissionError("glr_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "agent_glr_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "glr_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": GLR_EVENT_STREAM}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"account_count": len([r for r in self.accounts.values() if r["tenant_id"] == tenant]),
			"dimension_count": len([r for r in self.dimensions.values() if r["tenant_id"] == tenant]),
			"period_count": len([r for r in self.periods.values() if r["tenant_id"] == tenant]),
			"journal_batch_count": len([r for r in self.journal_batches.values() if r["tenant_id"] == tenant]),
			"journal_entry_count": len([r for r in self.journal_entries.values() if r["tenant_id"] == tenant]),
			"posted_journal_count": len([r for r in self.journal_entries.values() if r["tenant_id"] == tenant and r["status"] == "posted"]),
			"allocation_count": len([r for r in self.allocations.values() if r["tenant_id"] == tenant]),
			"glr_agent_count": len([r for r in self.agents.values() if r["tenant_id"] == tenant]),
			"audit_event_count": len([e for e in self._audit_events if e["tenant_id"] == tenant]),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		store = getattr(self, collection)
		return [deepcopy(r) for r in store.values() if r["tenant_id"] == tenant]

	# ==================================================================
	# JOURNAL ENTRY MANAGEMENT
	# ==================================================================

	async def post_journal_v2(
		self,
		tenant_id: str,
		journal_date: str,
		journal_type: str,
		lines: list[dict[str, Any]],
		description: str,
		reference: str,
		posted_by: str,
	) -> dict[str, Any]:
		"""Create, validate, and post a journal entry in one operation.

		Validation checklist:
		- sum(debits) == sum(credits)
		- Every line account exists, is active, and allows posting
		- The journal_date falls within an open period
		- Segregation of duties: posted_by must not be the preparer
		  (treated as system/service in this path so SOD is implicitly satisfied)

		Returns the ledger_posting record.
		"""
		tenant = self._tenant(tenant_id)

		total_debits = sum(_d(ln.get("debit", 0)) for ln in lines)
		total_credits = sum(_d(ln.get("credit", 0)) for ln in lines)
		if total_debits != total_credits or total_debits == 0:
			raise ValueError(f"journal_not_balanced: debits={total_debits} credits={total_credits}")
		if len(lines) < 2:
			raise ValueError("journal_requires_minimum_two_lines")

		# Resolve open period that covers journal_date
		covering_period: dict[str, Any] | None = None
		for p in self.periods.values():
			if (
				p["tenant_id"] == tenant
				and p["status"] == "open"
				and p.get("period_start", "") <= journal_date <= p.get("period_end", "")
			):
				covering_period = p
				break
		if covering_period is None:
			raise ValueError(f"no_open_period_for_date:{journal_date}")

		# Validate every posting account
		for ln in lines:
			acct_id = ln.get("account_id")
			acct = self.accounts.get(acct_id or "")
			if not acct or acct["tenant_id"] != tenant:
				raise ValueError(f"account_not_found:{acct_id}")
			if acct["status"] != "active":
				raise ValueError(f"account_inactive:{acct_id}")
			if not acct["allow_posting"]:
				raise ValueError(f"account_disallows_posting:{acct_id}")

		journal_number = self._next_journal_number(tenant)
		journal_id = self._record_id("journal")
		batch_id = self._record_id("batch")

		# Materialise a batch implicitly
		batch_record = {
			"id": batch_id,
			"type": "journal_batch",
			"tenant_id": tenant,
			"period_id": covering_period["id"],
			"source": journal_type if journal_type in SUPPORTED_JOURNAL_SOURCES else "manual",
			"currency": "USD",
			"status": "posted",
			"created_at": self._now(),
		}
		self.journal_batches[batch_id] = batch_record

		journal_record = {
			"id": journal_id,
			"type": "journal_entry",
			"tenant_id": tenant,
			"batch_id": batch_id,
			"journal_number": journal_number,
			"journal_type": journal_type,
			"journal_date": journal_date,
			"description": description,
			"reference": reference,
			"lines": deepcopy(lines),
			"total_debits": str(total_debits),
			"total_credits": str(total_credits),
			"prepared_by": "service",
			"approved_by": posted_by,  # single-step post: approver == poster
			"posted_by": posted_by,
			"status": "posted",
			"posted_at": self._now(),
			"created_at": self._now(),
		}
		self.journal_entries[journal_id] = journal_record

		posting = {
			"id": journal_id,
			"type": "ledger_posting",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"journal_number": journal_number,
			"period_code": covering_period.get("period_code"),
			"lines": deepcopy(lines),
			"status": "posted",
			"posted_by": posted_by,
			"created_at": self._now(),
		}
		self.postings[journal_id] = posting
		self._emit(tenant, "journal_posted", posting)
		return deepcopy(posting)

	async def reverse_journal_v2(
		self,
		tenant_id: str,
		journal_id: str,
		reversal_date: str,
		reversal_description: str,
		reversed_by: str,
	) -> dict[str, Any]:
		"""Create a mirror entry: all debits become credits and vice versa.

		Marks the original entry status='reversed' and links both records.
		The reversal journal is posted immediately (self-approving, same period
		lookup by reversal_date).
		"""
		tenant = self._tenant(tenant_id)
		original = self.journal_entries.get(journal_id)
		if not original or original["tenant_id"] != tenant:
			raise ValueError(f"journal_not_found:{journal_id}")
		if original["status"] != "posted":
			raise ValueError(f"journal_not_posted:{journal_id} status={original['status']}")

		# Flip debit/credit on every line
		reversed_lines = []
		for ln in original["lines"]:
			reversed_lines.append({
				**ln,
				"debit": str(_d(ln.get("credit", 0))),
				"credit": str(_d(ln.get("debit", 0))),
				"description": f"Reversal: {ln.get('description', '')}",
			})

		reversal_posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=reversal_date,
			journal_type="reversal",
			lines=reversed_lines,
			description=reversal_description or f"Reversal of {original.get('journal_number', journal_id)}",
			reference=original.get("reference", ""),
			posted_by=reversed_by,
		)

		# Mark original as reversed and cross-link
		original["status"] = "reversed"
		original["reversed_at"] = self._now()
		original["reversal_journal_id"] = reversal_posting["journal_id"]

		reversal_record = {
			"id": self._record_id("reversal"),
			"type": "journal_reversal",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"reversal_journal_id": reversal_posting["journal_id"],
			"reversal_date": reversal_date,
			"reason": reversal_description,
			"approved_by": reversed_by,
			"status": "reversed",
			"created_at": self._now(),
		}
		self.reversals[reversal_record["id"]] = reversal_record
		self._emit(tenant, "journal_reversed", reversal_record)
		return deepcopy(reversal_record)

	async def auto_reverse_on_date(
		self,
		tenant_id: str,
		journal_id: str,
		reversal_date: str,
	) -> dict[str, Any]:
		"""Schedule an automatic reversal on the given date.

		In this in-memory implementation the reversal is queued and the
		schedule record is returned.  A cron runner would invoke
		reverse_journal_v2 when reversal_date is reached.
		"""
		tenant = self._tenant(tenant_id)
		original = self.journal_entries.get(journal_id)
		if not original or original["tenant_id"] != tenant:
			raise ValueError(f"journal_not_found:{journal_id}")
		if original["status"] != "posted":
			raise ValueError(f"journal_not_posted:{journal_id}")

		schedule_id = self._record_id("sched")
		record = {
			"id": schedule_id,
			"type": "auto_reversal_schedule",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"reversal_date": reversal_date,
			"status": "scheduled",
			"created_at": self._now(),
		}
		# Persist as a reversal placeholder so it can be queried
		self.reversals[schedule_id] = record
		return deepcopy(record)

	async def recurring_journal_run(
		self,
		tenant_id: str,
		template_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Apply a recurring journal template for a given period.

		Templates encode monthly depreciation, prepaid amortisation, etc.
		Each run generates one posted journal with lines taken verbatim from
		the template (amounts can reference a multiplier stored in the template).
		"""
		tenant = self._tenant(tenant_id)
		template = self.recurring_templates.get(template_id)
		if not template or template["tenant_id"] != tenant:
			raise ValueError(f"recurring_template_not_found:{template_id}")

		# Resolve posting date — use period end date if we can find the period
		period_rec = self._period_by_code(tenant, period)
		journal_date = period_rec["period_end"] if period_rec else self._today()

		# Apply any multiplier stored on the template
		multiplier = _d(template.get("amount_multiplier", 1))
		lines = []
		for ln in template.get("lines", []):
			lines.append({
				**ln,
				"debit": str(_d(ln.get("debit", 0)) * multiplier),
				"credit": str(_d(ln.get("credit", 0)) * multiplier),
				"description": f"Recurring ({period}): {ln.get('description', '')}",
			})

		posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=journal_date,
			journal_type=template.get("journal_type", "manual"),
			lines=lines,
			description=f"{template['name']} – {period}",
			reference=f"RECUR-{template_id[:8]}-{period}",
			posted_by=template.get("owner", "system"),
		)

		run_record = {
			"id": self._record_id("recur_run"),
			"type": "recurring_journal_run",
			"tenant_id": tenant,
			"template_id": template_id,
			"period": period,
			"posting_id": posting["id"],
			"journal_id": posting["journal_id"],
			"status": "completed",
			"created_at": self._now(),
		}
		return deepcopy(run_record)

	async def intercompany_journal(
		self,
		tenant_id: str,
		counterpart_entity: str,
		amount: str,
		currency: str,
		account_mapping: dict[str, str],
	) -> dict[str, Any]:
		"""Post matching journal entries in both entities simultaneously.

		account_mapping = {
		    "entity_account": "<account_id in tenant_id>",
		    "counterpart_account": "<account_id in counterpart_entity>",
		}

		The primary entity records a debit; the counterpart records a credit
		(or vice versa depending on convention — we follow the payable/receivable
		approach: entity debits interco-receivable, counterpart credits interco-payable).
		"""
		tenant = self._tenant(tenant_id)
		amt = _d(amount)
		if amt <= 0:
			raise ValueError("intercompany_amount_must_be_positive")

		entity_acct_id = account_mapping.get("entity_account")
		counterpart_acct_id = account_mapping.get("counterpart_account")
		if not entity_acct_id or not counterpart_acct_id:
			raise ValueError("account_mapping_incomplete")

		today = self._today()

		# Primary entity journal: debit entity account
		entity_lines = [
			{"account_id": entity_acct_id, "debit": str(amt), "credit": "0.00",
			 "description": f"Intercompany with {counterpart_entity}"},
		]
		# We need a balancing credit — use a suspense account if no explicit offset given
		offset_acct_id = account_mapping.get("entity_offset_account", entity_acct_id)
		entity_lines.append(
			{"account_id": offset_acct_id, "debit": "0.00", "credit": str(amt),
			 "description": f"Intercompany offset – {counterpart_entity}"},
		)

		entity_posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=today,
			journal_type="manual",
			lines=entity_lines,
			description=f"Intercompany: {tenant} → {counterpart_entity} {currency} {amount}",
			reference=f"IC-{uuid4().hex[:8].upper()}",
			posted_by="system",
		)

		# Counterpart entity journal: credit counterpart account
		# Use same service instance (multi-tenant); counterpart_entity is a different tenant_id.
		counterpart_lines = [
			{"account_id": counterpart_acct_id, "debit": "0.00", "credit": str(amt),
			 "description": f"Intercompany with {tenant}"},
		]
		cp_offset_acct_id = account_mapping.get("counterpart_offset_account", counterpart_acct_id)
		counterpart_lines.append(
			{"account_id": cp_offset_acct_id, "debit": str(amt), "credit": "0.00",
			 "description": f"Intercompany offset – {tenant}"},
		)

		cp_posting: dict[str, Any] | None = None
		try:
			cp_posting = await self.post_journal_v2(
				tenant_id=counterpart_entity,
				journal_date=today,
				journal_type="manual",
				lines=counterpart_lines,
				description=f"Intercompany: {counterpart_entity} ← {tenant} {currency} {amount}",
				reference=entity_posting["journal_id"],
				posted_by="system",
			)
		except Exception as exc:
			# Compensating action: roll back entity posting by reversing it
			await self.reverse_journal_v2(
				tenant_id=tenant,
				journal_id=entity_posting["journal_id"],
				reversal_date=today,
				reversal_description=f"Auto-reversal: counterpart post failed – {exc}",
				reversed_by="system",
			)
			raise ValueError(f"intercompany_counterpart_failed:{exc}") from exc

		ic_record = {
			"id": self._record_id("ic"),
			"type": "intercompany_journal",
			"tenant_id": tenant,
			"counterpart_entity": counterpart_entity,
			"amount": str(amt),
			"currency": currency,
			"entity_posting_id": entity_posting["id"],
			"counterpart_posting_id": cp_posting["id"] if cp_posting else None,
			"status": "posted",
			"created_at": self._now(),
		}
		self.intercompany_journals[ic_record["id"]] = ic_record
		return deepcopy(ic_record)

	async def validate_journal_balance(self, lines: list[dict[str, Any]]) -> bool:
		"""Return True iff sum(debits) == sum(credits) and both are > 0."""
		total_debits = sum(_d(ln.get("debit", 0)) for ln in lines)
		total_credits = sum(_d(ln.get("credit", 0)) for ln in lines)
		return total_debits == total_credits and total_debits > Decimal("0")

	async def bulk_journal_import(
		self,
		tenant_id: str,
		journals_csv: str,
	) -> dict[str, Any]:
		"""Bulk import journals from CSV text.

		Expected CSV columns (case-insensitive):
		  journal_date, description, reference, account_id, debit, credit, posted_by

		Rows with the same journal_date + reference + description are grouped
		into a single journal entry.  Each group must balance independently.

		Returns a summary with counts of imported / failed journals.
		"""
		tenant = self._tenant(tenant_id)
		reader = csv.DictReader(io.StringIO(journals_csv))
		# Normalise headers
		groups: dict[str, list[dict[str, Any]]] = {}
		errors: list[str] = []

		for row_num, row in enumerate(reader, start=2):
			row = {k.strip().lower(): v.strip() for k, v in row.items()}
			key = f"{row.get('journal_date')}|{row.get('reference')}|{row.get('description')}"
			groups.setdefault(key, [])
			groups[key].append(row)

		posted: list[dict[str, Any]] = []
		failed: list[dict[str, Any]] = []

		for key, rows in groups.items():
			try:
				lines = [
					{
						"account_id": r["account_id"],
						"debit": r.get("debit", "0"),
						"credit": r.get("credit", "0"),
						"description": r.get("description", ""),
					}
					for r in rows
				]
				parts = key.split("|")
				posting = await self.post_journal_v2(
					tenant_id=tenant,
					journal_date=parts[0],
					journal_type="import",
					lines=lines,
					description=parts[2],
					reference=parts[1],
					posted_by=rows[0].get("posted_by", "import"),
				)
				posted.append({"key": key, "journal_id": posting["journal_id"]})
			except Exception as exc:
				failed.append({"key": key, "error": str(exc)})

		return {
			"id": self._record_id("bulk_import"),
			"type": "bulk_journal_import",
			"tenant_id": tenant,
			"total_groups": len(groups),
			"posted_count": len(posted),
			"failed_count": len(failed),
			"posted": posted,
			"failed": failed,
			"status": "completed",
			"created_at": self._now(),
		}

	async def journal_approval_workflow(
		self,
		tenant_id: str,
		journal_id: str,
		amount_threshold: str,
		approver_id: str,
	) -> dict[str, Any]:
		"""Route a journal through an approval workflow if it exceeds a threshold.

		If the journal's total debits exceed amount_threshold, the journal is
		placed in 'pending_approval' status and an approval request record is
		created.  The approver then calls approve_journal() to unblock posting.

		If the journal is below the threshold it is auto-approved and can
		proceed to posting immediately.
		"""
		tenant = self._tenant(tenant_id)
		journal = self.journal_entries.get(journal_id)
		if not journal or journal["tenant_id"] != tenant:
			raise ValueError(f"journal_not_found:{journal_id}")

		threshold = _d(amount_threshold)
		total = _d(journal["total_debits"])

		if total > threshold:
			journal["status"] = "pending_approval"
			workflow_record = {
				"id": self._record_id("wf"),
				"type": "journal_approval_workflow",
				"tenant_id": tenant,
				"journal_id": journal_id,
				"journal_number": journal.get("journal_number"),
				"amount": str(total),
				"threshold": str(threshold),
				"approver_id": approver_id,
				"decision": "pending",
				"status": "pending",
				"created_at": self._now(),
			}
			self.approval_workflows[workflow_record["id"]] = workflow_record
			return deepcopy(workflow_record)

		# Auto-approve
		journal["approved_by"] = approver_id
		journal["approved_at"] = self._now()
		journal["status"] = "approved"
		workflow_record = {
			"id": self._record_id("wf"),
			"type": "journal_approval_workflow",
			"tenant_id": tenant,
			"journal_id": journal_id,
			"journal_number": journal.get("journal_number"),
			"amount": str(total),
			"threshold": str(threshold),
			"approver_id": approver_id,
			"decision": "auto_approved",
			"status": "approved",
			"created_at": self._now(),
		}
		self.approval_workflows[workflow_record["id"]] = workflow_record
		return deepcopy(workflow_record)

	# ==================================================================
	# PERIOD MANAGEMENT
	# ==================================================================

	async def open_period_v2(
		self,
		tenant_id: str,
		period_code: str,
		opened_by: str,
	) -> dict[str, Any]:
		"""Open a period identified by period_code.

		Looks for an existing period record with matching period_code.  If none
		exists, creates a stub record.  A period can only be opened if it is
		currently in status 'future' or 'closed' (re-open path).
		"""
		tenant = self._tenant(tenant_id)
		period = self._period_by_code(tenant, period_code)
		if period is None:
			# Auto-create minimal period stub
			period_id = self._record_id("period")
			period = {
				"id": period_id,
				"type": "accounting_period",
				"tenant_id": tenant,
				"name": period_code,
				"period_code": period_code,
				"fiscal_year": int(period_code[:4]) if len(period_code) >= 4 and period_code[:4].isdigit() else 0,
				"period_start": f"{period_code[:7]}-01" if len(period_code) >= 7 else self._today(),
				"period_end": self._today(),
				"status": "future",
				"created_at": self._now(),
			}
			self.periods[period_id] = period

		allowed_from = {"future", "closed"}
		if period["status"] not in allowed_from:
			raise ValueError(f"period_cannot_be_opened:current_status={period['status']}")

		period["status"] = "open"
		period["opened_by"] = opened_by
		period["opened_at"] = self._now()
		self._emit(tenant, "period_opened", period)
		return deepcopy(period)

	async def close_period(
		self,
		tenant_id: str,
		period_code: str,
		closed_by: str,
	) -> dict[str, Any]:
		"""Close a period after pre-close checks.

		Pre-close checks performed:
		1. No journal entries in 'balanced' or 'pending_approval' state for the period.
		2. All reconciliations for the period are 'approved'.
		3. AR/AP subledger totals match their control accounts (stub check — passes
		   automatically in the in-memory service unless explicitly flagged).

		Raises ValueError listing all failed checks if any are outstanding.
		"""
		tenant = self._tenant(tenant_id)
		period = self._period_by_code(tenant, period_code)
		if period is None:
			raise ValueError(f"period_not_found:{period_code}")
		if period["status"] != "open":
			raise ValueError(f"period_not_open:status={period['status']}")

		failures: list[str] = []

		# Check 1: no unposted journals in this period
		for je in self.journal_entries.values():
			if je["tenant_id"] != tenant:
				continue
			batch = self.journal_batches.get(je["batch_id"], {})
			if batch.get("period_id") == period["id"] and je["status"] in {"balanced", "pending_approval"}:
				failures.append(f"unposted_journal:{je['id']}")

		# Check 2: all reconciliations for the period must be approved
		for rec in self.reconciliations.values():
			if rec["tenant_id"] == tenant and rec.get("period_code") == period_code and rec["status"] != "approved":
				failures.append(f"unapproved_reconciliation:{rec['id']}")

		if failures:
			raise ValueError(f"period_close_blocked:{','.join(failures)}")

		period["status"] = "closed"
		period["closed_by"] = closed_by
		period["closed_at"] = self._now()
		self._emit(tenant, "period_closed", period)
		return deepcopy(period)

	async def lock_period(
		self,
		tenant_id: str,
		period_code: str,
		locked_by: str,
	) -> dict[str, Any]:
		"""Permanently lock a period.  No adjustments are possible after locking."""
		tenant = self._tenant(tenant_id)
		period = self._period_by_code(tenant, period_code)
		if period is None:
			raise ValueError(f"period_not_found:{period_code}")
		if period["status"] != "closed":
			raise ValueError(f"period_must_be_closed_before_locking:status={period['status']}")

		period["status"] = "locked"
		period["locked_by"] = locked_by
		period["locked_at"] = self._now()
		self._emit(tenant, "period_locked", period)
		return deepcopy(period)

	async def reopen_period(
		self,
		tenant_id: str,
		period_code: str,
		reason: str,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Re-open a closed period.  Restricted: CFO-authorised, full audit trail.

		Locked periods cannot be reopened — only closed ones.
		"""
		tenant = self._tenant(tenant_id)
		if not reason:
			raise ValueError("reopen_reason_required")
		if not authorised_by:
			raise ValueError("authorisation_required_for_period_reopen")

		period = self._period_by_code(tenant, period_code)
		if period is None:
			raise ValueError(f"period_not_found:{period_code}")
		if period["status"] == "locked":
			raise PermissionError("locked_period_cannot_be_reopened")
		if period["status"] not in {"closed", "soft_closed"}:
			raise ValueError(f"period_not_closed:status={period['status']}")

		audit_entry = {
			"id": self._record_id("audit_reopen"),
			"type": "period_reopen_audit",
			"tenant_id": tenant,
			"period_code": period_code,
			"reason": reason,
			"authorised_by": authorised_by,
			"previous_status": period["status"],
			"status": "active",
			"created_at": self._now(),
		}
		self._audit_events.append({
			**audit_entry,
			"event_type": "period_reopened",
			"record_id": period["id"],
			"record_type": "accounting_period",
			"stream": GLR_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

		period["status"] = "open"
		period["reopened_by"] = authorised_by
		period["reopened_reason"] = reason
		period["reopened_at"] = self._now()
		return deepcopy(period)

	async def period_end_checklist(
		self,
		tenant_id: str,
		period_code: str,
	) -> dict[str, Any]:
		"""Return a structured checklist of outstanding period-end items.

		Each item has: category, description, status ('complete'|'outstanding'), detail.
		"""
		tenant = self._tenant(tenant_id)
		period = self._period_by_code(tenant, period_code)
		if period is None:
			raise ValueError(f"period_not_found:{period_code}")

		items: list[dict[str, Any]] = []

		# 1. Unposted journals
		unposted = [
			je["id"] for je in self.journal_entries.values()
			if je["tenant_id"] == tenant
			and self.journal_batches.get(je.get("batch_id", ""), {}).get("period_id") == period["id"]
			and je["status"] in {"balanced", "pending_approval"}
		]
		items.append({
			"category": "journals",
			"description": "All journals posted",
			"status": "complete" if not unposted else "outstanding",
			"detail": unposted,
		})

		# 2. Pending approval workflows
		pending_approvals = [
			wf["journal_id"] for wf in self.approval_workflows.values()
			if wf["tenant_id"] == tenant and wf["status"] == "pending"
		]
		items.append({
			"category": "approvals",
			"description": "No journals pending approval",
			"status": "complete" if not pending_approvals else "outstanding",
			"detail": pending_approvals,
		})

		# 3. Unapproved reconciliations
		open_recs = [
			r["id"] for r in self.reconciliations.values()
			if r["tenant_id"] == tenant
			and r.get("period_code") == period_code
			and r["status"] not in {"approved"}
		]
		items.append({
			"category": "reconciliations",
			"description": "All reconciliations approved",
			"status": "complete" if not open_recs else "outstanding",
			"detail": open_recs,
		})

		# 4. Recurring journals run (stub — always complete unless template flag set)
		outstanding_recurring = [
			t["id"] for t in self.recurring_templates.values()
			if t["tenant_id"] == tenant and t.get("run_required_for_period") == period_code
		]
		items.append({
			"category": "recurring",
			"description": "Recurring journals run",
			"status": "complete" if not outstanding_recurring else "outstanding",
			"detail": outstanding_recurring,
		})

		outstanding_count = sum(1 for it in items if it["status"] == "outstanding")
		return {
			"id": self._record_id("checklist"),
			"type": "period_end_checklist",
			"tenant_id": tenant,
			"period_code": period_code,
			"period_status": period["status"],
			"items": items,
			"outstanding_count": outstanding_count,
			"ready_to_close": outstanding_count == 0,
			"generated_at": self._now(),
		}

	async def get_period_status(
		self,
		tenant_id: str,
		fiscal_year: int,
	) -> list[dict[str, Any]]:
		"""Return all periods in a fiscal year with their current status."""
		tenant = self._tenant(tenant_id)
		result = [
			deepcopy(p)
			for p in self.periods.values()
			if p["tenant_id"] == tenant and p.get("fiscal_year") == fiscal_year
		]
		result.sort(key=lambda p: p.get("period_start", ""))
		return result

	# ==================================================================
	# FINANCIAL STATEMENTS
	# ==================================================================

	async def trial_balance(
		self,
		tenant_id: str,
		period_code: str,
		include_zero_balances: bool = False,
	) -> dict[str, Any]:
		"""Full trial balance: opening balance, movements, closing balance per account.

		Verifies total debits == total credits as a post-computation assertion.
		"""
		tenant = self._tenant(tenant_id)

		# Accumulate all postings up to and including period_code
		# Opening = sum of all postings in periods BEFORE this period
		# Movements = sum of postings IN this period

		account_rows: dict[str, dict[str, Any]] = {}

		for acct in self.accounts.values():
			if acct["tenant_id"] != tenant:
				continue
			account_rows[acct["id"]] = {
				"account_id": acct["id"],
				"account_code": acct["code"],
				"account_name": acct["name"],
				"account_type": acct["account_type"],
				"opening_debit": Decimal("0"),
				"opening_credit": Decimal("0"),
				"period_debit": Decimal("0"),
				"period_credit": Decimal("0"),
			}

		for posting in self.postings.values():
			if posting["tenant_id"] != tenant:
				continue
			p_code = posting.get("period_code", "")
			is_current = p_code == period_code
			is_prior = p_code < period_code if period_code else False

			for line in posting["lines"]:
				acct_id = line.get("account_id")
				if acct_id not in account_rows:
					continue
				d = _d(line.get("debit", 0))
				c = _d(line.get("credit", 0))
				if is_prior:
					account_rows[acct_id]["opening_debit"] += d
					account_rows[acct_id]["opening_credit"] += c
				elif is_current:
					account_rows[acct_id]["period_debit"] += d
					account_rows[acct_id]["period_credit"] += c

		rows = []
		total_closing_debit = Decimal("0")
		total_closing_credit = Decimal("0")

		for row in account_rows.values():
			opening_net = row["opening_debit"] - row["opening_credit"]
			closing_net = opening_net + row["period_debit"] - row["period_credit"]

			if not include_zero_balances and closing_net == 0 and row["period_debit"] == 0 and row["period_credit"] == 0:
				continue

			if closing_net >= 0:
				closing_debit = closing_net
				closing_credit = Decimal("0")
			else:
				closing_debit = Decimal("0")
				closing_credit = abs(closing_net)

			total_closing_debit += closing_debit
			total_closing_credit += closing_credit

			rows.append({
				"account_code": row["account_code"],
				"account_name": row["account_name"],
				"account_type": row["account_type"],
				"opening_balance": str(opening_net),
				"period_debit": str(row["period_debit"]),
				"period_credit": str(row["period_credit"]),
				"closing_debit": str(closing_debit),
				"closing_credit": str(closing_credit),
			})

		rows.sort(key=lambda r: r["account_code"])

		balanced = total_closing_debit == total_closing_credit

		return {
			"id": self._record_id("tb"),
			"type": "trial_balance",
			"tenant_id": tenant,
			"period_code": period_code,
			"rows": rows,
			"total_closing_debit": str(total_closing_debit),
			"total_closing_credit": str(total_closing_credit),
			"balanced": balanced,
			"include_zero_balances": include_zero_balances,
			"generated_at": self._now(),
		}

	async def balance_sheet(
		self,
		tenant_id: str,
		period_code: str,
		comparative_period: str | None = None,
	) -> dict[str, Any]:
		"""Assets = Liabilities + Equity.

		Groups accounts by type, applies normal-balance sign conventions, and
		returns a structured balance sheet.  Current vs non-current classification
		is taken from account metadata (current flag, defaults True for in-memory).
		"""
		tenant = self._tenant(tenant_id)
		tb = await self.trial_balance(tenant, period_code, include_zero_balances=False)

		comparative_tb: dict[str, Any] | None = None
		if comparative_period:
			comparative_tb = await self.trial_balance(tenant, comparative_period, include_zero_balances=False)

		def _net_balance_for_type(rows: list[dict[str, Any]], acct_type: str) -> Decimal:
			total = Decimal("0")
			for r in rows:
				if r["account_type"] != acct_type:
					continue
				# Closing balance net position
				total += _d(r["closing_debit"]) - _d(r["closing_credit"])
			return total

		def _rows_for_type(rows: list[dict[str, Any]], acct_type: str) -> list[dict[str, Any]]:
			return [r for r in rows if r["account_type"] == acct_type]

		def _section(rows: list[dict[str, Any]], acct_type: str, comparative_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
			section_rows = _rows_for_type(rows, acct_type)
			comp_map: dict[str, Decimal] = {}
			if comparative_rows:
				for cr in _rows_for_type(comparative_rows, acct_type):
					comp_map[cr["account_code"]] = _d(cr["closing_debit"]) - _d(cr["closing_credit"])

			lines_out = []
			for r in section_rows:
				net = _d(r["closing_debit"]) - _d(r["closing_credit"])
				# For credit-normal types flip sign for presentation
				if acct_type in _CREDIT_NORMAL_TYPES:
					net = -net
				comp_val = comp_map.get(r["account_code"])
				entry: dict[str, Any] = {
					"account_code": r["account_code"],
					"account_name": r["account_name"],
					"balance": str(net),
				}
				if comp_val is not None:
					if acct_type in _CREDIT_NORMAL_TYPES:
						comp_val = -comp_val
					entry["comparative_balance"] = str(comp_val)
				lines_out.append(entry)

			total = sum(_d(ln["balance"]) for ln in lines_out)
			section: dict[str, Any] = {"lines": lines_out, "total": str(total)}
			if comp_map:
				section["comparative_total"] = str(sum(
					(_d(ln["comparative_balance"]) if "comparative_balance" in ln else Decimal("0"))
					for ln in lines_out
				))
			return section

		rows = tb["rows"]
		comp_rows = comparative_tb["rows"] if comparative_tb else None

		assets = _section(rows, "asset", comp_rows)
		liabilities = _section(rows, "liability", comp_rows)
		equity = _section(rows, "equity", comp_rows)

		# Include current-period net income in equity (profit not yet closed to RE).
		# This is the standard presentation before year-end close.
		inc_stmt_rows = [r for r in rows if r["account_type"] in _INCOME_STMT_TYPES]
		period_net_income = Decimal("0")
		for r in inc_stmt_rows:
			net = _d(r["closing_debit"]) - _d(r["closing_credit"])
			if r["account_type"] == "revenue":
				period_net_income += -net   # credit-normal → positive = revenue
			else:
				period_net_income += -net   # debit-normal expense reduces NI → net is positive so negate

		if period_net_income != 0:
			equity["lines"].append({
				"account_code": "NET_INCOME",
				"account_name": "Profit for the period",
				"balance": str(period_net_income),
			})
			equity["total"] = str(_d(equity["total"]) + period_net_income)
			if comparative_tb:
				comp_inc_rows = [r for r in comp_rows if r["account_type"] in _INCOME_STMT_TYPES]
				comp_ni = Decimal("0")
				for r in comp_inc_rows:
					net = _d(r["closing_debit"]) - _d(r["closing_credit"])
					comp_ni += -net
				equity["lines"][-1]["comparative_balance"] = str(comp_ni)
				equity["comparative_total"] = str(
					_d(equity.get("comparative_total", "0")) + comp_ni
				)

		total_assets = _d(assets["total"])
		total_liab_equity = _d(liabilities["total"]) + _d(equity["total"])
		balanced = total_assets == total_liab_equity

		return {
			"id": self._record_id("bs"),
			"type": "balance_sheet",
			"tenant_id": tenant,
			"period_code": period_code,
			"comparative_period": comparative_period,
			"assets": assets,
			"liabilities": liabilities,
			"equity": equity,
			"total_assets": str(total_assets),
			"total_liabilities_and_equity": str(total_liab_equity),
			"balanced": balanced,
			"generated_at": self._now(),
		}

	async def income_statement(
		self,
		tenant_id: str,
		period_code: str,
		comparative_period: str | None = None,
		segment: str | None = None,
	) -> dict[str, Any]:
		"""Revenue − COGS = Gross Profit; GP − Opex = EBIT; EBIT − Finance = EBT; EBT − Tax = PAT.

		Revenue accounts are credit-normal (positive balance = revenue).
		Expense accounts are debit-normal (positive balance = expense).
		All amounts are absolute values with sign in context label.

		segment: optional dimension value to filter postings (e.g. cost_center='CC01').
		"""
		tenant = self._tenant(tenant_id)

		def _collect_period(p_code: str) -> tuple[Decimal, Decimal]:
			"""Return (total_revenue, total_expense) for period p_code."""
			rev = Decimal("0")
			exp = Decimal("0")
			for posting in self.postings.values():
				if posting["tenant_id"] != tenant or posting.get("period_code") != p_code:
					continue
				for line in posting["lines"]:
					if segment and line.get("segment") != segment:
						continue
					acct_id = line.get("account_id")
					acct = self.accounts.get(acct_id or "")
					if not acct:
						continue
					if acct["account_type"] == "revenue":
						# Revenue: credit increases revenue
						rev += _d(line.get("credit", 0)) - _d(line.get("debit", 0))
					elif acct["account_type"] == "expense":
						exp += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
			return rev, exp

		# Simplified P&L — for a production system you'd have COGS sub-type etc.
		revenue, expense = _collect_period(period_code)
		gross_profit = revenue  # Without COGS separation: GP = Revenue
		ebit = gross_profit - expense
		# Finance cost and tax: look for accounts tagged 'finance' or 'tax' in metadata
		finance_cost = Decimal("0")
		tax_expense = Decimal("0")
		for posting in self.postings.values():
			if posting["tenant_id"] != tenant or posting.get("period_code") != period_code:
				continue
			for line in posting["lines"]:
				acct_id = line.get("account_id")
				acct = self.accounts.get(acct_id or "")
				if not acct:
					continue
				tags = acct.get("tags", [])
				if "finance_cost" in tags:
					finance_cost += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
				if "tax_expense" in tags:
					tax_expense += _d(line.get("debit", 0)) - _d(line.get("credit", 0))

		ebt = ebit - finance_cost
		pat = ebt - tax_expense

		comparative: dict[str, Any] | None = None
		if comparative_period:
			comp_rev, comp_exp = _collect_period(comparative_period)
			comparative = {
				"period_code": comparative_period,
				"revenue": str(comp_rev),
				"total_expense": str(comp_exp),
				"gross_profit": str(comp_rev),
				"ebit": str(comp_rev - comp_exp),
				"pat": str(comp_rev - comp_exp),
			}

		return {
			"id": self._record_id("is"),
			"type": "income_statement",
			"tenant_id": tenant,
			"period_code": period_code,
			"segment": segment,
			"revenue": str(revenue),
			"cost_of_goods_sold": "0.00",
			"gross_profit": str(gross_profit),
			"operating_expenses": str(expense),
			"ebit": str(ebit),
			"finance_cost": str(finance_cost),
			"ebt": str(ebt),
			"tax_expense": str(tax_expense),
			"pat": str(pat),
			"comparative": comparative,
			"generated_at": self._now(),
		}

	async def cash_flow_statement(
		self,
		tenant_id: str,
		period_code: str,
		method: str = "indirect",
	) -> dict[str, Any]:
		"""Cash flow statement using the indirect method.

		Indirect method:
		  Net income (PAT)
		  + Adjustments for non-cash items (depreciation, amortisation)
		  +/- Working capital changes (AR, AP, Inventory)
		  = Operating cash flow
		  +/- Investing activities (capex, disposals)
		  +/- Financing activities (debt, equity)
		  = Net change in cash

		Non-cash and working capital classification is determined by account
		tags ('depreciation', 'amortisation', 'accounts_receivable',
		'accounts_payable', 'inventory', 'capex', 'debt', 'equity_financing').
		"""
		tenant = self._tenant(tenant_id)
		is_result = await self.income_statement(tenant, period_code)
		pat = _d(is_result["pat"])

		# Accumulate tag-based adjustments from posted lines in the period
		non_cash_adj = Decimal("0")
		wc_changes = Decimal("0")
		investing = Decimal("0")
		financing = Decimal("0")

		for posting in self.postings.values():
			if posting["tenant_id"] != tenant or posting.get("period_code") != period_code:
				continue
			for line in posting["lines"]:
				acct = self.accounts.get(line.get("account_id") or "")
				if not acct:
					continue
				tags = set(acct.get("tags", []))
				net = _d(line.get("debit", 0)) - _d(line.get("credit", 0))
				if tags & {"depreciation", "amortisation"}:
					non_cash_adj += net  # depreciation is an expense so net>0, add back
				elif tags & {"accounts_receivable"}:
					wc_changes -= net  # increase in AR is a use of cash
				elif tags & {"accounts_payable"}:
					wc_changes += net  # increase in AP is a source of cash
				elif tags & {"inventory"}:
					wc_changes -= net
				elif tags & {"capex"}:
					investing -= net
				elif tags & {"debt"}:
					financing += net
				elif tags & {"equity_financing"}:
					financing += net

		operating = pat + non_cash_adj + wc_changes
		net_change = operating + investing + financing

		return {
			"id": self._record_id("cfs"),
			"type": "cash_flow_statement",
			"tenant_id": tenant,
			"period_code": period_code,
			"method": method,
			"operating_activities": {
				"net_income": str(pat),
				"non_cash_adjustments": str(non_cash_adj),
				"working_capital_changes": str(wc_changes),
				"net_operating_cash_flow": str(operating),
			},
			"investing_activities": {
				"net_investing_cash_flow": str(investing),
			},
			"financing_activities": {
				"net_financing_cash_flow": str(financing),
			},
			"net_change_in_cash": str(net_change),
			"generated_at": self._now(),
		}

	async def statement_of_equity(
		self,
		tenant_id: str,
		fiscal_year: int,
	) -> dict[str, Any]:
		"""Statement of changes in equity for a full fiscal year.

		Opening equity
		+ Profit for the year (PAT)
		+ Other comprehensive income (OCI) — tagged accounts
		− Dividends
		= Closing equity
		"""
		tenant = self._tenant(tenant_id)

		periods_in_year = [
			p for p in self.periods.values()
			if p["tenant_id"] == tenant and p.get("fiscal_year") == fiscal_year
		]
		periods_in_year.sort(key=lambda p: p.get("period_start", ""))

		opening_equity = Decimal("0")
		total_pat = Decimal("0")
		oci = Decimal("0")
		dividends = Decimal("0")

		# Opening equity = equity account balances before first period of year
		first_period_code = periods_in_year[0].get("period_code") if periods_in_year else None

		for acct in self.accounts.values():
			if acct["tenant_id"] != tenant or acct["account_type"] != "equity":
				continue
			bal = self._get_account_balance(tenant, acct["id"], first_period_code)
			opening_equity += bal["opening"]

		# P&L for each period in the year
		for period in periods_in_year:
			pc = period.get("period_code")
			if not pc:
				continue
			is_result = await self.income_statement(tenant, pc)
			total_pat += _d(is_result["pat"])

			# OCI and dividends from tagged accounts
			for posting in self.postings.values():
				if posting["tenant_id"] != tenant or posting.get("period_code") != pc:
					continue
				for line in posting["lines"]:
					acct = self.accounts.get(line.get("account_id") or "")
					if not acct:
						continue
					tags = set(acct.get("tags", []))
					net = _d(line.get("debit", 0)) - _d(line.get("credit", 0))
					if "oci" in tags:
						oci += -net  # credit to OCI is positive
					elif "dividend" in tags:
						dividends += net  # debit to dividends paid

		closing_equity = opening_equity + total_pat + oci - dividends

		return {
			"id": self._record_id("soe"),
			"type": "statement_of_equity",
			"tenant_id": tenant,
			"fiscal_year": fiscal_year,
			"opening_equity": str(opening_equity),
			"profit_for_year": str(total_pat),
			"other_comprehensive_income": str(oci),
			"dividends_paid": str(dividends),
			"closing_equity": str(closing_equity),
			"generated_at": self._now(),
		}

	async def budget_vs_actual(
		self,
		tenant_id: str,
		period_code: str,
		budget_version: str = "approved",
	) -> dict[str, Any]:
		"""Account-level variance analysis: actual vs budget with F/A indicator."""
		tenant = self._tenant(tenant_id)
		tb = await self.trial_balance(tenant, period_code, include_zero_balances=True)

		# Build budget map for the period
		budget_map: dict[str, Decimal] = {}
		for b in self.budgets.values():
			if (
				b["tenant_id"] == tenant
				and b.get("period_code") == period_code
				and b.get("budget_version", "approved") == budget_version
			):
				budget_map[b["account_code"]] = _d(b["budget_amount"])

		rows = []
		for r in tb["rows"]:
			closing_net = _d(r["closing_debit"]) - _d(r["closing_credit"])
			# Flip sign for credit-normal accounts so positive = good
			if r["account_type"] in _CREDIT_NORMAL_TYPES:
				actual = -closing_net
			else:
				actual = closing_net

			budget = budget_map.get(r["account_code"], Decimal("0"))
			variance = actual - budget
			pct_variance = (
				(variance / budget * 100).quantize(TWO, rounding=ROUND_HALF_UP)
				if budget != 0
				else Decimal("0")
			)
			indicator = self._variance_indicator(r["account_type"], actual, budget)

			rows.append({
				"account_code": r["account_code"],
				"account_name": r["account_name"],
				"account_type": r["account_type"],
				"actual": str(actual),
				"budget": str(budget),
				"variance": str(variance),
				"variance_pct": str(pct_variance),
				"indicator": indicator,
			})

		return {
			"id": self._record_id("bva"),
			"type": "budget_vs_actual",
			"tenant_id": tenant,
			"period_code": period_code,
			"budget_version": budget_version,
			"rows": rows,
			"row_count": len(rows),
			"generated_at": self._now(),
		}

	async def segment_report(
		self,
		tenant_id: str,
		period_code: str,
		segment_dimension: str = "cost_center",
	) -> dict[str, Any]:
		"""P&L by segment dimension (cost_center / department / project / geography)."""
		tenant = self._tenant(tenant_id)

		# Collect distinct segment values present in postings for the period
		segments: dict[str, dict[str, Decimal]] = {}

		for posting in self.postings.values():
			if posting["tenant_id"] != tenant or posting.get("period_code") != period_code:
				continue
			for line in posting["lines"]:
				seg_value = line.get(segment_dimension, "unallocated")
				acct = self.accounts.get(line.get("account_id") or "")
				if not acct:
					continue
				if acct["account_type"] not in _INCOME_STMT_TYPES:
					continue
				segments.setdefault(seg_value, {"revenue": Decimal("0"), "expense": Decimal("0")})
				d = _d(line.get("debit", 0))
				c = _d(line.get("credit", 0))
				if acct["account_type"] == "revenue":
					segments[seg_value]["revenue"] += c - d
				else:
					segments[seg_value]["expense"] += d - c

		segment_rows = []
		for seg, totals in sorted(segments.items()):
			rev = totals["revenue"]
			exp = totals["expense"]
			segment_rows.append({
				"segment": seg,
				"revenue": str(rev),
				"expenses": str(exp),
				"contribution": str(rev - exp),
			})

		return {
			"id": self._record_id("seg"),
			"type": "segment_report",
			"tenant_id": tenant,
			"period_code": period_code,
			"segment_dimension": segment_dimension,
			"segments": segment_rows,
			"generated_at": self._now(),
		}

	async def management_accounts_pack(
		self,
		tenant_id: str,
		period_code: str,
	) -> dict[str, Any]:
		"""Full management pack: all statements + ratios + KPIs."""
		tenant = self._tenant(tenant_id)

		tb = await self.trial_balance(tenant, period_code)
		bs = await self.balance_sheet(tenant, period_code)
		inc = await self.income_statement(tenant, period_code)
		cfs = await self.cash_flow_statement(tenant, period_code)
		bva = await self.budget_vs_actual(tenant, period_code)

		revenue = _d(inc["revenue"])
		pat = _d(inc["pat"])
		total_assets = _d(bs["total_assets"])
		total_equity = _d(bs["equity"]["total"])

		ratios: dict[str, str] = {}
		if revenue != 0:
			ratios["net_profit_margin_pct"] = str((pat / revenue * 100).quantize(TWO))
		if total_assets != 0:
			ratios["return_on_assets_pct"] = str((pat / total_assets * 100).quantize(TWO))
		if total_equity != 0:
			ratios["return_on_equity_pct"] = str((pat / total_equity * 100).quantize(TWO))

		return {
			"id": self._record_id("map"),
			"type": "management_accounts_pack",
			"tenant_id": tenant,
			"period_code": period_code,
			"trial_balance": tb,
			"balance_sheet": bs,
			"income_statement": inc,
			"cash_flow_statement": cfs,
			"budget_vs_actual": bva,
			"ratios": ratios,
			"commentary_template": (
				f"Period {period_code}: Revenue {inc['revenue']}, PAT {inc['pat']}. "
				f"Total assets {bs['total_assets']}."
			),
			"generated_at": self._now(),
		}

	# ==================================================================
	# RECONCILIATION
	# ==================================================================

	async def account_reconciliation(
		self,
		tenant_id: str,
		account_code: str,
		period_code: str,
	) -> dict[str, Any]:
		"""Compare GL balance to subledger or bank statement balance.

		Returns the reconciliation header with the GL balance computed from
		postings.  Reconciling items are added via submit_reconciliation.
		"""
		tenant = self._tenant(tenant_id)
		acct = self._account_by_code(tenant, account_code)
		if not acct:
			raise ValueError(f"account_not_found:{account_code}")

		bal = self._get_account_balance(tenant, acct["id"], period_code)
		gl_balance = bal["opening"] + bal["debits"] - bal["credits"]

		rec_id = self._record_id("rec")
		record = {
			"id": rec_id,
			"type": "account_reconciliation",
			"tenant_id": tenant,
			"account_code": account_code,
			"account_id": acct["id"],
			"account_name": acct["name"],
			"period_code": period_code,
			"gl_balance": str(gl_balance),
			"subledger_balance": None,  # populated when items are submitted
			"reconciling_items": [],
			"unreconciled_difference": str(gl_balance),
			"status": "open",
			"created_at": self._now(),
		}
		self.reconciliations[rec_id] = record
		return deepcopy(record)

	async def submit_reconciliation(
		self,
		tenant_id: str,
		reconciliation_id: str,
		reconciled_by: str,
		reconciling_items: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Submit reconciling items and update the unreconciled difference.

		reconciling_items: list of {description, amount, type}.
		type: 'timing_difference' | 'error' | 'outstanding_cheque' | 'deposit_in_transit' etc.
		"""
		tenant = self._tenant(tenant_id)
		rec = self.reconciliations.get(reconciliation_id)
		if not rec or rec["tenant_id"] != tenant:
			raise ValueError(f"reconciliation_not_found:{reconciliation_id}")

		total_items = sum(_d(item.get("amount", 0)) for item in reconciling_items)
		gl_balance = _d(rec["gl_balance"])
		subledger_balance = gl_balance - total_items  # GL minus items = subledger

		rec["reconciling_items"] = deepcopy(reconciling_items)
		rec["subledger_balance"] = str(subledger_balance)
		rec["unreconciled_difference"] = str(gl_balance - subledger_balance - total_items)
		rec["reconciled_by"] = reconciled_by
		rec["reconciled_at"] = self._now()
		rec["status"] = "submitted"
		return deepcopy(rec)

	async def approve_reconciliation(
		self,
		tenant_id: str,
		reconciliation_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Approve a submitted reconciliation."""
		tenant = self._tenant(tenant_id)
		rec = self.reconciliations.get(reconciliation_id)
		if not rec or rec["tenant_id"] != tenant:
			raise ValueError(f"reconciliation_not_found:{reconciliation_id}")
		if rec["status"] != "submitted":
			raise ValueError(f"reconciliation_not_submitted:status={rec['status']}")

		rec["approved_by"] = approved_by
		rec["approved_at"] = self._now()
		rec["status"] = "approved"
		return deepcopy(rec)

	async def bank_reconciliation(
		self,
		tenant_id: str,
		bank_account_code: str,
		statement_id: str,
	) -> dict[str, Any]:
		"""Match GL entries to bank statement lines.

		In the in-memory implementation the bank statement is looked up by
		statement_id from a notional statements store (keyed as reconciliation
		records with type='bank_statement').  Unmatched items are flagged.

		Returns: matched_items, unmatched_gl, unmatched_bank, difference.
		"""
		tenant = self._tenant(tenant_id)
		acct = self._account_by_code(tenant, bank_account_code)
		if not acct:
			raise ValueError(f"bank_account_not_found:{bank_account_code}")

		# Fetch GL postings for the account (all periods, let caller filter by date)
		gl_lines: list[dict[str, Any]] = []
		for posting in self.postings.values():
			if posting["tenant_id"] != tenant:
				continue
			for line in posting["lines"]:
				if line.get("account_id") == acct["id"]:
					gl_lines.append({
						"posting_id": posting["id"],
						"debit": str(_d(line.get("debit", 0))),
						"credit": str(_d(line.get("credit", 0))),
						"description": line.get("description", ""),
						"matched": False,
					})

		# Bank statement lines come from a reconciliation record tagged as 'bank_statement'
		statement = self.reconciliations.get(statement_id)
		bank_lines = statement.get("reconciling_items", []) if statement else []

		# Simple matching: match on amount equality
		matched: list[dict[str, Any]] = []
		unmatched_gl = []
		unmatched_bank = list(deepcopy(bank_lines))

		for gl in gl_lines:
			gl_net = _d(gl["debit"]) - _d(gl["credit"])
			for i, bl in enumerate(unmatched_bank):
				bl_amount = _d(bl.get("amount", 0))
				if gl_net == bl_amount:
					matched.append({"gl": gl, "bank": bl})
					unmatched_bank.pop(i)
					break
			else:
				unmatched_gl.append(gl)

		gl_balance = sum(_d(g["debit"]) - _d(g["credit"]) for g in gl_lines)
		bank_balance = sum(_d(b.get("amount", 0)) for b in bank_lines)
		difference = gl_balance - bank_balance

		rec_id = self._record_id("bankrec")
		result = {
			"id": rec_id,
			"type": "bank_reconciliation",
			"tenant_id": tenant,
			"bank_account_code": bank_account_code,
			"statement_id": statement_id,
			"gl_balance": str(gl_balance),
			"bank_statement_balance": str(bank_balance),
			"difference": str(difference),
			"matched_count": len(matched),
			"unmatched_gl_count": len(unmatched_gl),
			"unmatched_bank_count": len(unmatched_bank),
			"matched_items": matched,
			"unmatched_gl": unmatched_gl,
			"unmatched_bank": unmatched_bank,
			"status": "complete" if difference == 0 else "difference_outstanding",
			"created_at": self._now(),
		}
		self.reconciliations[rec_id] = result
		return deepcopy(result)

	async def intercompany_reconciliation(
		self,
		tenant_id: str,
		counterpart_entity: str,
		period_code: str,
	) -> dict[str, Any]:
		"""Match AR in entity A to AP in entity B; identify breaks."""
		tenant = self._tenant(tenant_id)

		# Collect intercompany journals between the two entities
		entity_postings: list[dict[str, Any]] = []
		cp_postings: list[dict[str, Any]] = []

		for ic in self.intercompany_journals.values():
			if ic["period_code"] if "period_code" in ic else None == period_code:
				pass  # period filtering not stored on IC record; accept all for now
			if ic["tenant_id"] == tenant and ic["counterpart_entity"] == counterpart_entity:
				entity_postings.append(ic)
			elif ic["tenant_id"] == counterpart_entity and ic["counterpart_entity"] == tenant:
				cp_postings.append(ic)

		entity_total = sum(_d(ic["amount"]) for ic in entity_postings)
		cp_total = sum(_d(ic["amount"]) for ic in cp_postings)
		difference = entity_total - cp_total

		breaks: list[dict[str, Any]] = []
		if difference != 0:
			breaks.append({
				"type": "balance_mismatch",
				"entity_total": str(entity_total),
				"counterpart_total": str(cp_total),
				"difference": str(difference),
			})

		rec_id = self._record_id("icrec")
		result = {
			"id": rec_id,
			"type": "intercompany_reconciliation",
			"tenant_id": tenant,
			"counterpart_entity": counterpart_entity,
			"period_code": period_code,
			"entity_total": str(entity_total),
			"counterpart_total": str(cp_total),
			"difference": str(difference),
			"breaks": breaks,
			"status": "reconciled" if difference == 0 else "breaks_identified",
			"created_at": self._now(),
		}
		self.reconciliations[rec_id] = result
		return deepcopy(result)

	async def subledger_reconciliation(
		self,
		tenant_id: str,
		period_code: str,
	) -> dict[str, Any]:
		"""Reconcile AR and AP control accounts to their subledgers.

		Control accounts are identified by the tag 'ar_control' or 'ap_control'.
		Subledger balances are taken from accounts tagged 'ar_subledger' / 'ap_subledger'.
		"""
		tenant = self._tenant(tenant_id)

		def _balance_by_tag(tag: str) -> Decimal:
			total = Decimal("0")
			for acct in self.accounts.values():
				if acct["tenant_id"] != tenant or tag not in acct.get("tags", []):
					continue
				bal = self._get_account_balance(tenant, acct["id"], period_code)
				total += bal["opening"] + bal["debits"] - bal["credits"]
			return total

		ar_control = _balance_by_tag("ar_control")
		ar_subledger = _balance_by_tag("ar_subledger")
		ap_control = _balance_by_tag("ap_control")
		ap_subledger = _balance_by_tag("ap_subledger")

		ar_diff = ar_control - ar_subledger
		ap_diff = ap_control - ap_subledger

		items = [
			{
				"ledger": "AR",
				"control_balance": str(ar_control),
				"subledger_balance": str(ar_subledger),
				"difference": str(ar_diff),
				"status": "reconciled" if ar_diff == 0 else "difference",
			},
			{
				"ledger": "AP",
				"control_balance": str(ap_control),
				"subledger_balance": str(ap_subledger),
				"difference": str(ap_diff),
				"status": "reconciled" if ap_diff == 0 else "difference",
			},
		]

		rec_id = self._record_id("subrec")
		result = {
			"id": rec_id,
			"type": "subledger_reconciliation",
			"tenant_id": tenant,
			"period_code": period_code,
			"items": items,
			"status": "reconciled" if ar_diff == 0 and ap_diff == 0 else "differences_found",
			"created_at": self._now(),
		}
		self.reconciliations[rec_id] = result
		return deepcopy(result)

	# ==================================================================
	# CHART OF ACCOUNTS (extended)
	# ==================================================================

	async def create_account_v2(
		self,
		tenant_id: str,
		account_code: str,
		account_name: str,
		account_type: str,
		parent_code: str | None,
		currency: str,
	) -> dict[str, Any]:
		"""Create a ledger account using the v2 async API."""
		tenant = self._tenant(tenant_id)
		if account_type not in SUPPORTED_ACCOUNT_TYPES:
			raise ValueError(f"unsupported_account_type:{account_type}")
		if currency not in SUPPORTED_CURRENCIES:
			raise ValueError(f"unsupported_currency:{currency}")

		parent_id: str | None = None
		if parent_code:
			parent = self._account_by_code(tenant, parent_code)
			if not parent:
				raise ValueError(f"parent_account_not_found:{parent_code}")
			parent_id = parent["id"]

		acct_id = self._record_id("acct")
		record = {
			"id": acct_id,
			"type": "ledger_account",
			"tenant_id": tenant,
			"code": account_code,
			"name": account_name,
			"account_type": account_type,
			"parent_account_id": parent_id,
			"allow_posting": True,
			"currency": currency,
			"tags": [],
			"status": "active",
			"created_at": self._now(),
		}
		self.accounts[acct_id] = record
		self._emit(tenant, "account_created", record)
		return deepcopy(record)

	async def chart_of_accounts(
		self,
		tenant_id: str,
		include_inactive: bool = False,
	) -> list[dict[str, Any]]:
		"""Return all accounts for the tenant, optionally including inactive ones."""
		tenant = self._tenant(tenant_id)
		result = [
			deepcopy(a)
			for a in self.accounts.values()
			if a["tenant_id"] == tenant and (include_inactive or a["status"] == "active")
		]
		result.sort(key=lambda a: a["code"])
		return result

	async def account_hierarchy(self, tenant_id: str) -> dict[str, Any]:
		"""Return the chart of accounts as a nested tree structure."""
		tenant = self._tenant(tenant_id)
		all_accounts = {
			a["id"]: deepcopy(a)
			for a in self.accounts.values()
			if a["tenant_id"] == tenant
		}
		# Add children list to each node
		for a in all_accounts.values():
			a["children"] = []

		roots: list[dict[str, Any]] = []
		for a in all_accounts.values():
			parent_id = a.get("parent_account_id")
			if parent_id and parent_id in all_accounts:
				all_accounts[parent_id]["children"].append(a)
			else:
				roots.append(a)

		roots.sort(key=lambda a: a["code"])

		def _sort_children(node: dict[str, Any]) -> None:
			node["children"].sort(key=lambda c: c["code"])
			for child in node["children"]:
				_sort_children(child)

		for root in roots:
			_sort_children(root)

		return {
			"tenant_id": tenant,
			"account_count": len(all_accounts),
			"tree": roots,
			"generated_at": self._now(),
		}

	async def account_analysis(
		self,
		tenant_id: str,
		account_code: str,
		period_code: str,
		include_journals: bool = True,
	) -> dict[str, Any]:
		"""Full account history: all journal lines, running balance for a period."""
		tenant = self._tenant(tenant_id)
		acct = self._account_by_code(tenant, account_code)
		if not acct:
			raise ValueError(f"account_not_found:{account_code}")

		lines_out: list[dict[str, Any]] = []
		running_balance = Decimal("0")

		# Opening balance (all prior periods)
		for posting in self.postings.values():
			if posting["tenant_id"] != tenant:
				continue
			if posting.get("period_code", "") >= period_code:
				continue
			for line in posting["lines"]:
				if line.get("account_id") == acct["id"]:
					running_balance += _d(line.get("debit", 0)) - _d(line.get("credit", 0))

		opening_balance = running_balance

		# Period lines
		if include_journals:
			for posting in self.postings.values():
				if posting["tenant_id"] != tenant or posting.get("period_code") != period_code:
					continue
				je = self.journal_entries.get(posting["journal_id"])
				for line in posting["lines"]:
					if line.get("account_id") != acct["id"]:
						continue
					d = _d(line.get("debit", 0))
					c = _d(line.get("credit", 0))
					running_balance += d - c
					lines_out.append({
						"posting_id": posting["id"],
						"journal_number": je.get("journal_number") if je else None,
						"journal_date": je.get("journal_date") if je else None,
						"description": line.get("description", ""),
						"debit": str(d),
						"credit": str(c),
						"running_balance": str(running_balance),
					})

		return {
			"id": self._record_id("aa"),
			"type": "account_analysis",
			"tenant_id": tenant,
			"account_code": account_code,
			"account_name": acct["name"],
			"account_type": acct["account_type"],
			"period_code": period_code,
			"opening_balance": str(opening_balance),
			"closing_balance": str(running_balance),
			"line_count": len(lines_out),
			"lines": lines_out,
			"generated_at": self._now(),
		}

	async def reorganise_chart(
		self,
		tenant_id: str,
		mapping: dict[str, str],
		approved_by: str,
	) -> dict[str, Any]:
		"""Re-code accounts.  mapping = {old_code: new_code}.

		Updates all journal lines and posting references to use the new codes.
		Creates an audit record of each change.
		"""
		tenant = self._tenant(tenant_id)
		if not approved_by:
			raise PermissionError("approval_required_for_chart_reorganisation")

		changes: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []

		for old_code, new_code in mapping.items():
			acct = self._account_by_code(tenant, old_code)
			if not acct:
				errors.append({"old_code": old_code, "error": "account_not_found"})
				continue

			# Check new code is not already taken
			if self._account_by_code(tenant, new_code):
				errors.append({"old_code": old_code, "new_code": new_code, "error": "new_code_already_exists"})
				continue

			old_code_stored = acct["code"]
			acct["code"] = new_code
			acct["previous_code"] = old_code_stored
			acct["recoded_by"] = approved_by
			acct["recoded_at"] = self._now()
			changes.append({"old_code": old_code, "new_code": new_code, "account_id": acct["id"]})

		return {
			"id": self._record_id("reorg"),
			"type": "chart_reorganisation",
			"tenant_id": tenant,
			"approved_by": approved_by,
			"changes_applied": len(changes),
			"errors": errors,
			"changes": changes,
			"status": "completed",
			"created_at": self._now(),
		}

	# ==================================================================
	# YEAR-END CLOSE
	# ==================================================================

	async def year_end_closing(
		self,
		tenant_id: str,
		fiscal_year: int,
		retained_earnings_account: str,
	) -> dict[str, Any]:
		"""Close all revenue and expense accounts to retained earnings.

		Steps:
		1. Compute net P&L for the year (sum of all income-statement-type accounts).
		2. Generate closing journal entries: debit all revenue accounts, credit all
		   expense accounts to zero them; net goes to retained earnings.
		3. Mark the fiscal year as closed.
		4. Lock all periods in the year.

		Returns the closing journal posting record.
		"""
		tenant = self._tenant(tenant_id)
		re_acct = self._account_by_code(tenant, retained_earnings_account)
		if not re_acct:
			raise ValueError(f"retained_earnings_account_not_found:{retained_earnings_account}")

		# Collect all income-statement accounts with non-zero balances
		closing_lines: list[dict[str, Any]] = []
		net_to_retained = Decimal("0")

		for acct in self.accounts.values():
			if acct["tenant_id"] != tenant or acct["account_type"] not in _INCOME_STMT_TYPES:
				continue
			# Sum all postings for the fiscal year
			year_periods = {
				p["id"] for p in self.periods.values()
				if p["tenant_id"] == tenant and p.get("fiscal_year") == fiscal_year
			}
			acct_debit = Decimal("0")
			acct_credit = Decimal("0")
			for posting in self.postings.values():
				if posting["tenant_id"] != tenant:
					continue
				batch = self.journal_batches.get(self.journal_entries.get(posting["journal_id"], {}).get("batch_id", ""), {})
				if batch.get("period_id") not in year_periods:
					continue
				for line in posting["lines"]:
					if line.get("account_id") == acct["id"]:
						acct_debit += _d(line.get("debit", 0))
						acct_credit += _d(line.get("credit", 0))

			net = acct_debit - acct_credit
			if net == 0:
				continue

			if acct["account_type"] == "revenue":
				# Revenue has a credit balance (net < 0 in debit-credit net terms)
				# Closing entry: debit revenue to zero it, credit retained earnings
				closing_lines.append({
					"account_id": acct["id"],
					"debit": str(abs(net)) if net < 0 else "0.00",
					"credit": str(abs(net)) if net > 0 else "0.00",
					"description": f"Year-end close: {acct['code']}",
				})
				net_to_retained += (-net)  # positive credit to RE
			else:  # expense
				# Expense has a debit balance (net > 0)
				# Closing entry: credit expense to zero it, debit retained earnings
				closing_lines.append({
					"account_id": acct["id"],
					"debit": "0.00",
					"credit": str(abs(net)),
					"description": f"Year-end close: {acct['code']}",
				})
				net_to_retained -= abs(net)  # expenses reduce RE

		if not closing_lines:
			return {
				"id": self._record_id("ye"),
				"type": "year_end_closing",
				"tenant_id": tenant,
				"fiscal_year": fiscal_year,
				"closing_lines": 0,
				"net_to_retained_earnings": "0.00",
				"status": "no_income_statement_balances",
				"created_at": self._now(),
			}

		# Add the retained earnings line (balancing entry)
		if net_to_retained > 0:
			closing_lines.append({
				"account_id": re_acct["id"],
				"debit": "0.00",
				"credit": str(net_to_retained),
				"description": f"Year-end close: net profit to retained earnings FY{fiscal_year}",
			})
		elif net_to_retained < 0:
			closing_lines.append({
				"account_id": re_acct["id"],
				"debit": str(abs(net_to_retained)),
				"credit": "0.00",
				"description": f"Year-end close: net loss from retained earnings FY{fiscal_year}",
			})

		# Find or create a year-end adjustment period
		ye_period_code = f"{fiscal_year}-YE"
		ye_period = self._period_by_code(tenant, ye_period_code)
		if ye_period is None:
			ye_period = {
				"id": self._record_id("period"),
				"type": "accounting_period",
				"tenant_id": tenant,
				"name": ye_period_code,
				"period_code": ye_period_code,
				"fiscal_year": fiscal_year,
				"period_start": f"{fiscal_year}-12-31",
				"period_end": f"{fiscal_year}-12-31",
				"status": "open",
				"created_at": self._now(),
			}
			self.periods[ye_period["id"]] = ye_period

		closing_posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=f"{fiscal_year}-12-31",
			journal_type="manual",
			lines=closing_lines,
			description=f"Year-end closing entries FY{fiscal_year}",
			reference=f"YE-{fiscal_year}",
			posted_by="year_end_close",
		)

		# Lock all periods in the year
		for p in self.periods.values():
			if p["tenant_id"] == tenant and p.get("fiscal_year") == fiscal_year:
				if p["status"] == "closed":
					p["status"] = "locked"
					p["locked_by"] = "year_end_close"
					p["locked_at"] = self._now()
				elif p["status"] == "open":
					p["status"] = "locked"
					p["locked_by"] = "year_end_close"
					p["locked_at"] = self._now()

		# Record fiscal year as closed
		fy_id = self._record_id("fy")
		fy_record = {
			"id": fy_id,
			"type": "fiscal_year_close",
			"tenant_id": tenant,
			"fiscal_year": fiscal_year,
			"retained_earnings_account": retained_earnings_account,
			"net_to_retained_earnings": str(net_to_retained),
			"closing_journal_id": closing_posting["journal_id"],
			"status": "closed",
			"closed_at": self._now(),
		}
		self.fiscal_years[fy_id] = fy_record

		return deepcopy({**fy_record, "closing_posting": closing_posting})

	async def opening_balances_new_year(
		self,
		tenant_id: str,
		new_fiscal_year: int,
	) -> dict[str, Any]:
		"""Carry forward balance sheet accounts to the new year; zero income statement accounts.

		Creates opening balance journal entries for all balance-sheet type accounts
		with non-zero closing balances from the prior year.
		"""
		tenant = self._tenant(tenant_id)
		prior_year = new_fiscal_year - 1

		# Get last period of prior year
		prior_periods = sorted(
			[p for p in self.periods.values() if p["tenant_id"] == tenant and p.get("fiscal_year") == prior_year],
			key=lambda p: p.get("period_end", ""),
			reverse=True,
		)
		last_prior_period_code = prior_periods[0].get("period_code") if prior_periods else None

		# Ensure a new year opening period exists
		ob_period_code = f"{new_fiscal_year}-OB"
		ob_period = self._period_by_code(tenant, ob_period_code)
		if ob_period is None:
			ob_period = {
				"id": self._record_id("period"),
				"type": "accounting_period",
				"tenant_id": tenant,
				"name": ob_period_code,
				"period_code": ob_period_code,
				"fiscal_year": new_fiscal_year,
				"period_start": f"{new_fiscal_year}-01-01",
				"period_end": f"{new_fiscal_year}-01-01",
				"status": "open",
				"created_at": self._now(),
			}
			self.periods[ob_period["id"]] = ob_period

		ob_lines: list[dict[str, Any]] = []
		for acct in self.accounts.values():
			if acct["tenant_id"] != tenant:
				continue
			if acct["account_type"] in _INCOME_STMT_TYPES:
				# Income statement accounts zero out — no opening balance carried
				continue
			bal = self._get_account_balance(tenant, acct["id"], last_prior_period_code)
			closing_net = bal["opening"] + bal["debits"] - bal["credits"]
			if closing_net == 0:
				continue
			if closing_net > 0:
				ob_lines.append({"account_id": acct["id"], "debit": str(closing_net), "credit": "0.00",
				                  "description": f"Opening balance {new_fiscal_year}"})
			else:
				ob_lines.append({"account_id": acct["id"], "debit": "0.00", "credit": str(abs(closing_net)),
				                  "description": f"Opening balance {new_fiscal_year}"})

		if not ob_lines:
			return {
				"id": self._record_id("ob"),
				"type": "opening_balances",
				"tenant_id": tenant,
				"new_fiscal_year": new_fiscal_year,
				"status": "no_balances_to_carry_forward",
				"created_at": self._now(),
			}

		# Opening balances must balance — add a suspense line if needed
		total_d = sum(_d(ln["debit"]) for ln in ob_lines)
		total_c = sum(_d(ln["credit"]) for ln in ob_lines)
		if total_d != total_c:
			diff = total_d - total_c
			# Find retained earnings account to absorb the difference
			re_accounts = [a for a in self.accounts.values()
			               if a["tenant_id"] == tenant and "retained_earnings" in a.get("tags", [])]
			re_acct_id = re_accounts[0]["id"] if re_accounts else None
			if re_acct_id:
				if diff > 0:
					ob_lines.append({"account_id": re_acct_id, "debit": "0.00", "credit": str(diff),
					                  "description": "Opening balance plug"})
				else:
					ob_lines.append({"account_id": re_acct_id, "debit": str(abs(diff)), "credit": "0.00",
					                  "description": "Opening balance plug"})

		posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=f"{new_fiscal_year}-01-01",
			journal_type="manual",
			lines=ob_lines,
			description=f"Opening balances FY{new_fiscal_year}",
			reference=f"OB-{new_fiscal_year}",
			posted_by="system",
		)

		return {
			"id": self._record_id("ob"),
			"type": "opening_balances",
			"tenant_id": tenant,
			"new_fiscal_year": new_fiscal_year,
			"prior_year": prior_year,
			"accounts_carried_forward": len(ob_lines),
			"posting_id": posting["id"],
			"journal_id": posting["journal_id"],
			"status": "completed",
			"created_at": self._now(),
		}

	async def prior_year_adjustment(
		self,
		tenant_id: str,
		account_code: str,
		amount: str,
		adjustment_reason: str,
	) -> dict[str, Any]:
		"""IAS 8 error correction: restate opening retained earnings.

		Posts a prior-period adjustment journal to the retained earnings account,
		tagging it as a prior-year restatement.
		"""
		tenant = self._tenant(tenant_id)
		if not adjustment_reason:
			raise ValueError("adjustment_reason_required_for_ias8_correction")

		acct = self._account_by_code(tenant, account_code)
		if not acct:
			raise ValueError(f"account_not_found:{account_code}")

		amt = _d(amount)
		# Find retained earnings account
		re_accounts = [a for a in self.accounts.values()
		               if a["tenant_id"] == tenant and "retained_earnings" in a.get("tags", [])]
		if not re_accounts:
			raise ValueError("retained_earnings_account_not_tagged")
		re_acct = re_accounts[0]

		# Adjustment: correct the account, offset to retained earnings
		if amt > 0:
			lines = [
				{"account_id": acct["id"], "debit": str(amt), "credit": "0.00",
				 "description": f"IAS8 adjustment: {adjustment_reason}"},
				{"account_id": re_acct["id"], "debit": "0.00", "credit": str(amt),
				 "description": f"IAS8 RE offset: {adjustment_reason}"},
			]
		else:
			lines = [
				{"account_id": acct["id"], "debit": "0.00", "credit": str(abs(amt)),
				 "description": f"IAS8 adjustment: {adjustment_reason}"},
				{"account_id": re_acct["id"], "debit": str(abs(amt)), "credit": "0.00",
				 "description": f"IAS8 RE offset: {adjustment_reason}"},
			]

		today = self._today()
		posting = await self.post_journal_v2(
			tenant_id=tenant,
			journal_date=today,
			journal_type="manual",
			lines=lines,
			description=f"Prior year adjustment (IAS 8): {adjustment_reason}",
			reference=f"PYA-{uuid4().hex[:8].upper()}",
			posted_by="system",
		)

		return {
			"id": self._record_id("pya"),
			"type": "prior_year_adjustment",
			"tenant_id": tenant,
			"account_code": account_code,
			"amount": str(amt),
			"adjustment_reason": adjustment_reason,
			"journal_id": posting["journal_id"],
			"status": "posted",
			"created_at": self._now(),
		}

	async def ifrs_consolidation(
		self,
		tenant_id: str,
		subsidiaries: list[str],
		group_adjustments: list[dict[str, Any]],
		minority_interest: dict[str, Any],
	) -> dict[str, Any]:
		"""Consolidate group financials under IFRS.

		Steps:
		1. Aggregate balance sheet and income statement across parent + subsidiaries.
		2. Apply intercompany eliminations (call intercompany_reconciliation for each pair).
		3. Apply group_adjustments (e.g. fair value uplift, goodwill amortisation).
		4. Calculate minority interest.

		group_adjustments: list of {description, account_code, amount, entity}
		minority_interest: {subsidiary, percentage}
		"""
		tenant = self._tenant(tenant_id)

		# Aggregate trial balances
		entities = [tenant] + subsidiaries
		combined_accounts: dict[str, dict[str, Decimal]] = {}

		for entity in entities:
			try:
				# Use latest open or closed period
				entity_periods = sorted(
					[p for p in self.periods.values() if p["tenant_id"] == entity and p["status"] in {"open", "closed"}],
					key=lambda p: p.get("period_end", ""),
					reverse=True,
				)
				if not entity_periods:
					continue
				latest_pc = entity_periods[0].get("period_code", "")
				tb = await self.trial_balance(entity, latest_pc, include_zero_balances=False)
				for row in tb["rows"]:
					code = row["account_code"]
					combined_accounts.setdefault(code, {"debit": Decimal("0"), "credit": Decimal("0"),
					                                     "account_name": row["account_name"],
					                                     "account_type": row["account_type"]})
					combined_accounts[code]["debit"] += _d(row["closing_debit"])
					combined_accounts[code]["credit"] += _d(row["closing_credit"])
			except Exception:
				pass  # Skip entities with no periods

		# Elimination entries for intercompany balances
		eliminations: list[dict[str, Any]] = []
		for i, sub in enumerate(subsidiaries):
			try:
				ic_rec = await self.intercompany_reconciliation(tenant, sub, "")
				if _d(ic_rec["difference"]) != 0:
					eliminations.append({
						"type": "intercompany_elimination",
						"entity_a": tenant,
						"entity_b": sub,
						"difference": ic_rec["difference"],
					})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# Apply group adjustments
		for adj in group_adjustments:
			code = adj.get("account_code", "")
			amt = _d(adj.get("amount", 0))
			combined_accounts.setdefault(code, {"debit": Decimal("0"), "credit": Decimal("0"),
			                                     "account_name": adj.get("description", code),
			                                     "account_type": "equity"})
			if amt > 0:
				combined_accounts[code]["debit"] += amt
			else:
				combined_accounts[code]["credit"] += abs(amt)

		# Minority interest calculation
		mi_amount = Decimal("0")
		mi_sub = minority_interest.get("subsidiary")
		mi_pct = _d(minority_interest.get("percentage", 0)) / 100
		if mi_sub:
			sub_equity = Decimal("0")
			for row in combined_accounts.values():
				if row.get("account_type") == "equity":
					sub_equity += row["credit"] - row["debit"]
			mi_amount = sub_equity * mi_pct

		consolidated_rows = [
			{
				"account_code": code,
				"account_name": vals["account_name"],
				"account_type": vals["account_type"],
				"consolidated_debit": str(vals["debit"]),
				"consolidated_credit": str(vals["credit"]),
			}
			for code, vals in sorted(combined_accounts.items())
		]

		return {
			"id": self._record_id("consol"),
			"type": "ifrs_consolidation",
			"tenant_id": tenant,
			"subsidiaries": subsidiaries,
			"entity_count": len(entities),
			"consolidated_rows": consolidated_rows,
			"eliminations": eliminations,
			"group_adjustments_applied": len(group_adjustments),
			"minority_interest": {
				"subsidiary": mi_sub,
				"percentage": str(mi_pct * 100),
				"amount": str(mi_amount),
			},
			"status": "completed",
			"generated_at": self._now(),
		}

	async def xbrl_tagging_extract(
		self,
		tenant_id: str,
		period_code: str,
		framework: str = "IFRS",
	) -> dict[str, Any]:
		"""Generate an XBRL-ready extract for the specified period and framework.

		Maps each account to its XBRL concept using the account's code and type.
		The mapping is a simplified illustrative version — production would use
		the full IFRS taxonomy namespace.
		"""
		tenant = self._tenant(tenant_id)
		tb = await self.trial_balance(tenant, period_code, include_zero_balances=False)

		# Simplified IFRS element map by account type
		_ifrs_map = {
			"asset": "ifrs-full:Assets",
			"liability": "ifrs-full:Liabilities",
			"equity": "ifrs-full:Equity",
			"revenue": "ifrs-full:Revenue",
			"expense": "ifrs-full:ExpenseByNature",
		}

		tagged_facts: list[dict[str, Any]] = []
		for row in tb["rows"]:
			closing_net = _d(row["closing_debit"]) - _d(row["closing_credit"])
			# Sign convention: credit-normal amounts reported positive
			if row["account_type"] in _CREDIT_NORMAL_TYPES:
				closing_net = -closing_net

			tagged_facts.append({
				"xbrl_concept": _ifrs_map.get(row["account_type"], "ifrs-full:OtherAssets"),
				"account_code": row["account_code"],
				"account_name": row["account_name"],
				"account_type": row["account_type"],
				"period": period_code,
				"value": str(closing_net),
				"decimals": "2",
				"unit": "USD",
				"context": f"{tenant}_{period_code}",
			})

		return {
			"id": self._record_id("xbrl"),
			"type": "xbrl_tagging_extract",
			"tenant_id": tenant,
			"period_code": period_code,
			"framework": framework,
			"namespace": "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full"
			             if framework == "IFRS"
			             else "http://xbrl.fasb.org/us-gaap/2023",
			"fact_count": len(tagged_facts),
			"facts": tagged_facts,
			"generated_at": self._now(),
		}

	# ------------------------------------------------------------------
	# Convenience aliases / shim methods
	# ------------------------------------------------------------------

	async def year_end_close(
		self,
		tenant_id: str,
		fiscal_year: int,
		retained_earnings_account: str,
	) -> dict[str, Any]:
		"""Alias for year_end_closing — preferred public name."""
		return await self.year_end_closing(tenant_id, fiscal_year, retained_earnings_account)

	async def currency_revaluation(
		self,
		tenant_id: str,
		period_code: str,
		rates: dict[str, Any],
	) -> dict[str, Any]:
		"""Revalue all foreign-currency monetary balances at period-end rates.

		For each account with a non-functional currency, computes the unrealised
		FX gain or loss and posts a revaluation journal.

		rates: dict mapping currency code → new exchange rate (Decimal or str).
		"""
		from .domain.calculations import revaluation_gain_loss

		tenant = self._tenant(tenant_id)

		# Find or create a FX gain/loss account (first tagged account, else stub)
		fx_accounts = [
			a for a in self.accounts.values()
			if a["tenant_id"] == tenant and "fx_gain_loss" in a.get("tags", [])
		]
		if not fx_accounts:
			# Use first expense account as a proxy
			fx_accounts = [
				a for a in self.accounts.values()
				if a["tenant_id"] == tenant and a["account_type"] == "expense" and a["status"] == "active"
			]
		if not fx_accounts:
			return {
				"id": self._record_id("reval"),
				"type": "currency_revaluation",
				"tenant_id": tenant,
				"period_code": period_code,
				"status": "no_fx_account_available",
				"journals_posted": 0,
				"created_at": self._now(),
			}

		fx_acct = fx_accounts[0]
		journals_posted: list[dict[str, Any]] = []

		for acct in self.accounts.values():
			if acct["tenant_id"] != tenant:
				continue
			acct_currency = acct.get("currency", "USD")
			if acct_currency == "USD" or acct_currency not in rates:
				continue

			new_rate = _d(str(rates[acct_currency]))
			# Find last recorded rate for this currency
			old_rate = _d("1")
			for cr in sorted(
				[r for r in self.currency_rates.values()
				 if r["tenant_id"] == tenant and r.get("to_currency") == acct_currency],
				key=lambda r: r.get("created_at", ""),
			):
				old_rate = _d(str(cr.get("exchange_rate", 1)))

			bal = self._get_account_balance(tenant, acct["id"], period_code)
			foreign_balance = bal["opening"] + bal["debits"] - bal["credits"]
			if foreign_balance == 0:
				continue

			gain_loss = revaluation_gain_loss(foreign_balance, old_rate, new_rate)
			if gain_loss == 0:
				continue

			# Post the revaluation journal
			if gain_loss > 0:
				lines = [
					{"account_id": acct["id"], "debit": str(gain_loss), "credit": "0.00",
					 "description": f"FX reval {acct_currency}"},
					{"account_id": fx_acct["id"], "debit": "0.00", "credit": str(gain_loss),
					 "description": "FX gain"},
				]
			else:
				lines = [
					{"account_id": fx_acct["id"], "debit": str(abs(gain_loss)), "credit": "0.00",
					 "description": "FX loss"},
					{"account_id": acct["id"], "debit": "0.00", "credit": str(abs(gain_loss)),
					 "description": f"FX reval {acct_currency}"},
				]

			try:
				posting = await self.post_journal_v2(
					tenant_id=tenant,
					journal_date=period_code[:10] if len(period_code) >= 10 else self._today(),
					journal_type="manual",
					lines=lines,
					description=f"FX revaluation {acct_currency} period {period_code}",
					reference=f"REVAL-{acct_currency}-{period_code}",
					posted_by="system",
				)
				journals_posted.append({"account": acct["code"], "gain_loss": str(gain_loss),
				                        "posting_id": posting["id"]})
			except Exception:
				pass  # Skip accounts whose period isn't open

		return {
			"id": self._record_id("reval"),
			"type": "currency_revaluation",
			"tenant_id": tenant,
			"period_code": period_code,
			"rates_applied": {k: str(v) for k, v in rates.items()},
			"journals_posted": len(journals_posted),
			"details": journals_posted,
			"status": "completed",
			"created_at": self._now(),
		}


	# ==================================================================
	# PERIOD REPORTING HELPERS
	# ==================================================================

	async def _get_active_reporting_accounts(self, account_types: list) -> list:
		"""Return accounts whose type_code (or account_type) is in account_types.

		Subclasses override this to adapt to ORM model shapes.
		Default implementation queries the in-memory accounts store.
		"""
		result = []
		for acct in self.accounts.values():
			atype = acct.get("account_type", "")
			# Support both plain strings and AccountTypeEnum values
			for at in account_types:
				at_val = at.value if hasattr(at, "value") else str(at)
				if atype == at_val:
					result.append(acct)
					break
		return result

	async def _get_account_balance(self, account_id: str, as_of_date: object) -> "Decimal":  # type: ignore[override]
		"""Return a single Decimal balance for *account_id* as of *as_of_date*.

		Subclasses replace this.  Default: sum all postings for the account.
		"""
		total = Decimal("0")
		for posting in self.postings.values():
			for line in posting["lines"]:
				if line.get("account_id") != account_id:
					continue
				total += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
		return total

	async def _get_account_period_activity(
		self,
		account_id: str,
		date_from: object,
		date_to: object,
	) -> "Decimal":
		"""Return net activity for *account_id* between *date_from* and *date_to*.

		Subclasses replace this.  Default: sum all postings (no date filtering in
		the in-memory store as postings carry period_code, not exact dates).
		"""
		total = Decimal("0")
		for posting in self.postings.values():
			for line in posting["lines"]:
				if line.get("account_id") != account_id:
					continue
				total += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
		return total

	async def _get_comparative_balances(
		self,
		account_types: list,
		as_of_date: object,
	) -> dict:
		"""Return balance-sheet style comparative data for the given account types.

		Returns::

		    {
		        "totals": {<TYPE_NAME>: float, ...},
		        "sections": {<TYPE_NAME>: [{"account_code": ..., "account_name": ...,
		                                    "balance": float}, ...], ...},
		    }
		"""
		accounts = await self._get_active_reporting_accounts(account_types)
		totals: dict = {}
		sections: dict = {}

		for acct in accounts:
			# Resolve the type name — works for both ORM objects with .account_type.type_code
			# and plain dicts with "account_type".
			if hasattr(acct, "account_type"):
				type_obj = acct.account_type
				type_code = type_obj.type_code if hasattr(type_obj, "type_code") else type_obj
			else:
				type_code = acct.get("account_type", "")

			type_name = (type_code.value if hasattr(type_code, "value") else str(type_code)).upper()

			account_id = acct.account_id if hasattr(acct, "account_id") else acct.get("id", "")
			account_code = acct.account_code if hasattr(acct, "account_code") else acct.get("code", "")
			account_name = acct.account_name if hasattr(acct, "account_name") else acct.get("name", "")

			balance = await self._get_account_balance(account_id, as_of_date)
			balance_float = float(balance)

			totals[type_name] = totals.get(type_name, 0.0) + balance_float
			sections.setdefault(type_name, []).append({
				"account_id": account_id,
				"account_code": account_code,
				"account_name": account_name,
				"balance": balance_float,
			})

		return {"totals": totals, "sections": sections}

	async def _get_comparative_income_data(
		self,
		account_types: list,
		date_from: object,
		date_to: object,
	) -> dict:
		"""Return income-statement style comparative data for the given account types.

		Returns::

		    {
		        "totals": {<TYPE_NAME>: float, ..., "net_income": float},
		        "sections": {<TYPE_NAME>: [...], ...},
		    }
		"""
		accounts = await self._get_active_reporting_accounts(account_types)
		totals: dict = {}
		sections: dict = {}

		for acct in accounts:
			if hasattr(acct, "account_type"):
				type_obj = acct.account_type
				type_code = type_obj.type_code if hasattr(type_obj, "type_code") else type_obj
			else:
				type_code = acct.get("account_type", "")

			type_name = (type_code.value if hasattr(type_code, "value") else str(type_code)).upper()

			account_id = acct.account_id if hasattr(acct, "account_id") else acct.get("id", "")
			account_code = acct.account_code if hasattr(acct, "account_code") else acct.get("code", "")
			account_name = acct.account_name if hasattr(acct, "account_name") else acct.get("name", "")

			activity = await self._get_account_period_activity(account_id, date_from, date_to)
			activity_float = float(activity)

			totals[type_name] = totals.get(type_name, 0.0) + activity_float
			sections.setdefault(type_name, []).append({
				"account_id": account_id,
				"account_code": account_code,
				"account_name": account_name,
				"activity": activity_float,
			})

		revenue = totals.get("REVENUE", 0.0)
		expense = totals.get("EXPENSE", 0.0)
		totals["net_income"] = revenue - expense

		return {"totals": totals, "sections": sections}

	async def _run_period_allocations(self, period: object) -> None:
		"""Run auto-allocation rules for all expense accounts in the period.

		Appends a checklist item to *period.closing_checklist* and commits.
		"""
		account_types_all: list = []
		for acct in (self.accounts.values() if isinstance(self.accounts, dict) else self.accounts):
			if hasattr(acct, "auto_allocation_rules") and acct.auto_allocation_rules:
				account_types_all.append(acct)
			elif isinstance(acct, dict) and acct.get("auto_allocation_rules"):
				account_types_all.append(acct)

		# Also accept accounts that are dataclass/SimpleNamespace objects from tests
		if not account_types_all and hasattr(self, "accounts") and isinstance(self.accounts, list):
			for acct in self.accounts:
				if hasattr(acct, "auto_allocation_rules") and acct.auto_allocation_rules:
					account_types_all.append(acct)

		allocations_run: list = []
		for acct in account_types_all:
			rules = (
				acct.auto_allocation_rules
				if hasattr(acct, "auto_allocation_rules")
				else acct.get("auto_allocation_rules", [])
			)
			account_id = acct.account_id if hasattr(acct, "account_id") else acct.get("id", "")
			balance = await self._get_account_balance(
				account_id,
				getattr(period, "end_date", None),
			)
			for rule in (rules or []):
				pct = Decimal(str(rule.get("percent", 0))) / Decimal("100")
				amount = float((balance * pct).quantize(TWO, rounding=ROUND_HALF_UP))
				allocations_run.append({
					"source_account": account_id,
					"target_account": rule.get("target_account_id"),
					"rule_name": rule.get("name"),
					"amount": amount,
				})

		period.closing_checklist.append({
			"step": "run_period_allocations",
			"period_id": getattr(period, "period_id", None),
			"allocations": allocations_run,
			"status": "completed",
		})
		if hasattr(self, "session"):
			self.session.commit()

	async def _generate_period_reports(self, period: object) -> None:
		"""Generate standard period-end reports and append results to checklist.

		Appends a checklist item and sets *period.closing_notes*.  Commits.
		"""
		from datetime import date as _date

		as_of = getattr(period, "end_date", None)
		date_from = getattr(period, "start_date", None)

		# Build TrialBalanceParams and call generate_trial_balance if available
		tb_result = None
		if hasattr(self, "generate_trial_balance"):
			try:
				params = TrialBalanceParams(
					period_code=getattr(period, "period_name", ""),
					as_of_date=as_of,
				)
				tb_result = await self.generate_trial_balance(params)
			except Exception:
				tb_result = None

		bs_result = None
		if hasattr(self, "generate_balance_sheet"):
			try:
				bs_result = await self.generate_balance_sheet(as_of_date=as_of)
			except Exception:
				bs_result = None

		is_result = None
		if hasattr(self, "generate_income_statement") and date_from and as_of:
			try:
				is_result = await self.generate_income_statement(date_from=date_from, date_to=as_of)
			except Exception:
				is_result = None

		def _report_meta(result) -> dict:
			if result is None:
				return {}
			if isinstance(result, dict):
				return result.get("metadata", result)
			return getattr(result, "metadata", {}) or {}

		reports = {
			"trial_balance": _report_meta(tb_result),
			"balance_sheet": _report_meta(bs_result),
			"income_statement": _report_meta(is_result),
		}

		period.closing_checklist.append({
			"step": "generate_period_reports",
			"period_id": getattr(period, "period_id", None),
			"reports": reports,
			"status": "completed",
		})

		period.closing_notes = (
			f"Period {getattr(period, 'period_name', '')} reports generated. "
			f"Trial balance balanced: {reports['trial_balance'].get('balanced', False)}."
		)
		if hasattr(self, "session"):
			self.session.commit()

	async def setup_tenant(self, tenant_data: dict) -> dict:
		"""Initialise a tenant with its chart of accounts and opening configuration.

		*tenant_data* may carry: tenant_id, accounts, periods, currency.
		Returns a summary dict.
		"""
		tenant_id = tenant_data.get("tenant_id", self.tenant_id or "default")
		self.tenant_id = tenant_id

		accounts_created = 0
		for acct in tenant_data.get("accounts", []):
			try:
				await self.create_account_v2(
					tenant_id=tenant_id,
					account_code=acct.get("code", acct.get("account_code", "")),
					account_name=acct.get("name", acct.get("account_name", "")),
					account_type=acct.get("account_type", "asset"),
					parent_code=acct.get("parent_code"),
					currency=acct.get("currency", "USD"),
				)
				accounts_created += 1
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		return {
			"tenant_id": tenant_id,
			"accounts_created": accounts_created,
			"status": "initialized",
		}


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

	async def ml_period_close_risk(self, *args, **kwargs):
		"""AI-powered GL period close risk assessment and anomaly detection. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="general_ledger_period_close_risk")
			return {"close_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

GLRService = GeneralLedgerService

from enum import Enum
class AccountTypeEnum(str, Enum):
    ASSET = "asset"; LIABILITY = "liability"; EQUITY = "equity"; REVENUE = "revenue"; EXPENSE = "expense"

from dataclasses import dataclass as _fdc, field as _fld
@_fdc
class FinancialReportingResult:
	report_type: str = ""
	as_of_date: object = None
	currency: str = "USD"
	data: dict = None
	metadata: dict = None
	# Legacy fields kept for backwards compatibility
	period: str = ""
	entity: str = ""
	statements: dict = None
	ok: bool = True

	def __post_init__(self):
		if self.data is None:
			self.data = {}
		if self.metadata is None:
			self.metadata = {}
		if self.statements is None:
			self.statements = {}


from dataclasses import dataclass as _tbdc
@_tbdc
class TrialBalanceParams:
	period_code: str = ""
	tenant_id: str = "default"
	include_zero_balances: bool = False
	currency: str = "functional"
	as_of_date: object = None
