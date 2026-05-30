"""Dependency-light General Ledger lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

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
except ImportError:  # pragma: no cover - supports direct file loading in tests
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


class GeneralLedgerService:
	"""In-memory executable service for the GLR lifecycle packet."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
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
		self._audit_events: list[dict[str, Any]] = []
		self._idempotency_keys: set[str] = set()

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

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
		total_debits = sum(Decimal(str(line.get("debit", 0))) for line in lines)
		total_credits = sum(Decimal(str(line.get("credit", 0))) for line in lines)
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
				account_totals[account]["debit"] += Decimal(str(line.get("debit", 0)))
				account_totals[account]["credit"] += Decimal(str(line.get("credit", 0)))
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
			"account_count": len([record for record in self.accounts.values() if record["tenant_id"] == tenant]),
			"dimension_count": len([record for record in self.dimensions.values() if record["tenant_id"] == tenant]),
			"period_count": len([record for record in self.periods.values() if record["tenant_id"] == tenant]),
			"journal_batch_count": len([record for record in self.journal_batches.values() if record["tenant_id"] == tenant]),
			"journal_entry_count": len([record for record in self.journal_entries.values() if record["tenant_id"] == tenant]),
			"posted_journal_count": len([record for record in self.journal_entries.values() if record["tenant_id"] == tenant and record["status"] == "posted"]),
			"allocation_count": len([record for record in self.allocations.values() if record["tenant_id"] == tenant]),
			"glr_agent_count": len([record for record in self.agents.values() if record["tenant_id"] == tenant]),
			"audit_event_count": len([event for event in self._audit_events if event["tenant_id"] == tenant]),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		store = getattr(self, collection)
		return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]


GLRService = GeneralLedgerService
