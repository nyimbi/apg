"""Legal Billing & Time Tracking — async service layer."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
from copy import deepcopy
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

ACTIVITY_CODES = {
	"L110", "L120", "L130", "L140", "L160", "L190",  # Case assessment
	"L210", "L220", "L230", "L240", "L250",            # Pleadings
	"L310", "L320", "L330", "L340", "L350",            # Discovery
	"L410", "L420", "L430", "L440", "L450",            # Trial prep
	"A101", "A102", "A103", "A104",                     # Project management
}
DISBURSEMENT_TYPES = {"court_fee", "expert_fee", "travel", "postage", "copy", "translation", "filing_fee", "other"}
INVOICE_STATUSES = {"draft", "submitted", "approved", "sent", "paid", "overdue", "written_off", "disputed"}
TRUST_TRANSACTION_TYPES = {"deposit", "withdrawal", "transfer", "fee_application", "interest", "refund"}


class LegalBillingService:
	"""In-memory async service for legal billing and time tracking."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.time_entries: dict[str, dict[str, Any]] = {}
		self.disbursements: dict[str, dict[str, Any]] = {}
		self.invoices: dict[str, dict[str, Any]] = {}
		self.trust_accounts: dict[str, dict[str, Any]] = {}
		self.trust_transactions: dict[str, dict[str, Any]] = {}
		self.rate_cards: dict[str, dict[str, Any]] = {}
		self.write_offs: dict[str, dict[str, Any]] = {}
		self._invoice_sequence: int = 1000
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}{uuid4().hex[:12]}"

	def _tenant(self, tenant_id: str | None = None) -> str:
		val = tenant_id or self.tenant_id
		guard_tenant_id(val)
		return val

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		})

	def _next_invoice_number(self, tenant: str) -> str:
		self._invoice_sequence += 1
		year = datetime.utcnow().year
		return f"INV-{year}-{self._invoice_sequence:05d}"

	# ── Health & Describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "leg_bil",
			"status": "healthy",
			"time_entries": len(self.time_entries),
			"open_invoices": sum(1 for i in self.invoices.values() if i["status"] in {"sent", "overdue"}),
			"trust_accounts": len(self.trust_accounts),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_bil",
			"name": "Legal Billing & Time Tracking",
			"domain": "legal",
			"version": "1.0.0",
			"disbursement_types": sorted(DISBURSEMENT_TYPES),
			"invoice_statuses": sorted(INVOICE_STATUSES),
			"trust_transaction_types": sorted(TRUST_TRANSACTION_TYPES),
		}

	# ── Time Entries ─────────────────────────────────────────────────────────

	async def create_time_entry(
		self,
		tenant_id: str,
		matter_id: str,
		attorney_id: str,
		date: str,
		hours: float,
		rate: float,
		activity_code: str,
		description: str,
		billable: bool = True,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Capture a billable time entry."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(matter_id, "matter_id")
		guard_non_empty_string(description, "description")
		if hours <= 0:
			raise ValueError("hours must be positive")
		if rate < 0:
			raise ValueError("rate cannot be negative")
		record: dict[str, Any] = {
			"id": self._id("te-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"attorney_id": attorney_id,
			"date": date,
			"hours": hours,
			"rate": rate,
			"amount": round(hours * rate, 2),
			"activity_code": activity_code,
			"description": description,
			"billable": billable,
			"currency": currency,
			"status": "draft",
			"invoice_id": None,
			"created_at": self._now(),
			"updated_at": None,
		}
		self.time_entries[record["id"]] = record
		self._emit(tenant, "time_entry_created", record["id"], {"matter_id": matter_id, "hours": hours})
		_log.info("time entry created tenant=%s id=%s hours=%.2f", tenant, record["id"], hours)
		return deepcopy(record)

	async def get_time_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		te = self.time_entries.get(entry_id)
		if not te or te["tenant_id"] != tenant:
			raise KeyError(f"time entry {entry_id} not found")
		return deepcopy(te)

	async def list_time_entries(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		attorney_id: str | None = None,
		status: str | None = None,
		billable: bool | None = None,
		date_from: str | None = None,
		date_to: str | None = None,
	) -> list[dict[str, Any]]:
		"""List time entries with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(te) for te in self.time_entries.values() if te["tenant_id"] == tenant]
		if matter_id:
			items = [te for te in items if te["matter_id"] == matter_id]
		if attorney_id:
			items = [te for te in items if te["attorney_id"] == attorney_id]
		if status:
			items = [te for te in items if te["status"] == status]
		if billable is not None:
			items = [te for te in items if te["billable"] == billable]
		if date_from:
			items = [te for te in items if te["date"] >= date_from]
		if date_to:
			items = [te for te in items if te["date"] <= date_to]
		return sorted(items, key=lambda te: te["date"])

	async def update_time_entry(self, tenant_id: str, entry_id: str, **updates: Any) -> dict[str, Any]:
		"""Update a draft time entry."""
		tenant = self._tenant(tenant_id)
		te = self.time_entries.get(entry_id)
		if not te or te["tenant_id"] != tenant:
			raise KeyError(f"time entry {entry_id} not found")
		if te["status"] not in {"draft"}:
			raise ValueError("only draft entries can be updated")
		allowed = {"hours", "rate", "description", "billable", "activity_code"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				te[k] = v
		te["amount"] = round(te["hours"] * te["rate"], 2)
		te["updated_at"] = self._now()
		self._emit(tenant, "time_entry_updated", entry_id, updates)
		return deepcopy(te)

	async def submit_time_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		"""Submit a time entry for approval."""
		tenant = self._tenant(tenant_id)
		te = self.time_entries.get(entry_id)
		if not te or te["tenant_id"] != tenant:
			raise KeyError(f"time entry {entry_id} not found")
		te["status"] = "submitted"
		te["submitted_at"] = self._now()
		self._emit(tenant, "time_entry_submitted", entry_id)
		return deepcopy(te)

	async def approve_time_entry(self, tenant_id: str, entry_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a submitted time entry."""
		tenant = self._tenant(tenant_id)
		te = self.time_entries.get(entry_id)
		if not te or te["tenant_id"] != tenant:
			raise KeyError(f"time entry {entry_id} not found")
		te["status"] = "approved"
		te["approved_by"] = approved_by
		te["approved_at"] = self._now()
		self._emit(tenant, "time_entry_approved", entry_id, {"approved_by": approved_by})
		return deepcopy(te)

	async def delete_time_entry(self, tenant_id: str, entry_id: str) -> dict[str, Any]:
		"""Write off a time entry."""
		tenant = self._tenant(tenant_id)
		te = self.time_entries.get(entry_id)
		if not te or te["tenant_id"] != tenant:
			raise KeyError(f"time entry {entry_id} not found")
		te["status"] = "written_off"
		self._emit(tenant, "time_entry_written_off", entry_id)
		return deepcopy(te)

	# ── Disbursements ────────────────────────────────────────────────────────

	async def create_disbursement(
		self,
		tenant_id: str,
		matter_id: str,
		recorded_by_id: str,
		date: str,
		amount: float,
		disbursement_type: str,
		description: str,
		currency: str = "KES",
		receipt_reference: str = "",
		billable: bool = True,
	) -> dict[str, Any]:
		"""Record a disbursement."""
		tenant = self._tenant(tenant_id)
		if disbursement_type not in DISBURSEMENT_TYPES:
			raise ValueError(f"disbursement_type must be one of {DISBURSEMENT_TYPES}")
		if amount <= 0:
			raise ValueError("amount must be positive")
		record: dict[str, Any] = {
			"id": self._id("dis-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"recorded_by_id": recorded_by_id,
			"date": date,
			"amount": amount,
			"currency": currency,
			"disbursement_type": disbursement_type,
			"description": description,
			"receipt_reference": receipt_reference,
			"billable": billable,
			"status": "pending",
			"invoice_id": None,
			"created_at": self._now(),
		}
		self.disbursements[record["id"]] = record
		self._emit(tenant, "disbursement_recorded", record["id"], {"matter_id": matter_id, "amount": amount})
		return deepcopy(record)

	async def get_disbursement(self, tenant_id: str, disbursement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		d = self.disbursements.get(disbursement_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"disbursement {disbursement_id} not found")
		return deepcopy(d)

	async def list_disbursements(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		disbursement_type: str | None = None,
		billable: bool | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.disbursements.values() if d["tenant_id"] == tenant]
		if matter_id:
			items = [d for d in items if d["matter_id"] == matter_id]
		if disbursement_type:
			items = [d for d in items if d["disbursement_type"] == disbursement_type]
		if billable is not None:
			items = [d for d in items if d["billable"] == billable]
		return items

	async def update_disbursement(self, tenant_id: str, disbursement_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		d = self.disbursements.get(disbursement_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"disbursement {disbursement_id} not found")
		allowed = {"amount", "description", "receipt_reference", "billable"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				d[k] = v
		self._emit(tenant, "disbursement_updated", disbursement_id, updates)
		return deepcopy(d)

	async def delete_disbursement(self, tenant_id: str, disbursement_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		d = self.disbursements.get(disbursement_id)
		if not d or d["tenant_id"] != tenant:
			raise KeyError(f"disbursement {disbursement_id} not found")
		d["status"] = "cancelled"
		self._emit(tenant, "disbursement_cancelled", disbursement_id)
		return deepcopy(d)

	# ── Invoices ─────────────────────────────────────────────────────────────

	async def create_invoice(
		self,
		tenant_id: str,
		matter_id: str,
		client_id: str,
		billing_period_start: str,
		billing_period_end: str,
		due_date: str,
		time_entry_ids: list[str] | None = None,
		disbursement_ids: list[str] | None = None,
		discount_amount: float = 0.0,
		discount_reason: str = "",
		notes: str = "",
		currency: str = "KES",
		tax_rate: float = 16.0,  # Kenya VAT
	) -> dict[str, Any]:
		"""Generate an invoice for a matter."""
		tenant = self._tenant(tenant_id)
		te_ids = time_entry_ids or []
		dis_ids = disbursement_ids or []
		fees_amount = 0.0
		for te_id in te_ids:
			te = self.time_entries.get(te_id)
			if not te or te["tenant_id"] != tenant:
				raise KeyError(f"time entry {te_id} not found")
			if te["status"] not in {"approved"}:
				raise ValueError(f"time entry {te_id} must be approved before invoicing")
			fees_amount += te["amount"]
		disbursements_amount = 0.0
		for d_id in dis_ids:
			d = self.disbursements.get(d_id)
			if not d or d["tenant_id"] != tenant:
				raise KeyError(f"disbursement {d_id} not found")
			disbursements_amount += d["amount"]
		subtotal = fees_amount + disbursements_amount - discount_amount
		tax_amount = round(subtotal * tax_rate / 100, 2)
		total_amount = round(subtotal + tax_amount, 2)
		invoice_number = self._next_invoice_number(tenant)
		invoice: dict[str, Any] = {
			"id": self._id("inv-"),
			"tenant_id": tenant,
			"invoice_number": invoice_number,
			"matter_id": matter_id,
			"client_id": client_id,
			"billing_period_start": billing_period_start,
			"billing_period_end": billing_period_end,
			"due_date": due_date,
			"time_entry_ids": list(te_ids),
			"disbursement_ids": list(dis_ids),
			"fees_amount": fees_amount,
			"disbursements_amount": disbursements_amount,
			"discount_amount": discount_amount,
			"subtotal": subtotal,
			"tax_rate": tax_rate,
			"tax_amount": tax_amount,
			"total_amount": total_amount,
			"currency": currency,
			"notes": notes,
			"status": "draft",
			"approved_by_id": None,
			"sent_at": None,
			"paid_at": None,
			"created_at": self._now(),
		}
		self.invoices[invoice["id"]] = invoice
		# Mark time entries and disbursements as billed
		for te_id in te_ids:
			te = self.time_entries[te_id]
			te["status"] = "billed"
			te["invoice_id"] = invoice["id"]
		for d_id in dis_ids:
			d = self.disbursements[d_id]
			d["status"] = "billed"
			d["invoice_id"] = invoice["id"]
		self._emit(tenant, "invoice_created", invoice["id"], {"number": invoice_number, "total": total_amount})
		_log.info("invoice created tenant=%s id=%s total=%.2f %s", tenant, invoice["id"], total_amount, currency)
		return deepcopy(invoice)

	async def get_invoice(self, tenant_id: str, invoice_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		return deepcopy(inv)

	async def list_invoices(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		client_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(i) for i in self.invoices.values() if i["tenant_id"] == tenant]
		if matter_id:
			items = [i for i in items if i["matter_id"] == matter_id]
		if client_id:
			items = [i for i in items if i["client_id"] == client_id]
		if status:
			items = [i for i in items if i["status"] == status]
		return items

	async def update_invoice(self, tenant_id: str, invoice_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		if inv["status"] not in {"draft"}:
			raise ValueError("only draft invoices can be updated")
		allowed = {"due_date", "notes", "discount_amount", "discount_reason"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				inv[k] = v
		self._emit(tenant, "invoice_updated", invoice_id, updates)
		return deepcopy(inv)

	async def approve_invoice(self, tenant_id: str, invoice_id: str, approved_by_id: str) -> dict[str, Any]:
		"""Approve an invoice for sending."""
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		inv["status"] = "approved"
		inv["approved_by_id"] = approved_by_id
		inv["approved_at"] = self._now()
		self._emit(tenant, "invoice_approved", invoice_id, {"approved_by": approved_by_id})
		return deepcopy(inv)

	async def send_invoice(self, tenant_id: str, invoice_id: str) -> dict[str, Any]:
		"""Mark invoice as sent to client."""
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		if inv["status"] != "approved":
			raise ValueError("invoice must be approved before sending")
		inv["status"] = "sent"
		inv["sent_at"] = self._now()
		self._emit(tenant, "invoice_sent", invoice_id)
		return deepcopy(inv)

	async def record_payment(self, tenant_id: str, invoice_id: str, payment_reference: str) -> dict[str, Any]:
		"""Record payment receipt for an invoice."""
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		if inv["status"] not in {"sent", "overdue"}:
			raise ValueError("can only record payment for sent or overdue invoices")
		inv["status"] = "paid"
		inv["paid_at"] = self._now()
		inv["payment_reference"] = payment_reference
		self._emit(tenant, "invoice_paid", invoice_id, {"reference": payment_reference})
		return deepcopy(inv)

	async def delete_invoice(self, tenant_id: str, invoice_id: str) -> dict[str, Any]:
		"""Write off / cancel an invoice."""
		tenant = self._tenant(tenant_id)
		inv = self.invoices.get(invoice_id)
		if not inv or inv["tenant_id"] != tenant:
			raise KeyError(f"invoice {invoice_id} not found")
		inv["status"] = "written_off"
		self._emit(tenant, "invoice_written_off", invoice_id)
		return deepcopy(inv)

	# ── Trust Accounts ────────────────────────────────────────────────────────

	async def create_trust_account(
		self,
		tenant_id: str,
		matter_id: str,
		client_id: str,
		account_name: str,
		bank_name: str,
		account_number: str,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Open a client trust account."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(account_name, "account_name")
		guard_non_empty_string(account_number, "account_number")
		acct: dict[str, Any] = {
			"id": self._id("ta-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"client_id": client_id,
			"account_name": account_name,
			"bank_name": bank_name,
			"account_number": account_number,
			"currency": currency,
			"balance": 0.0,
			"status": "active",
			"created_at": self._now(),
		}
		self.trust_accounts[acct["id"]] = acct
		self._emit(tenant, "trust_account_opened", acct["id"], {"client_id": client_id})
		return deepcopy(acct)

	async def get_trust_account(self, tenant_id: str, account_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		acct = self.trust_accounts.get(account_id)
		if not acct or acct["tenant_id"] != tenant:
			raise KeyError(f"trust account {account_id} not found")
		return deepcopy(acct)

	async def list_trust_accounts(self, tenant_id: str, client_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.trust_accounts.values() if a["tenant_id"] == tenant]
		if client_id:
			items = [a for a in items if a["client_id"] == client_id]
		return items

	async def update_trust_account(self, tenant_id: str, account_id: str, **updates: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		acct = self.trust_accounts.get(account_id)
		if not acct or acct["tenant_id"] != tenant:
			raise KeyError(f"trust account {account_id} not found")
		allowed = {"account_name", "bank_name"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				acct[k] = v
		self._emit(tenant, "trust_account_updated", account_id, updates)
		return deepcopy(acct)

	async def delete_trust_account(self, tenant_id: str, account_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		acct = self.trust_accounts.get(account_id)
		if not acct or acct["tenant_id"] != tenant:
			raise KeyError(f"trust account {account_id} not found")
		if acct["balance"] != 0:
			raise ValueError("cannot close trust account with non-zero balance")
		acct["status"] = "closed"
		self._emit(tenant, "trust_account_closed", account_id)
		return deepcopy(acct)

	async def trust_transaction(
		self,
		tenant_id: str,
		trust_account_id: str,
		transaction_type: str,
		amount: float,
		date: str,
		description: str,
		authorized_by_id: str,
		reference: str = "",
	) -> dict[str, Any]:
		"""Record a trust account transaction."""
		tenant = self._tenant(tenant_id)
		acct = self.trust_accounts.get(trust_account_id)
		if not acct or acct["tenant_id"] != tenant:
			raise KeyError(f"trust account {trust_account_id} not found")
		if transaction_type not in TRUST_TRANSACTION_TYPES:
			raise ValueError(f"transaction_type must be one of {TRUST_TRANSACTION_TYPES}")
		if amount <= 0:
			raise ValueError("amount must be positive")
		debit_types = {"withdrawal", "fee_application", "transfer"}
		if transaction_type in debit_types and acct["balance"] < amount:
			raise ValueError("insufficient trust account balance")
		if transaction_type in debit_types:
			acct["balance"] = round(acct["balance"] - amount, 2)
		else:
			acct["balance"] = round(acct["balance"] + amount, 2)
		txn: dict[str, Any] = {
			"id": self._id("ttx-"),
			"tenant_id": tenant,
			"trust_account_id": trust_account_id,
			"transaction_type": transaction_type,
			"amount": amount,
			"running_balance": acct["balance"],
			"date": date,
			"description": description,
			"reference": reference,
			"authorized_by_id": authorized_by_id,
			"status": "completed",
			"created_at": self._now(),
		}
		self.trust_transactions[txn["id"]] = txn
		self._emit(tenant, "trust_transaction_recorded", txn["id"], {
			"account_id": trust_account_id,
			"type": transaction_type,
			"amount": amount,
			"balance": acct["balance"],
		})
		return deepcopy(txn)

	async def list_trust_transactions(self, tenant_id: str, trust_account_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(t) for t in self.trust_transactions.values()
			if t["tenant_id"] == tenant and t["trust_account_id"] == trust_account_id
		]

	# ── Rate Cards ────────────────────────────────────────────────────────────

	async def set_rate_card(
		self,
		tenant_id: str,
		attorney_id: str,
		hourly_rate: float,
		currency: str = "KES",
		effective_from: str = "",
	) -> dict[str, Any]:
		"""Set or update billing rate for an attorney."""
		tenant = self._tenant(tenant_id)
		if hourly_rate < 0:
			raise ValueError("hourly_rate cannot be negative")
		card: dict[str, Any] = {
			"id": self._id("rc-"),
			"tenant_id": tenant,
			"attorney_id": attorney_id,
			"hourly_rate": hourly_rate,
			"currency": currency,
			"effective_from": effective_from or self._now()[:10],
			"status": "active",
			"created_at": self._now(),
		}
		self.rate_cards[attorney_id] = card
		self._emit(tenant, "rate_card_set", card["id"], {"attorney_id": attorney_id, "rate": hourly_rate})
		return deepcopy(card)

	async def get_rate_card(self, tenant_id: str, attorney_id: str) -> dict[str, Any] | None:
		tenant = self._tenant(tenant_id)
		card = self.rate_cards.get(attorney_id)
		if card and card["tenant_id"] == tenant:
			return deepcopy(card)
		return None

	# ── Analytics ────────────────────────────────────────────────────────────

	async def billing_dashboard(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		invoices = [i for i in self.invoices.values() if i["tenant_id"] == tenant]
		time_entries = [te for te in self.time_entries.values() if te["tenant_id"] == tenant]
		today = date.today().isoformat()
		outstanding = sum(i["total_amount"] for i in invoices if i["status"] in {"sent", "overdue"})
		collected = sum(i["total_amount"] for i in invoices if i["status"] == "paid")
		return {
			"tenant_id": tenant,
			"total_invoices": len(invoices),
			"outstanding_amount": outstanding,
			"collected_amount": collected,
			"draft_time_entries": sum(1 for te in time_entries if te["status"] == "draft"),
			"total_billed_hours": sum(te["hours"] for te in time_entries if te["status"] == "billed"),
			"trust_balance_total": sum(a["balance"] for a in self.trust_accounts.values() if a["tenant_id"] == tenant),
			"overdue_invoices": sum(1 for i in invoices if i["status"] == "sent" and i.get("due_date", "9999") < today),
			"generated_at": self._now(),
		}

	async def matter_billing_summary(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Billing summary for a specific matter."""
		tenant = self._tenant(tenant_id)
		time_entries = [te for te in self.time_entries.values() if te["tenant_id"] == tenant and te["matter_id"] == matter_id]
		disbursements = [d for d in self.disbursements.values() if d["tenant_id"] == tenant and d["matter_id"] == matter_id]
		invoices = [i for i in self.invoices.values() if i["tenant_id"] == tenant and i["matter_id"] == matter_id]
		return {
			"matter_id": matter_id,
			"total_hours": sum(te["hours"] for te in time_entries),
			"billable_hours": sum(te["hours"] for te in time_entries if te["billable"]),
			"fees_total": sum(te["amount"] for te in time_entries if te["billable"]),
			"disbursements_total": sum(d["amount"] for d in disbursements if d["billable"]),
			"invoiced_total": sum(i["total_amount"] for i in invoices),
			"paid_total": sum(i["total_amount"] for i in invoices if i["status"] == "paid"),
			"outstanding_total": sum(i["total_amount"] for i in invoices if i["status"] in {"sent", "overdue"}),
			"generated_at": self._now(),
		}

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

