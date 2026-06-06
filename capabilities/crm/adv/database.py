"""CRM Advanced Analytics — database utilities.

Provides an in-memory DatabaseManager with typed async CRUD operations for
CRMLead, CRMAccount, CRMOpportunity, and CRMActivity — no Postgres required
for tests and local development.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import (
	CRMAccount,
	CRMActivity,
	CRMLead,
	CRMOpportunity,
)


class DatabaseManager:
	"""In-memory database manager for CRM records.

	All write operations are async to match the production asyncpg interface;
	the implementation is synchronous in-memory so no event loop or pool is
	required.
	"""

	def __init__(self) -> None:
		self._initialized: bool = True
		# typed stores
		self._leads: dict[str, CRMLead] = {}
		self._accounts: dict[str, CRMAccount] = {}
		self._opportunities: dict[str, CRMOpportunity] = {}
		self._activities: dict[str, CRMActivity] = {}
		self._contacts: dict[str, Any] = {}
		# low-level table store retained for backward-compat
		self._tables: dict[str, list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------
	# Backward-compat low-level helpers
	# ------------------------------------------------------------------

	def create_table(self, name: str) -> None:
		self._tables.setdefault(name, [])

	def insert(self, table: str, record: dict[str, Any]) -> None:
		self._tables.setdefault(table, []).append(record)

	def query(self, table: str, **filters) -> list[dict[str, Any]]:
		rows = self._tables.get(table, [])
		return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]

	def delete(self, table: str, **filters) -> int:
		rows = self._tables.get(table, [])
		before = len(rows)
		self._tables[table] = [
			r for r in rows if not all(r.get(k) == v for k, v in filters.items())
		]
		return before - len(self._tables[table])

	# ------------------------------------------------------------------
	# CRMLead
	# ------------------------------------------------------------------

	async def create_lead(self, lead: CRMLead) -> CRMLead:
		self._leads[lead.id] = lead
		return deepcopy(lead)

	async def get_lead(self, lead_id: str, tenant_id: str) -> CRMLead | None:
		rec = self._leads.get(lead_id)
		if rec is None or rec.tenant_id != tenant_id:
			return None
		return deepcopy(rec)

	async def update_lead(
		self, lead_id: str, updates: dict[str, Any], tenant_id: str
	) -> CRMLead:
		rec = self._leads.get(lead_id)
		if rec is None or rec.tenant_id != tenant_id:
			raise KeyError(f"Lead not found: {lead_id}")
		data = rec.model_dump()
		data.update(updates)
		updated = rec.__class__(**data)
		self._leads[lead_id] = updated
		return deepcopy(updated)

	async def list_leads(
		self,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		search_term: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		items = [r for r in self._leads.values() if r.tenant_id == tenant_id]
		if filters:
			for k, v in filters.items():
				if v is not None:
					items = [r for r in items if getattr(r, k, None) == v]
		if search_term:
			q = search_term.lower()
			items = [
				r for r in items
				if q in (r.first_name or "").lower()
				or q in (r.last_name or "").lower()
				or q in (r.email or "").lower()
				or q in (r.company or "").lower()
			]
		total = len(items)
		start = (page - 1) * page_size
		return {"items": [deepcopy(r) for r in items[start: start + page_size]], "total_count": total}

	# ------------------------------------------------------------------
	# CRMAccount
	# ------------------------------------------------------------------

	async def create_account(self, account: CRMAccount) -> CRMAccount:
		self._accounts[account.id] = account
		return deepcopy(account)

	async def get_account(self, account_id: str, tenant_id: str) -> CRMAccount | None:
		rec = self._accounts.get(account_id)
		if rec is None or rec.tenant_id != tenant_id:
			return None
		return deepcopy(rec)

	async def update_account(
		self, account_id: str, updates: dict[str, Any], tenant_id: str
	) -> CRMAccount:
		rec = self._accounts.get(account_id)
		if rec is None or rec.tenant_id != tenant_id:
			raise KeyError(f"Account not found: {account_id}")
		data = rec.model_dump()
		data.update(updates)
		updated = rec.__class__(**data)
		self._accounts[account_id] = updated
		return deepcopy(updated)

	async def list_accounts(
		self,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		search_term: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		items = [r for r in self._accounts.values() if r.tenant_id == tenant_id]
		if filters:
			for k, v in filters.items():
				if v is not None:
					items = [r for r in items if getattr(r, k, None) == v]
		if search_term:
			q = search_term.lower()
			items = [
				r for r in items
				if q in (r.account_name or "").lower()
				or q in (r.industry or "").lower()
			]
		total = len(items)
		start = (page - 1) * page_size
		return {"items": [deepcopy(r) for r in items[start: start + page_size]], "total_count": total}

	# ------------------------------------------------------------------
	# CRMOpportunity
	# ------------------------------------------------------------------

	async def create_opportunity(self, opportunity: CRMOpportunity) -> CRMOpportunity:
		self._opportunities[opportunity.id] = opportunity
		return deepcopy(opportunity)

	async def get_opportunity(self, opp_id: str, tenant_id: str) -> CRMOpportunity | None:
		rec = self._opportunities.get(opp_id)
		if rec is None or rec.tenant_id != tenant_id:
			return None
		return deepcopy(rec)

	async def update_opportunity(
		self, opp_id: str, updates: dict[str, Any], tenant_id: str
	) -> CRMOpportunity:
		rec = self._opportunities.get(opp_id)
		if rec is None or rec.tenant_id != tenant_id:
			raise KeyError(f"Opportunity not found: {opp_id}")
		data = rec.model_dump()
		data.update(updates)
		updated = rec.__class__(**data)
		self._opportunities[opp_id] = updated
		return deepcopy(updated)

	async def list_opportunities(
		self,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		search_term: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		items = [r for r in self._opportunities.values() if r.tenant_id == tenant_id]
		if filters:
			for k, v in filters.items():
				if v is not None:
					items = [r for r in items if getattr(r, k, None) == v]
		if search_term:
			q = search_term.lower()
			items = [
				r for r in items
				if q in (r.opportunity_name or "").lower()
			]
		total = len(items)
		start = (page - 1) * page_size
		return {"items": [deepcopy(r) for r in items[start: start + page_size]], "total_count": total}

	# ------------------------------------------------------------------
	# CRMActivity
	# ------------------------------------------------------------------

	async def create_activity(self, activity: CRMActivity) -> CRMActivity:
		self._activities[activity.id] = activity
		return deepcopy(activity)

	async def get_activity(self, activity_id: str, tenant_id: str) -> CRMActivity | None:
		rec = self._activities.get(activity_id)
		if rec is None or rec.tenant_id != tenant_id:
			return None
		return deepcopy(rec)

	async def list_activities(
		self,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		search_term: str | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		items = [r for r in self._activities.values() if r.tenant_id == tenant_id]
		if filters:
			for k, v in filters.items():
				if v is not None:
					items = [r for r in items if getattr(r, k, None) == v]
		if search_term:
			q = search_term.lower()
			items = [
				r for r in items
				if q in (r.subject or "").lower()
			]
		total = len(items)
		start = (page - 1) * page_size
		return {"items": [deepcopy(r) for r in items[start: start + page_size]], "total_count": total}

	# ------------------------------------------------------------------
	# Contact helpers (used by import_export)
	# ------------------------------------------------------------------

	async def find_contacts_by_emails(
		self, tenant_id: str, emails: list[str]
	) -> list[Any]:
		email_set = set(e.lower() for e in emails)
		return [
			c for c in self._contacts.values()
			if c.get("tenant_id") == tenant_id and (c.get("email") or "").lower() in email_set
		]

	async def bulk_create_contacts(
		self, contacts: list[dict[str, Any]]
	) -> dict[str, Any]:
		errors: list[dict[str, Any]] = []
		success = 0
		for c in contacts:
			try:
				cid = c.get("id") or c.get("email") or str(len(self._contacts))
				self._contacts[cid] = c
				success += 1
			except Exception as exc:
				errors.append({"contact": c, "error": str(exc)})
		return {"success_count": success, "error_count": len(errors), "errors": errors}

	async def get_contact(self, contact_id: str, tenant_id: str) -> Any | None:
		rec = self._contacts.get(contact_id)
		if rec is None or rec.get("tenant_id") != tenant_id:
			return None
		return rec

	async def list_contacts(
		self,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		limit: int = 10000,
	) -> dict[str, Any]:
		items = [c for c in self._contacts.values() if c.get("tenant_id") == tenant_id]
		return {"items": items[:limit], "total_count": len(items)}

	# ------------------------------------------------------------------
	# Count helpers
	# ------------------------------------------------------------------

	def count_leads(self, tenant_id: str) -> int:
		return sum(1 for r in self._leads.values() if r.tenant_id == tenant_id)

	def count_accounts(self, tenant_id: str) -> int:
		return sum(1 for r in self._accounts.values() if r.tenant_id == tenant_id)

	def count_opportunities(self, tenant_id: str) -> int:
		return sum(1 for r in self._opportunities.values() if r.tenant_id == tenant_id)

	def count_activities(self, tenant_id: str) -> int:
		return sum(1 for r in self._activities.values() if r.tenant_id == tenant_id)

	def count_contacts(self, tenant_id: str) -> int:
		return sum(1 for c in self._contacts.values() if c.get("tenant_id") == tenant_id)
