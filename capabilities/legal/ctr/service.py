"""Contract Lifecycle Management — async service layer."""
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

CONTRACT_TYPES = {
	"nda", "msa", "sow", "lease", "employment", "vendor", "partnership",
	"licensing", "service", "supply", "loan", "settlement",
}
CONTRACT_STATUSES = {"draft", "under_review", "approved", "executed", "active", "expired", "terminated", "archived"}
APPROVAL_STATUSES = {"pending", "approved", "rejected", "withdrawn"}


class ContractLifecycleService:
	"""In-memory async service for contract lifecycle management."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.contracts: dict[str, dict[str, Any]] = {}
		self.redlines: dict[str, dict[str, Any]] = {}
		self.obligations: dict[str, dict[str, Any]] = {}
		self.approvals: dict[str, dict[str, Any]] = {}
		self.versions: dict[str, list[dict[str, Any]]] = {}
		self.signatories: dict[str, dict[str, Any]] = {}
		self.renewals: dict[str, dict[str, Any]] = {}
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

	# ── Health & Describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "leg_ctr",
			"status": "healthy",
			"contract_count": len(self.contracts),
			"pending_approvals": sum(1 for a in self.approvals.values() if a["status"] == "pending"),
			"expiring_soon": sum(
				1 for c in self.contracts.values()
				if c.get("expiry_date") and c["expiry_date"] <= date.today().replace(day=date.today().day + 30).isoformat()
				and c["status"] == "active"
			),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_ctr",
			"name": "Contract Lifecycle Management",
			"domain": "legal",
			"version": "1.0.0",
			"contract_types": sorted(CONTRACT_TYPES),
			"statuses": sorted(CONTRACT_STATUSES),
		}

	# ── Contracts ────────────────────────────────────────────────────────────

	async def create_contract(
		self,
		tenant_id: str,
		title: str,
		contract_type: str,
		counterparty_id: str,
		owner_id: str,
		effective_date: str,
		expiry_date: str | None = None,
		auto_renew: bool = False,
		renewal_notice_days: int = 30,
		value: float | None = None,
		currency: str = "KES",
		jurisdiction: str = "",
		governing_law: str = "",
		description: str = "",
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Draft a new contract."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(counterparty_id, "counterparty_id")
		if contract_type not in CONTRACT_TYPES:
			raise ValueError(f"contract_type must be one of {CONTRACT_TYPES}")
		record: dict[str, Any] = {
			"id": self._id("ctr-"),
			"tenant_id": tenant,
			"title": title,
			"contract_type": contract_type,
			"counterparty_id": counterparty_id,
			"owner_id": owner_id,
			"effective_date": effective_date,
			"expiry_date": expiry_date,
			"auto_renew": auto_renew,
			"renewal_notice_days": renewal_notice_days,
			"value": value,
			"currency": currency,
			"jurisdiction": jurisdiction,
			"governing_law": governing_law,
			"description": description,
			"status": "draft",
			"version": 1,
			"tags": list(tags or []),
			"document_ids": [],
			"obligation_count": 0,
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
			"signed_at": None,
			"executed_at": None,
		}
		self.contracts[record["id"]] = record
		self.versions[record["id"]] = [deepcopy(record)]
		self._emit(tenant, "contract_created", record["id"], {"title": title, "type": contract_type})
		_log.info("contract created tenant=%s id=%s type=%s", tenant, record["id"], contract_type)
		return deepcopy(record)

	async def get_contract(self, tenant_id: str, contract_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		return deepcopy(c)

	async def list_contracts(
		self,
		tenant_id: str,
		status: str | None = None,
		contract_type: str | None = None,
		counterparty_id: str | None = None,
		owner_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List contracts with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.contracts.values() if c["tenant_id"] == tenant]
		if status:
			items = [c for c in items if c["status"] == status]
		if contract_type:
			items = [c for c in items if c["contract_type"] == contract_type]
		if counterparty_id:
			items = [c for c in items if c["counterparty_id"] == counterparty_id]
		if owner_id:
			items = [c for c in items if c["owner_id"] == owner_id]
		return items

	async def update_contract(self, tenant_id: str, contract_id: str, **updates: Any) -> dict[str, Any]:
		"""Update draft contract fields."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		if c["status"] not in {"draft", "under_review"}:
			raise ValueError(f"cannot update contract in status {c['status']}")
		allowed = {
			"title", "description", "expiry_date", "auto_renew", "renewal_notice_days",
			"value", "governing_law", "jurisdiction", "tags", "metadata",
		}
		for k, v in updates.items():
			if k in allowed and v is not None:
				c[k] = v
		c["version"] = c.get("version", 1) + 1
		c["updated_at"] = self._now()
		self.versions[contract_id].append(deepcopy(c))
		self._emit(tenant, "contract_updated", contract_id, {"version": c["version"]})
		return deepcopy(c)

	async def submit_for_review(self, tenant_id: str, contract_id: str, submitted_by: str) -> dict[str, Any]:
		"""Move contract to under_review status."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		if c["status"] != "draft":
			raise ValueError("only draft contracts can be submitted for review")
		c["status"] = "under_review"
		c["submitted_by"] = submitted_by
		c["submitted_at"] = self._now()
		c["updated_at"] = self._now()
		self._emit(tenant, "contract_submitted_for_review", contract_id, {"submitted_by": submitted_by})
		return deepcopy(c)

	async def execute_contract(self, tenant_id: str, contract_id: str, executed_by: str) -> dict[str, Any]:
		"""Mark contract as fully executed."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		if c["status"] != "approved":
			raise ValueError("contract must be approved before execution")
		c["status"] = "active"
		c["executed_at"] = self._now()
		c["executed_by"] = executed_by
		c["updated_at"] = self._now()
		self._emit(tenant, "contract_executed", contract_id, {"executed_by": executed_by})
		return deepcopy(c)

	async def terminate_contract(self, tenant_id: str, contract_id: str, reason: str, terminated_by: str) -> dict[str, Any]:
		"""Terminate an active contract."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		c["status"] = "terminated"
		c["termination_reason"] = reason
		c["terminated_by"] = terminated_by
		c["terminated_at"] = self._now()
		c["updated_at"] = self._now()
		self._emit(tenant, "contract_terminated", contract_id, {"reason": reason, "by": terminated_by})
		return deepcopy(c)

	async def delete_contract(self, tenant_id: str, contract_id: str) -> dict[str, Any]:
		"""Archive a contract."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		c["status"] = "archived"
		c["updated_at"] = self._now()
		self._emit(tenant, "contract_archived", contract_id)
		return deepcopy(c)

	async def get_contract_versions(self, tenant_id: str, contract_id: str) -> list[dict[str, Any]]:
		"""Return version history for a contract."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		return [deepcopy(v) for v in self.versions.get(contract_id, [])]

	# ── Redlining ────────────────────────────────────────────────────────────

	async def create_redline(
		self,
		tenant_id: str,
		contract_id: str,
		reviewer_id: str,
		section_ref: str,
		original_text: str,
		proposed_text: str,
		comment: str = "",
		change_type: str = "modification",
	) -> dict[str, Any]:
		"""Add a redline comment to a contract."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		redline: dict[str, Any] = {
			"id": self._id("rdl-"),
			"tenant_id": tenant,
			"contract_id": contract_id,
			"reviewer_id": reviewer_id,
			"section_ref": section_ref,
			"original_text": original_text,
			"proposed_text": proposed_text,
			"comment": comment,
			"change_type": change_type,
			"status": "pending",
			"resolved_by_id": None,
			"resolved_at": None,
			"created_at": self._now(),
		}
		self.redlines[redline["id"]] = redline
		self._emit(tenant, "redline_created", redline["id"], {"contract_id": contract_id, "section": section_ref})
		return deepcopy(redline)

	async def resolve_redline(
		self,
		tenant_id: str,
		redline_id: str,
		decision: str,  # accepted | rejected
		resolved_by_id: str,
	) -> dict[str, Any]:
		"""Accept or reject a redline."""
		tenant = self._tenant(tenant_id)
		rdl = self.redlines.get(redline_id)
		if not rdl or rdl["tenant_id"] != tenant:
			raise KeyError(f"redline {redline_id} not found")
		if decision not in {"accepted", "rejected"}:
			raise ValueError("decision must be accepted or rejected")
		rdl["status"] = decision
		rdl["resolved_by_id"] = resolved_by_id
		rdl["resolved_at"] = self._now()
		self._emit(tenant, f"redline_{decision}", redline_id, {"resolved_by": resolved_by_id})
		return deepcopy(rdl)

	async def list_redlines(self, tenant_id: str, contract_id: str) -> list[dict[str, Any]]:
		"""List redlines for a contract."""
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(r) for r in self.redlines.values()
			if r["tenant_id"] == tenant and r["contract_id"] == contract_id
		]

	# ── Approvals ────────────────────────────────────────────────────────────

	async def create_approval(
		self,
		tenant_id: str,
		contract_id: str,
		approver_id: str,
		approval_level: int = 1,
		comments: str = "",
	) -> dict[str, Any]:
		"""Request an approval for a contract."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		approval: dict[str, Any] = {
			"id": self._id("apr-"),
			"tenant_id": tenant,
			"contract_id": contract_id,
			"approver_id": approver_id,
			"approval_level": approval_level,
			"comments": comments,
			"status": "pending",
			"decided_at": None,
			"created_at": self._now(),
		}
		self.approvals[approval["id"]] = approval
		self._emit(tenant, "approval_requested", approval["id"], {"contract_id": contract_id, "approver": approver_id})
		return deepcopy(approval)

	async def decide_approval(
		self,
		tenant_id: str,
		approval_id: str,
		decision: str,  # approved | rejected
		comments: str = "",
	) -> dict[str, Any]:
		"""Record approval/rejection decision."""
		tenant = self._tenant(tenant_id)
		appr = self.approvals.get(approval_id)
		if not appr or appr["tenant_id"] != tenant:
			raise KeyError(f"approval {approval_id} not found")
		if decision not in {"approved", "rejected"}:
			raise ValueError("decision must be approved or rejected")
		appr["status"] = decision
		appr["comments"] = comments or appr["comments"]
		appr["decided_at"] = self._now()
		# If approved, check if all levels are approved and advance contract
		contract_id = appr["contract_id"]
		c = self.contracts.get(contract_id)
		if c and decision == "approved":
			all_approved = all(
				a["status"] == "approved"
				for a in self.approvals.values()
				if a["contract_id"] == contract_id
			)
			if all_approved:
				c["status"] = "approved"
				c["updated_at"] = self._now()
				self._emit(tenant, "contract_approved", contract_id)
		elif c and decision == "rejected":
			c["status"] = "under_review"
			c["updated_at"] = self._now()
		self._emit(tenant, f"approval_{decision}", approval_id)
		return deepcopy(appr)

	async def list_approvals(self, tenant_id: str, contract_id: str) -> list[dict[str, Any]]:
		"""List approvals for a contract."""
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(a) for a in self.approvals.values()
			if a["tenant_id"] == tenant and a["contract_id"] == contract_id
		]

	# ── Obligations ──────────────────────────────────────────────────────────

	async def create_obligation(
		self,
		tenant_id: str,
		contract_id: str,
		title: str,
		description: str,
		obligor: str,
		owner_id: str,
		due_date: str | None = None,
		recurrence: str | None = None,
	) -> dict[str, Any]:
		"""Add a contractual obligation tracker."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		guard_non_empty_string(title, "title")
		obligation: dict[str, Any] = {
			"id": self._id("obl-"),
			"tenant_id": tenant,
			"contract_id": contract_id,
			"title": title,
			"description": description,
			"obligor": obligor,
			"owner_id": owner_id,
			"due_date": due_date,
			"recurrence": recurrence,
			"status": "active",
			"last_fulfilled_at": None,
			"created_at": self._now(),
		}
		self.obligations[obligation["id"]] = obligation
		c["obligation_count"] = c.get("obligation_count", 0) + 1
		self._emit(tenant, "obligation_created", obligation["id"], {"contract_id": contract_id})
		return deepcopy(obligation)

	async def fulfill_obligation(self, tenant_id: str, obligation_id: str, fulfilled_by: str) -> dict[str, Any]:
		"""Record fulfillment of an obligation."""
		tenant = self._tenant(tenant_id)
		obl = self.obligations.get(obligation_id)
		if not obl or obl["tenant_id"] != tenant:
			raise KeyError(f"obligation {obligation_id} not found")
		obl["last_fulfilled_at"] = self._now()
		obl["fulfilled_by"] = fulfilled_by
		if not obl.get("recurrence"):
			obl["status"] = "fulfilled"
		self._emit(tenant, "obligation_fulfilled", obligation_id)
		return deepcopy(obl)

	async def list_obligations(self, tenant_id: str, contract_id: str) -> list[dict[str, Any]]:
		"""List obligations for a contract."""
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(o) for o in self.obligations.values()
			if o["tenant_id"] == tenant and o["contract_id"] == contract_id
		]

	# ── E-Signature ──────────────────────────────────────────────────────────

	async def add_signatory(
		self,
		tenant_id: str,
		contract_id: str,
		signatory_id: str,
		signatory_name: str,
		signatory_role: str,
		signatory_order: int = 1,
	) -> dict[str, Any]:
		"""Add a signatory to the contract signing workflow."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		sig: dict[str, Any] = {
			"id": self._id("sig-"),
			"tenant_id": tenant,
			"contract_id": contract_id,
			"signatory_id": signatory_id,
			"signatory_name": signatory_name,
			"signatory_role": signatory_role,
			"signatory_order": signatory_order,
			"status": "pending",
			"signed_at": None,
			"ip_address": None,
			"created_at": self._now(),
		}
		self.signatories[sig["id"]] = sig
		self._emit(tenant, "signatory_added", sig["id"], {"contract_id": contract_id, "signatory": signatory_id})
		return deepcopy(sig)

	async def record_signature(
		self,
		tenant_id: str,
		signatory_id: str,
		contract_id: str,
		ip_address: str = "",
	) -> dict[str, Any]:
		"""Record an e-signature for a contract signatory."""
		tenant = self._tenant(tenant_id)
		sig = next(
			(s for s in self.signatories.values()
			 if s["tenant_id"] == tenant and s["contract_id"] == contract_id and s["signatory_id"] == signatory_id),
			None,
		)
		if not sig:
			raise KeyError(f"signatory {signatory_id} not found for contract {contract_id}")
		sig["status"] = "signed"
		sig["signed_at"] = self._now()
		sig["ip_address"] = ip_address
		# Check if all signatories have signed
		all_signed = all(
			s["status"] == "signed"
			for s in self.signatories.values()
			if s["contract_id"] == contract_id and s["tenant_id"] == tenant
		)
		c = self.contracts.get(contract_id)
		if c and all_signed:
			c["signed_at"] = self._now()
			self._emit(tenant, "contract_fully_signed", contract_id)
		self._emit(tenant, "signature_recorded", sig["id"], {"signatory": signatory_id})
		return deepcopy(sig)

	# ── Renewals ─────────────────────────────────────────────────────────────

	async def schedule_renewal(
		self,
		tenant_id: str,
		contract_id: str,
		renewal_date: str,
		new_expiry_date: str,
		renewal_value: float | None = None,
		notes: str = "",
	) -> dict[str, Any]:
		"""Schedule a contract renewal."""
		tenant = self._tenant(tenant_id)
		c = self.contracts.get(contract_id)
		if not c or c["tenant_id"] != tenant:
			raise KeyError(f"contract {contract_id} not found")
		renewal: dict[str, Any] = {
			"id": self._id("rnw-"),
			"tenant_id": tenant,
			"contract_id": contract_id,
			"renewal_date": renewal_date,
			"new_expiry_date": new_expiry_date,
			"renewal_value": renewal_value,
			"notes": notes,
			"status": "scheduled",
			"created_at": self._now(),
		}
		self.renewals[renewal["id"]] = renewal
		self._emit(tenant, "renewal_scheduled", renewal["id"], {"contract_id": contract_id})
		return deepcopy(renewal)

	async def execute_renewal(self, tenant_id: str, renewal_id: str, executed_by: str) -> dict[str, Any]:
		"""Execute a scheduled renewal."""
		tenant = self._tenant(tenant_id)
		renewal = self.renewals.get(renewal_id)
		if not renewal or renewal["tenant_id"] != tenant:
			raise KeyError(f"renewal {renewal_id} not found")
		contract_id = renewal["contract_id"]
		c = self.contracts.get(contract_id)
		if c:
			c["expiry_date"] = renewal["new_expiry_date"]
			if renewal.get("renewal_value"):
				c["value"] = renewal["renewal_value"]
			c["updated_at"] = self._now()
			c["status"] = "active"
		renewal["status"] = "executed"
		renewal["executed_by"] = executed_by
		renewal["executed_at"] = self._now()
		self._emit(tenant, "renewal_executed", renewal_id)
		return deepcopy(renewal)

	async def list_expiring_contracts(self, tenant_id: str, days_ahead: int = 30) -> list[dict[str, Any]]:
		"""Return active contracts expiring within N days."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		items = [
			deepcopy(c) for c in self.contracts.values()
			if c["tenant_id"] == tenant
			and c["status"] == "active"
			and c.get("expiry_date")
			and c["expiry_date"] >= today
		]
		return sorted(items, key=lambda c: c["expiry_date"])[:50]

	# ── Analytics ────────────────────────────────────────────────────────────

	async def contract_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate contract metrics."""
		tenant = self._tenant(tenant_id)
		contracts = [c for c in self.contracts.values() if c["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		total_value = 0.0
		for c in contracts:
			by_type[c["contract_type"]] = by_type.get(c["contract_type"], 0) + 1
			by_status[c["status"]] = by_status.get(c["status"], 0) + 1
			if c.get("value"):
				total_value += c["value"]
		return {
			"tenant_id": tenant,
			"total_contracts": len(contracts),
			"by_type": by_type,
			"by_status": by_status,
			"total_value": total_value,
			"pending_approvals": sum(1 for a in self.approvals.values() if a["tenant_id"] == tenant and a["status"] == "pending"),
			"open_redlines": sum(1 for r in self.redlines.values() if r["tenant_id"] == tenant and r["status"] == "pending"),
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

