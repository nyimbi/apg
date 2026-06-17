"""Policy Administration Service (ins_pol).

Manages the full policy lifecycle: issuance, endorsements, renewals,
cancellations, reinstatements and document generation.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import logging
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

SUPPORTED_STATUSES = {"draft", "active", "lapsed", "cancelled", "expired", "reinstated", "pending_renewal"}
SUPPORTED_ENDORSEMENT_TYPES = {
	"name_change", "address_change", "sum_insured_change", "premium_adjustment",
	"beneficiary_change", "coverage_extension", "coverage_reduction", "vehicle_change",
}
SUPPORTED_CANCELLATION_TYPES = {"voluntary", "non_payment", "fraud", "regulatory", "mutual_agreement"}
SUPPORTED_DOC_TYPES = {
	"policy_schedule", "certificate_of_insurance", "renewal_notice",
	"cancellation_notice", "endorsement_schedule", "claim_form",
}
SUPPORTED_PRODUCT_CODES = {
	"motor_comprehensive", "motor_third_party", "fire_industrial", "fire_domestic",
	"marine_cargo", "marine_hull", "life_whole", "life_term", "health_individual",
	"health_group", "travel", "engineering", "liability_public", "liability_employers",
}


class PolicyAdministrationService:
	"""In-memory executable service for the Policy Administration lifecycle."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.policies: dict[str, dict[str, Any]] = {}
		self.endorsements: dict[str, dict[str, Any]] = {}
		self.renewals: dict[str, dict[str, Any]] = {}
		self.cancellations: dict[str, dict[str, Any]] = {}
		self.reinstatements: dict[str, dict[str, Any]] = {}
		self.documents: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	# ── Helpers ───────────────────────────────────────────────────────────────

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

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"details": details or {},
			"created_at": self._now(),
		})

	def _get_policy(self, policy_id: str, tenant: str) -> dict[str, Any]:
		pol = self.policies.get(policy_id)
		if not pol or pol["tenant_id"] != tenant:
			raise KeyError(f"policy_not_found:{policy_id}")
		return pol

	# ── Policy CRUD ───────────────────────────────────────────────────────────

	async def create_policy(
		self,
		tenant_id: str,
		policy_number: str,
		product_code: str,
		insured_name: str,
		insured_id: str,
		sum_insured: Decimal,
		inception_date: str,
		expiry_date: str,
		premium: Decimal,
		underwriter_id: str,
		currency: str = "KES",
		agent_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Issue a new insurance policy."""
		tenant = self._tenant(tenant_id)
		if not policy_number:
			raise ValueError("policy_number_required")
		if product_code not in SUPPORTED_PRODUCT_CODES:
			raise ValueError(f"unsupported_product_code:{product_code}")
		existing = next((p for p in self.policies.values() if p["policy_number"] == policy_number and p["tenant_id"] == tenant), None)
		if existing:
			raise ValueError(f"policy_number_duplicate:{policy_number}")
		record: dict[str, Any] = {
			"id": self._record_id("pol"),
			"type": "ins_policy",
			"policy_number": policy_number,
			"product_code": product_code,
			"insured_name": insured_name,
			"insured_id": insured_id,
			"sum_insured": Decimal(str(sum_insured)),
			"currency": currency,
			"inception_date": inception_date,
			"expiry_date": expiry_date,
			"premium": Decimal(str(premium)),
			"underwriter_id": underwriter_id,
			"agent_id": agent_id,
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
			"updated_at": None,
			"metadata": deepcopy(metadata or {}),
		}
		self.policies[record["id"]] = record
		self._emit(tenant, "policy_issued", record["id"], "ins_policy", {"policy_number": policy_number, "product_code": product_code})
		_log.info("Policy issued: %s tenant=%s", policy_number, tenant)
		return deepcopy(record)

	async def get_policy(self, tenant_id: str, policy_id: str) -> dict[str, Any]:
		"""Retrieve a policy by ID."""
		tenant = self._tenant(tenant_id)
		return deepcopy(self._get_policy(policy_id, tenant))

	async def get_policy_by_number(self, tenant_id: str, policy_number: str) -> dict[str, Any]:
		"""Retrieve a policy by policy number."""
		tenant = self._tenant(tenant_id)
		pol = next((p for p in self.policies.values() if p["policy_number"] == policy_number and p["tenant_id"] == tenant), None)
		if not pol:
			raise KeyError(f"policy_not_found:{policy_number}")
		return deepcopy(pol)

	async def update_policy(self, tenant_id: str, policy_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update mutable policy fields."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		allowed = {"sum_insured", "premium", "expiry_date", "agent_id", "metadata"}
		for key, value in updates.items():
			if key not in allowed:
				raise ValueError(f"field_not_updatable:{key}")
			pol[key] = value
		pol["updated_at"] = self._now()
		self._emit(tenant, "policy_updated", policy_id, "ins_policy", {"fields": list(updates.keys())})
		return deepcopy(pol)

	async def delete_policy(self, tenant_id: str, policy_id: str, reason: str) -> dict[str, Any]:
		"""Soft-delete (void) a draft policy."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] != "draft":
			raise PermissionError("only_draft_policies_can_be_deleted")
		pol["status"] = "voided"
		pol["voided_reason"] = reason
		pol["voided_at"] = self._now()
		self._emit(tenant, "policy_voided", policy_id, "ins_policy", {"reason": reason})
		return deepcopy(pol)

	async def list_policies(self, tenant_id: str, status: str | None = None, product_code: str | None = None) -> list[dict[str, Any]]:
		"""List policies for a tenant, optionally filtered."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.policies.values() if p["tenant_id"] == tenant]
		if status:
			items = [p for p in items if p["status"] == status]
		if product_code:
			items = [p for p in items if p["product_code"] == product_code]
		return items

	async def list_policies_by_insured(self, tenant_id: str, insured_id: str) -> list[dict[str, Any]]:
		"""List all policies for a given insured party."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(p) for p in self.policies.values() if p["tenant_id"] == tenant and p["insured_id"] == insured_id]

	# ── Endorsements ──────────────────────────────────────────────────────────

	async def create_endorsement(
		self,
		tenant_id: str,
		policy_id: str,
		endorsement_type: str,
		effective_date: str,
		description: str,
		change_in_premium: Decimal = Decimal("0"),
		change_in_sum_insured: Decimal = Decimal("0"),
		requested_by: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a policy endorsement (mid-term change)."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] not in {"active", "reinstated"}:
			raise PermissionError("policy_must_be_active_for_endorsement")
		if endorsement_type not in SUPPORTED_ENDORSEMENT_TYPES:
			raise ValueError(f"unsupported_endorsement_type:{endorsement_type}")
		record: dict[str, Any] = {
			"id": self._record_id("end"),
			"type": "ins_endorsement",
			"policy_id": policy_id,
			"policy_number": pol["policy_number"],
			"endorsement_type": endorsement_type,
			"effective_date": effective_date,
			"description": description,
			"change_in_premium": Decimal(str(change_in_premium)),
			"change_in_sum_insured": Decimal(str(change_in_sum_insured)),
			"requested_by": requested_by,
			"status": "approved",
			"tenant_id": tenant,
			"created_at": self._now(),
			"metadata": deepcopy(metadata or {}),
		}
		self.endorsements[record["id"]] = record
		# Apply changes to policy
		if change_in_sum_insured:
			pol["sum_insured"] = pol["sum_insured"] + Decimal(str(change_in_sum_insured))
		if change_in_premium:
			pol["premium"] = pol["premium"] + Decimal(str(change_in_premium))
		pol["updated_at"] = self._now()
		self._emit(tenant, "endorsement_issued", record["id"], "ins_endorsement", {"policy_id": policy_id, "type": endorsement_type})
		return deepcopy(record)

	async def get_endorsement(self, tenant_id: str, endorsement_id: str) -> dict[str, Any]:
		"""Retrieve an endorsement by ID."""
		tenant = self._tenant(tenant_id)
		end = self.endorsements.get(endorsement_id)
		if not end or end["tenant_id"] != tenant:
			raise KeyError(f"endorsement_not_found:{endorsement_id}")
		return deepcopy(end)

	async def list_endorsements(self, tenant_id: str, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List endorsements, optionally filtered by policy."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.endorsements.values() if e["tenant_id"] == tenant]
		if policy_id:
			items = [e for e in items if e["policy_id"] == policy_id]
		return items

	# ── Renewals ──────────────────────────────────────────────────────────────

	async def initiate_renewal(
		self,
		tenant_id: str,
		policy_id: str,
		new_expiry_date: str,
		new_premium: Decimal,
		initiated_by: str,
		renewal_terms: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Initiate a policy renewal."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] not in {"active", "lapsed"}:
			raise PermissionError("policy_must_be_active_or_lapsed_for_renewal")
		record: dict[str, Any] = {
			"id": self._record_id("ren"),
			"type": "ins_renewal",
			"policy_id": policy_id,
			"policy_number": pol["policy_number"],
			"previous_expiry_date": pol["expiry_date"],
			"new_expiry_date": new_expiry_date,
			"previous_premium": pol["premium"],
			"new_premium": Decimal(str(new_premium)),
			"initiated_by": initiated_by,
			"renewal_terms": deepcopy(renewal_terms or {}),
			"status": "confirmed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.renewals[record["id"]] = record
		pol["expiry_date"] = new_expiry_date
		pol["premium"] = Decimal(str(new_premium))
		pol["status"] = "active"
		pol["updated_at"] = self._now()
		self._emit(tenant, "policy_renewed", record["id"], "ins_renewal", {"policy_id": policy_id, "new_expiry": new_expiry_date})
		return deepcopy(record)

	async def get_renewal(self, tenant_id: str, renewal_id: str) -> dict[str, Any]:
		"""Retrieve a renewal record."""
		tenant = self._tenant(tenant_id)
		ren = self.renewals.get(renewal_id)
		if not ren or ren["tenant_id"] != tenant:
			raise KeyError(f"renewal_not_found:{renewal_id}")
		return deepcopy(ren)

	async def list_renewals(self, tenant_id: str, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List renewals, optionally filtered by policy."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.renewals.values() if r["tenant_id"] == tenant]
		if policy_id:
			items = [r for r in items if r["policy_id"] == policy_id]
		return items

	async def list_due_renewals(self, tenant_id: str, days_ahead: int = 30) -> list[dict[str, Any]]:
		"""Return policies due for renewal within the next N days."""
		tenant = self._tenant(tenant_id)
		from datetime import timedelta
		cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
		today = date.today().isoformat()
		return [
			deepcopy(p) for p in self.policies.values()
			if p["tenant_id"] == tenant
			and p["status"] == "active"
			and today <= p["expiry_date"] <= cutoff
		]

	# ── Cancellations ─────────────────────────────────────────────────────────

	async def cancel_policy(
		self,
		tenant_id: str,
		policy_id: str,
		cancellation_date: str,
		reason: str,
		cancellation_type: str = "voluntary",
		refund_premium: bool = True,
		authorised_by: str = "",
	) -> dict[str, Any]:
		"""Cancel a policy."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] in {"cancelled", "voided", "expired"}:
			raise PermissionError(f"policy_already_{pol['status']}")
		if cancellation_type not in SUPPORTED_CANCELLATION_TYPES:
			raise ValueError(f"unsupported_cancellation_type:{cancellation_type}")
		# Calculate pro-rata refund
		refund_amount = Decimal("0")
		if refund_premium:
			try:
				inc = date.fromisoformat(pol["inception_date"])
				exp = date.fromisoformat(pol["expiry_date"])
				can = date.fromisoformat(cancellation_date)
				total_days = (exp - inc).days or 1
				remaining_days = max((exp - can).days, 0)
				refund_amount = (pol["premium"] * remaining_days / total_days).quantize(Decimal("0.01"))
			except Exception as exc:
				_log.error("Pro-rata refund calculation failed: %s", exc)
		record: dict[str, Any] = {
			"id": self._record_id("can"),
			"type": "ins_cancellation",
			"policy_id": policy_id,
			"policy_number": pol["policy_number"],
			"cancellation_date": cancellation_date,
			"reason": reason,
			"cancellation_type": cancellation_type,
			"refund_amount": refund_amount,
			"authorised_by": authorised_by,
			"status": "processed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.cancellations[record["id"]] = record
		pol["status"] = "cancelled"
		pol["cancellation_id"] = record["id"]
		pol["updated_at"] = self._now()
		self._emit(tenant, "policy_cancelled", record["id"], "ins_cancellation", {"policy_id": policy_id, "reason": reason})
		return deepcopy(record)

	async def list_cancellations(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all cancellation records for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(c) for c in self.cancellations.values() if c["tenant_id"] == tenant]

	# ── Reinstatements ────────────────────────────────────────────────────────

	async def reinstate_policy(
		self,
		tenant_id: str,
		policy_id: str,
		reinstatement_date: str,
		outstanding_premium: Decimal,
		reason: str,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Reinstate a lapsed or cancelled policy."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] not in {"lapsed", "cancelled"}:
			raise PermissionError("policy_must_be_lapsed_or_cancelled_for_reinstatement")
		record: dict[str, Any] = {
			"id": self._record_id("rst"),
			"type": "ins_reinstatement",
			"policy_id": policy_id,
			"policy_number": pol["policy_number"],
			"reinstatement_date": reinstatement_date,
			"outstanding_premium": Decimal(str(outstanding_premium)),
			"reason": reason,
			"authorised_by": authorised_by,
			"status": "approved",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.reinstatements[record["id"]] = record
		pol["status"] = "reinstated"
		pol["reinstatement_id"] = record["id"]
		pol["updated_at"] = self._now()
		self._emit(tenant, "policy_reinstated", record["id"], "ins_reinstatement", {"policy_id": policy_id})
		return deepcopy(record)

	async def list_reinstatements(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List reinstatement records."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.reinstatements.values() if r["tenant_id"] == tenant]

	# ── Document generation ───────────────────────────────────────────────────

	async def generate_document(
		self,
		tenant_id: str,
		policy_id: str,
		document_type: str,
		generated_by: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Generate a policy document."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if document_type not in SUPPORTED_DOC_TYPES:
			raise ValueError(f"unsupported_document_type:{document_type}")
		record: dict[str, Any] = {
			"id": self._record_id("doc"),
			"type": "ins_document",
			"document_type": document_type,
			"policy_id": policy_id,
			"policy_number": pol["policy_number"],
			"insured_name": pol["insured_name"],
			"generated_by": generated_by,
			"file_reference": f"{document_type}_{pol['policy_number']}_{self._now()[:10]}.pdf",
			"status": "generated",
			"tenant_id": tenant,
			"created_at": self._now(),
			"metadata": deepcopy(metadata or {}),
		}
		self.documents[record["id"]] = record
		self._emit(tenant, "document_generated", record["id"], "ins_document", {"policy_id": policy_id, "doc_type": document_type})
		return deepcopy(record)

	async def list_documents(self, tenant_id: str, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List documents, optionally filtered by policy."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.documents.values() if d["tenant_id"] == tenant]
		if policy_id:
			items = [d for d in items if d["policy_id"] == policy_id]
		return items

	# ── Status transitions ────────────────────────────────────────────────────

	async def lapse_policy(self, tenant_id: str, policy_id: str, reason: str = "non_payment") -> dict[str, Any]:
		"""Mark a policy as lapsed due to non-payment."""
		tenant = self._tenant(tenant_id)
		pol = self._get_policy(policy_id, tenant)
		if pol["status"] != "active":
			raise PermissionError("only_active_policies_can_lapse")
		pol["status"] = "lapsed"
		pol["lapse_reason"] = reason
		pol["lapsed_at"] = self._now()
		pol["updated_at"] = self._now()
		self._emit(tenant, "policy_lapsed", policy_id, "ins_policy", {"reason": reason})
		return deepcopy(pol)

	async def expire_policies(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Batch-expire all policies past their expiry date."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		expired = []
		for pol in self.policies.values():
			if pol["tenant_id"] == tenant and pol["status"] == "active" and pol["expiry_date"] < today:
				pol["status"] = "expired"
				pol["expired_at"] = self._now()
				pol["updated_at"] = self._now()
				self._emit(tenant, "policy_expired", pol["id"], "ins_policy", {})
				expired.append(deepcopy(pol))
		return expired

	# ── Analytics & Health ────────────────────────────────────────────────────

	async def portfolio_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return portfolio-level summary metrics."""
		tenant = self._tenant(tenant_id)
		pols = [p for p in self.policies.values() if p["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		by_product: dict[str, int] = {}
		total_sum_insured = Decimal("0")
		total_premium = Decimal("0")
		for p in pols:
			by_status[p["status"]] = by_status.get(p["status"], 0) + 1
			by_product[p["product_code"]] = by_product.get(p["product_code"], 0) + 1
			if p["status"] == "active":
				total_sum_insured += p["sum_insured"]
				total_premium += p["premium"]
		return {
			"tenant_id": tenant,
			"total_policies": len(pols),
			"by_status": by_status,
			"by_product": by_product,
			"active_sum_insured": str(total_sum_insured),
			"active_premium_income": str(total_premium),
			"endorsement_count": len([e for e in self.endorsements.values() if e["tenant_id"] == tenant]),
			"renewal_count": len([r for r in self.renewals.values() if r["tenant_id"] == tenant]),
			"cancellation_count": len([c for c in self.cancellations.values() if c["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "ins_pol",
			"status": "healthy",
			"policy_count": len(self.policies),
			"endorsement_count": len(self.endorsements),
			"document_count": len(self.documents),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Describe this capability."""
		return {
			"capability_id": "ins_pol",
			"name": "Policy Administration",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_products": list(SUPPORTED_PRODUCT_CODES),
			"supported_endorsement_types": list(SUPPORTED_ENDORSEMENT_TYPES),
			"supported_doc_types": list(SUPPORTED_DOC_TYPES),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return audit trail for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	async def search_policies(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Full-text search across policy number, insured name, and metadata."""
		tenant = self._tenant(tenant_id)
		q = query.lower()
		return [
			deepcopy(p) for p in self.policies.values()
			if p["tenant_id"] == tenant
			and (q in p["policy_number"].lower() or q in p["insured_name"].lower())
		]

	async def bulk_issue_policies(self, tenant_id: str, policies: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-issue multiple policies."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		for pol_data in policies:
			try:
				rec = await self.create_policy(
					tenant_id=tenant,
					policy_number=pol_data["policy_number"],
					product_code=pol_data["product_code"],
					insured_name=pol_data["insured_name"],
					insured_id=pol_data["insured_id"],
					sum_insured=Decimal(str(pol_data["sum_insured"])),
					inception_date=pol_data["inception_date"],
					expiry_date=pol_data["expiry_date"],
					premium=Decimal(str(pol_data["premium"])),
					underwriter_id=pol_data["underwriter_id"],
					currency=pol_data.get("currency", "KES"),
					agent_id=pol_data.get("agent_id"),
				)
				results.append(rec)
			except Exception as exc:
				_log.error("Bulk policy issuance failed for %s: %s", pol_data.get("policy_number"), exc)
				errors.append({"input": pol_data, "error": str(exc)})
		return {"processed": len(results), "failed": len(errors), "policies": results, "errors": errors}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

