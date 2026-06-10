"""Returns & Reverse Logistics async service (scm_rrl)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_rrl"
RMA_STATUSES = {"pending", "approved", "received", "processing", "resolved", "rejected", "closed"}
REASON_CODES = {"defective", "wrong_item", "not_as_described", "changed_mind", "damaged_in_transit"}
DISPOSAL_METHODS = {"recycle", "destroy", "donate", "auction", "landfill"}
RESOLUTIONS = {"refund", "replacement", "credit", "repair"}
CONDITIONS = {"like_new", "good", "fair", "poor", "scrap"}


class ReturnsService:
	"""Async service for RMA processing, refurbishment workflows,
	disposal management, credit notes and reverse shipment tracking."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.rmas: dict[str, dict[str, Any]] = {}
		self.refurbishments: dict[str, dict[str, Any]] = {}
		self.disposals: dict[str, dict[str, Any]] = {}
		self.credit_notes: dict[str, dict[str, Any]] = {}
		self.reverse_shipments: dict[str, dict[str, Any]] = {}
		self.return_inspections: dict[str, dict[str, Any]] = {}
		self._rma_seq: int = 3000
		self._cn_seq: int = 7000
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _tenant(self, tenant_id: str | None = None) -> str:
		t = tenant_id or self.tenant_id
		if not t:
			raise PermissionError("tenant_context_required")
		return t

	def _next_rma_number(self, tenant: str) -> str:
		self._rma_seq += 1
		return f"RMA-{tenant[:4].upper()}-{self._rma_seq:06d}"

	def _next_cn_number(self, tenant: str) -> str:
		self._cn_seq += 1
		return f"CN-{tenant[:4].upper()}-{self._cn_seq:06d}"

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, status: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
		})

	# ── Health & describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"open_rmas": sum(1 for r in self.rmas.values() if r["status"] not in {"resolved", "closed", "rejected"}),
			"pending_refurbishments": sum(1 for r in self.refurbishments.values() if r["status"] == "pending"),
			"credit_notes_issued": sum(1 for c in self.credit_notes.values() if c["status"] == "issued"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "RMA processing, refurbishment workflow, disposal management, credit notes, reverse shipment tracking",
			"rma_statuses": sorted(RMA_STATUSES),
			"reason_codes": sorted(REASON_CODES),
			"disposal_methods": sorted(DISPOSAL_METHODS),
			"resolutions": sorted(RESOLUTIONS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── RMA management ────────────────────────────────────────────────────────

	async def create_rma(
		self,
		order_id: str,
		customer_id: str,
		items: list[dict[str, Any]],
		reason_code: str,
		description: str | None = None,
		requested_resolution: str = "refund",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a Return Merchandise Authorisation."""
		tenant = self._tenant(tenant_id)
		if reason_code not in REASON_CODES:
			raise ValueError(f"reason_code must be one of {REASON_CODES}")
		if requested_resolution not in RESOLUTIONS:
			raise ValueError(f"requested_resolution must be one of {RESOLUTIONS}")
		if not items:
			raise ValueError("RMA must include at least one item")
		record: dict[str, Any] = {
			"id": self._id("rma"),
			"type": "scm_rrl_rma",
			"tenant_id": tenant,
			"rma_number": self._next_rma_number(tenant),
			"order_id": order_id,
			"customer_id": customer_id,
			"items": deepcopy(items),
			"reason_code": reason_code,
			"description": description,
			"requested_resolution": requested_resolution,
			"resolution": None,
			"status": "pending",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.rmas[record["id"]] = record
		self._emit(tenant, "rma_created", record["id"], "scm_rrl_rma", "pending")
		return deepcopy(record)

	async def approve_rma(
		self,
		rma_id: str,
		approved_by: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Approve an RMA request."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		if rma["status"] != "pending":
			raise ValueError("only pending RMAs can be approved")
		rma["status"] = "approved"
		rma["approved_by"] = approved_by
		rma["approval_notes"] = notes
		rma["approved_at"] = self._now()
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_approved", rma_id, "scm_rrl_rma", "approved")
		return deepcopy(rma)

	async def reject_rma(
		self,
		rma_id: str,
		rejected_by: str,
		reason: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Reject an RMA request."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		rma["status"] = "rejected"
		rma["rejected_by"] = rejected_by
		rma["rejection_reason"] = reason
		rma["rejected_at"] = self._now()
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_rejected", rma_id, "scm_rrl_rma", "rejected")
		return deepcopy(rma)

	async def receive_return(
		self,
		rma_id: str,
		received_by: str,
		condition_notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark returned goods as physically received."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		if rma["status"] != "approved":
			raise ValueError("only approved RMAs can receive goods")
		rma["status"] = "received"
		rma["received_by"] = received_by
		rma["condition_notes"] = condition_notes
		rma["received_at"] = self._now()
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_goods_received", rma_id, "scm_rrl_rma", "received")
		return deepcopy(rma)

	async def resolve_rma(
		self,
		rma_id: str,
		resolution: str,
		resolved_by: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Resolve an RMA with a specified outcome."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		if resolution not in RESOLUTIONS:
			raise ValueError(f"resolution must be one of {RESOLUTIONS}")
		rma["status"] = "resolved"
		rma["resolution"] = resolution
		rma["resolved_by"] = resolved_by
		rma["resolution_notes"] = notes
		rma["resolved_at"] = self._now()
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_resolved", rma_id, "scm_rrl_rma", "resolved")
		return deepcopy(rma)

	async def list_rmas(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		customer_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List RMAs."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rmas.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		if customer_id:
			items = [r for r in items if r["customer_id"] == customer_id]
		return items

	async def get_rma(self, rma_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single RMA."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		return deepcopy(rma)

	async def update_rma(
		self,
		rma_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update RMA notes or resolution."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		allowed = {"status", "resolution", "notes"}
		for k, v in updates.items():
			if k in allowed:
				rma[k] = v
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_updated", rma_id, "scm_rrl_rma", rma["status"])
		return deepcopy(rma)

	async def delete_rma(self, rma_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Close an RMA."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		rma["status"] = "closed"
		rma["updated_at"] = self._now()
		self._emit(tenant, "rma_closed", rma_id, "scm_rrl_rma", "closed")
		return deepcopy(rma)

	# ── Return inspection ─────────────────────────────────────────────────────

	async def create_inspection(
		self,
		rma_id: str,
		inspector: str,
		items_inspected: list[dict[str, Any]],
		overall_condition: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a quality inspection of returned goods."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		if overall_condition not in CONDITIONS:
			raise ValueError(f"overall_condition must be one of {CONDITIONS}")
		record: dict[str, Any] = {
			"id": self._id("insp"),
			"type": "scm_rrl_inspection",
			"tenant_id": tenant,
			"rma_id": rma_id,
			"inspector": inspector,
			"items_inspected": deepcopy(items_inspected),
			"overall_condition": overall_condition,
			"notes": notes,
			"status": "completed",
			"inspected_at": self._now(),
			"created_at": self._now(),
		}
		self.return_inspections[record["id"]] = record
		self._emit(tenant, "inspection_completed", record["id"], "scm_rrl_inspection", "completed")
		return deepcopy(record)

	# ── Refurbishment workflow ────────────────────────────────────────────────

	async def create_refurbishment(
		self,
		rma_id: str,
		sku: str,
		condition_received: str,
		refurbishment_actions: list[str],
		assigned_to: str | None = None,
		estimated_cost: float | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a refurbishment work order for a returned item."""
		tenant = self._tenant(tenant_id)
		if condition_received not in CONDITIONS:
			raise ValueError(f"condition_received must be one of {CONDITIONS}")
		record: dict[str, Any] = {
			"id": self._id("refurb"),
			"type": "scm_rrl_refurbishment",
			"tenant_id": tenant,
			"rma_id": rma_id,
			"sku": sku,
			"condition_received": condition_received,
			"condition_after": None,
			"refurbishment_actions": refurbishment_actions,
			"assigned_to": assigned_to,
			"estimated_cost": estimated_cost,
			"actual_cost": None,
			"status": "pending",
			"created_at": self._now(),
			"completed_at": None,
		}
		self.refurbishments[record["id"]] = record
		self._emit(tenant, "refurbishment_created", record["id"], "scm_rrl_refurbishment", "pending")
		return deepcopy(record)

	async def complete_refurbishment(
		self,
		refurb_id: str,
		condition_after: str,
		actual_cost: float,
		completed_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark a refurbishment as complete."""
		tenant = self._tenant(tenant_id)
		refurb = self.refurbishments.get(refurb_id)
		if not refurb or refurb["tenant_id"] != tenant:
			raise KeyError(f"refurbishment '{refurb_id}' not found")
		if condition_after not in CONDITIONS:
			raise ValueError(f"condition_after must be one of {CONDITIONS}")
		refurb["status"] = "completed"
		refurb["condition_after"] = condition_after
		refurb["actual_cost"] = actual_cost
		refurb["completed_by"] = completed_by
		refurb["completed_at"] = self._now()
		self._emit(tenant, "refurbishment_completed", refurb_id, "scm_rrl_refurbishment", "completed")
		return deepcopy(refurb)

	async def list_refurbishments(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List refurbishment work orders."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.refurbishments.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── Disposal management ───────────────────────────────────────────────────

	async def create_disposal(
		self,
		rma_id: str,
		sku: str,
		quantity: float,
		disposal_method: str,
		reason: str,
		authorised_by: str,
		disposal_cost: float | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register disposal of unrepairable returned goods."""
		tenant = self._tenant(tenant_id)
		if disposal_method not in DISPOSAL_METHODS:
			raise ValueError(f"disposal_method must be one of {DISPOSAL_METHODS}")
		record: dict[str, Any] = {
			"id": self._id("disp"),
			"type": "scm_rrl_disposal",
			"tenant_id": tenant,
			"rma_id": rma_id,
			"sku": sku,
			"quantity": quantity,
			"disposal_method": disposal_method,
			"reason": reason,
			"authorised_by": authorised_by,
			"disposal_cost": disposal_cost,
			"status": "pending",
			"created_at": self._now(),
			"disposed_at": None,
		}
		self.disposals[record["id"]] = record
		self._emit(tenant, "disposal_created", record["id"], "scm_rrl_disposal", "pending")
		return deepcopy(record)

	async def complete_disposal(
		self,
		disposal_id: str,
		disposed_by: str,
		disposal_certificate: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Mark disposal as executed."""
		tenant = self._tenant(tenant_id)
		disposal = self.disposals.get(disposal_id)
		if not disposal or disposal["tenant_id"] != tenant:
			raise KeyError(f"disposal '{disposal_id}' not found")
		disposal["status"] = "completed"
		disposal["disposed_by"] = disposed_by
		disposal["disposal_certificate"] = disposal_certificate
		disposal["disposed_at"] = self._now()
		self._emit(tenant, "disposal_completed", disposal_id, "scm_rrl_disposal", "completed")
		return deepcopy(disposal)

	async def list_disposals(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List disposal records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.disposals.values() if d["tenant_id"] == tenant]
		if status:
			items = [d for d in items if d["status"] == status]
		return items

	# ── Credit notes ──────────────────────────────────────────────────────────

	async def issue_credit_note(
		self,
		rma_id: str,
		customer_id: str,
		amount: float,
		reason: str,
		issued_by: str,
		currency: str = "USD",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Issue a credit note against a resolved RMA."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		if amount <= 0:
			raise ValueError("credit note amount must be positive")
		record: dict[str, Any] = {
			"id": self._id("cn"),
			"type": "scm_rrl_credit_note",
			"tenant_id": tenant,
			"credit_note_number": self._next_cn_number(tenant),
			"rma_id": rma_id,
			"customer_id": customer_id,
			"amount": amount,
			"currency": currency,
			"reason": reason,
			"issued_by": issued_by,
			"status": "issued",
			"issued_at": self._now(),
		}
		self.credit_notes[record["id"]] = record
		self._emit(tenant, "credit_note_issued", record["id"], "scm_rrl_credit_note", "issued")
		return deepcopy(record)

	async def list_credit_notes(
		self,
		customer_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List credit notes."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.credit_notes.values() if c["tenant_id"] == tenant]
		if customer_id:
			items = [c for c in items if c["customer_id"] == customer_id]
		return items

	# ── Reverse shipments ─────────────────────────────────────────────────────

	async def create_reverse_shipment(
		self,
		rma_id: str,
		carrier_id: str,
		pickup_address: dict[str, Any],
		destination_address: dict[str, Any],
		weight_kg: float | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a reverse shipment for collecting returned goods."""
		tenant = self._tenant(tenant_id)
		rma = self.rmas.get(rma_id)
		if not rma or rma["tenant_id"] != tenant:
			raise KeyError(f"rma '{rma_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("revshp"),
			"type": "scm_rrl_reverse_shipment",
			"tenant_id": tenant,
			"rma_id": rma_id,
			"carrier_id": carrier_id,
			"pickup_address": deepcopy(pickup_address),
			"destination_address": deepcopy(destination_address),
			"weight_kg": weight_kg,
			"tracking_number": f"RVS{uuid4().hex[:10].upper()}",
			"status": "booked",
			"created_at": self._now(),
		}
		self.reverse_shipments[record["id"]] = record
		self._emit(tenant, "reverse_shipment_created", record["id"], "scm_rrl_reverse_shipment", "booked")
		return deepcopy(record)

	async def list_reverse_shipments(
		self,
		rma_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List reverse shipments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.reverse_shipments.values() if s["tenant_id"] == tenant]
		if rma_id:
			items = [s for s in items if s["rma_id"] == rma_id]
		return items

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def returns_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate returns metrics."""
		tenant = self._tenant(tenant_id)
		all_rmas = [r for r in self.rmas.values() if r["tenant_id"] == tenant]
		by_reason: dict[str, int] = {}
		by_status: dict[str, int] = {}
		by_resolution: dict[str, int] = {}
		for r in all_rmas:
			by_reason[r["reason_code"]] = by_reason.get(r["reason_code"], 0) + 1
			by_status[r["status"]] = by_status.get(r["status"], 0) + 1
			if r.get("resolution"):
				by_resolution[r["resolution"]] = by_resolution.get(r["resolution"], 0) + 1
		total_credit = sum(c["amount"] for c in self.credit_notes.values() if c["tenant_id"] == tenant)
		return {
			"tenant_id": tenant,
			"total_rmas": len(all_rmas),
			"by_reason": by_reason,
			"by_status": by_status,
			"by_resolution": by_resolution,
			"total_credit_issued": round(total_credit, 2),
			"pending_refurbishments": sum(1 for r in self.refurbishments.values() if r["tenant_id"] == tenant and r["status"] == "pending"),
			"pending_disposals": sum(1 for d in self.disposals.values() if d["tenant_id"] == tenant and d["status"] == "pending"),
			"generated_at": self._now(),
		}

	async def bulk_create_rmas(
		self,
		rmas_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create multiple RMAs."""
		tenant = self._tenant(tenant_id)
		tasks = [self.create_rma(tenant_id=tenant, **r) for r in rmas_data]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "rmas": results, "errors": errors}
