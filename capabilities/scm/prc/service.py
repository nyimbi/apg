"""Procurement Management async service (scm_prc)."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_prc"
PO_STATUSES = {"draft", "sent", "acknowledged", "partially_received", "received", "invoiced", "closed", "cancelled"}
RFQ_STATUSES = {"draft", "issued", "responses_received", "awarded", "cancelled"}
MATCH_RESULTS = {"matched", "partial", "disputed"}


class ProcurementService:
	"""Async service for RFQ, purchase orders, three-way match,
	vendor evaluation, contract compliance and spend analytics."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.rfqs: dict[str, dict[str, Any]] = {}
		self.rfq_responses: dict[str, dict[str, Any]] = {}
		self.purchase_orders: dict[str, dict[str, Any]] = {}
		self.receipts: dict[str, dict[str, Any]] = {}
		self.three_way_matches: dict[str, dict[str, Any]] = {}
		self.vendor_evaluations: dict[str, dict[str, Any]] = {}
		self.contracts: dict[str, dict[str, Any]] = {}
		self.spend_records: dict[str, dict[str, Any]] = {}
		self._rfq_seq: int = 5000
		self._po_seq: int = 8000
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

	def _next_rfq_number(self, tenant: str) -> str:
		self._rfq_seq += 1
		return f"RFQ-{tenant[:4].upper()}-{self._rfq_seq:06d}"

	def _next_po_number(self, tenant: str) -> str:
		self._po_seq += 1
		return f"PO-{tenant[:4].upper()}-{self._po_seq:06d}"

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
			"rfq_count": len(self.rfqs),
			"open_pos": sum(1 for p in self.purchase_orders.values() if p["status"] not in {"closed", "cancelled"}),
			"pending_matches": sum(1 for m in self.three_way_matches.values() if m["status"] == "pending"),
			"active_contracts": sum(1 for c in self.contracts.values() if c["status"] == "active"),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"domain": "scm",
			"version": "1.0.0",
			"description": "RFQ, purchase order, three-way match, vendor evaluation, contract compliance, spend analytics",
			"po_statuses": sorted(PO_STATUSES),
			"rfq_statuses": sorted(RFQ_STATUSES),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── RFQ management ────────────────────────────────────────────────────────

	async def create_rfq(
		self,
		title: str,
		lines: list[dict[str, Any]],
		vendor_ids: list[str] | None = None,
		deadline: str | None = None,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a request for quotation."""
		tenant = self._tenant(tenant_id)
		if not lines:
			raise ValueError("RFQ must have at least one line")
		record: dict[str, Any] = {
			"id": self._id("rfq"),
			"type": "scm_prc_rfq",
			"tenant_id": tenant,
			"rfq_number": self._next_rfq_number(tenant),
			"title": title,
			"lines": deepcopy(lines),
			"vendor_ids": vendor_ids or [],
			"deadline": deadline,
			"notes": notes,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.rfqs[record["id"]] = record
		self._emit(tenant, "rfq_created", record["id"], "scm_prc_rfq", "draft")
		return deepcopy(record)

	async def issue_rfq(self, rfq_id: str, issued_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Issue an RFQ to vendors."""
		tenant = self._tenant(tenant_id)
		rfq = self.rfqs.get(rfq_id)
		if not rfq or rfq["tenant_id"] != tenant:
			raise KeyError(f"rfq '{rfq_id}' not found")
		if rfq["status"] != "draft":
			raise ValueError("only draft RFQs can be issued")
		rfq["status"] = "issued"
		rfq["issued_by"] = issued_by
		rfq["issued_at"] = self._now()
		rfq["updated_at"] = self._now()
		self._emit(tenant, "rfq_issued", rfq_id, "scm_prc_rfq", "issued")
		return deepcopy(rfq)

	async def record_rfq_response(
		self,
		rfq_id: str,
		vendor_id: str,
		quoted_lines: list[dict[str, Any]],
		total_quoted_amount: float,
		currency: str = "USD",
		valid_until: str | None = None,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a vendor's response to an RFQ."""
		tenant = self._tenant(tenant_id)
		rfq = self.rfqs.get(rfq_id)
		if not rfq or rfq["tenant_id"] != tenant:
			raise KeyError(f"rfq '{rfq_id}' not found")
		record: dict[str, Any] = {
			"id": self._id("rfqr"),
			"type": "scm_prc_rfq_response",
			"tenant_id": tenant,
			"rfq_id": rfq_id,
			"vendor_id": vendor_id,
			"quoted_lines": deepcopy(quoted_lines),
			"total_quoted_amount": total_quoted_amount,
			"currency": currency,
			"valid_until": valid_until,
			"notes": notes,
			"status": "received",
			"created_at": self._now(),
		}
		self.rfq_responses[record["id"]] = record
		rfq["status"] = "responses_received"
		rfq["updated_at"] = self._now()
		self._emit(tenant, "rfq_response_received", record["id"], "scm_prc_rfq_response", "received")
		return deepcopy(record)

	async def award_rfq(
		self,
		rfq_id: str,
		winning_vendor_id: str,
		awarded_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Award an RFQ to a vendor."""
		tenant = self._tenant(tenant_id)
		rfq = self.rfqs.get(rfq_id)
		if not rfq or rfq["tenant_id"] != tenant:
			raise KeyError(f"rfq '{rfq_id}' not found")
		rfq["status"] = "awarded"
		rfq["winning_vendor_id"] = winning_vendor_id
		rfq["awarded_by"] = awarded_by
		rfq["awarded_at"] = self._now()
		rfq["updated_at"] = self._now()
		self._emit(tenant, "rfq_awarded", rfq_id, "scm_prc_rfq", "awarded")
		return deepcopy(rfq)

	async def list_rfqs(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List RFQs."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.rfqs.values() if r["tenant_id"] == tenant]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	async def get_rfq(self, rfq_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single RFQ."""
		tenant = self._tenant(tenant_id)
		rfq = self.rfqs.get(rfq_id)
		if not rfq or rfq["tenant_id"] != tenant:
			raise KeyError(f"rfq '{rfq_id}' not found")
		return deepcopy(rfq)

	# ── Purchase orders ───────────────────────────────────────────────────────

	async def create_purchase_order(
		self,
		vendor_id: str,
		lines: list[dict[str, Any]],
		rfq_id: str | None = None,
		payment_terms: str = "NET30",
		delivery_address: dict[str, Any] | None = None,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a purchase order."""
		tenant = self._tenant(tenant_id)
		if not lines:
			raise ValueError("PO must have at least one line")
		total_value = sum(
			float(l.get("quantity", 0)) * float(l.get("unit_price", 0))
			for l in lines
		)
		enriched = [
			{**l, "line_total": round(float(l.get("quantity", 0)) * float(l.get("unit_price", 0)), 4), "received_quantity": 0.0}
			for l in lines
		]
		record: dict[str, Any] = {
			"id": self._id("po"),
			"type": "scm_prc_purchase_order",
			"tenant_id": tenant,
			"po_number": self._next_po_number(tenant),
			"vendor_id": vendor_id,
			"lines": enriched,
			"total_value": round(total_value, 4),
			"currency": lines[0].get("currency", "USD") if lines else "USD",
			"rfq_id": rfq_id,
			"payment_terms": payment_terms,
			"delivery_address": deepcopy(delivery_address or {}),
			"notes": notes,
			"status": "draft",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.purchase_orders[record["id"]] = record
		self._emit(tenant, "po_created", record["id"], "scm_prc_purchase_order", "draft")
		return deepcopy(record)

	async def send_purchase_order(self, po_id: str, sent_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Send a PO to the vendor."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		if po["status"] != "draft":
			raise ValueError("only draft POs can be sent")
		po["status"] = "sent"
		po["sent_by"] = sent_by
		po["sent_at"] = self._now()
		po["updated_at"] = self._now()
		self._emit(tenant, "po_sent", po_id, "scm_prc_purchase_order", "sent")
		return deepcopy(po)

	async def acknowledge_purchase_order(self, po_id: str, vendor_reference: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Record vendor acknowledgement of a PO."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		po["status"] = "acknowledged"
		po["vendor_reference"] = vendor_reference
		po["acknowledged_at"] = self._now()
		po["updated_at"] = self._now()
		self._emit(tenant, "po_acknowledged", po_id, "scm_prc_purchase_order", "acknowledged")
		return deepcopy(po)

	async def receive_purchase_order(
		self,
		po_id: str,
		received_lines: list[dict[str, Any]],
		received_by: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record goods receipt against a PO."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		receipt_id = self._id("rcpt")
		receipt: dict[str, Any] = {
			"id": receipt_id,
			"type": "scm_prc_receipt",
			"tenant_id": tenant,
			"po_id": po_id,
			"received_lines": deepcopy(received_lines),
			"received_by": received_by,
			"total_received_value": sum(
				float(l.get("quantity", 0)) * float(l.get("unit_price", 0))
				for l in received_lines
			),
			"status": "received",
			"received_at": self._now(),
		}
		self.receipts[receipt_id] = receipt
		# update PO line received quantities
		received_map = {l["sku"]: float(l.get("quantity", 0)) for l in received_lines}
		all_received = True
		for line in po["lines"]:
			line["received_quantity"] = line.get("received_quantity", 0.0) + received_map.get(line["sku"], 0.0)
			if line["received_quantity"] < line["quantity"]:
				all_received = False
		po["status"] = "received" if all_received else "partially_received"
		po["updated_at"] = self._now()
		self._emit(tenant, "po_received", po_id, "scm_prc_purchase_order", po["status"])
		return deepcopy(receipt)

	async def list_purchase_orders(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
		vendor_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List purchase orders."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.purchase_orders.values() if p["tenant_id"] == tenant]
		if status:
			items = [p for p in items if p["status"] == status]
		if vendor_id:
			items = [p for p in items if p["vendor_id"] == vendor_id]
		return items

	async def get_purchase_order(self, po_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Fetch a single purchase order."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		return deepcopy(po)

	async def update_purchase_order(
		self,
		po_id: str,
		updates: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update a purchase order."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		allowed = {"status", "payment_terms", "notes"}
		for k, v in updates.items():
			if k in allowed:
				po[k] = v
		po["updated_at"] = self._now()
		self._emit(tenant, "po_updated", po_id, "scm_prc_purchase_order", po["status"])
		return deepcopy(po)

	async def delete_purchase_order(self, po_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a draft purchase order."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		if po["status"] not in {"draft"}:
			raise ValueError("only draft POs can be deleted")
		po["status"] = "cancelled"
		po["updated_at"] = self._now()
		self._emit(tenant, "po_cancelled", po_id, "scm_prc_purchase_order", "cancelled")
		return deepcopy(po)

	# ── Three-way match ───────────────────────────────────────────────────────

	async def create_three_way_match(
		self,
		po_id: str,
		receipt_id: str,
		invoice_number: str,
		invoiced_amount: float,
		currency: str = "USD",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Perform three-way match: PO vs receipt vs invoice."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		receipt = self.receipts.get(receipt_id)
		if not receipt or receipt["tenant_id"] != tenant:
			raise KeyError(f"receipt '{receipt_id}' not found")
		po_amount = po["total_value"]
		received_amount = receipt.get("total_received_value", 0.0)
		variance = round(abs(invoiced_amount - po_amount), 4)
		tolerance = po_amount * 0.01  # 1% tolerance
		if variance <= tolerance:
			match_result = "matched"
		elif variance <= po_amount * 0.05:
			match_result = "partial"
		else:
			match_result = "disputed"
		record: dict[str, Any] = {
			"id": self._id("3wm"),
			"type": "scm_prc_three_way_match",
			"tenant_id": tenant,
			"po_id": po_id,
			"receipt_id": receipt_id,
			"invoice_number": invoice_number,
			"po_amount": po_amount,
			"received_amount": received_amount,
			"invoiced_amount": invoiced_amount,
			"variance": variance,
			"currency": currency,
			"match_result": match_result,
			"status": "pending" if match_result != "matched" else "approved",
			"created_at": self._now(),
		}
		self.three_way_matches[record["id"]] = record
		if match_result == "matched":
			po["status"] = "invoiced"
			po["updated_at"] = self._now()
		self._emit(tenant, f"three_way_match_{match_result}", record["id"], "scm_prc_three_way_match", record["status"])
		return deepcopy(record)

	async def resolve_three_way_match(
		self,
		match_id: str,
		resolution: str,
		resolved_by: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Resolve a disputed or partial three-way match."""
		tenant = self._tenant(tenant_id)
		match = self.three_way_matches.get(match_id)
		if not match or match["tenant_id"] != tenant:
			raise KeyError(f"match '{match_id}' not found")
		if resolution not in {"approved", "rejected"}:
			raise ValueError("resolution must be 'approved' or 'rejected'")
		match["status"] = resolution
		match["resolved_by"] = resolved_by
		match["resolution_notes"] = notes
		match["resolved_at"] = self._now()
		self._emit(tenant, f"three_way_match_{resolution}", match_id, "scm_prc_three_way_match", resolution)
		return deepcopy(match)

	async def list_three_way_matches(
		self,
		tenant_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List three-way matches."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(m) for m in self.three_way_matches.values() if m["tenant_id"] == tenant]
		if status:
			items = [m for m in items if m["status"] == status]
		return items

	# ── Vendor evaluation ─────────────────────────────────────────────────────

	async def create_vendor_evaluation(
		self,
		vendor_id: str,
		period: str,
		quality_score: float,
		delivery_score: float,
		price_score: float,
		service_score: float,
		evaluated_by: str,
		notes: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Record a periodic vendor evaluation."""
		tenant = self._tenant(tenant_id)
		for score in (quality_score, delivery_score, price_score, service_score):
			if not 0 <= score <= 10:
				raise ValueError("all scores must be between 0 and 10")
		overall = round((quality_score + delivery_score + price_score + service_score) / 4, 2)
		record: dict[str, Any] = {
			"id": self._id("veval"),
			"type": "scm_prc_vendor_evaluation",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"period": period,
			"quality_score": quality_score,
			"delivery_score": delivery_score,
			"price_score": price_score,
			"service_score": service_score,
			"overall_score": overall,
			"evaluated_by": evaluated_by,
			"notes": notes,
			"status": "completed",
			"created_at": self._now(),
		}
		self.vendor_evaluations[record["id"]] = record
		self._emit(tenant, "vendor_evaluation_completed", record["id"], "scm_prc_vendor_evaluation", "completed")
		return deepcopy(record)

	async def list_vendor_evaluations(
		self,
		vendor_id: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List vendor evaluations."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.vendor_evaluations.values() if e["tenant_id"] == tenant]
		if vendor_id:
			items = [e for e in items if e["vendor_id"] == vendor_id]
		return items

	# ── Contract management ───────────────────────────────────────────────────

	async def create_contract(
		self,
		vendor_id: str,
		contract_reference: str,
		start_date: str,
		end_date: str,
		value: float,
		currency: str = "USD",
		terms: dict[str, Any] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a procurement contract with a vendor."""
		tenant = self._tenant(tenant_id)
		record: dict[str, Any] = {
			"id": self._id("contract"),
			"type": "scm_prc_contract",
			"tenant_id": tenant,
			"vendor_id": vendor_id,
			"contract_reference": contract_reference,
			"start_date": start_date,
			"end_date": end_date,
			"value": value,
			"currency": currency,
			"terms": deepcopy(terms or {}),
			"status": "active",
			"created_at": self._now(),
		}
		self.contracts[record["id"]] = record
		self._emit(tenant, "contract_created", record["id"], "scm_prc_contract", "active")
		return deepcopy(record)

	async def list_contracts(
		self,
		vendor_id: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""List contracts."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.contracts.values() if c["tenant_id"] == tenant]
		if vendor_id:
			items = [c for c in items if c["vendor_id"] == vendor_id]
		if status:
			items = [c for c in items if c["status"] == status]
		return items

	async def check_contract_compliance(
		self,
		po_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Check if a PO is covered by an active vendor contract."""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		vendor_contracts = [
			c for c in self.contracts.values()
			if c["tenant_id"] == tenant
			and c["vendor_id"] == po["vendor_id"]
			and c["status"] == "active"
		]
		covered = bool(vendor_contracts)
		return {
			"po_id": po_id,
			"vendor_id": po["vendor_id"],
			"contract_covered": covered,
			"matching_contracts": [c["id"] for c in vendor_contracts],
			"checked_at": self._now(),
		}

	# ── Spend analytics ───────────────────────────────────────────────────────

	async def spend_analytics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate spend by vendor and category."""
		tenant = self._tenant(tenant_id)
		pos = [p for p in self.purchase_orders.values() if p["tenant_id"] == tenant]
		total_spend = sum(p["total_value"] for p in pos)
		by_vendor: dict[str, float] = {}
		by_status: dict[str, int] = {}
		for p in pos:
			by_vendor[p["vendor_id"]] = round(by_vendor.get(p["vendor_id"], 0.0) + p["total_value"], 4)
			by_status[p["status"]] = by_status.get(p["status"], 0) + 1
		top_vendors = sorted(by_vendor.items(), key=lambda x: x[1], reverse=True)[:5]
		return {
			"tenant_id": tenant,
			"total_spend": round(total_spend, 2),
			"total_pos": len(pos),
			"by_status": by_status,
			"top_vendors": [{"vendor_id": v, "spend": s} for v, s in top_vendors],
			"active_contracts": sum(1 for c in self.contracts.values() if c["tenant_id"] == tenant and c["status"] == "active"),
			"generated_at": self._now(),
		}

	async def procurement_dashboard(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return procurement KPI dashboard."""
		tenant = self._tenant(tenant_id)
		pos = [p for p in self.purchase_orders.values() if p["tenant_id"] == tenant]
		matches = [m for m in self.three_way_matches.values() if m["tenant_id"] == tenant]
		matched_count = sum(1 for m in matches if m["match_result"] == "matched")
		match_rate = round(matched_count / len(matches) * 100, 1) if matches else 0.0
		return {
			"tenant_id": tenant,
			"open_rfqs": sum(1 for r in self.rfqs.values() if r["tenant_id"] == tenant and r["status"] not in {"awarded", "cancelled"}),
			"open_pos": sum(1 for p in pos if p["status"] not in {"closed", "cancelled"}),
			"three_way_match_rate_pct": match_rate,
			"disputed_invoices": sum(1 for m in matches if m["match_result"] == "disputed"),
			"vendor_count": len({p["vendor_id"] for p in pos}),
			"generated_at": self._now(),
		}

	async def bulk_create_purchase_orders(
		self,
		orders_data: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Bulk-create multiple purchase orders."""
		tenant = self._tenant(tenant_id)
		tasks = [self.create_purchase_order(tenant_id=tenant, **o) for o in orders_data]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		results, errors = [], []
		for item in raw:
			if isinstance(item, Exception):
				errors.append(str(item))
			else:
				results.append(item)
		return {"created": len(results), "failed": len(errors), "purchase_orders": results, "errors": errors}
