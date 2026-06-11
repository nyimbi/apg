"""Procurement Management async service (scm_prc)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

CAPABILITY_ID = "scm_prc"
PO_STATUSES = {"draft", "sent", "acknowledged", "partially_received", "received", "invoiced", "closed", "cancelled"}
RFQ_STATUSES = {"draft", "issued", "responses_received", "awarded", "cancelled"}
MATCH_RESULTS = {"matched", "partial", "disputed"}
CONTRACT_ALERT_THRESHOLDS = (0.80, 0.95)  # fractions of contract value
DEFAULT_SLA_HOURS: dict[str, int] = {
	"po_acknowledgement": 48,
	"disputed_invoice_resolution": 120,
	"rfq_response": 168,
	"goods_receipt": 72,
}


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
		self.exchange_rates: dict[str, float] = {}  # "USD/KES": 130.5
		self.sla_config: dict[str, int] = dict(DEFAULT_SLA_HOURS)  # overridable per tenant
		self._rfq_seq: int = 5000
		self._po_seq: int = 8000
		self._audit_events: list[dict[str, Any]] = []
		self._audit_chain_hash: str = ""  # tamper-evident chain

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
		event: dict[str, Any] = {
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"status": status,
			"capability_id": CAPABILITY_ID,
			"emitted_at": self._now(),
			"prev_hash": self._audit_chain_hash,
		}
		event_bytes = json.dumps(event, sort_keys=True).encode()
		self._audit_chain_hash = hashlib.sha256(event_bytes).hexdigest()
		event["hash"] = self._audit_chain_hash
		self._audit_events.append(event)

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

	# ── Contract spend-down tracking ──────────────────────────────────────────

	async def get_contract_spend_status(
		self,
		contract_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return consumed vs ceiling value for a contract and emit alerts at 80%/95%.

		Aggregates all non-cancelled POs raised against vendors covered by the contract
		that were created after the contract start date and before the contract end date.
		"""
		tenant = self._tenant(tenant_id)
		contract = self.contracts.get(contract_id)
		if not contract or contract["tenant_id"] != tenant:
			raise KeyError(f"contract '{contract_id}' not found")

		consumed = sum(
			p["total_value"]
			for p in self.purchase_orders.values()
			if p["tenant_id"] == tenant
			and p["vendor_id"] == contract["vendor_id"]
			and p["status"] not in {"cancelled"}
			and p["created_at"] >= contract["start_date"]
			and p["created_at"] <= contract["end_date"]
		)
		ceiling = contract["value"]
		utilisation = round(consumed / ceiling, 4) if ceiling else 0.0
		alert_level: str | None = None
		for threshold in sorted(CONTRACT_ALERT_THRESHOLDS, reverse=True):
			if utilisation >= threshold:
				alert_level = f"{int(threshold * 100)}pct"
				self._emit(
					tenant,
					f"contract_nearing_limit_{alert_level}",
					contract_id,
					"scm_prc_contract",
					"active",
				)
				break

		return {
			"contract_id": contract_id,
			"vendor_id": contract["vendor_id"],
			"ceiling": ceiling,
			"currency": contract["currency"],
			"consumed": round(consumed, 4),
			"remaining": round(ceiling - consumed, 4),
			"utilisation_pct": round(utilisation * 100, 2),
			"alert_level": alert_level,
			"checked_at": self._now(),
		}

	# ── RFQ comparative scoring ───────────────────────────────────────────────

	async def score_rfq_responses(
		self,
		rfq_id: str,
		weights: dict[str, float] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Produce a weighted scorecard across all vendor responses to an RFQ.

		Default weights: price=0.50, lead_time=0.25, quality=0.15, sustainability=0.10.
		Each response must carry the corresponding keys in its ``quoted_lines`` or at
		the top-level response dict (callers can set ``lead_time_days``, ``quality_score``,
		``sustainability_score`` when recording responses).
		Returns ranked recommendations with scores and an auditable rationale.
		"""
		tenant = self._tenant(tenant_id)
		rfq = self.rfqs.get(rfq_id)
		if not rfq or rfq["tenant_id"] != tenant:
			raise KeyError(f"rfq '{rfq_id}' not found")

		w = {"price": 0.50, "lead_time": 0.25, "quality": 0.15, "sustainability": 0.10}
		if weights:
			w.update(weights)
		total_w = sum(w.values())
		if abs(total_w - 1.0) > 1e-6:
			raise ValueError(f"weights must sum to 1.0 (got {total_w})")

		responses = [
			deepcopy(r)
			for r in self.rfq_responses.values()
			if r["tenant_id"] == tenant and r["rfq_id"] == rfq_id
		]
		if not responses:
			return {"rfq_id": rfq_id, "ranked": [], "weights": w, "scored_at": self._now()}

		# normalise price (lower is better → invert)
		prices = [r["total_quoted_amount"] for r in responses]
		min_price, max_price = min(prices), max(prices)
		price_range = (max_price - min_price) or 1.0

		scored = []
		for r in responses:
			price_norm = 1.0 - (r["total_quoted_amount"] - min_price) / price_range  # 1 = best price
			lead_time_days = float(r.get("lead_time_days", 14))
			lead_norm = max(0.0, 1.0 - lead_time_days / 60.0)  # cap at 60 days
			quality_norm = float(r.get("quality_score", 5.0)) / 10.0
			sustainability_norm = float(r.get("sustainability_score", 5.0)) / 10.0

			composite = round(
				w["price"] * price_norm
				+ w["lead_time"] * lead_norm
				+ w["quality"] * quality_norm
				+ w["sustainability"] * sustainability_norm,
				4,
			)
			scored.append({
				"response_id": r["id"],
				"vendor_id": r["vendor_id"],
				"total_quoted_amount": r["total_quoted_amount"],
				"currency": r["currency"],
				"lead_time_days": lead_time_days,
				"quality_score": quality_norm * 10,
				"sustainability_score": sustainability_norm * 10,
				"composite_score": composite,
			})

		ranked = sorted(scored, key=lambda x: x["composite_score"], reverse=True)
		for i, entry in enumerate(ranked):
			entry["rank"] = i + 1

		return {
			"rfq_id": rfq_id,
			"weights": w,
			"ranked": ranked,
			"recommended_vendor": ranked[0]["vendor_id"] if ranked else None,
			"scored_at": self._now(),
		}

	# ── Multi-currency normalisation ──────────────────────────────────────────

	async def set_exchange_rate(
		self,
		from_currency: str,
		to_currency: str,
		rate: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Store a dated exchange rate for multi-currency spend normalisation.

		Rates are stored as ``FROM/TO`` keys (e.g. ``"EUR/USD": 1.08``).
		A reverse rate is derived automatically.
		"""
		tenant = self._tenant(tenant_id)
		if rate <= 0:
			raise ValueError("exchange rate must be positive")
		key = f"{from_currency.upper()}/{to_currency.upper()}"
		rev_key = f"{to_currency.upper()}/{from_currency.upper()}"
		self.exchange_rates[key] = round(rate, 6)
		self.exchange_rates[rev_key] = round(1.0 / rate, 6)
		self._emit(tenant, "exchange_rate_updated", key, "scm_prc_exchange_rate", "active")
		return {"key": key, "rate": self.exchange_rates[key], "reverse_key": rev_key, "reverse_rate": self.exchange_rates[rev_key], "updated_at": self._now()}

	async def normalised_spend_analytics(
		self,
		reporting_currency: str = "USD",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Aggregate spend normalised to a single reporting currency.

		For each PO, converts ``total_value`` from its transaction currency to
		``reporting_currency`` using stored exchange rates.  POs whose currency
		has no stored rate are included in a ``unconverted`` bucket with the
		original amount and currency noted.
		"""
		tenant = self._tenant(tenant_id)
		reporting_currency = reporting_currency.upper()
		by_vendor: dict[str, float] = {}
		unconverted: list[dict[str, Any]] = []
		total_normalised = 0.0

		for p in self.purchase_orders.values():
			if p["tenant_id"] != tenant or p["status"] == "cancelled":
				continue
			tx_currency = p.get("currency", "USD").upper()
			value = p["total_value"]
			if tx_currency == reporting_currency:
				normalised = value
			else:
				rate_key = f"{tx_currency}/{reporting_currency}"
				rate = self.exchange_rates.get(rate_key)
				if rate is None:
					unconverted.append({"po_id": p["id"], "currency": tx_currency, "amount": value})
					continue
				normalised = round(value * rate, 4)
			total_normalised += normalised
			by_vendor[p["vendor_id"]] = round(by_vendor.get(p["vendor_id"], 0.0) + normalised, 4)

		top_vendors = sorted(by_vendor.items(), key=lambda x: x[1], reverse=True)[:5]
		return {
			"tenant_id": tenant,
			"reporting_currency": reporting_currency,
			"total_normalised_spend": round(total_normalised, 2),
			"by_vendor": by_vendor,
			"top_vendors": [{"vendor_id": v, "spend": s} for v, s in top_vendors],
			"unconverted_pos": unconverted,
			"generated_at": self._now(),
		}

	# ── SLA monitoring ────────────────────────────────────────────────────────

	async def configure_sla(
		self,
		sla_config: dict[str, int],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Override default SLA hours for this service instance.

		Valid keys: ``po_acknowledgement``, ``disputed_invoice_resolution``,
		``rfq_response``, ``goods_receipt``.  Values are hours (int > 0).
		"""
		tenant = self._tenant(tenant_id)
		valid_keys = set(DEFAULT_SLA_HOURS.keys())
		for k, v in sla_config.items():
			if k not in valid_keys:
				raise ValueError(f"unknown SLA key '{k}'; valid: {sorted(valid_keys)}")
			if not isinstance(v, int) or v <= 0:
				raise ValueError(f"SLA value for '{k}' must be a positive integer (hours)")
			self.sla_config[k] = v
		self._emit(tenant, "sla_config_updated", "sla", "scm_prc_sla", "active")
		return {"sla_config": dict(self.sla_config), "updated_at": self._now()}

	async def check_sla_breaches(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Scan all open documents for SLA breaches and near-misses.

		Returns per-document status: ``ok``, ``warning`` (>75% of SLA elapsed),
		or ``breached`` (elapsed > SLA).  Only evaluates documents in transitional
		states where a response is still awaited.
		"""
		tenant = self._tenant(tenant_id)
		now_dt = datetime.now(timezone.utc)

		def _elapsed_hours(ts: str | None) -> float:
			if not ts:
				return 0.0
			try:
				dt = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
				return (now_dt - dt).total_seconds() / 3600
			except Exception:
				return 0.0

		def _classify(elapsed: float, sla_hours: int) -> str:
			ratio = elapsed / sla_hours if sla_hours else 0.0
			if ratio > 1.0:
				return "breached"
			if ratio > 0.75:
				return "warning"
			return "ok"

		breaches: list[dict[str, Any]] = []

		# POs waiting for acknowledgement
		for po in self.purchase_orders.values():
			if po["tenant_id"] != tenant or po["status"] != "sent":
				continue
			elapsed = _elapsed_hours(po.get("sent_at"))
			sla = self.sla_config["po_acknowledgement"]
			status = _classify(elapsed, sla)
			if status != "ok":
				breaches.append({
					"type": "po_acknowledgement",
					"record_id": po["id"],
					"po_number": po.get("po_number"),
					"vendor_id": po["vendor_id"],
					"elapsed_hours": round(elapsed, 1),
					"sla_hours": sla,
					"status": status,
				})

		# Disputed three-way matches waiting for resolution
		for m in self.three_way_matches.values():
			if m["tenant_id"] != tenant or m["status"] not in {"pending"}:
				continue
			elapsed = _elapsed_hours(m.get("created_at"))
			sla = self.sla_config["disputed_invoice_resolution"]
			status = _classify(elapsed, sla)
			if status != "ok":
				breaches.append({
					"type": "disputed_invoice_resolution",
					"record_id": m["id"],
					"invoice_number": m.get("invoice_number"),
					"match_result": m.get("match_result"),
					"elapsed_hours": round(elapsed, 1),
					"sla_hours": sla,
					"status": status,
				})

		# Issued RFQs with no responses yet
		for rfq in self.rfqs.values():
			if rfq["tenant_id"] != tenant or rfq["status"] != "issued":
				continue
			elapsed = _elapsed_hours(rfq.get("issued_at"))
			sla = self.sla_config["rfq_response"]
			status = _classify(elapsed, sla)
			if status != "ok":
				breaches.append({
					"type": "rfq_response",
					"record_id": rfq["id"],
					"rfq_number": rfq.get("rfq_number"),
					"elapsed_hours": round(elapsed, 1),
					"sla_hours": sla,
					"status": status,
				})

		return {
			"tenant_id": tenant,
			"total_issues": len(breaches),
			"breached": [b for b in breaches if b["status"] == "breached"],
			"warnings": [b for b in breaches if b["status"] == "warning"],
			"checked_at": self._now(),
		}

	# ── Tamper-evident audit verification ─────────────────────────────────────

	async def verify_audit_chain(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Verify the SHA-256 chain integrity of all audit events for a tenant.

		Recomputes each event's hash from its content and the previous event's hash.
		Returns ``valid=True`` only if every link in the chain is intact, making
		unauthorised event deletion or modification detectable.
		"""
		tenant = self._tenant(tenant_id)
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		if not events:
			return {"valid": True, "event_count": 0, "checked_at": self._now()}

		prev_hash = ""
		broken_at: int | None = None
		for i, event in enumerate(events):
			check_event = {k: v for k, v in event.items() if k != "hash"}
			check_event["prev_hash"] = prev_hash
			computed = hashlib.sha256(json.dumps(check_event, sort_keys=True).encode()).hexdigest()
			if computed != event.get("hash"):
				broken_at = i
				break
			prev_hash = computed

		return {
			"valid": broken_at is None,
			"event_count": len(events),
			"broken_at_index": broken_at,
			"checked_at": self._now(),
		}

	# ── Delivery schedule tracking ────────────────────────────────────────────

	async def set_po_delivery_schedule(
		self,
		po_id: str,
		schedule: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Attach a delivery schedule to a PO.

		``schedule`` is a list of milestone dicts, each with keys:
		  - ``sku``: str
		  - ``expected_date``: ISO date string
		  - ``expected_quantity``: float
		  - ``actual_date``: str | None (populated on receipt)
		  - ``actual_quantity``: float | None
		"""
		tenant = self._tenant(tenant_id)
		po = self.purchase_orders.get(po_id)
		if not po or po["tenant_id"] != tenant:
			raise KeyError(f"po '{po_id}' not found")
		for milestone in schedule:
			if "sku" not in milestone or "expected_date" not in milestone:
				raise ValueError("each schedule milestone requires 'sku' and 'expected_date'")
		po["delivery_schedule"] = deepcopy(schedule)
		po["updated_at"] = self._now()
		self._emit(tenant, "po_delivery_schedule_set", po_id, "scm_prc_purchase_order", po["status"])
		return deepcopy(po)

	async def get_delivery_performance(
		self,
		tenant_id: str | None = None,
		vendor_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute on-time delivery rate per vendor from PO delivery schedules.

		A milestone is 'on time' when ``actual_date <= expected_date`` and
		``actual_quantity >= expected_quantity``.  Milestones without an
		``actual_date`` are counted as pending (not included in rate calculation).
		"""
		tenant = self._tenant(tenant_id)
		vendor_stats: dict[str, dict[str, int]] = {}

		for po in self.purchase_orders.values():
			if po["tenant_id"] != tenant:
				continue
			if vendor_id and po["vendor_id"] != vendor_id:
				continue
			vnd = po["vendor_id"]
			if vnd not in vendor_stats:
				vendor_stats[vnd] = {"on_time": 0, "late": 0, "pending": 0}

			for milestone in po.get("delivery_schedule", []):
				actual_date = milestone.get("actual_date")
				expected_date = milestone.get("expected_date", "")
				actual_qty = float(milestone.get("actual_quantity") or 0.0)
				expected_qty = float(milestone.get("expected_quantity", 0.0))
				if not actual_date:
					vendor_stats[vnd]["pending"] += 1
					continue
				if actual_date <= expected_date and actual_qty >= expected_qty:
					vendor_stats[vnd]["on_time"] += 1
				else:
					vendor_stats[vnd]["late"] += 1

		result_vendors = []
		for vnd, stats in vendor_stats.items():
			completed = stats["on_time"] + stats["late"]
			otr = round(stats["on_time"] / completed * 100, 1) if completed else None
			result_vendors.append({
				"vendor_id": vnd,
				"on_time": stats["on_time"],
				"late": stats["late"],
				"pending": stats["pending"],
				"on_time_rate_pct": otr,
			})

		result_vendors.sort(key=lambda x: (x["on_time_rate_pct"] or -1), reverse=True)
		return {
			"tenant_id": tenant,
			"vendors": result_vendors,
			"generated_at": self._now(),
		}

	# ── Process cycle time analytics ──────────────────────────────────────────

	async def procurement_cycle_times(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compute mean cycle times across key procurement process steps.

		Measures:
		  - ``rfq_to_award_hours``: RFQ created_at → awarded_at
		  - ``po_draft_to_sent_hours``: PO created_at → sent_at
		  - ``po_sent_to_acknowledged_hours``: sent_at → acknowledged_at
		  - ``po_sent_to_received_hours``: sent_at → latest receipt created_at

		Returns mean, min, max per step to surface bottlenecks.
		"""
		tenant = self._tenant(tenant_id)

		def _hours_between(start: str | None, end: str | None) -> float | None:
			if not start or not end:
				return None
			try:
				s = datetime.fromisoformat(start.rstrip("Z")).replace(tzinfo=timezone.utc)
				e = datetime.fromisoformat(end.rstrip("Z")).replace(tzinfo=timezone.utc)
				return round((e - s).total_seconds() / 3600, 2)
			except Exception:
				return None

		def _stats(values: list[float]) -> dict[str, float | None]:
			if not values:
				return {"mean": None, "min": None, "max": None, "count": 0}
			return {
				"mean": round(sum(values) / len(values), 2),
				"min": round(min(values), 2),
				"max": round(max(values), 2),
				"count": len(values),
			}

		rfq_to_award: list[float] = []
		for r in self.rfqs.values():
			if r["tenant_id"] != tenant:
				continue
			h = _hours_between(r.get("created_at"), r.get("awarded_at"))
			if h is not None:
				rfq_to_award.append(h)

		po_draft_to_sent: list[float] = []
		po_sent_to_ack: list[float] = []
		po_sent_to_received: list[float] = []

		# build receipt lookup: po_id → earliest receipt timestamp
		receipt_by_po: dict[str, str] = {}
		for rcpt in self.receipts.values():
			if rcpt["tenant_id"] != tenant:
				continue
			pid = rcpt["po_id"]
			existing = receipt_by_po.get(pid)
			if existing is None or rcpt["received_at"] < existing:
				receipt_by_po[pid] = rcpt["received_at"]

		for po in self.purchase_orders.values():
			if po["tenant_id"] != tenant:
				continue
			h1 = _hours_between(po.get("created_at"), po.get("sent_at"))
			if h1 is not None:
				po_draft_to_sent.append(h1)
			h2 = _hours_between(po.get("sent_at"), po.get("acknowledged_at"))
			if h2 is not None:
				po_sent_to_ack.append(h2)
			h3 = _hours_between(po.get("sent_at"), receipt_by_po.get(po["id"]))
			if h3 is not None:
				po_sent_to_received.append(h3)

		return {
			"tenant_id": tenant,
			"rfq_to_award_hours": _stats(rfq_to_award),
			"po_draft_to_sent_hours": _stats(po_draft_to_sent),
			"po_sent_to_acknowledged_hours": _stats(po_sent_to_ack),
			"po_sent_to_received_hours": _stats(po_sent_to_received),
			"generated_at": self._now(),
		}
