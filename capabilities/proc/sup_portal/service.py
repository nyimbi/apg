"""APG Supplier Self-Service Portal service."""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string
from .models import SpSupplierProfile, SpQuote, SpInvoice, SpDeliveryConfirmation, SpDispute

_log = logging.getLogger(__name__)

DISPUTE_SLA_DAYS = 5


class SpService:
	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._suppliers: dict[str, SpSupplierProfile] = {}
		self._quotes: dict[str, SpQuote] = {}
		self._invoices: dict[str, SpInvoice] = {}
		self._deliveries: dict[str, SpDeliveryConfirmation] = {}
		self._disputes: dict[str, SpDispute] = {}

	async def register_supplier(self, supplier_id: str, company_name: str, contact_name: str, email: str, tenant_id: str | None = None) -> SpSupplierProfile:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		guard_non_empty_string(company_name, "company_name")
		profile = SpSupplierProfile(tenant_id=tid, supplier_id=supplier_id, company_name=company_name, contact_name=contact_name, email=email)
		self._suppliers[f"{tid}:{supplier_id}"] = profile
		_log.info("Supplier %s registered: %s", supplier_id, company_name)
		return profile

	async def submit_quote(self, supplier_id: str, rfq_id: str, line_items: list[dict], total_amount: float, currency: str = "KES", tenant_id: str | None = None) -> SpQuote:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		quote = SpQuote(tenant_id=tid, supplier_id=supplier_id, rfq_id=rfq_id, line_items=line_items, total_amount=total_amount, currency=currency)
		self._quotes[quote.id] = quote
		_log.info("Quote %s submitted by supplier %s for RFQ %s", quote.id, supplier_id, rfq_id)
		return quote

	async def submit_invoice(self, supplier_id: str, po_number: str, invoice_number: str, line_items: list[dict], total_amount: float, currency: str = "KES", tenant_id: str | None = None) -> SpInvoice:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		invoice = SpInvoice(tenant_id=tid, supplier_id=supplier_id, po_number=po_number, invoice_number=invoice_number, line_items=line_items, total_amount=total_amount, currency=currency)
		self._invoices[invoice.id] = invoice
		_log.info("Invoice %s submitted by supplier %s for PO %s", invoice.id, supplier_id, po_number)
		return invoice

	async def confirm_delivery(self, supplier_id: str, po_number: str, delivery_date: datetime, items_delivered: list[dict], delivery_note: str = "", tenant_id: str | None = None) -> SpDeliveryConfirmation:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		confirmation = SpDeliveryConfirmation(tenant_id=tid, supplier_id=supplier_id, po_number=po_number, delivery_date=delivery_date, items_delivered=items_delivered, delivery_note=delivery_note)
		self._deliveries[confirmation.id] = confirmation
		_log.info("Delivery confirmed for PO %s by supplier %s", po_number, supplier_id)
		return confirmation

	async def raise_dispute(self, supplier_id: str, reference_id: str, reference_type: str, dispute_reason: str, details: str = "", tenant_id: str | None = None) -> SpDispute:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		dispute = SpDispute(
			tenant_id=tid, supplier_id=supplier_id,
			reference_id=reference_id, reference_type=reference_type,
			dispute_reason=dispute_reason, details=details,
			sla_due_at=datetime.now(timezone.utc) + timedelta(days=DISPUTE_SLA_DAYS),
		)
		self._disputes[dispute.id] = dispute
		_log.info("Dispute %s raised by supplier %s", dispute.id, supplier_id)
		return dispute

	async def resolve_dispute(self, dispute_id: str, resolution: str, tenant_id: str | None = None) -> SpDispute:
		dispute = self._disputes.get(dispute_id)
		assert dispute is not None, f"Dispute {dispute_id} not found"
		dispute.resolution = resolution
		dispute.status = "resolved"
		dispute.resolved_at = datetime.now(timezone.utc)
		return dispute

	async def get_supplier_dashboard(self, supplier_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tid = tenant_id or self._tenant_id
		invoices = [i for i in self._invoices.values() if i.tenant_id == tid and i.supplier_id == supplier_id]
		disputes = [d for d in self._disputes.values() if d.tenant_id == tid and d.supplier_id == supplier_id and d.status == "open"]
		return {
			"supplier_id": supplier_id,
			"pending_invoices": sum(1 for i in invoices if i.payment_status == "unpaid"),
			"total_invoiced": sum(i.total_amount for i in invoices),
			"open_disputes": len(disputes),
			"overdue_disputes": sum(1 for d in disputes if d.sla_due_at and d.sla_due_at < datetime.now(timezone.utc)),
		}
