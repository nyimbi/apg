"""Premium & Billing Service (ins_prm).

Premium calculation, instalment management, collections, reconciliation, refunds.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

SUPPORTED_FREQUENCIES = {"annual", "semi_annual", "quarterly", "monthly"}
SUPPORTED_PAYMENT_METHODS = {"mpesa", "bank_transfer", "card", "cash", "cheque", "direct_debit", "bancassurance"}
INSTALMENT_MAP = {"annual": 1, "semi_annual": 2, "quarterly": 4, "monthly": 12}


class PremiumBillingService:
	"""In-memory executable service for Premium & Billing."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.schedules: dict[str, dict[str, Any]] = {}
		self.instalments: dict[str, dict[str, Any]] = {}
		self.collections: dict[str, dict[str, Any]] = {}
		self.refunds: dict[str, dict[str, Any]] = {}
		self.reconciliations: dict[str, dict[str, Any]] = {}
		self.debit_orders: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

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

	def _generate_instalment_dates(self, inception_date: str, frequency: str, count: int) -> list[str]:
		"""Generate due dates for instalments."""
		start = date.fromisoformat(inception_date)
		dates = []
		if frequency == "annual":
			dates = [start.isoformat()]
		elif frequency == "semi_annual":
			dates = [start.isoformat(), (start + timedelta(days=182)).isoformat()]
		elif frequency == "quarterly":
			dates = [(start + timedelta(days=91 * i)).isoformat() for i in range(count)]
		elif frequency == "monthly":
			dates = [(start + timedelta(days=30 * i)).isoformat() for i in range(count)]
		return dates[:count]

	# ── Premium Schedules ─────────────────────────────────────────────────────

	async def create_schedule(
		self,
		tenant_id: str,
		policy_id: str,
		policy_number: str,
		total_premium: Decimal,
		frequency: str,
		inception_date: str,
		expiry_date: str,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Create a premium payment schedule with instalments."""
		tenant = self._tenant(tenant_id)
		if frequency not in SUPPORTED_FREQUENCIES:
			raise ValueError(f"unsupported_frequency:{frequency}")
		count = INSTALMENT_MAP[frequency]
		per_instalment = (Decimal(str(total_premium)) / count).quantize(Decimal("0.01"))
		# Last instalment absorbs rounding
		remainder = Decimal(str(total_premium)) - per_instalment * count
		due_dates = self._generate_instalment_dates(inception_date, frequency, count)
		record: dict[str, Any] = {
			"id": self._record_id("sch"),
			"type": "prm_schedule",
			"policy_id": policy_id,
			"policy_number": policy_number,
			"total_premium": Decimal(str(total_premium)),
			"frequency": frequency,
			"instalment_count": count,
			"currency": currency,
			"inception_date": inception_date,
			"expiry_date": expiry_date,
			"collected_amount": Decimal("0"),
			"outstanding_amount": Decimal(str(total_premium)),
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.schedules[record["id"]] = record
		# Create instalment records
		for i, due_date in enumerate(due_dates):
			amount = per_instalment + (remainder if i == count - 1 else Decimal("0"))
			inst: dict[str, Any] = {
				"id": self._record_id("inst"),
				"type": "prm_instalment",
				"schedule_id": record["id"],
				"policy_id": policy_id,
				"instalment_number": i + 1,
				"due_date": due_date,
				"amount": amount,
				"currency": currency,
				"status": "pending",
				"paid_at": None,
				"tenant_id": tenant,
				"created_at": self._now(),
			}
			self.instalments[inst["id"]] = inst
		self._emit(tenant, "premium_schedule_created", record["id"], "prm_schedule", {"policy_id": policy_id, "total": str(total_premium)})
		_log.info("Premium schedule created: policy=%s freq=%s tenant=%s", policy_number, frequency, tenant)
		return deepcopy(record)

	async def get_schedule(self, tenant_id: str, schedule_id: str) -> dict[str, Any]:
		"""Retrieve a premium schedule."""
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		return deepcopy(sch)

	async def list_schedules(self, tenant_id: str, policy_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List premium schedules."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.schedules.values() if s["tenant_id"] == tenant]
		if policy_id:
			items = [s for s in items if s["policy_id"] == policy_id]
		if status:
			items = [s for s in items if s["status"] == status]
		return items

	async def update_schedule(self, tenant_id: str, schedule_id: str, updates: dict[str, Any]) -> dict[str, Any]:
		"""Update schedule fields."""
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		allowed = {"status", "metadata"}
		for k, v in updates.items():
			if k in allowed:
				sch[k] = v
		self._emit(tenant, "schedule_updated", schedule_id, "prm_schedule", {})
		return deepcopy(sch)

	async def delete_schedule(self, tenant_id: str, schedule_id: str) -> dict[str, Any]:
		"""Cancel a premium schedule."""
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")
		if sch["collected_amount"] > 0:
			raise PermissionError("cannot_delete_schedule_with_collections")
		sch["status"] = "cancelled"
		sch["cancelled_at"] = self._now()
		self._emit(tenant, "schedule_cancelled", schedule_id, "prm_schedule", {})
		return deepcopy(sch)

	# ── Instalments ───────────────────────────────────────────────────────────

	async def list_instalments(self, tenant_id: str, schedule_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List instalments."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(i) for i in self.instalments.values() if i["tenant_id"] == tenant]
		if schedule_id:
			items = [i for i in items if i["schedule_id"] == schedule_id]
		if status:
			items = [i for i in items if i["status"] == status]
		return items

	async def get_instalment(self, tenant_id: str, instalment_id: str) -> dict[str, Any]:
		"""Retrieve an instalment."""
		tenant = self._tenant(tenant_id)
		inst = self.instalments.get(instalment_id)
		if not inst or inst["tenant_id"] != tenant:
			raise KeyError(f"instalment_not_found:{instalment_id}")
		return deepcopy(inst)

	async def list_overdue_instalments(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all overdue unpaid instalments."""
		tenant = self._tenant(tenant_id)
		today = date.today().isoformat()
		return [
			deepcopy(i) for i in self.instalments.values()
			if i["tenant_id"] == tenant and i["status"] == "pending" and i["due_date"] < today
		]

	# ── Collections ───────────────────────────────────────────────────────────

	async def collect_payment(
		self,
		tenant_id: str,
		instalment_id: str,
		payment_method: str,
		payment_reference: str,
		amount: Decimal,
		collected_by: str,
	) -> dict[str, Any]:
		"""Record a premium collection against an instalment."""
		tenant = self._tenant(tenant_id)
		inst = self.instalments.get(instalment_id)
		if not inst or inst["tenant_id"] != tenant:
			raise KeyError(f"instalment_not_found:{instalment_id}")
		if inst["status"] == "paid":
			raise PermissionError("instalment_already_paid")
		if payment_method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		collected = Decimal(str(amount))
		if collected <= 0:
			raise ValueError("amount_must_be_positive")
		record: dict[str, Any] = {
			"id": self._record_id("col"),
			"type": "prm_collection",
			"instalment_id": instalment_id,
			"schedule_id": inst["schedule_id"],
			"policy_id": inst["policy_id"],
			"payment_method": payment_method,
			"payment_reference": payment_reference,
			"amount": collected,
			"currency": inst["currency"],
			"collected_by": collected_by,
			"status": "received",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.collections[record["id"]] = record
		inst["status"] = "paid"
		inst["paid_at"] = self._now()
		inst["collection_id"] = record["id"]
		# Update schedule totals
		sch = self.schedules.get(inst["schedule_id"])
		if sch:
			sch["collected_amount"] = sch["collected_amount"] + collected
			sch["outstanding_amount"] = sch["outstanding_amount"] - collected
			if sch["outstanding_amount"] <= 0:
				sch["status"] = "fully_paid"
		self._emit(tenant, "premium_collected", record["id"], "prm_collection", {"instalment_id": instalment_id, "amount": str(collected)})
		return deepcopy(record)

	async def list_collections(self, tenant_id: str, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List collection records."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(c) for c in self.collections.values() if c["tenant_id"] == tenant]
		if policy_id:
			items = [c for c in items if c["policy_id"] == policy_id]
		return items

	# ── Refunds ───────────────────────────────────────────────────────────────

	async def process_refund(
		self,
		tenant_id: str,
		policy_id: str,
		refund_amount: Decimal,
		reason: str,
		payee_account: str,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Process a premium refund."""
		tenant = self._tenant(tenant_id)
		amount = Decimal(str(refund_amount))
		if amount <= 0:
			raise ValueError("refund_amount_must_be_positive")
		# Verify there is sufficient collected premium
		collected = sum(
			c["amount"] for c in self.collections.values()
			if c["tenant_id"] == tenant and c["policy_id"] == policy_id
		)
		refunded = sum(
			r["refund_amount"] for r in self.refunds.values()
			if r["tenant_id"] == tenant and r["policy_id"] == policy_id
		)
		available = collected - refunded
		if amount > available:
			raise ValueError(f"refund_exceeds_collected_premium:available={available}")
		record: dict[str, Any] = {
			"id": self._record_id("ref"),
			"type": "prm_refund",
			"policy_id": policy_id,
			"refund_amount": amount,
			"reason": reason,
			"payee_account": payee_account,
			"authorised_by": authorised_by,
			"status": "processed",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.refunds[record["id"]] = record
		self._emit(tenant, "premium_refunded", record["id"], "prm_refund", {"policy_id": policy_id, "amount": str(amount)})
		return deepcopy(record)

	async def list_refunds(self, tenant_id: str, policy_id: str | None = None) -> list[dict[str, Any]]:
		"""List refunds."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.refunds.values() if r["tenant_id"] == tenant]
		if policy_id:
			items = [r for r in items if r["policy_id"] == policy_id]
		return items

	# ── Reconciliation ────────────────────────────────────────────────────────

	async def reconcile_period(
		self,
		tenant_id: str,
		period_start: str,
		period_end: str,
		reconciled_by: str,
	) -> dict[str, Any]:
		"""Reconcile premium collections for a period."""
		tenant = self._tenant(tenant_id)
		period_collections = [
			c for c in self.collections.values()
			if c["tenant_id"] == tenant and period_start <= c["created_at"][:10] <= period_end
		]
		period_refunds = [
			r for r in self.refunds.values()
			if r["tenant_id"] == tenant and period_start <= r["created_at"][:10] <= period_end
		]
		total_collected = sum(c["amount"] for c in period_collections)
		total_refunded = sum(r["refund_amount"] for r in period_refunds)
		net = total_collected - total_refunded
		record: dict[str, Any] = {
			"id": self._record_id("recon"),
			"type": "prm_reconciliation",
			"period_start": period_start,
			"period_end": period_end,
			"total_collected": total_collected,
			"total_refunded": total_refunded,
			"net_premium": net,
			"collection_count": len(period_collections),
			"refund_count": len(period_refunds),
			"reconciled_by": reconciled_by,
			"status": "reconciled",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.reconciliations[record["id"]] = record
		self._emit(tenant, "period_reconciled", record["id"], "prm_reconciliation", {"period": f"{period_start}/{period_end}"})
		return deepcopy(record)

	async def list_reconciliations(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List reconciliation records."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.reconciliations.values() if r["tenant_id"] == tenant]

	# ── Debit Orders ──────────────────────────────────────────────────────────

	async def setup_debit_order(
		self,
		tenant_id: str,
		policy_id: str,
		schedule_id: str,
		bank_account: str,
		bank_code: str,
		collection_day: int,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Register a recurring debit order for automated premium collection."""
		tenant = self._tenant(tenant_id)
		if not (1 <= collection_day <= 28):
			raise ValueError("collection_day_must_be_between_1_and_28")
		record: dict[str, Any] = {
			"id": self._record_id("dbt"),
			"type": "prm_debit_order",
			"policy_id": policy_id,
			"schedule_id": schedule_id,
			"bank_account": bank_account,
			"bank_code": bank_code,
			"collection_day": collection_day,
			"authorised_by": authorised_by,
			"status": "active",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.debit_orders[record["id"]] = record
		self._emit(tenant, "debit_order_setup", record["id"], "prm_debit_order", {"policy_id": policy_id})
		return deepcopy(record)

	async def cancel_debit_order(self, tenant_id: str, debit_order_id: str, reason: str) -> dict[str, Any]:
		"""Cancel a debit order mandate."""
		tenant = self._tenant(tenant_id)
		do = self.debit_orders.get(debit_order_id)
		if not do or do["tenant_id"] != tenant:
			raise KeyError(f"debit_order_not_found:{debit_order_id}")
		do["status"] = "cancelled"
		do["cancellation_reason"] = reason
		do["cancelled_at"] = self._now()
		self._emit(tenant, "debit_order_cancelled", debit_order_id, "prm_debit_order", {})
		return deepcopy(do)

	async def list_debit_orders(self, tenant_id: str, active_only: bool = True) -> list[dict[str, Any]]:
		"""List debit order mandates."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.debit_orders.values() if d["tenant_id"] == tenant]
		if active_only:
			items = [d for d in items if d["status"] == "active"]
		return items

	# ── Premium Calculation ───────────────────────────────────────────────────

	async def calculate_premium(
		self,
		tenant_id: str,
		product_code: str,
		sum_insured: Decimal,
		base_rate: Decimal,
		loadings: dict[str, Decimal] | None = None,
		discounts: dict[str, Decimal] | None = None,
	) -> dict[str, Any]:
		"""Calculate gross premium with loadings and discounts."""
		tenant = self._tenant(tenant_id)
		si = Decimal(str(sum_insured))
		rate = Decimal(str(base_rate))
		loading_total = sum((Decimal(str(v)) for v in (loadings or {}).values()), Decimal("0"))
		discount_total = sum((Decimal(str(v)) for v in (discounts or {}).values()), Decimal("0"))
		net_rate = rate + loading_total - discount_total
		if net_rate < Decimal("0"):
			net_rate = Decimal("0")
		gross_premium = (si * net_rate).quantize(Decimal("0.01"))
		return {
			"product_code": product_code,
			"sum_insured": str(si),
			"base_rate": str(rate),
			"loading_total": str(loading_total),
			"discount_total": str(discount_total),
			"net_rate": str(net_rate),
			"gross_premium": str(gross_premium),
			"computed_at": self._now(),
		}

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def billing_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Summary of premium billing metrics."""
		tenant = self._tenant(tenant_id)
		scheds = [s for s in self.schedules.values() if s["tenant_id"] == tenant]
		insts = [i for i in self.instalments.values() if i["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"schedule_count": len(scheds),
			"total_billed": str(sum(s["total_premium"] for s in scheds)),
			"total_collected": str(sum(s["collected_amount"] for s in scheds)),
			"total_outstanding": str(sum(s["outstanding_amount"] for s in scheds)),
			"overdue_instalments": sum(1 for i in insts if i["status"] == "pending" and i["due_date"] < date.today().isoformat()),
			"collection_count": len([c for c in self.collections.values() if c["tenant_id"] == tenant]),
			"refund_count": len([r for r in self.refunds.values() if r["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ins_prm",
			"status": "healthy",
			"schedule_count": len(self.schedules),
			"instalment_count": len(self.instalments),
			"collection_count": len(self.collections),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"capability_id": "ins_prm",
			"name": "Premium & Billing",
			"version": "1.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_frequencies": list(SUPPORTED_FREQUENCIES),
			"supported_payment_methods": list(SUPPORTED_PAYMENT_METHODS),
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
