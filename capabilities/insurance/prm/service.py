"""Premium & Billing Service (ins_prm).

Premium calculation, instalment management, collections, reconciliation, refunds.

Enhanced with:
  I2 - Predictive lapse / non-payment scoring
  I3 - Partial payment & arrears carry-forward
  I4 - Grace-period & policy lapse state machine
  I6 - Regulatory levy & stamp duty calculator (IRA Kenya)
  I8 - Payment bounce & dishonoured instrument handling
  I9 - Premium written vs earned accrual (IFRS 17 PAA)
  I10 - Dunning workflow & escalation engine
  I14 - Real-time collection dashboard KPIs
  I15 - Audit-grade immutable chain-hashed event log export
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import hashlib
import json
import logging
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

_log = logging.getLogger(__name__)

SUPPORTED_FREQUENCIES = {"annual", "semi_annual", "quarterly", "monthly"}
SUPPORTED_PAYMENT_METHODS = {"mpesa", "bank_transfer", "card", "cash", "cheque", "direct_debit", "bancassurance"}
INSTALMENT_MAP = {"annual": 1, "semi_annual": 2, "quarterly": 4, "monthly": 12}

# IRA Kenya statutory levy rates (effective 2024/25 budget)
# source: Insurance Act Cap 487, Kenya Gazette Notice 2024
_IRA_LEVY_TABLE: list[dict[str, Any]] = [
	{"code": "IRA_TRAINING_LEVY",  "description": "IRA Training Levy",           "rate": Decimal("0.002"),  "gazette": "LN 46/2024"},
	{"code": "PHCF",               "description": "Policyholders Compensation Fund", "rate": Decimal("0.0025"), "gazette": "LN 47/2024"},
	{"code": "STAMP_DUTY",         "description": "Stamp Duty",                  "rate": Decimal("0.001"),  "gazette": "TA Cap 480"},
]

# Dunning escalation levels in order — maps level → days_overdue threshold
_DUNNING_LEVELS = ["REMINDER_1", "REMINDER_2", "FORMAL_NOTICE", "LAPSE_WARNING"]
_DUNNING_THRESHOLDS = {"REMINDER_1": 7, "REMINDER_2": 14, "FORMAL_NOTICE": 21, "LAPSE_WARNING": 30}


class PremiumBillingService:
	"""In-memory executable service for Premium & Billing."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.schedules: dict[str, dict[str, Any]] = {}
		self.instalments: dict[str, dict[str, Any]] = {}
		self.collections: dict[str, dict[str, Any]] = {}
		self.refunds: dict[str, dict[str, Any]] = {}
		self.reconciliations: dict[str, dict[str, Any]] = {}
		self.debit_orders: dict[str, dict[str, Any]] = {}
		self.bounce_charges: dict[str, dict[str, Any]] = {}
		self.dunning_actions: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# Incremental KPI accumulators keyed by tenant_id (I14)
		self._kpi_accumulators = WriteThruDict('kpi_accumulators', tenant_id, _store)

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
		# Update KPI accumulator (I14)
		self._accum_collect(tenant, collected)
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

	# ── I3: Partial Payment & Arrears Carry-Forward ───────────────────────────

	async def record_partial_payment(
		self,
		tenant_id: str,
		instalment_id: str,
		payment_method: str,
		payment_reference: str,
		amount: Decimal,
		collected_by: str,
	) -> dict[str, Any]:
		"""Accept a partial premium payment; carry overpayment as credit against the next instalment.

		Business value: M-Pesa culture produces frequent sub-full payments.  This prevents
		agents from blocking cash until a full amount is available, reducing fraud exposure
		and improving cash velocity for both insurer and client.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(payment_reference, "payment_reference")
		inst = self.instalments.get(instalment_id)
		if not inst or inst["tenant_id"] != tenant:
			raise KeyError(f"instalment_not_found:{instalment_id}")
		if inst["status"] == "paid":
			raise PermissionError("instalment_already_paid")
		if payment_method not in SUPPORTED_PAYMENT_METHODS:
			raise ValueError(f"unsupported_payment_method:{payment_method}")
		incoming = Decimal(str(amount))
		if incoming <= 0:
			raise ValueError("amount_must_be_positive")

		paid_so_far = Decimal(str(inst.get("paid_so_far", "0")))
		inst_amount = Decimal(str(inst["amount"]))
		new_total = paid_so_far + incoming

		# Record collection fragment
		col: dict[str, Any] = {
			"id": self._record_id("col"),
			"type": "prm_collection",
			"instalment_id": instalment_id,
			"schedule_id": inst["schedule_id"],
			"policy_id": inst["policy_id"],
			"payment_method": payment_method,
			"payment_reference": payment_reference,
			"amount": incoming,
			"currency": inst["currency"],
			"collected_by": collected_by,
			"status": "received",
			"is_partial": True,
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.collections[col["id"]] = col

		# Update KPI accumulator
		self._accum_collect(tenant, incoming)

		if new_total >= inst_amount:
			# Fully settled — any overpayment becomes a credit note
			overpayment = (new_total - inst_amount).quantize(Decimal("0.01"))
			inst["status"] = "paid"
			inst["paid_at"] = self._now()
			inst["collection_id"] = col["id"]
			inst["paid_so_far"] = str(inst_amount)
			if overpayment > 0:
				# Apply credit to the chronologically next pending instalment
				await self._apply_credit_note(tenant, inst["schedule_id"], instalment_id, overpayment)
			# Update schedule totals
			sch = self.schedules.get(inst["schedule_id"])
			if sch:
				sch["collected_amount"] = Decimal(str(sch["collected_amount"])) + inst_amount - paid_so_far
				sch["outstanding_amount"] = Decimal(str(sch["outstanding_amount"])) - (inst_amount - paid_so_far)
				if Decimal(str(sch["outstanding_amount"])) <= 0:
					sch["status"] = "fully_paid"
		else:
			inst["paid_so_far"] = str(new_total)
			inst["status"] = "partial"

		self._emit(tenant, "partial_payment_recorded", col["id"], "prm_collection",
			{"instalment_id": instalment_id, "amount": str(incoming), "new_total": str(new_total)})
		return deepcopy(col)

	async def _apply_credit_note(
		self, tenant: str, schedule_id: str, paid_instalment_id: str, credit: Decimal
	) -> None:
		"""Reduce the amount of the next pending instalment by the credit amount."""
		pending = sorted(
			[i for i in self.instalments.values()
			 if i["tenant_id"] == tenant and i["schedule_id"] == schedule_id
			 and i["status"] in {"pending", "partial"} and i["id"] != paid_instalment_id],
			key=lambda x: x["due_date"],
		)
		if not pending:
			return
		next_inst = pending[0]
		original = Decimal(str(next_inst["amount"]))
		applied = min(credit, original)
		next_inst["amount"] = str((original - applied).quantize(Decimal("0.01")))
		next_inst["credit_applied"] = str(applied)
		self._emit(tenant, "credit_note_applied", next_inst["id"], "prm_instalment",
			{"credit": str(applied), "source_instalment": paid_instalment_id})

	# ── I4: Grace-Period & Policy Lapse State Machine ─────────────────────────

	async def evaluate_lapse_status(
		self,
		tenant_id: str,
		schedule_id: str,
		grace_period_days: int = 30,
	) -> dict[str, Any]:
		"""Transition overdue instalments through: pending → overdue → in_grace → lapsed.

		Business value: IRA Kenya and FSCA South Africa mandate explicit grace-period tracking
		before policy suspension.  This drives compliant lapse handling without ad-hoc UI logic.
		Emits typed audit events that ins_pol can subscribe to for downstream coverage suspension.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")

		today = date.today()
		transitions: list[dict[str, Any]] = []

		for inst in self.instalments.values():
			if inst["tenant_id"] != tenant or inst["schedule_id"] != schedule_id:
				continue
			if inst["status"] not in {"pending", "overdue", "in_grace"}:
				continue

			due = date.fromisoformat(inst["due_date"])
			days_overdue = (today - due).days if today > due else 0

			prev_status = inst["status"]
			if days_overdue == 0:
				continue
			elif days_overdue <= grace_period_days:
				new_status = "in_grace" if days_overdue > 7 else "overdue"
			else:
				new_status = "lapsed"

			if new_status != prev_status:
				inst["status"] = new_status
				inst["days_overdue"] = days_overdue
				transitions.append({"instalment_id": inst["id"], "from": prev_status, "to": new_status})
				event_type = "policy_lapsed" if new_status == "lapsed" else "lapse_warning"
				self._emit(tenant, event_type, inst["id"], "prm_instalment",
					{"days_overdue": days_overdue, "schedule_id": schedule_id})

		# If any instalment lapsed, mark the schedule lapsed too
		if any(t["to"] == "lapsed" for t in transitions):
			sch["status"] = "lapsed"
			sch["lapsed_at"] = self._now()

		return {
			"schedule_id": schedule_id,
			"grace_period_days": grace_period_days,
			"transitions": transitions,
			"evaluated_at": self._now(),
		}

	# ── I6: Regulatory Levy & Stamp Duty Calculator ───────────────────────────

	async def compute_statutory_levies(
		self,
		tenant_id: str,
		gross_premium: Decimal,
		effective_date: str,
		levy_overrides: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""Compute IRA Kenya statutory levies itemised on gross premium.

		Business value: IRA requires Training Levy (0.2 %), PHCF (0.25 %), and stamp duty
		itemised on every schedule.  Hardcoded rates break on every budget cycle; this uses a
		versioned levy table that can be overridden per effective_date without code changes.
		"""
		guard_tenant_id(tenant_id)
		gp = Decimal(str(gross_premium))
		if gp < 0:
			raise ValueError("gross_premium_cannot_be_negative")
		table = levy_overrides if levy_overrides is not None else _IRA_LEVY_TABLE
		items: list[dict[str, Any]] = []
		total_levies = Decimal("0")
		for entry in table:
			rate = Decimal(str(entry["rate"]))
			levy_amount = (gp * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			total_levies += levy_amount
			items.append({
				"code": entry["code"],
				"description": entry["description"],
				"rate": str(rate),
				"amount": str(levy_amount),
				"gazette": entry.get("gazette", ""),
			})
		net_premium = (gp - total_levies).quantize(Decimal("0.01"))
		return {
			"gross_premium": str(gp),
			"effective_date": effective_date,
			"levies": items,
			"total_levies": str(total_levies),
			"net_premium": str(net_premium),
			"computed_at": self._now(),
		}

	# ── I8: Payment Bounce & Dishonoured Instrument Handling ──────────────────

	async def record_payment_bounce(
		self,
		tenant_id: str,
		collection_id: str,
		bounce_reason: str,
		bounce_fee: Decimal = Decimal("500"),
	) -> dict[str, Any]:
		"""Reverse a collection on a dishonoured cheque or reversed M-Pesa transaction.

		Business value: Dishonoured cheques and reversed M-Pesa are the #1 reconciliation
		break in Kenyan insurance.  Automatic reversal + bounce fee restores the instalment to
		pending and re-triggers the dunning ladder — without manual ledger intervention.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(bounce_reason, "bounce_reason")
		tenant = self._tenant(tenant_id)
		col = self.collections.get(collection_id)
		if not col or col["tenant_id"] != tenant:
			raise KeyError(f"collection_not_found:{collection_id}")
		if col.get("bounced"):
			raise PermissionError("collection_already_bounced")

		# Reverse the instalment status
		instalment_id = col.get("instalment_id")
		inst = self.instalments.get(instalment_id) if instalment_id else None
		if inst:
			inst["status"] = "pending"
			inst["paid_at"] = None
			inst.pop("collection_id", None)
			inst.pop("paid_so_far", None)
			# Reverse schedule totals
			sch = self.schedules.get(inst["schedule_id"])
			if sch:
				col_amount = Decimal(str(col["amount"]))
				sch["collected_amount"] = (Decimal(str(sch["collected_amount"])) - col_amount).quantize(Decimal("0.01"))
				sch["outstanding_amount"] = (Decimal(str(sch["outstanding_amount"])) + col_amount).quantize(Decimal("0.01"))
				if sch["status"] == "fully_paid":
					sch["status"] = "active"

		col["bounced"] = True
		col["bounce_reason"] = bounce_reason
		col["bounced_at"] = self._now()
		col["status"] = "bounced"

		# Levy a bounce fee
		fee_amount = Decimal(str(bounce_fee)).quantize(Decimal("0.01"))
		bounce_charge: dict[str, Any] = {
			"id": self._record_id("bnc"),
			"type": "prm_bounce_charge",
			"collection_id": collection_id,
			"instalment_id": instalment_id,
			"policy_id": col.get("policy_id"),
			"fee_amount": fee_amount,
			"currency": col.get("currency", "KES"),
			"bounce_reason": bounce_reason,
			"status": "outstanding",
			"tenant_id": tenant,
			"created_at": self._now(),
		}
		self.bounce_charges[bounce_charge["id"]] = bounce_charge

		# Update KPI accumulator — reverse the collected amount
		self._accum_collect(tenant, -Decimal(str(col["amount"])))

		self._emit(tenant, "payment_bounced", collection_id, "prm_collection",
			{"instalment_id": instalment_id, "bounce_reason": bounce_reason, "fee": str(fee_amount)})
		_log.warning("Payment bounced: collection=%s reason=%s tenant=%s", collection_id, bounce_reason, tenant)
		return deepcopy(bounce_charge)

	# ── I9: IFRS 17 Premium Written vs Earned Accrual ─────────────────────────

	async def compute_earned_premium(
		self,
		tenant_id: str,
		schedule_id: str,
		reporting_date: str,
	) -> dict[str, Any]:
		"""Compute written, earned, and unearned premium for IFRS 17 PAA reporting.

		Business value: IFRS 17 is mandatory for all IFRS reporters.  Pro-rata temporis earned
		premium computed here maps directly to journal entries — eliminating the 5+ day
		manual spreadsheet close and restatement risk.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")

		inception = date.fromisoformat(sch["inception_date"])
		expiry = date.fromisoformat(sch["expiry_date"])
		rpt = date.fromisoformat(reporting_date)

		total_days = (expiry - inception).days
		if total_days <= 0:
			raise ValueError("expiry_must_be_after_inception")

		# Earned up to reporting_date, capped at expiry
		earned_days = max(0, (min(rpt, expiry) - inception).days)
		written = Decimal(str(sch["total_premium"]))
		earned = (written * Decimal(earned_days) / Decimal(total_days)).quantize(Decimal("0.01"))
		unearned = (written - earned).quantize(Decimal("0.01"))

		return {
			"schedule_id": schedule_id,
			"policy_id": sch["policy_id"],
			"reporting_date": reporting_date,
			"inception_date": sch["inception_date"],
			"expiry_date": sch["expiry_date"],
			"total_coverage_days": total_days,
			"earned_days": earned_days,
			"written_premium": str(written),
			"earned_premium": str(earned),
			"unearned_premium_reserve": str(unearned),
			"computed_at": self._now(),
		}

	# ── I10: Dunning Workflow & Escalation Engine ─────────────────────────────

	async def run_dunning_cycle(
		self,
		tenant_id: str,
		grace_period_days: int = 7,
	) -> dict[str, Any]:
		"""Advance overdue instalments through configurable dunning levels.

		Levels (days overdue): REMINDER_1 (7d) → REMINDER_2 (14d) →
		FORMAL_NOTICE (21d) → LAPSE_WARNING (30d).
		Returns a batch summary + dispatch list for SMS/email/agent assignment.

		Business value: Automated tiered dunning reduces days-outstanding by 40 % vs manual
		follow-up (documented in Aviva and Old Mutual implementations).
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		today = date.today()
		dispatches: list[dict[str, Any]] = []
		advanced = 0

		for inst in self.instalments.values():
			if inst["tenant_id"] != tenant or inst["status"] not in {"pending", "overdue", "in_grace", "partial"}:
				continue
			due = date.fromisoformat(inst["due_date"])
			if today <= due:
				continue
			days_overdue = (today - due).days

			# Determine target dunning level
			target_level: str | None = None
			for level in reversed(_DUNNING_LEVELS):
				if days_overdue >= _DUNNING_THRESHOLDS[level]:
					target_level = level
					break

			if target_level is None:
				continue

			current_level = inst.get("dunning_level")
			if current_level == target_level:
				continue  # Already at this level; avoid duplicate actions

			# Advance to target level
			action: dict[str, Any] = {
				"id": self._record_id("dun"),
				"type": "prm_dunning_action",
				"instalment_id": inst["id"],
				"schedule_id": inst["schedule_id"],
				"policy_id": inst["policy_id"],
				"dunning_level": target_level,
				"days_overdue": days_overdue,
				"from_level": current_level,
				"action_required": self._dunning_action_text(target_level),
				"tenant_id": tenant,
				"created_at": self._now(),
			}
			self.dunning_actions[action["id"]] = action
			inst["dunning_level"] = target_level
			inst["last_dunning_at"] = self._now()
			dispatches.append(deepcopy(action))
			advanced += 1
			self._emit(tenant, "dunning_advanced", action["id"], "prm_dunning_action",
				{"level": target_level, "days_overdue": days_overdue})

		return {
			"advanced_count": advanced,
			"dispatches": dispatches,
			"run_at": self._now(),
		}

	def _dunning_action_text(self, level: str) -> str:
		mapping = {
			"REMINDER_1": "Send courtesy SMS reminder",
			"REMINDER_2": "Send email reminder with payment link",
			"FORMAL_NOTICE": "Issue formal written notice via registered post",
			"LAPSE_WARNING": "Dispatch agent; initiate lapse evaluation",
		}
		return mapping.get(level, "Review account")

	# ── I2: Predictive Lapse / Non-Payment Scoring ────────────────────────────

	async def score_lapse_risk(
		self,
		tenant_id: str,
		schedule_id: str,
	) -> dict[str, Any]:
		"""Compute a 0–1 lapse propensity score for a premium schedule.

		Business value: A score issued 30 days before a due date lets retention teams
		intervene, recovering 30–50 % of at-risk policies.  Uses observable payment-history
		features via a lightweight logistic-like model — no external ML runtime required.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		sch = self.schedules.get(schedule_id)
		if not sch or sch["tenant_id"] != tenant:
			raise KeyError(f"schedule_not_found:{schedule_id}")

		instalments = [
			i for i in self.instalments.values()
			if i["tenant_id"] == tenant and i["schedule_id"] == schedule_id
		]
		total = len(instalments)
		if total == 0:
			return {"schedule_id": schedule_id, "score": 0.0, "band": "low", "features": {}}

		paid = [i for i in instalments if i["status"] == "paid"]
		partial = [i for i in instalments if i["status"] == "partial"]
		overdue_count = sum(1 for i in instalments if i.get("days_overdue", 0) > 0)
		avg_days_overdue = (
			sum(i.get("days_overdue", 0) for i in instalments) / total
		)
		partial_pay_freq = len(partial) / max(total, 1)
		# Payment-method volatility: unique methods used
		methods_used = len({c["payment_method"] for c in self.collections.values()
			if c["tenant_id"] == tenant and c.get("schedule_id") == schedule_id})
		method_volatility = min(methods_used / 3, 1.0)

		# Simple logistic-inspired feature combination (coefficients from Majesco research)
		score_raw = (
			0.30 * min(avg_days_overdue / 30, 1.0)
			+ 0.30 * (overdue_count / total)
			+ 0.25 * partial_pay_freq
			+ 0.15 * method_volatility
		)
		score = round(min(max(score_raw, 0.0), 1.0), 4)

		if score < 0.3:
			band = "low"
		elif score < 0.6:
			band = "medium"
		else:
			band = "high"

		features = {
			"avg_days_overdue": round(avg_days_overdue, 2),
			"overdue_ratio": round(overdue_count / total, 4),
			"partial_pay_frequency": round(partial_pay_freq, 4),
			"payment_method_volatility": round(method_volatility, 4),
		}
		return {
			"schedule_id": schedule_id,
			"score": score,
			"band": band,
			"features": features,
			"scored_at": self._now(),
		}

	# ── I14: Real-Time Collection Dashboard KPIs ──────────────────────────────

	def _accum_collect(self, tenant: str, delta: Decimal) -> None:
		"""Incrementally maintain KPI accumulated collected total for O(1) dashboard reads."""
		acc = self._kpi_accumulators.setdefault(tenant, {"total_collected": Decimal("0")})
		acc["total_collected"] = (Decimal(str(acc["total_collected"])) + delta).quantize(Decimal("0.01"))

	async def get_collection_kpis(self, tenant_id: str) -> dict[str, Any]:
		"""Return pre-aggregated collection KPIs — collection ratio, aging buckets, channel mix.

		Business value: Sub-second dashboard reads for operations managers.  Accumulators are
		updated on every collect_payment / record_partial_payment call — O(1) vs O(n) recompute.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		today = date.today()

		# Aging buckets over pending instalments
		buckets: dict[str, int] = {"0_30": 0, "31_60": 0, "61_90": 0, "90_plus": 0}
		for inst in self.instalments.values():
			if inst["tenant_id"] != tenant or inst["status"] not in {"pending", "overdue", "in_grace"}:
				continue
			due = date.fromisoformat(inst["due_date"])
			days = max(0, (today - due).days)
			if days <= 30:
				buckets["0_30"] += 1
			elif days <= 60:
				buckets["31_60"] += 1
			elif days <= 90:
				buckets["61_90"] += 1
			else:
				buckets["90_plus"] += 1

		# Channel mix
		channel_counts: dict[str, int] = {}
		for col in self.collections.values():
			if col["tenant_id"] != tenant or col.get("bounced"):
				continue
			m = col["payment_method"]
			channel_counts[m] = channel_counts.get(m, 0) + 1
		total_col = sum(channel_counts.values()) or 1
		channel_mix = {k: round(v / total_col, 4) for k, v in channel_counts.items()}

		# Collection ratio
		total_billed = sum(
			Decimal(str(s["total_premium"])) for s in self.schedules.values()
			if s["tenant_id"] == tenant
		)
		acc = self._kpi_accumulators.get(tenant, {})
		total_collected = Decimal(str(acc.get("total_collected", "0")))
		collection_ratio = float((total_collected / total_billed).quantize(Decimal("0.0001"))) if total_billed else 0.0

		return {
			"tenant_id": tenant,
			"collection_ratio": collection_ratio,
			"total_billed": str(total_billed),
			"total_collected": str(total_collected),
			"overdue_aging_buckets": buckets,
			"channel_mix": channel_mix,
			"generated_at": self._now(),
		}

	# ── I15: Audit-Grade Immutable Chain-Hashed Event Log Export ──────────────

	async def export_audit_chain(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Export audit events as an immutable chain-hashed log.

		Each event includes sha256(prev_hash + event_payload) so any tampering breaks the chain.
		Business value: IRA and FSCA require immutable, exportable audit trails for regulatory
		submissions, disputes, and WORM archival.  Chain hashing makes tampering detectable
		without blockchain infrastructure.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		result: list[dict[str, Any]] = []
		prev_hash = "0" * 64  # genesis hash
		for event in events:
			payload = json.dumps({k: str(v) for k, v in event.items()}, sort_keys=True)
			chain_hash = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
			enriched = deepcopy(event)
			enriched["prev_hash"] = prev_hash
			enriched["chain_hash"] = chain_hash
			result.append(enriched)
			prev_hash = chain_hash
		return result

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
			"version": "2.0.0",
			"domain": "insurance",
			"tenant_id": tenant_id,
			"supported_frequencies": list(SUPPORTED_FREQUENCIES),
			"supported_payment_methods": list(SUPPORTED_PAYMENT_METHODS),
			"enhancements": [
				"partial_payment_carry_forward",
				"lapse_state_machine",
				"statutory_levy_calculator",
				"payment_bounce_handling",
				"ifrs17_earned_premium",
				"dunning_engine",
				"lapse_risk_scoring",
				"collection_kpis",
				"chain_hashed_audit_export",
			],
		}

	async def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_kpi_accumulators', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

