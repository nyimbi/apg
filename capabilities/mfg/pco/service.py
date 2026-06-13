"""Async service layer for APG Product Costing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import evaluate_capability_rules
	from .models import MfPcoCostRecord, MfPcoVarianceRecord, MfPcoPeriodClose
except ImportError:
	from capability_contract import evaluate_capability_rules  # type: ignore
	from models import MfPcoCostRecord, MfPcoVarianceRecord, MfPcoPeriodClose  # type: ignore

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


class MfgPcoService:
	"""Product Costing service — async, in-memory."""

	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._cost_records: dict[str, MfPcoCostRecord] = {}
		self._variances: dict[str, MfPcoVarianceRecord] = {}
		self._period_closes: dict[str, MfPcoPeriodClose] = {}

	# ------------------------------------------------------------------ #
	# Cost Records
	# ------------------------------------------------------------------ #

	async def create_cost_record(
		self,
		item_id: str,
		item_code: str,
		cost_type: str = "standard",
		cost_version: str = "1",
		currency: str = "USD",
		material_cost: float = 0.0,
		labour_cost: float = 0.0,
		overhead_cost: float = 0.0,
		subcontract_cost: float = 0.0,
		tooling_cost: float = 0.0,
		bom_id: str | None = None,
		routing_id: str | None = None,
		effective_from: str | None = None,
		created_by: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> MfPcoCostRecord:
		ctx = {"tenant_context_present": True, "operation": "create_cost_record", "item_present": bool(item_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Cost record creation denied: {decision['actions']}")

		total = material_cost + labour_cost + overhead_cost + subcontract_cost + tooling_cost
		rec = MfPcoCostRecord(
			tenant_id=self._tenant_id,
			item_id=item_id,
			item_code=item_code,
			cost_type=cost_type,
			cost_version=cost_version,
			currency=currency,
			material_cost=material_cost,
			labour_cost=labour_cost,
			overhead_cost=overhead_cost,
			subcontract_cost=subcontract_cost,
			tooling_cost=tooling_cost,
			total_cost=total,
			bom_id=bom_id,
			routing_id=routing_id,
			effective_from=effective_from,
			created_by=created_by,
			metadata=metadata or {},
		)
		self._cost_records[rec.id] = rec
		return rec

	async def rollup_cost(self, cost_record_id: str, bom_components: list[dict[str, Any]]) -> MfPcoCostRecord:
		"""
		Roll up material cost from BOM components.
		bom_components: list of {item_id, quantity, unit_cost}
		"""
		rec = self._cost_records.get(cost_record_id)
		if not rec:
			raise KeyError(f"Cost record not found: {cost_record_id}")

		ctx = {"tenant_context_present": True, "operation": "update_cost_record", "cost_status_frozen": rec.status == "frozen"}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Cost rollup denied: {decision['actions']}")

		rolled_material = sum(c.get("quantity", 0) * c.get("unit_cost", 0) for c in bom_components)
		rec.material_cost = rolled_material
		rec.total_cost = rolled_material + rec.labour_cost + rec.overhead_cost + rec.subcontract_cost + rec.tooling_cost
		rec.rolled_up_at = _now()
		return rec

	async def freeze_cost_record(self, cost_record_id: str) -> MfPcoCostRecord:
		rec = self._cost_records.get(cost_record_id)
		if not rec:
			raise KeyError(f"Cost record not found: {cost_record_id}")
		rec.status = "frozen"
		rec.frozen_at = _now()
		return rec

	async def activate_cost_record(self, cost_record_id: str) -> MfPcoCostRecord:
		rec = self._cost_records.get(cost_record_id)
		if not rec:
			raise KeyError(f"Cost record not found: {cost_record_id}")
		# Deactivate previous active record for same item
		for r in self._cost_records.values():
			if r.item_id == rec.item_id and r.status == "active" and r.id != cost_record_id:
				r.status = "archived"
		rec.status = "active"
		return rec

	async def get_cost_record(self, cost_record_id: str) -> MfPcoCostRecord:
		if cost_record_id not in self._cost_records:
			raise KeyError(f"Cost record not found: {cost_record_id}")
		return self._cost_records[cost_record_id]

	async def get_active_cost(self, item_id: str) -> MfPcoCostRecord | None:
		for rec in self._cost_records.values():
			if rec.item_id == item_id and rec.status == "active":
				return rec
		return None

	async def list_cost_records(self, item_id: str | None = None, cost_type: str | None = None, status: str | None = None) -> list[MfPcoCostRecord]:
		recs = list(self._cost_records.values())
		if item_id:
			recs = [r for r in recs if r.item_id == item_id]
		if cost_type:
			recs = [r for r in recs if r.cost_type == cost_type]
		if status:
			recs = [r for r in recs if r.status == status]
		return recs

	# ------------------------------------------------------------------ #
	# Variance Analysis
	# ------------------------------------------------------------------ #

	async def record_variance(
		self,
		work_order_id: str,
		item_id: str,
		item_code: str,
		variance_type: str,
		cost_element: str,
		standard_cost: float,
		actual_cost: float,
		period: str,
		metadata: dict[str, Any] | None = None,
	) -> MfPcoVarianceRecord:
		variance_amount = actual_cost - standard_cost
		variance_pct = (variance_amount / standard_cost * 100) if standard_cost else None

		var = MfPcoVarianceRecord(
			tenant_id=self._tenant_id,
			work_order_id=work_order_id,
			item_id=item_id,
			item_code=item_code,
			variance_type=variance_type,
			cost_element=cost_element,
			standard_cost=standard_cost,
			actual_cost=actual_cost,
			variance_amount=variance_amount,
			variance_pct=round(variance_pct, 2) if variance_pct is not None else None,
			period=period,
			metadata=metadata or {},
		)
		self._variances[var.id] = var
		return var

	async def list_variances(self, period: str | None = None, item_id: str | None = None, variance_type: str | None = None, limit: int = 200, offset: int = 0) -> list[MfPcoVarianceRecord]:
		vars_ = list(self._variances.values())
		if period:
			vars_ = [v for v in vars_ if v.period == period]
		if item_id:
			vars_ = [v for v in vars_ if v.item_id == item_id]
		if variance_type:
			vars_ = [v for v in vars_ if v.variance_type == variance_type]
		return vars_[offset : offset + limit]

	async def get_variance_summary(self, period: str) -> dict[str, Any]:
		vars_ = [v for v in self._variances.values() if v.period == period]
		total = sum(v.variance_amount for v in vars_)
		by_type: dict[str, float] = {}
		for v in vars_:
			by_type[v.variance_type] = by_type.get(v.variance_type, 0.0) + v.variance_amount
		return {"period": period, "total_variance": round(total, 2), "count": len(vars_), "by_type": {k: round(v, 2) for k, v in by_type.items()}}

	# ------------------------------------------------------------------ #
	# Period Close
	# ------------------------------------------------------------------ #

	async def initiate_period_close(self, period: str, created_by: str = "system") -> MfPcoPeriodClose:
		# Calculate totals
		vars_ = [v for v in self._variances.values() if v.period == period]
		total_var = sum(v.variance_amount for v in vars_)

		pc = MfPcoPeriodClose(tenant_id=self._tenant_id, period=period, total_variances=round(total_var, 2), variance_records_count=len(vars_), created_by=created_by)
		self._period_closes[pc.id] = pc
		return pc

	async def approve_period_close(self, period_close_id: str, approver_id: str) -> MfPcoPeriodClose:
		pc = self._period_closes.get(period_close_id)
		if not pc:
			raise KeyError(f"Period close not found: {period_close_id}")
		ctx = {"tenant_context_present": True, "operation": "close_period", "approval_present": bool(approver_id)}
		decision = evaluate_capability_rules(ctx)
		if decision["decision"] == "deny":
			raise ValueError(f"Period close denied: {decision['actions']}")
		pc.status = "completed"
		pc.approver_id = approver_id
		pc.approved_at = _now()
		pc.closed_at = _now()
		return pc

	# ------------------------------------------------------------------ #
	# Dashboard
	# ------------------------------------------------------------------ #

	async def get_dashboard_summary(self) -> dict[str, Any]:
		recs = list(self._cost_records.values())
		vars_ = list(self._variances.values())
		closes = list(self._period_closes.values())
		return {
			"tenant_id": self._tenant_id,
			"cost_records": {"total": len(recs), "active": sum(1 for r in recs if r.status == "active"), "frozen": sum(1 for r in recs if r.status == "frozen")},
			"variances": {"total": len(vars_), "total_amount": round(sum(v.variance_amount for v in vars_), 2)},
			"period_closes": {"total": len(closes), "completed": sum(1 for c in closes if c.status == "completed")},
		}
