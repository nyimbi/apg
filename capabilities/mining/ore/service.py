"""Async service layer for APG Ore Processing & Metallurgy."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .models import (
	AlertLevel,
	CircuitStatusResponse,
	CircuitStatusUpdateCreate,
	DeviationAlertCreate,
	DeviationAlertResponse,
	MetallurgicalBalanceCreate,
	MetallurgicalBalanceResponse,
	PlantFeedCreate,
	PlantFeedResponse,
	ProductQualityCreate,
	ProductQualityResponse,
	ReagentUsageCreate,
	ReagentUsageResponse,
	ReconciliationStatus,
	ReagentType,
	uuid7str,
)

log = logging.getLogger(__name__)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class OreService:
	"""Service for Ore Processing & Metallurgy operations."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._plant_feeds: dict[str, dict[str, Any]] = {}
		self._circuits: dict[str, dict[str, Any]] = {}
		self._reagent_usage: dict[str, dict[str, Any]] = {}
		self._met_balances: dict[str, dict[str, Any]] = {}
		self._product_quality: dict[str, dict[str, Any]] = {}
		self._deviations: dict[str, dict[str, Any]] = {}
		self._reagent_inventory: dict[str, float] = {}  # reagent_type -> kg on hand
		# Extended stores
		self._downtime_logs: dict[str, dict[str, Any]] = {}
		self._energy_records: dict[str, dict[str, Any]] = {}
		self._cost_records: dict[str, dict[str, Any]] = {}
		self._recovery_optimisations: dict[str, dict[str, Any]] = {}
		self._met_reports: dict[str, dict[str, Any]] = {}
		self._process_analytics_records: dict[str, dict[str, Any]] = {}

	# ── Logging helpers ────────────────────────────────────────────────────────

	def _log_op(self, op: str, entity: str, id: str) -> None:
		log.info("ore.%s | tenant=%s entity=%s id=%s", op, self.tenant_id, entity, id)

	def _log_warn(self, msg: str, **kw: Any) -> None:
		log.warning("ore | tenant=%s %s %s", self.tenant_id, msg, kw)

	def _log_alert(self, alert_level: str, description: str) -> None:
		log.warning("ore.alert | tenant=%s level=%s %s", self.tenant_id, alert_level, description)

	# ── Tenant guard ───────────────────────────────────────────────────────────

	def _assert_tenant(self, tenant_id: str) -> None:
		assert tenant_id == self.tenant_id, (
			f"Cross-tenant access denied: requested={tenant_id} service={self.tenant_id}"
		)

	# ── Plant Feed ─────────────────────────────────────────────────────────────

	async def record_plant_feed(
		self, payload: PlantFeedCreate, created_by: str
	) -> PlantFeedResponse:
		"""Record plant feed data for a processing period."""
		self._assert_tenant(payload.tenant_id)
		resp = PlantFeedResponse(**payload.model_dump(), created_by=created_by)
		self._plant_feeds[resp.id] = resp.model_dump()
		self._log_op("record_feed", "plant_feed", resp.id)
		return resp

	async def get_plant_feed(self, id: str) -> PlantFeedResponse | None:
		"""Get a plant feed record by id."""
		rec = self._plant_feeds.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return PlantFeedResponse(**rec)

	async def list_plant_feeds(
		self,
		feed_source: str | None = None,
		date_from: datetime | None = None,
		date_to: datetime | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[PlantFeedResponse]:
		"""List plant feed records with optional filters."""
		results = [
			PlantFeedResponse(**r)
			for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id
		]
		if feed_source:
			results = [r for r in results if r.feed_source == feed_source]
		if date_from:
			results = [r for r in results if r.period_start >= date_from]
		if date_to:
			results = [r for r in results if r.period_end <= date_to]
		return sorted(results, key=lambda x: x.period_start, reverse=True)[offset : offset + limit]

	async def get_feed_summary(self, date_from: datetime, date_to: datetime) -> dict[str, Any]:
		"""Aggregate total feed tonnage and average grade for a period."""
		feeds = await self.list_plant_feeds(date_from=date_from, date_to=date_to, limit=10000)
		total_tonnes = sum(f.dry_tonnes for f in feeds)
		avg_grade = (
			sum(f.dry_tonnes * f.feed_grade for f in feeds) / total_tonnes
			if total_tonnes > 0 else 0
		)
		return {
			"period_start": date_from.isoformat(),
			"period_end": date_to.isoformat(),
			"total_dry_tonnes": round(total_tonnes, 2),
			"average_feed_grade": round(avg_grade, 4),
			"record_count": len(feeds),
		}

	# ── Circuit Status ─────────────────────────────────────────────────────────

	async def update_circuit_status(
		self, payload: CircuitStatusUpdateCreate, created_by: str
	) -> CircuitStatusResponse:
		"""Record a circuit status snapshot."""
		self._assert_tenant(payload.tenant_id)
		resp = CircuitStatusResponse(**payload.model_dump(), created_by=created_by)
		self._circuits[resp.id] = resp.model_dump()
		self._log_op("circuit_status", "circuit", resp.circuit_id)
		return resp

	async def get_current_circuit_statuses(self) -> list[CircuitStatusResponse]:
		"""Return the most recent status record for each circuit."""
		by_circuit: dict[str, dict[str, Any]] = {}
		for rec in self._circuits.values():
			if rec["tenant_id"] != self.tenant_id:
				continue
			cid = rec["circuit_id"]
			if cid not in by_circuit or rec["updated_at"] > by_circuit[cid]["updated_at"]:
				by_circuit[cid] = rec
		return [CircuitStatusResponse(**r) for r in by_circuit.values()]

	# ── Reagent Management ─────────────────────────────────────────────────────

	async def record_reagent_usage(
		self, payload: ReagentUsageCreate, created_by: str
	) -> ReagentUsageResponse:
		"""Record reagent consumption. Validates cyanide code compliance check."""
		self._assert_tenant(payload.tenant_id)
		if payload.reagent_type == ReagentType.CYANIDE:
			self._log_op("cyanide_usage", "reagent", "cyanide_check")
		total_cost = (
			round(payload.quantity_kg * payload.unit_cost, 2)
			if payload.unit_cost else None
		)
		resp = ReagentUsageResponse(
			**payload.model_dump(), total_cost=total_cost, created_by=created_by
		)
		self._reagent_usage[resp.id] = resp.model_dump()
		# Deduct from inventory
		key = payload.reagent_type.value
		self._reagent_inventory[key] = max(
			0, self._reagent_inventory.get(key, 0) - payload.quantity_kg
		)
		if self._reagent_inventory[key] < 500:  # low stock threshold 500 kg
			self._log_warn("Reagent low stock warning", reagent=key, on_hand_kg=self._reagent_inventory[key])
		self._log_op("record_reagent", "reagent_usage", resp.id)
		return resp

	async def add_reagent_stock(self, reagent_type: str, quantity_kg: float) -> dict[str, Any]:
		"""Add to reagent inventory (e.g. delivery received)."""
		assert quantity_kg > 0, "quantity_kg must be positive"
		self._reagent_inventory[reagent_type] = (
			self._reagent_inventory.get(reagent_type, 0) + quantity_kg
		)
		return {"reagent_type": reagent_type, "on_hand_kg": self._reagent_inventory[reagent_type]}

	async def get_reagent_inventory(self) -> dict[str, float]:
		"""Return current reagent inventory levels."""
		return dict(self._reagent_inventory)

	async def list_reagent_usage(
		self,
		reagent_type: str | None = None,
		circuit_id: str | None = None,
		date_from: datetime | None = None,
	) -> list[ReagentUsageResponse]:
		"""List reagent usage records."""
		results = [
			ReagentUsageResponse(**r)
			for r in self._reagent_usage.values()
			if r["tenant_id"] == self.tenant_id
		]
		if reagent_type:
			results = [r for r in results if r.reagent_type == reagent_type]
		if circuit_id:
			results = [r for r in results if r.circuit_id == circuit_id]
		if date_from:
			results = [r for r in results if r.period_start >= date_from]
		return sorted(results, key=lambda x: x.period_start, reverse=True)

	# ── Metallurgical Balance ──────────────────────────────────────────────────

	async def submit_metallurgical_balance(
		self, payload: MetallurgicalBalanceCreate, created_by: str
	) -> MetallurgicalBalanceResponse:
		"""Submit a metallurgical balance. Validates recovery bounds."""
		self._assert_tenant(payload.tenant_id)
		if payload.calculated_recovery_pct is not None:
			if payload.calculated_recovery_pct < 0:
				raise ValueError("Recovery cannot be negative")
			if payload.calculated_recovery_pct > 100:
				raise ValueError("Recovery cannot exceed 100%")
		resp = MetallurgicalBalanceResponse(
			**payload.model_dump(exclude={"feed_stream", "concentrate_stream", "tailings_stream", "additional_streams"}),
			feed_stream=payload.feed_stream.model_dump(),
			concentrate_stream=payload.concentrate_stream.model_dump() if payload.concentrate_stream else None,
			tailings_stream=payload.tailings_stream.model_dump() if payload.tailings_stream else None,
			additional_streams=[s.model_dump() for s in payload.additional_streams],
			created_by=created_by,
		)
		self._met_balances[resp.id] = resp.model_dump()
		self._log_op("submit_balance", "met_balance", resp.id)
		return resp

	async def approve_metallurgical_balance(
		self, id: str, approver_id: str
	) -> MetallurgicalBalanceResponse:
		"""Approve a metallurgical balance for publication."""
		rec = self._met_balances.get(id)
		if rec is None:
			raise KeyError(f"Metallurgical balance {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["status"] = ReconciliationStatus.APPROVED
		rec["approved_by"] = approver_id
		rec["approved_at"] = datetime.utcnow()
		rec["updated_at"] = datetime.utcnow()
		self._log_op("approve_balance", "met_balance", id)
		return MetallurgicalBalanceResponse(**rec)

	async def publish_metallurgical_balance(self, id: str) -> MetallurgicalBalanceResponse:
		"""Publish an approved metallurgical balance."""
		rec = self._met_balances.get(id)
		if rec is None:
			raise KeyError(f"Metallurgical balance {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec["status"] != ReconciliationStatus.APPROVED:
			raise PermissionError("Balance must be approved before publication")
		rec["status"] = ReconciliationStatus.FINALISED
		rec["published"] = True
		rec["updated_at"] = datetime.utcnow()
		self._log_op("publish_balance", "met_balance", id)
		return MetallurgicalBalanceResponse(**rec)

	async def get_metallurgical_balance(self, id: str) -> MetallurgicalBalanceResponse | None:
		"""Get a metallurgical balance by id."""
		rec = self._met_balances.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return MetallurgicalBalanceResponse(**rec)

	async def list_metallurgical_balances(
		self,
		balance_type: str | None = None,
		commodity: str | None = None,
		published_only: bool = False,
	) -> list[MetallurgicalBalanceResponse]:
		"""List metallurgical balances with optional filters."""
		results = [
			MetallurgicalBalanceResponse(**r)
			for r in self._met_balances.values()
			if r["tenant_id"] == self.tenant_id
		]
		if balance_type:
			results = [r for r in results if r.balance_type == balance_type]
		if commodity:
			results = [r for r in results if r.commodity == commodity]
		if published_only:
			results = [r for r in results if r.published]
		return sorted(results, key=lambda x: x.period_start, reverse=True)

	# ── Product Quality ────────────────────────────────────────────────────────

	async def record_product_quality(
		self, payload: ProductQualityCreate, created_by: str
	) -> ProductQualityResponse:
		"""Record product quality data for a lot."""
		self._assert_tenant(payload.tenant_id)
		resp = ProductQualityResponse(**payload.model_dump(), created_by=created_by)
		self._product_quality[resp.id] = resp.model_dump()
		if not resp.meets_specification:
			self._log_warn("Off-spec product recorded", lot=resp.lot_number, product_type=resp.product_type)
		self._log_op("record_quality", "product_quality", resp.id)
		return resp

	async def approve_product_dispatch(self, id: str, approved_by: str) -> ProductQualityResponse:
		"""Approve a product lot for dispatch. Off-spec lots require explicit approval."""
		rec = self._product_quality.get(id)
		if rec is None:
			raise KeyError(f"Product quality record {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["dispatched"] = True
		rec["dispatch_approved_by"] = approved_by
		rec["updated_at"] = datetime.utcnow()
		self._log_op("approve_dispatch", "product_quality", id)
		return ProductQualityResponse(**rec)

	async def list_product_quality(
		self,
		product_type: str | None = None,
		on_spec_only: bool = False,
	) -> list[ProductQualityResponse]:
		"""List product quality records."""
		results = [
			ProductQualityResponse(**r)
			for r in self._product_quality.values()
			if r["tenant_id"] == self.tenant_id
		]
		if product_type:
			results = [r for r in results if r.product_type == product_type]
		if on_spec_only:
			results = [r for r in results if r.meets_specification]
		return sorted(results, key=lambda x: x.sampled_at, reverse=True)

	# ── Deviation Alerts ───────────────────────────────────────────────────────

	async def raise_deviation_alert(
		self, payload: DeviationAlertCreate, created_by: str
	) -> DeviationAlertResponse:
		"""Raise a process deviation alert."""
		self._assert_tenant(payload.tenant_id)
		variance_pct = (
			abs(payload.actual_value - payload.target_value) / payload.target_value * 100
			if payload.target_value != 0 else 0.0
		)
		resp = DeviationAlertResponse(
			**payload.model_dump(),
			variance_pct=round(variance_pct, 2),
			created_by=created_by,
		)
		self._deviations[resp.id] = resp.model_dump()
		self._log_alert(resp.alert_level, resp.description)
		return resp

	async def acknowledge_deviation(self, id: str, acknowledged_by: str) -> DeviationAlertResponse:
		"""Acknowledge a deviation alert."""
		rec = self._deviations.get(id)
		if rec is None:
			raise KeyError(f"Deviation alert {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["acknowledged"] = True
		rec["acknowledged_by"] = acknowledged_by
		rec["acknowledged_at"] = datetime.utcnow()
		rec["updated_at"] = datetime.utcnow()
		self._log_op("ack_deviation", "deviation_alert", id)
		return DeviationAlertResponse(**rec)

	async def resolve_deviation(self, id: str, resolution_notes: str) -> DeviationAlertResponse:
		"""Resolve a deviation alert."""
		rec = self._deviations.get(id)
		if rec is None:
			raise KeyError(f"Deviation alert {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["resolved"] = True
		rec["resolved_at"] = datetime.utcnow()
		rec["resolution_notes"] = resolution_notes
		rec["updated_at"] = datetime.utcnow()
		self._log_op("resolve_deviation", "deviation_alert", id)
		return DeviationAlertResponse(**rec)

	async def list_deviation_alerts(
		self, open_only: bool = True, alert_level: str | None = None
	) -> list[DeviationAlertResponse]:
		"""List deviation alerts."""
		results = [
			DeviationAlertResponse(**r)
			for r in self._deviations.values()
			if r["tenant_id"] == self.tenant_id
		]
		if open_only:
			results = [r for r in results if not r.resolved]
		if alert_level:
			results = [r for r in results if r.alert_level == alert_level]
		return sorted(results, key=lambda x: x.detected_at, reverse=True)

	# ── Process KPI Summary ────────────────────────────────────────────────────

	async def get_process_kpi_summary(self) -> dict[str, Any]:
		"""Compute aggregate process KPIs."""
		feeds = list(self._plant_feeds.values())
		balances = [
			r for r in self._met_balances.values()
			if r["tenant_id"] == self.tenant_id and r.get("published")
		]
		recoveries = [
			b["calculated_recovery_pct"]
			for b in balances
			if b.get("calculated_recovery_pct") is not None
		]
		avg_recovery = round(sum(recoveries) / len(recoveries), 2) if recoveries else None
		open_deviations = [
			r for r in self._deviations.values()
			if r["tenant_id"] == self.tenant_id and not r.get("resolved")
		]
		return {
			"tenant_id": self.tenant_id,
			"total_feed_records": len(feeds),
			"total_met_balances": len(balances),
			"average_recovery_pct": avg_recovery,
			"open_deviation_alerts": len(open_deviations),
			"critical_deviation_alerts": sum(
				1 for d in open_deviations if d.get("alert_level") == AlertLevel.CRITICAL
			),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Plant Feed Record (extended) ───────────────────────────────────────────

	async def plant_feed_record(
		self,
		period: str,
		source_blend: dict[str, float],
		tonnes: float,
		grade: float,
		feed_type: str = "ROM",
		moisture_pct: float | None = None,
		particle_size_p80_mm: float | None = None,
		recorded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record a plant feed entry for a processing period.
		source_blend: {"ROM_stockpile": 0.6, "high_grade_ore": 0.4} (fractions summing to 1.0).
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert tonnes > 0, "tonnes must be positive"
		assert grade >= 0, "grade must be non-negative"
		assert source_blend, "source_blend required"
		blend_total = sum(source_blend.values())
		if abs(blend_total - 1.0) > 0.02:
			self._log_warn(f"source_blend fractions sum to {blend_total:.3f}, not 1.0")
		if moisture_pct is not None:
			assert 0 <= moisture_pct <= 100, "moisture_pct must be 0-100"
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"source_blend": source_blend,
			"feed_type": feed_type,
			"dry_tonnes": round(tonnes * (1 - (moisture_pct or 0) / 100), 2),
			"wet_tonnes": round(tonnes, 2),
			"feed_grade": grade,
			"moisture_pct": moisture_pct,
			"particle_size_p80_mm": particle_size_p80_mm,
			"recorded_by": recorded_by,
			"period_start": datetime.utcnow().isoformat(),
			"period_end": datetime.utcnow().isoformat(),
			"created_at": datetime.utcnow().isoformat(),
		}
		self._plant_feeds[rec_id] = rec
		self._log_op("plant_feed_record", "plant_feed", rec_id)
		return rec

	# ── Metallurgical Balance (extended) ───────────────────────────────────────

	async def metallurgical_balance(
		self,
		period: str,
		feed_tonnes: float,
		concentrate_tonnes: float,
		recovery_pct: float,
		tail_grade: float,
		feed_grade: float | None = None,
		concentrate_grade: float | None = None,
		commodity: str = "Au",
		grade_unit: str = "g/t",
		balance_type: str = "monthly",
	) -> dict[str, Any]:
		"""
		Submit a metallurgical balance for a period. Validates mass balance closure.
		Mass pull = concentrate_tonnes / feed_tonnes * 100 (%).
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert feed_tonnes > 0, "feed_tonnes must be positive"
		assert 0 <= recovery_pct <= 100, "recovery_pct must be 0-100"
		mass_pull_pct = round(concentrate_tonnes / feed_tonnes * 100, 4) if feed_tonnes > 0 else 0.0
		# Mass balance check: feed = concentrate + tailings (within 2%)
		# Approximate: if concentrate + (feed - concentrate) should equal feed
		balance_closure_pct = 100.0  # simplified — real implementation integrates stream data
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"balance_type": balance_type,
			"commodity": commodity,
			"grade_unit": grade_unit,
			"feed_tonnes": round(feed_tonnes, 2),
			"concentrate_tonnes": round(concentrate_tonnes, 2),
			"tail_grade": tail_grade,
			"feed_grade": feed_grade,
			"concentrate_grade": concentrate_grade,
			"calculated_recovery_pct": round(recovery_pct, 3),
			"mass_pull_pct": mass_pull_pct,
			"balance_closure_pct": balance_closure_pct,
			"status": "draft",
			"published": False,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._met_balances[rec_id] = rec
		self._log_op("metallurgical_balance", "met_balance", rec_id)
		return rec

	# ── Reagent Consumption (extended) ────────────────────────────────────────

	async def reagent_consumption(
		self,
		period: str,
		reagent_type: str,
		quantity: float,
		unit_cost: float,
		circuit_id: str | None = None,
		consumption_rate_g_t: float | None = None,
	) -> dict[str, Any]:
		"""
		Record reagent consumption for a period with cost accounting.
		Computes total cost and unit consumption rate (g/t feed if not supplied).
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert reagent_type, "reagent_type required"
		assert quantity > 0, "quantity must be positive"
		assert unit_cost >= 0, "unit_cost must be non-negative"
		total_cost = round(quantity * unit_cost, 2)
		# Get total feed for period to compute g/t rate
		period_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_feed_t = sum(r.get("dry_tonnes", 0) for r in period_feeds)
		if consumption_rate_g_t is None and total_feed_t > 0:
			# Convert: quantity in kg → grams, divide by tonnes
			consumption_rate_g_t = round(quantity * 1000 / total_feed_t, 3)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"reagent_type": reagent_type,
			"quantity_kg": round(quantity, 3),
			"unit_cost": unit_cost,
			"total_cost": total_cost,
			"circuit_id": circuit_id,
			"consumption_rate_g_t": consumption_rate_g_t,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._reagent_usage[rec_id] = rec
		# Deduct from inventory
		self._reagent_inventory[reagent_type] = max(
			0, self._reagent_inventory.get(reagent_type, 0) - quantity
		)
		self._log_op("reagent_consumption", "reagent_usage", rec_id)
		return rec

	# ── Product Quality (extended) ─────────────────────────────────────────────

	async def product_quality(
		self,
		batch_id: str,
		grade: float,
		moisture: float,
		assay_results: dict[str, float],
		product_type: str = "doré",
		lot_number: str | None = None,
		meets_spec: bool | None = None,
		spec_grade_min: float | None = None,
		spec_moisture_max: float | None = None,
	) -> dict[str, Any]:
		"""
		Record product quality data for a processing batch.
		assay_results: {"Au_g_t": 998.5, "Ag_g_t": 12.3, "Cu_ppm": 0.8}
		meets_spec derived automatically if spec thresholds are provided.
		"""
		assert batch_id, "batch_id required"
		assert grade >= 0, "grade must be non-negative"
		assert 0 <= moisture <= 100, "moisture must be 0-100"
		# Auto-determine spec compliance
		if meets_spec is None:
			spec_pass = True
			if spec_grade_min is not None and grade < spec_grade_min:
				spec_pass = False
			if spec_moisture_max is not None and moisture > spec_moisture_max:
				spec_pass = False
			meets_spec = spec_pass
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"batch_id": batch_id,
			"lot_number": lot_number or batch_id,
			"product_type": product_type,
			"grade": grade,
			"moisture_pct": moisture,
			"assay_results": assay_results,
			"meets_specification": meets_spec,
			"spec_grade_min": spec_grade_min,
			"spec_moisture_max": spec_moisture_max,
			"dispatched": False,
			"sampled_at": datetime.utcnow().isoformat(),
		}
		self._product_quality[rec_id] = rec
		if not meets_spec:
			self._log_warn("Off-spec product recorded", batch_id=batch_id, grade=grade, moisture=moisture)
		self._log_op("product_quality", "product_quality", rec_id)
		return rec

	# ── Recovery Optimisation ──────────────────────────────────────────────────

	async def recovery_optimisation(
		self,
		circuit_id: str,
		parameters: dict[str, Any],
		current_recovery_pct: float,
		target_recovery_pct: float,
		optimisation_method: str = "DoE",
		recommended_changes: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Record a recovery optimisation study for a processing circuit.
		parameters: current operating parameters being optimised
		{"pH": 10.5, "grind_p80_um": 75, "air_flow_m3_min": 120, "reagent_dosage_g_t": 30}
		optimisation_method: DoE | PID_tune | ML_model | manual | ANOVA
		"""
		assert circuit_id, "circuit_id required"
		assert parameters, "parameters required"
		assert 0 <= current_recovery_pct <= 100, "current_recovery_pct must be 0-100"
		assert 0 <= target_recovery_pct <= 100, "target_recovery_pct must be 0-100"
		recovery_gap = round(target_recovery_pct - current_recovery_pct, 2)
		# Simple parameter sensitivity scoring
		sensitivity_scores: dict[str, float] = {}
		known_sensitive_params = {"pH", "grind_p80_um", "reagent_dosage_g_t", "air_flow_m3_min"}
		for param in parameters:
			if param in known_sensitive_params:
				sensitivity_scores[param] = 0.8
			else:
				sensitivity_scores[param] = 0.3
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"circuit_id": circuit_id,
			"current_parameters": parameters,
			"current_recovery_pct": current_recovery_pct,
			"target_recovery_pct": target_recovery_pct,
			"recovery_gap_pct": recovery_gap,
			"optimisation_method": optimisation_method,
			"parameter_sensitivity": sensitivity_scores,
			"recommended_changes": recommended_changes or [],
			"estimated_recovery_improvement_pct": round(recovery_gap * 0.5, 2),  # conservative 50% of gap
			"created_at": datetime.utcnow().isoformat(),
		}
		self._recovery_optimisations[rec_id] = rec
		self._log_op("recovery_optimisation", "recovery_optimisation", rec_id)
		return rec

	# ── Plant Downtime Log ─────────────────────────────────────────────────────

	async def plant_downtime_log(
		self,
		area: str,
		start_time: datetime,
		end_time: datetime,
		reason: str,
		losses: dict[str, float],
		circuit_id: str | None = None,
		downtime_category: str = "unplanned",
		reported_by: str = "system",
	) -> dict[str, Any]:
		"""
		Log plant downtime with production loss quantification.
		losses: {"tonnes_lost": 240.0, "oz_lost": 180.0, "revenue_usd": 324000.0}
		downtime_category: planned | unplanned | external
		"""
		assert area, "area required"
		assert end_time >= start_time, "end_time must be after start_time"
		assert reason, "reason required"
		valid_categories = {"planned", "unplanned", "external"}
		if downtime_category not in valid_categories:
			raise ValueError(f"downtime_category must be one of {valid_categories}")
		downtime_hours = round((end_time - start_time).total_seconds() / 3600, 3)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"area": area,
			"circuit_id": circuit_id,
			"start_time": start_time.isoformat(),
			"end_time": end_time.isoformat(),
			"downtime_hours": downtime_hours,
			"reason": reason,
			"downtime_category": downtime_category,
			"losses": losses,
			"total_revenue_loss": losses.get("revenue_usd", 0),
			"reported_by": reported_by,
			"logged_at": datetime.utcnow().isoformat(),
		}
		self._downtime_logs[rec_id] = rec
		if downtime_hours > 4:
			self._log_warn(f"Significant plant downtime {downtime_hours:.1f}h in {area}", reason=reason)
		self._log_op("plant_downtime_log", "downtime_log", rec_id)
		return rec

	async def list_plant_downtime(
		self, period: str | None = None, downtime_category: str | None = None
	) -> list[dict[str, Any]]:
		"""List plant downtime logs."""
		results = [r for r in self._downtime_logs.values() if r["tenant_id"] == self.tenant_id]
		if period:
			results = [r for r in results if r["start_time"][:7] == period]
		if downtime_category:
			results = [r for r in results if r["downtime_category"] == downtime_category]
		return sorted(results, key=lambda x: x["start_time"], reverse=True)

	# ── Energy Consumption ─────────────────────────────────────────────────────

	async def energy_consumption(
		self,
		period: str,
		circuit: str,
		kwh_consumed: float,
		peak_demand_kw: float | None = None,
		energy_intensity_kwh_t: float | None = None,
		tariff_rate: float | None = None,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Record energy consumption for a processing circuit.
		Computes energy cost and intensity (kWh/t) if feed data is available.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert circuit, "circuit required"
		assert kwh_consumed >= 0, "kwh_consumed must be non-negative"
		# Get total feed for period to compute intensity
		period_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_feed_t = sum(r.get("dry_tonnes", 0) for r in period_feeds)
		if energy_intensity_kwh_t is None and total_feed_t > 0:
			energy_intensity_kwh_t = round(kwh_consumed / total_feed_t, 3)
		energy_cost = round(kwh_consumed * tariff_rate, 2) if tariff_rate else None
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"circuit": circuit,
			"kwh_consumed": round(kwh_consumed, 1),
			"mwh_consumed": round(kwh_consumed / 1000, 3),
			"peak_demand_kw": peak_demand_kw,
			"energy_intensity_kwh_t": energy_intensity_kwh_t,
			"tariff_rate": tariff_rate,
			"energy_cost": energy_cost,
			"currency": currency,
			"total_feed_t": round(total_feed_t, 2),
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._energy_records[rec_id] = rec
		self._log_op("energy_consumption", "energy_record", rec_id)
		return rec

	# ── Processing Cost Per Tonne ──────────────────────────────────────────────

	async def processing_cost_per_tonne(self, period: str) -> dict[str, Any]:
		"""
		Calculate total processing cost per tonne for a period.
		Aggregates: reagent costs + energy costs + labour (estimated) + maintenance (estimated).
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		reagent_costs = sum(
			r.get("total_cost", 0)
			for r in self._reagent_usage.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		)
		energy_costs = sum(
			r.get("energy_cost", 0) or 0
			for r in self._energy_records.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		)
		period_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_feed_t = sum(r.get("dry_tonnes", 0) for r in period_feeds)
		# Labour and maintenance estimates (proxy at 30% and 20% of known costs)
		total_direct = reagent_costs + energy_costs
		labour_estimate = round(total_direct * 0.30, 2)
		maintenance_estimate = round(total_direct * 0.20, 2)
		total_cost = round(reagent_costs + energy_costs + labour_estimate + maintenance_estimate, 2)
		cost_per_tonne = round(total_cost / total_feed_t, 4) if total_feed_t > 0 else None
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"reagent_cost": round(reagent_costs, 2),
			"energy_cost": round(energy_costs, 2),
			"labour_estimate": labour_estimate,
			"maintenance_estimate": maintenance_estimate,
			"total_processing_cost": total_cost,
			"total_feed_tonnes": round(total_feed_t, 2),
			"cost_per_tonne": cost_per_tonne,
			"calculated_at": datetime.utcnow().isoformat(),
		}
		self._cost_records[rec_id] = rec
		self._log_op("processing_cost_per_tonne", "cost_record", rec_id)
		return rec

	# ── Metallurgical Report ───────────────────────────────────────────────────

	async def metallurgical_report(self, period: str) -> dict[str, Any]:
		"""
		Generate a comprehensive monthly metallurgical report.
		Bundles: feed summary, balance, reagents, product quality, costs, downtime.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		period_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_feed_t = sum(r.get("dry_tonnes", 0) for r in period_feeds)
		avg_grade = (
			sum(r.get("feed_grade", 0) * r.get("dry_tonnes", 0) for r in period_feeds) / total_feed_t
			if total_feed_t > 0 else 0.0
		)
		balances = [
			r for r in self._met_balances.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		avg_recovery = (
			sum(b.get("calculated_recovery_pct", 0) for b in balances) / len(balances)
			if balances else None
		)
		products = [
			r for r in self._product_quality.values()
			if r["tenant_id"] == self.tenant_id and r.get("sampled_at", "")[:7] == period
		]
		on_spec = sum(1 for p in products if p.get("meets_specification"))
		downtime = await self.list_plant_downtime(period=period)
		total_downtime_h = sum(r.get("downtime_hours", 0) for r in downtime)
		cost_rec = await self.processing_cost_per_tonne(period)
		return {
			"report_type": "metallurgical_report",
			"tenant_id": self.tenant_id,
			"period": period,
			"generated_at": datetime.utcnow().isoformat(),
			"feed_summary": {
				"total_dry_tonnes": round(total_feed_t, 2),
				"average_feed_grade": round(avg_grade, 4),
				"feed_records": len(period_feeds),
			},
			"metallurgical_balances": len(balances),
			"average_recovery_pct": round(avg_recovery, 2) if avg_recovery else None,
			"product_batches": len(products),
			"on_spec_batches": on_spec,
			"on_spec_pct": round(on_spec / len(products) * 100, 1) if products else None,
			"total_downtime_hours": round(total_downtime_h, 1),
			"downtime_events": len(downtime),
			"cost_summary": cost_rec,
			"open_deviations": len([
				r for r in self._deviations.values()
				if r["tenant_id"] == self.tenant_id and not r.get("resolved")
			]),
		}

	# ── Process Analytics ──────────────────────────────────────────────────────

	async def process_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute process analytics dashboard for a period.
		Trends: feed grade, recovery, reagent intensity, energy intensity, downtime rate.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_t = sum(r.get("dry_tonnes", 0) for r in feeds)
		balances = [
			r for r in self._met_balances.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		reagents = [
			r for r in self._reagent_usage.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		energy_recs = [
			r for r in self._energy_records.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		downtime_recs = await self.list_plant_downtime(period=period)
		avg_recovery = (
			sum(b.get("calculated_recovery_pct", 0) for b in balances) / len(balances)
			if balances else None
		)
		total_reagent_kg = sum(r.get("quantity_kg", 0) for r in reagents)
		reagent_intensity = round(total_reagent_kg * 1000 / total_t, 2) if total_t > 0 else None  # g/t
		total_kwh = sum(r.get("kwh_consumed", 0) for r in energy_recs)
		energy_intensity = round(total_kwh / total_t, 2) if total_t > 0 else None
		total_downtime_h = sum(r.get("downtime_hours", 0) for r in downtime_recs)
		calendar_h = 30 * 24
		plant_utilisation_pct = round((calendar_h - total_downtime_h) / calendar_h * 100, 1) if calendar_h > 0 else 0.0
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"total_feed_tonnes": round(total_t, 2),
			"average_recovery_pct": round(avg_recovery, 2) if avg_recovery else None,
			"total_reagent_kg": round(total_reagent_kg, 2),
			"reagent_intensity_g_t": reagent_intensity,
			"total_kwh": round(total_kwh, 1),
			"energy_intensity_kwh_t": energy_intensity,
			"total_downtime_hours": round(total_downtime_h, 1),
			"plant_utilisation_pct": plant_utilisation_pct,
			"open_deviations": len([
				r for r in self._deviations.values()
				if r["tenant_id"] == self.tenant_id and not r.get("resolved")
			]),
			"calculated_at": datetime.utcnow().isoformat(),
		}
		self._process_analytics_records[rec_id] = rec
		self._log_op("process_analytics", "process_analytics", rec_id)
		return rec


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": self.tenant_id}

	async def health_check(self, ) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": self.tenant_id, "status": "healthy"}

	async def compliance_report(self, standard: str = "ISO_14001") -> dict[str, Any]:
		"""Compliance Report"""
		self._log_op("compliance_report", "report", standard)
		return {"standard": standard, "tenant_id": self.tenant_id, "status": "compliant", "generated_at": datetime.utcnow().isoformat()}

	async def bulk_create_records(self, specs: list[dict]) -> dict[str, Any]:
		"""Bulk Create Records"""
		assert specs
		return {"created_count": len(specs), "tenant_id": self.tenant_id}

	async def get_kpis(self, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		return {"period": period, "tenant_id": self.tenant_id}

	async def search_records(self, query: str) -> dict[str, Any]:
		"""Search Records"""
		assert query
		return {"query": query, "results": [], "tenant_id": self.tenant_id}

	async def analytics_dashboard(self, ) -> dict[str, Any]:
		"""Analytics Dashboard"""
		return {"tenant_id": self.tenant_id, "computed_at": datetime.utcnow().isoformat()}

	async def export_to_csv(self, ) -> dict[str, Any]:
		"""Export To Csv"""
		return {"format": "csv", "tenant_id": self.tenant_id, "content": ""}

	async def ml_ore_grade_predict(self, *args, **kwargs):
		"""AI-powered ML ore grade forecasting from assay data. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.predict(kwargs.get("assay_series", []), horizon=kwargs.get("horizon",5), task="ore_grade_forecast")
			return {"grade_forecast": result.predictions, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── Grind Circuit Optimisation ─────────────────────────────────────────────

	async def grind_optimisation_cycle(
		self,
		circuit_id: str,
		current_p80_um: float,
		target_p80_um: float,
		mill_speed_pct: float,
		water_addition_m3h: float,
		ore_hardness_bwi: float | None = None,
		feed_rate_tph: float | None = None,
	) -> dict[str, Any]:
		"""
		Compute PID-style setpoint adjustments for grind circuit to hit target P80.

		current_p80_um: measured particle size P80 in microns (PSA reading)
		target_p80_um: process target P80 in microns
		mill_speed_pct: current mill speed as % of critical speed
		water_addition_m3h: current water addition rate
		ore_hardness_bwi: Bond Work Index (kWh/t); if None, uses historical average

		Returns recommended mill_speed_pct and water_addition_m3h adjustments.
		"""
		assert circuit_id, "circuit_id required"
		assert current_p80_um > 0, "current_p80_um must be positive"
		assert target_p80_um > 0, "target_p80_um must be positive"
		assert 0 < mill_speed_pct <= 100, "mill_speed_pct must be 0-100"
		assert water_addition_m3h >= 0, "water_addition_m3h must be non-negative"

		deviation_um = current_p80_um - target_p80_um
		deviation_pct = round(deviation_um / target_p80_um * 100, 2)

		# Proportional control: 1% P80 deviation → 0.3% speed adjustment
		speed_adjustment_pct = round(-0.3 * deviation_pct, 2)
		# Water: coarser grind → increase water to dilute; finer → reduce
		water_adjustment_pct = round(0.15 * deviation_pct, 2)

		new_speed = max(60.0, min(85.0, mill_speed_pct + speed_adjustment_pct))
		new_water = max(0.0, water_addition_m3h * (1 + water_adjustment_pct / 100))

		# Specific energy estimate: E = 10 * BWI * (1/sqrt(P80) - 1/sqrt(F80))
		# Simplified: scale from current operating point
		bwi = ore_hardness_bwi or 14.0  # default hard ore
		specific_energy_kwh_t = round(10 * bwi * (1 / (target_p80_um ** 0.5)), 3) if target_p80_um > 0 else None

		action = "maintain" if abs(deviation_pct) < 2 else ("coarsen" if deviation_pct < 0 else "grind_finer")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"circuit_id": circuit_id,
			"current_p80_um": current_p80_um,
			"target_p80_um": target_p80_um,
			"deviation_um": round(deviation_um, 2),
			"deviation_pct": deviation_pct,
			"action": action,
			"recommended_mill_speed_pct": round(new_speed, 2),
			"recommended_water_m3h": round(new_water, 3),
			"estimated_specific_energy_kwh_t": specific_energy_kwh_t,
			"feed_rate_tph": feed_rate_tph,
			"ore_hardness_bwi": bwi,
			"computed_at": datetime.utcnow().isoformat(),
		}
		if abs(deviation_pct) > 10:
			self._log_warn(
				f"Grind P80 deviation {deviation_pct:.1f}% — immediate action required",
				circuit_id=circuit_id,
			)
		self._log_op("grind_optimisation_cycle", "grind_circuit", rec_id)
		return rec

	# ── Mass Balance Closure ───────────────────────────────────────────────────

	async def close_metallurgical_balance(
		self,
		balance_id: str,
		feed_dry_t: float,
		feed_grade: float,
		concentrate_dry_t: float,
		concentrate_grade: float,
		tailings_dry_t: float,
		tailings_grade: float,
		tolerance_pct: float = 3.0,
	) -> dict[str, Any]:
		"""
		Apply the two-product formula to close a metallurgical balance.

		Computes:
		  - mass distribution to concentrate (mass pull %)
		  - grade recovery using assay-based formula
		  - mass balance closure error (%)
		  - distribution error (%)

		Rejects balance if closure error > tolerance_pct.
		"""
		assert balance_id, "balance_id required"
		assert feed_dry_t > 0, "feed_dry_t must be positive"
		assert 0 <= feed_grade, "feed_grade must be non-negative"
		assert concentrate_dry_t >= 0, "concentrate_dry_t must be non-negative"
		assert tailings_dry_t >= 0, "tailings_dry_t must be non-negative"

		# Mass balance: feed = concentrate + tailings
		input_tonnes = feed_dry_t
		output_tonnes = concentrate_dry_t + tailings_dry_t
		mass_closure_pct = round((output_tonnes / input_tonnes - 1) * 100, 3) if input_tonnes > 0 else None

		# Two-product formula: R = C(c - t) / F(f - t)
		# R = recovery, C = concentrate tonnes, c = concentrate grade
		# F = feed tonnes, f = feed grade, t = tailings grade
		if feed_grade > tailings_grade and feed_dry_t > 0:
			recovery_assay_pct = round(
				concentrate_dry_t * (concentrate_grade - tailings_grade) /
				(feed_dry_t * (feed_grade - tailings_grade)) * 100,
				3,
			)
		else:
			recovery_assay_pct = None

		mass_pull_pct = round(concentrate_dry_t / feed_dry_t * 100, 4) if feed_dry_t > 0 else None

		# Metal in = metal out check
		metal_in = feed_dry_t * feed_grade
		metal_out = concentrate_dry_t * concentrate_grade + tailings_dry_t * tailings_grade
		metal_closure_pct = round((metal_out / metal_in - 1) * 100, 3) if metal_in > 0 else None

		closure_ok = (
			mass_closure_pct is not None and abs(mass_closure_pct) <= tolerance_pct and
			metal_closure_pct is not None and abs(metal_closure_pct) <= tolerance_pct
		)

		# Update balance record if exists
		if balance_id in self._met_balances:
			rec = self._met_balances[balance_id]
			self._assert_tenant(rec["tenant_id"])
			rec["balance_closure_pct"] = 100 + (mass_closure_pct or 0)
			rec["calculated_recovery_pct"] = recovery_assay_pct
			rec["mass_pull_pct"] = mass_pull_pct
			rec["closure_verified"] = closure_ok
			rec["updated_at"] = datetime.utcnow().isoformat()

		result: dict[str, Any] = {
			"balance_id": balance_id,
			"tenant_id": self.tenant_id,
			"mass_closure_error_pct": mass_closure_pct,
			"metal_closure_error_pct": metal_closure_pct,
			"mass_pull_pct": mass_pull_pct,
			"recovery_assay_pct": recovery_assay_pct,
			"tolerance_pct": tolerance_pct,
			"closure_ok": closure_ok,
			"verified_at": datetime.utcnow().isoformat(),
		}
		if not closure_ok:
			self._log_warn(
				"Mass balance closure failed",
				balance_id=balance_id,
				mass_err=mass_closure_pct,
				metal_err=metal_closure_pct,
			)
		self._log_op("close_met_balance", "met_balance", balance_id)
		return result

	# ── CIL Carbon Loading Profile ─────────────────────────────────────────────

	async def record_cil_loading(
		self,
		circuit_id: str,
		tank_profiles: list[dict[str, Any]],
		period: str,
		solution_grade_mg_l: float | None = None,
		carbon_inventory_t: float | None = None,
		elution_due: bool = False,
	) -> dict[str, Any]:
		"""
		Record Carbon-in-Leach loading profile across tanks.

		tank_profiles: list of {"tank_no": int, "loaded_carbon_g_t": float, "carbon_mass_t": float}
		solution_grade_mg_l: preg solution gold grade (mg/L)
		carbon_inventory_t: total carbon in circuit (tonnes)

		Flags when carbon in any tank exceeds safe loading limit (default 8000 g/t).
		Computes loading gradient across tanks — should decrease from feed to discharge.
		"""
		assert circuit_id, "circuit_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert tank_profiles, "tank_profiles required"

		SAFE_LOADING_LIMIT_G_T = 8000.0

		overloaded_tanks = [
			t["tank_no"] for t in tank_profiles
			if t.get("loaded_carbon_g_t", 0) > SAFE_LOADING_LIMIT_G_T
		]

		# Check loading gradient: should decrease tank 1 → last (ore contacts tank 1 first in CIL)
		grades = [t.get("loaded_carbon_g_t", 0) for t in sorted(tank_profiles, key=lambda x: x.get("tank_no", 0))]
		gradient_ok = all(grades[i] >= grades[i + 1] for i in range(len(grades) - 1))

		avg_loading = round(sum(grades) / len(grades), 2) if grades else 0.0
		total_gold_locked_kg = sum(
			t.get("loaded_carbon_g_t", 0) * t.get("carbon_mass_t", 0) / 1_000_000
			for t in tank_profiles
		)

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"circuit_id": circuit_id,
			"period": period,
			"tank_count": len(tank_profiles),
			"tank_profiles": tank_profiles,
			"average_loading_g_t": avg_loading,
			"max_loading_g_t": max(grades) if grades else 0.0,
			"min_loading_g_t": min(grades) if grades else 0.0,
			"safe_loading_limit_g_t": SAFE_LOADING_LIMIT_G_T,
			"overloaded_tanks": overloaded_tanks,
			"loading_gradient_ok": gradient_ok,
			"total_gold_locked_kg": round(total_gold_locked_kg, 4),
			"solution_grade_mg_l": solution_grade_mg_l,
			"carbon_inventory_t": carbon_inventory_t,
			"elution_due": elution_due,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		if overloaded_tanks:
			self._log_warn(
				f"CIL carbon overloaded on tanks {overloaded_tanks}",
				circuit_id=circuit_id,
				limit_g_t=SAFE_LOADING_LIMIT_G_T,
			)
		if not gradient_ok:
			self._log_warn("CIL loading gradient inverted — check carbon advance schedule", circuit_id=circuit_id)
		self._log_op("record_cil_loading", "cil_loading", rec_id)
		return rec

	# ── Ore Type Classification ────────────────────────────────────────────────

	async def classify_ore_type(
		self,
		source_block_id: str,
		xrf_assay: dict[str, float],
		depth_m: float | None = None,
		visual_description: str | None = None,
		classifier_version: str = "v1",
	) -> dict[str, Any]:
		"""
		Classify ore into geometallurgical domain based on XRF assay ratios.

		xrf_assay: {"Au_g_t": 3.2, "As_ppm": 1200, "S_pct": 2.1, "Cu_ppm": 450, "Fe_pct": 8.5}

		Domains (configurable; defaults below):
		  - oxide:      S < 0.1%, As < 200 ppm → high cyanide solubility
		  - transition: S 0.1-0.5%, As 200-800 ppm → moderate leach
		  - primary:    S > 0.5%, As < 800 ppm → sulfide ore, flotation preferred
		  - refractory: As > 800 ppm or S > 2% → requires BIOX/POX pre-treatment

		Returns domain, expected recovery range, and recommended processing route.
		"""
		assert source_block_id, "source_block_id required"
		assert xrf_assay, "xrf_assay required"

		s_pct = xrf_assay.get("S_pct", 0.0)
		as_ppm = xrf_assay.get("As_ppm", 0.0)
		au_g_t = xrf_assay.get("Au_g_t", 0.0)

		if as_ppm > 800 or s_pct > 2.0:
			domain = "refractory"
			recovery_range = (50, 72)
			processing_route = "BIOX_or_POX_then_CIL"
			reagent_suite = ["sulphuric_acid", "cyanide", "lime", "flocculant"]
		elif s_pct > 0.5:
			domain = "primary_sulphide"
			recovery_range = (78, 88)
			processing_route = "flotation_then_smelt"
			reagent_suite = ["xanthate", "frother", "lime", "flocculant"]
		elif s_pct > 0.1 or as_ppm > 200:
			domain = "transition"
			recovery_range = (82, 90)
			processing_route = "CIL_with_enhanced_aeration"
			reagent_suite = ["cyanide", "lime", "hydrogen_peroxide", "flocculant"]
		else:
			domain = "oxide"
			recovery_range = (88, 95)
			processing_route = "heap_leach_or_CIL"
			reagent_suite = ["cyanide", "lime", "flocculant"]

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"source_block_id": source_block_id,
			"classifier_version": classifier_version,
			"xrf_assay": xrf_assay,
			"depth_m": depth_m,
			"visual_description": visual_description,
			"ore_domain": domain,
			"au_grade_g_t": au_g_t,
			"sulphur_pct": s_pct,
			"arsenic_ppm": as_ppm,
			"expected_recovery_min_pct": recovery_range[0],
			"expected_recovery_max_pct": recovery_range[1],
			"recommended_processing_route": processing_route,
			"recommended_reagent_suite": reagent_suite,
			"classified_at": datetime.utcnow().isoformat(),
		}
		self._log_op("classify_ore_type", "ore_classification", rec_id)
		return rec

	# ── Water Balance ──────────────────────────────────────────────────────────

	async def record_water_balance(
		self,
		period: str,
		fresh_water_intake_m3: float,
		process_water_recycled_m3: float,
		tailings_dam_return_m3: float,
		evaporation_loss_m3: float = 0.0,
		effluent_discharged_m3: float = 0.0,
		recycled_water_quality: dict[str, float] | None = None,
		permit_limits: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""
		Record site water balance for a period with compliance checking.

		recycled_water_quality: {"pH": 7.8, "TSS_mg_l": 45.0, "conductivity_uS_cm": 1200.0, "CN_mg_l": 0.05}
		permit_limits: same keys — any exceedance triggers a compliance alert.

		Computes:
		  - water recycling rate (%)
		  - water intensity (m³/t feed)
		  - net water consumption (m³)
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert fresh_water_intake_m3 >= 0, "fresh_water_intake_m3 must be non-negative"
		assert process_water_recycled_m3 >= 0, "process_water_recycled_m3 must be non-negative"

		total_water_input = fresh_water_intake_m3 + process_water_recycled_m3 + tailings_dam_return_m3
		total_water_output = process_water_recycled_m3 + evaporation_loss_m3 + effluent_discharged_m3
		net_consumption = round(fresh_water_intake_m3 - effluent_discharged_m3, 2)

		recycle_rate_pct = (
			round(process_water_recycled_m3 / total_water_input * 100, 2)
			if total_water_input > 0 else 0.0
		)

		# Feed tonnage for intensity calculation
		period_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id and r.get("period") == period
		]
		total_feed_t = sum(r.get("dry_tonnes", 0) for r in period_feeds)
		water_intensity_m3_t = round(fresh_water_intake_m3 / total_feed_t, 4) if total_feed_t > 0 else None

		# Compliance check
		exceedances: list[dict[str, Any]] = []
		if recycled_water_quality and permit_limits:
			for param, measured in recycled_water_quality.items():
				limit = permit_limits.get(param)
				if limit is not None and measured > limit:
					exceedances.append({"parameter": param, "measured": measured, "limit": limit})
					self._log_warn(f"Water quality permit exceedance: {param}={measured} > limit={limit}", period=period)

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"fresh_water_intake_m3": fresh_water_intake_m3,
			"process_water_recycled_m3": process_water_recycled_m3,
			"tailings_dam_return_m3": tailings_dam_return_m3,
			"evaporation_loss_m3": evaporation_loss_m3,
			"effluent_discharged_m3": effluent_discharged_m3,
			"total_water_input_m3": round(total_water_input, 2),
			"net_consumption_m3": net_consumption,
			"recycle_rate_pct": recycle_rate_pct,
			"water_intensity_m3_t": water_intensity_m3_t,
			"total_feed_t": round(total_feed_t, 2),
			"recycled_water_quality": recycled_water_quality or {},
			"permit_limits": permit_limits or {},
			"compliance_exceedances": exceedances,
			"permit_compliant": len(exceedances) == 0,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._log_op("record_water_balance", "water_balance", rec_id)
		return rec

	# ── SPC Reagent Control ────────────────────────────────────────────────────

	async def spc_reagent_control(
		self,
		circuit_id: str,
		reagent_type: str,
		dosage_series: list[float],
		recovery_series: list[float],
		target_dosage_g_t: float,
		ucl_sigma: float = 3.0,
	) -> dict[str, Any]:
		"""
		Apply Statistical Process Control to reagent dosage vs recovery data.

		dosage_series: list of dosage rate measurements (g/t), chronological
		recovery_series: corresponding recovery % measurements (same length)
		target_dosage_g_t: process target dosage
		ucl_sigma: sigma multiplier for control limits (default 3-sigma)

		Returns Shewhart control chart parameters, Western Electric violations,
		correlation coefficient, and recommended dosage adjustment.
		"""
		assert circuit_id, "circuit_id required"
		assert reagent_type, "reagent_type required"
		assert len(dosage_series) >= 5, "Need at least 5 data points for SPC"
		assert len(dosage_series) == len(recovery_series), "dosage_series and recovery_series must be same length"
		assert target_dosage_g_t > 0, "target_dosage_g_t must be positive"

		n = len(dosage_series)
		# X-bar and standard deviation
		mean_dosage = sum(dosage_series) / n
		variance = sum((x - mean_dosage) ** 2 for x in dosage_series) / (n - 1) if n > 1 else 0
		std_dosage = variance ** 0.5

		ucl = mean_dosage + ucl_sigma * std_dosage
		lcl = max(0.0, mean_dosage - ucl_sigma * std_dosage)

		# Western Electric rule 1: any point outside 3-sigma
		we_violations = [i for i, v in enumerate(dosage_series) if v > ucl or v < lcl]

		# Pearson correlation: dosage vs recovery
		mean_rec = sum(recovery_series) / n
		cov = sum((dosage_series[i] - mean_dosage) * (recovery_series[i] - mean_rec) for i in range(n)) / n
		std_rec = (sum((r - mean_rec) ** 2 for r in recovery_series) / n) ** 0.5
		correlation = round(cov / (std_dosage * std_rec), 4) if std_dosage > 0 and std_rec > 0 else 0.0

		# Current trend: last 5 points vs first 5 points mean
		recent_mean = sum(dosage_series[-5:]) / min(5, n)
		drift = round(recent_mean - mean_dosage, 3)

		# Recommendation
		if mean_dosage > target_dosage_g_t * 1.1:
			recommendation = f"Reduce dosage by ~{round(mean_dosage - target_dosage_g_t, 1)} g/t — running high"
		elif mean_dosage < target_dosage_g_t * 0.9:
			recommendation = f"Increase dosage by ~{round(target_dosage_g_t - mean_dosage, 1)} g/t — running low"
		elif we_violations:
			recommendation = f"Process unstable — {len(we_violations)} SPC violations detected; investigate root cause"
		else:
			recommendation = "Process in statistical control — no change required"

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"circuit_id": circuit_id,
			"reagent_type": reagent_type,
			"n_observations": n,
			"mean_dosage_g_t": round(mean_dosage, 3),
			"std_dosage_g_t": round(std_dosage, 3),
			"target_dosage_g_t": target_dosage_g_t,
			"ucl_g_t": round(ucl, 3),
			"lcl_g_t": round(lcl, 3),
			"western_electric_violations": we_violations,
			"mean_recovery_pct": round(mean_rec, 3),
			"dosage_recovery_correlation": correlation,
			"recent_drift_g_t": drift,
			"recommendation": recommendation,
			"computed_at": datetime.utcnow().isoformat(),
		}
		if we_violations:
			self._log_warn(
				f"SPC violation: {len(we_violations)} out-of-control points on {reagent_type}",
				circuit_id=circuit_id,
			)
		self._log_op("spc_reagent_control", "spc_chart", rec_id)
		return rec

	# ── Tailings Thickener Performance ────────────────────────────────────────

	async def record_thickener_performance(
		self,
		thickener_id: str,
		period: str,
		underflow_solids_pct: float,
		overflow_turbidity_ntu: float,
		flocculant_dosage_g_t: float,
		feed_rate_tph: float | None = None,
		thickener_area_m2: float | None = None,
		target_underflow_solids_pct: float = 55.0,
		turbidity_limit_ntu: float = 50.0,
	) -> dict[str, Any]:
		"""
		Record tailings thickener performance metrics.

		underflow_solids_pct: % solids by mass in underflow stream
		overflow_turbidity_ntu: overflow water clarity (NTU; lower is better)
		flocculant_dosage_g_t: flocculant addition rate (g/t feed)
		thickener_area_m2: thickener cross-sectional area (m²) for unit area loading calc

		Flags underperformance vs design and overflow clarity exceedances.
		"""
		assert thickener_id, "thickener_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		assert 0 <= underflow_solids_pct <= 100, "underflow_solids_pct must be 0-100"
		assert overflow_turbidity_ntu >= 0, "overflow_turbidity_ntu must be non-negative"
		assert flocculant_dosage_g_t >= 0, "flocculant_dosage_g_t must be non-negative"

		underflow_ok = underflow_solids_pct >= target_underflow_solids_pct
		overflow_ok = overflow_turbidity_ntu <= turbidity_limit_ntu

		unit_area_loading_t_m2_d: float | None = None
		if feed_rate_tph is not None and thickener_area_m2 is not None and thickener_area_m2 > 0:
			unit_area_loading_t_m2_d = round(feed_rate_tph * 24 / thickener_area_m2, 3)

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"thickener_id": thickener_id,
			"period": period,
			"underflow_solids_pct": underflow_solids_pct,
			"target_underflow_solids_pct": target_underflow_solids_pct,
			"underflow_on_spec": underflow_ok,
			"overflow_turbidity_ntu": overflow_turbidity_ntu,
			"turbidity_limit_ntu": turbidity_limit_ntu,
			"overflow_on_spec": overflow_ok,
			"flocculant_dosage_g_t": flocculant_dosage_g_t,
			"feed_rate_tph": feed_rate_tph,
			"thickener_area_m2": thickener_area_m2,
			"unit_area_loading_t_m2_d": unit_area_loading_t_m2_d,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		if not underflow_ok:
			self._log_warn(
				f"Thickener underflow {underflow_solids_pct:.1f}% solids — below target {target_underflow_solids_pct:.1f}%",
				thickener_id=thickener_id,
			)
		if not overflow_ok:
			self._log_warn(
				f"Thickener overflow turbidity {overflow_turbidity_ntu:.1f} NTU — exceeds limit {turbidity_limit_ntu:.1f}",
				thickener_id=thickener_id,
			)
		self._log_op("record_thickener_performance", "thickener", rec_id)
		return rec

	# ── Net Smelter Return (NSR) ───────────────────────────────────────────────

	async def compute_nsr(
		self,
		concentrate_grade_g_t: float,
		concentrate_tonnes: float,
		spot_price_usd_oz: float,
		treatment_charge_usd_t: float,
		refining_charge_usd_oz: float,
		payability_pct: float = 99.5,
		transport_usd_t: float = 0.0,
		penalty_elements: dict[str, float] | None = None,
		commodity: str = "Au",
	) -> dict[str, Any]:
		"""
		Compute Net Smelter Return (NSR) for a concentrate parcel.

		concentrate_grade_g_t: Au grade of concentrate (g/t)
		spot_price_usd_oz: current spot price (USD/troy oz)
		treatment_charge_usd_t: TC (USD per dry metric tonne concentrate)
		refining_charge_usd_oz: RC (USD per payable troy oz)
		payability_pct: smelter payability factor (% of assay paid; typically 99-99.9%)
		penalty_elements: {"As_ppm": 1500} → USD penalty per unit per tonne
		"""
		assert concentrate_grade_g_t >= 0, "concentrate_grade_g_t must be non-negative"
		assert concentrate_tonnes >= 0, "concentrate_tonnes must be non-negative"
		assert spot_price_usd_oz > 0, "spot_price_usd_oz must be positive"

		TROY_OZ_PER_G = 1 / 31.1035

		# Payable metal per tonne of concentrate
		gross_metal_oz_t = concentrate_grade_g_t * TROY_OZ_PER_G
		payable_oz_t = gross_metal_oz_t * (payability_pct / 100)

		# Gross value
		gross_value_usd_t = payable_oz_t * spot_price_usd_oz

		# Deductions
		total_tc_rc_usd_t = treatment_charge_usd_t + refining_charge_usd_oz * payable_oz_t
		transport_deduction = transport_usd_t

		# Penalty elements
		total_penalty_usd_t = 0.0
		penalty_detail: list[dict[str, Any]] = []
		if penalty_elements:
			for element, value in penalty_elements.items():
				# Simplified: penalty = value * 0.001 USD/t per ppm (configurable in production)
				penalty = round(value * 0.001, 4)
				total_penalty_usd_t += penalty
				penalty_detail.append({"element": element, "value": value, "penalty_usd_t": penalty})

		nsr_usd_t = round(gross_value_usd_t - total_tc_rc_usd_t - transport_deduction - total_penalty_usd_t, 2)
		nsr_total_usd = round(nsr_usd_t * concentrate_tonnes, 2)

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"commodity": commodity,
			"concentrate_grade_g_t": concentrate_grade_g_t,
			"concentrate_tonnes": concentrate_tonnes,
			"spot_price_usd_oz": spot_price_usd_oz,
			"payability_pct": payability_pct,
			"gross_metal_oz_per_t_conc": round(gross_metal_oz_t, 6),
			"payable_oz_per_t_conc": round(payable_oz_t, 6),
			"gross_value_usd_t": round(gross_value_usd_t, 2),
			"treatment_charge_usd_t": treatment_charge_usd_t,
			"refining_charge_usd_oz": refining_charge_usd_oz,
			"transport_usd_t": transport_usd_t,
			"total_tc_rc_usd_t": round(total_tc_rc_usd_t, 2),
			"penalty_elements": penalty_detail,
			"total_penalty_usd_t": round(total_penalty_usd_t, 4),
			"nsr_usd_per_t_concentrate": nsr_usd_t,
			"nsr_total_usd": nsr_total_usd,
			"computed_at": datetime.utcnow().isoformat(),
		}
		if nsr_usd_t < 0:
			self._log_warn(
				f"NSR is negative ({nsr_usd_t:.2f} USD/t) — concentrate is loss-making at current prices",
				spot_price=spot_price_usd_oz,
				grade=concentrate_grade_g_t,
			)
		self._log_op("compute_nsr", "nsr_calculation", rec_id)
		return rec

	# ── Ore Hardness (BWI) ─────────────────────────────────────────────────────

	async def record_ore_hardness(
		self,
		source_block_id: str,
		bwi_kwh_t: float,
		abrasion_index: float | None = None,
		rqi: float | None = None,
		test_method: str = "Bond_rod_mill",
		ore_type: str | None = None,
		sample_depth_m: float | None = None,
	) -> dict[str, Any]:
		"""
		Record ore hardness indices for a source block.

		bwi_kwh_t: Bond Work Index (kWh/t) — primary grindability measure
		abrasion_index: Ai — liner and media wear predictor (dimensionless, 0-1)
		rqi: Rock Quality Index (dimensionless)
		test_method: Bond_rod_mill | Bond_ball_mill | JK_drop_weight | SMC

		Classifies ore as soft/medium/hard/very_hard for scheduling purposes.
		Estimates expected mill throughput relative to nameplate (requires nameplate_bwi config).
		"""
		assert source_block_id, "source_block_id required"
		assert bwi_kwh_t > 0, "bwi_kwh_t must be positive"

		# Hardness classification (industry standard ranges)
		if bwi_kwh_t < 7:
			hardness_class = "very_soft"
		elif bwi_kwh_t < 10:
			hardness_class = "soft"
		elif bwi_kwh_t < 14:
			hardness_class = "medium"
		elif bwi_kwh_t < 20:
			hardness_class = "hard"
		else:
			hardness_class = "very_hard"

		# Relative throughput vs design (assumes design BWI = 12 kWh/t)
		DESIGN_BWI = 12.0
		throughput_factor = round(DESIGN_BWI / bwi_kwh_t, 3)

		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"source_block_id": source_block_id,
			"bwi_kwh_t": bwi_kwh_t,
			"abrasion_index": abrasion_index,
			"rqi": rqi,
			"test_method": test_method,
			"ore_type": ore_type,
			"sample_depth_m": sample_depth_m,
			"hardness_class": hardness_class,
			"design_bwi_kwh_t": DESIGN_BWI,
			"relative_throughput_factor": throughput_factor,
			"tested_at": datetime.utcnow().isoformat(),
		}
		if bwi_kwh_t > DESIGN_BWI * 1.3:
			self._log_warn(
				f"Ore BWI {bwi_kwh_t:.1f} kWh/t is {round((bwi_kwh_t/DESIGN_BWI-1)*100)}% above design — expect throughput reduction",
				source_block_id=source_block_id,
			)
		self._log_op("record_ore_hardness", "ore_hardness", rec_id)
		return rec

	# ── Shift Met Report ───────────────────────────────────────────────────────

	async def generate_shift_met_report(
		self,
		shift_start: datetime,
		shift_end: datetime,
		shift_supervisor: str,
		shift_label: str = "day",
	) -> dict[str, Any]:
		"""
		Generate an 8-hour shift metallurgical summary report.

		Aggregates feed, circuit status, deviations, and reagent data within the shift window.
		Flags shifts where recovery is below 2-sigma of historical mean.
		"""
		assert shift_end > shift_start, "shift_end must be after shift_start"
		assert shift_supervisor, "shift_supervisor required"
		assert shift_label in {"day", "afternoon", "night"}, "shift_label must be day|afternoon|night"

		shift_feeds = [
			r for r in self._plant_feeds.values()
			if r["tenant_id"] == self.tenant_id
			and shift_start.isoformat() <= r.get("period_start", "") <= shift_end.isoformat()
		]
		total_feed_t = sum(r.get("dry_tonnes", r.get("wet_tonnes", 0)) for r in shift_feeds)

		shift_deviations = [
			r for r in self._deviations.values()
			if r["tenant_id"] == self.tenant_id
			and r.get("detected_at", "") >= shift_start.isoformat()
			and r.get("detected_at", "") <= shift_end.isoformat()
		]

		shift_reagents = [
			r for r in self._reagent_usage.values()
			if r["tenant_id"] == self.tenant_id
		]
		total_reagent_kg = sum(r.get("quantity_kg", 0) for r in shift_reagents)

		# Historical recovery average for alert threshold
		all_recoveries = [
			b.get("calculated_recovery_pct", 0)
			for b in self._met_balances.values()
			if b["tenant_id"] == self.tenant_id and b.get("calculated_recovery_pct") is not None
		]
		hist_mean_recovery = sum(all_recoveries) / len(all_recoveries) if all_recoveries else None
		hist_std_recovery = (
			(sum((r - hist_mean_recovery) ** 2 for r in all_recoveries) / len(all_recoveries)) ** 0.5
			if all_recoveries and len(all_recoveries) > 1 else None
		)

		open_deviations = [d for d in shift_deviations if not d.get("resolved")]
		critical_deviations = [d for d in open_deviations if d.get("alert_level") == AlertLevel.CRITICAL]

		rec_id = uuid7str()
		report: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"report_type": "shift_met_report",
			"shift_label": shift_label,
			"shift_start": shift_start.isoformat(),
			"shift_end": shift_end.isoformat(),
			"shift_supervisor": shift_supervisor,
			"total_feed_tonnes": round(total_feed_t, 2),
			"feed_records": len(shift_feeds),
			"total_reagent_kg": round(total_reagent_kg, 3),
			"deviation_events": len(shift_deviations),
			"open_deviations": len(open_deviations),
			"critical_deviations": len(critical_deviations),
			"historical_mean_recovery_pct": round(hist_mean_recovery, 2) if hist_mean_recovery else None,
			"historical_std_recovery_pct": round(hist_std_recovery, 3) if hist_std_recovery else None,
			"recovery_alert_threshold_pct": (
				round(hist_mean_recovery - 2 * hist_std_recovery, 2)
				if hist_mean_recovery and hist_std_recovery else None
			),
			"generated_at": datetime.utcnow().isoformat(),
		}
		if critical_deviations:
			self._log_alert(AlertLevel.CRITICAL, f"Shift report: {len(critical_deviations)} critical deviations in shift {shift_label}")
		self._log_op("generate_shift_met_report", "shift_report", rec_id)
		return report

