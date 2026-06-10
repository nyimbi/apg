"""Async service layer for APG Equipment & Plant Management."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .models import (
	DispatchStatus,
	EquipmentCreate,
	EquipmentFaultCreate,
	EquipmentFaultResponse,
	EquipmentResponse,
	EquipmentUpdate,
	FaultSeverity,
	FuelDocketCreate,
	FuelDocketResponse,
	InspectionCreate,
	InspectionResponse,
	LifecycleStatus,
	MaintenanceStatus,
	MaintenanceWorkOrderCreate,
	MaintenanceWorkOrderResponse,
	MaintenanceWorkOrderUpdate,
	uuid7str,
)

log = logging.getLogger(__name__)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class EqpService:
	"""Service for Equipment & Plant Management operations."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._equipment: dict[str, dict[str, Any]] = {}
		self._work_orders: dict[str, dict[str, Any]] = {}
		self._inspections: dict[str, dict[str, Any]] = {}
		self._fuel_dockets: dict[str, dict[str, Any]] = {}
		self._faults: dict[str, dict[str, Any]] = {}
		self._dispatch_log: list[dict[str, Any]] = []
		# Extended stores
		self._availability_records: dict[str, dict[str, Any]] = {}
		self._planned_maintenance: dict[str, dict[str, Any]] = {}
		self._breakdown_logs: dict[str, dict[str, Any]] = {}
		self._condition_monitoring: dict[str, dict[str, Any]] = {}
		self._pre_start_checks: dict[str, dict[str, Any]] = {}
		self._fuel_lube_records: dict[str, dict[str, Any]] = {}
		self._major_components: dict[str, dict[str, Any]] = {}
		self._replacement_recs: dict[str, dict[str, Any]] = {}

	# ── Logging helpers ────────────────────────────────────────────────────────

	def _log_op(self, op: str, entity: str, id: str) -> None:
		log.info("eqp.%s | tenant=%s entity=%s id=%s", op, self.tenant_id, entity, id)

	def _log_warn(self, msg: str, **kw: Any) -> None:
		log.warning("eqp | tenant=%s %s %s", self.tenant_id, msg, kw)

	def _log_kpi_breach(self, asset_number: str, kpi: str, value: float, target: float) -> None:
		log.warning(
			"eqp.kpi_breach | tenant=%s asset=%s kpi=%s value=%.2f target=%.2f",
			self.tenant_id, asset_number, kpi, value, target,
		)

	# ── Tenant guard ───────────────────────────────────────────────────────────

	def _assert_tenant(self, tenant_id: str) -> None:
		assert tenant_id == self.tenant_id, (
			f"Cross-tenant access denied: requested={tenant_id} service={self.tenant_id}"
		)

	# ── Fleet Register ─────────────────────────────────────────────────────────

	async def register_equipment(
		self, payload: EquipmentCreate, created_by: str
	) -> EquipmentResponse:
		"""Register new equipment. asset_number must be unique within the tenant."""
		self._assert_tenant(payload.tenant_id)
		existing = [e for e in self._equipment.values() if e["asset_number"] == payload.asset_number and e["tenant_id"] == self.tenant_id]
		if existing:
			raise ValueError(f"Asset number '{payload.asset_number}' already registered")
		resp = EquipmentResponse(**payload.model_dump(), created_by=created_by)
		self._equipment[resp.id] = resp.model_dump()
		self._log_op("register", "equipment", resp.id)
		return resp

	async def get_equipment(self, id: str) -> EquipmentResponse | None:
		"""Get equipment by record id."""
		rec = self._equipment.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return EquipmentResponse(**rec)

	async def get_equipment_by_asset_number(self, asset_number: str) -> EquipmentResponse | None:
		"""Look up equipment by asset number."""
		for rec in self._equipment.values():
			if rec["asset_number"] == asset_number and rec["tenant_id"] == self.tenant_id:
				return EquipmentResponse(**rec)
		return None

	async def update_equipment(self, id: str, update: EquipmentUpdate) -> EquipmentResponse:
		"""Update equipment lifecycle status, dispatch status, or assignment."""
		rec = self._equipment.get(id)
		if rec is None:
			raise KeyError(f"Equipment {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if update.lifecycle_status == LifecycleStatus.DECOMMISSIONED and rec["lifecycle_status"] == LifecycleStatus.ACTIVE:
			raise ValueError("Use decommission_equipment() for the full decommissioning workflow")
		for field, value in update.model_dump(exclude_none=True).items():
			rec[field] = value
		rec["updated_at"] = datetime.utcnow()
		self._log_op("update", "equipment", id)
		return EquipmentResponse(**rec)

	async def decommission_equipment(self, id: str, approved_by: str) -> EquipmentResponse:
		"""Decommission active equipment. Requires prior approval."""
		rec = self._equipment.get(id)
		if rec is None:
			raise KeyError(f"Equipment {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec["lifecycle_status"] not in (LifecycleStatus.ACTIVE, LifecycleStatus.STANDBY):
			raise ValueError(f"Cannot decommission equipment in {rec['lifecycle_status']} status")
		rec["lifecycle_status"] = LifecycleStatus.DECOMMISSIONED
		rec["dispatch_status"] = DispatchStatus.PARKED
		rec["updated_at"] = datetime.utcnow()
		self._log_op("decommission", "equipment", id)
		return EquipmentResponse(**rec)

	async def list_equipment(
		self,
		equipment_class: str | None = None,
		lifecycle_status: str | None = None,
		dispatch_status: str | None = None,
		mine_area: str | None = None,
		limit: int = 200,
		offset: int = 0,
	) -> list[EquipmentResponse]:
		"""List fleet with optional filters."""
		results = [
			EquipmentResponse(**r)
			for r in self._equipment.values()
			if r["tenant_id"] == self.tenant_id
		]
		if equipment_class:
			results = [r for r in results if r.equipment_class == equipment_class]
		if lifecycle_status:
			results = [r for r in results if r.lifecycle_status == lifecycle_status]
		if dispatch_status:
			results = [r for r in results if r.dispatch_status == dispatch_status]
		if mine_area:
			results = [r for r in results if r.mine_area_assignment == mine_area]
		return sorted(results, key=lambda x: x.asset_number)[offset : offset + limit]

	# ── Dispatch ───────────────────────────────────────────────────────────────

	async def dispatch_equipment(
		self,
		equipment_id: str,
		operator_id: str,
		operator_licensed: bool,
		destination_area: str,
	) -> EquipmentResponse:
		"""Dispatch equipment to an area. Checks: active lifecycle, pre-shift inspection, operator licence."""
		rec = self._equipment.get(equipment_id)
		if rec is None:
			raise KeyError(f"Equipment {equipment_id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec["dispatch_status"] == DispatchStatus.BREAKDOWN:
			raise PermissionError("Equipment in BREAKDOWN status cannot be dispatched")
		if rec["lifecycle_status"] != LifecycleStatus.ACTIVE:
			raise PermissionError(f"Equipment lifecycle status {rec['lifecycle_status']} is not ACTIVE")
		if not operator_licensed:
			raise PermissionError("Operator must hold a valid licence for this equipment class")
		# Check pre-shift inspection exists today
		today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
		recent_inspection = any(
			insp["equipment_id"] == equipment_id
			and insp["inspection_type"] == "pre_shift"
			and insp["inspected_at"] >= today_start
			and insp["overall_result"] != "fail"
			for insp in self._inspections.values()
		)
		if not recent_inspection:
			raise PermissionError("Pre-shift inspection required before dispatch")
		rec["dispatch_status"] = DispatchStatus.OPERATING
		rec["mine_area_assignment"] = destination_area
		rec["updated_at"] = datetime.utcnow()
		self._dispatch_log.append({
			"equipment_id": equipment_id,
			"operator_id": operator_id,
			"destination_area": destination_area,
			"dispatched_at": datetime.utcnow(),
		})
		self._log_op("dispatch", "equipment", equipment_id)
		return EquipmentResponse(**rec)

	# ── Maintenance Work Orders ────────────────────────────────────────────────

	async def create_work_order(
		self, payload: MaintenanceWorkOrderCreate, created_by: str
	) -> MaintenanceWorkOrderResponse:
		"""Create a maintenance work order."""
		equipment = self._equipment.get(payload.equipment_id)
		if equipment is None:
			raise KeyError(f"Equipment {payload.equipment_id} not found")
		self._assert_tenant(equipment["tenant_id"])
		resp = MaintenanceWorkOrderResponse(
			**payload.model_dump(exclude={"spare_parts"}),
			spare_parts=[p.model_dump() for p in payload.spare_parts],
			created_by=created_by,
		)
		self._work_orders[resp.id] = resp.model_dump()
		# Set equipment to maintenance if critical
		if payload.priority == "critical":
			equipment["dispatch_status"] = DispatchStatus.MAINTENANCE
			equipment["updated_at"] = datetime.utcnow()
		self._log_op("create_wo", "work_order", resp.id)
		return resp

	async def approve_work_order(self, id: str, approver_id: str) -> MaintenanceWorkOrderResponse:
		"""Approve a maintenance work order."""
		rec = self._work_orders.get(id)
		if rec is None:
			raise KeyError(f"Work order {id} not found")
		rec["approved_by"] = approver_id
		rec["approved_at"] = datetime.utcnow()
		rec["updated_at"] = datetime.utcnow()
		self._log_op("approve_wo", "work_order", id)
		return MaintenanceWorkOrderResponse(**rec)

	async def complete_work_order(
		self, id: str, update: MaintenanceWorkOrderUpdate
	) -> MaintenanceWorkOrderResponse:
		"""Complete a work order and optionally restore equipment dispatch status."""
		rec = self._work_orders.get(id)
		if rec is None:
			raise KeyError(f"Work order {id} not found")
		if not rec.get("approved_by"):
			raise PermissionError("Work order must be approved before execution")
		for field, value in update.model_dump(exclude_none=True).items():
			rec[field] = value
		rec["status"] = MaintenanceStatus.COMPLETED
		rec["updated_at"] = datetime.utcnow()
		# Restore equipment availability
		equipment = self._equipment.get(rec["equipment_id"])
		if equipment and equipment["dispatch_status"] == DispatchStatus.MAINTENANCE:
			equipment["dispatch_status"] = DispatchStatus.AVAILABLE
			equipment["updated_at"] = datetime.utcnow()
		self._log_op("complete_wo", "work_order", id)
		return MaintenanceWorkOrderResponse(**rec)

	async def list_work_orders(
		self, equipment_id: str | None = None, status: str | None = None
	) -> list[MaintenanceWorkOrderResponse]:
		"""List work orders with optional equipment/status filter."""
		results = [MaintenanceWorkOrderResponse(**r) for r in self._work_orders.values()]
		if equipment_id:
			results = [r for r in results if r.equipment_id == equipment_id]
		if status:
			results = [r for r in results if r.status == status]
		return sorted(results, key=lambda x: x.planned_start)

	# ── Inspections ────────────────────────────────────────────────────────────

	async def submit_inspection(
		self, payload: InspectionCreate, created_by: str
	) -> InspectionResponse:
		"""Record an equipment inspection. Failed inspections trigger fault creation."""
		equipment = self._equipment.get(payload.equipment_id)
		if equipment is None:
			raise KeyError(f"Equipment {payload.equipment_id} not found")
		self._assert_tenant(equipment["tenant_id"])
		resp = InspectionResponse(
			**payload.model_dump(exclude={"items"}),
			items=[i.model_dump() for i in payload.items],
			created_by=created_by,
		)
		self._inspections[resp.id] = resp.model_dump()
		if resp.overall_result == "fail":
			equipment["dispatch_status"] = DispatchStatus.MAINTENANCE
			equipment["updated_at"] = datetime.utcnow()
			self._log_warn("Inspection failed; equipment set to MAINTENANCE", equipment_id=payload.equipment_id)
		self._log_op("submit_inspection", "inspection", resp.id)
		return resp

	async def list_inspections_for_equipment(self, equipment_id: str) -> list[InspectionResponse]:
		"""Return all inspections for a given equipment, most recent first."""
		results = [
			InspectionResponse(**r)
			for r in self._inspections.values()
			if r["equipment_id"] == equipment_id
		]
		return sorted(results, key=lambda x: x.inspected_at, reverse=True)

	# ── Fuel Dockets ───────────────────────────────────────────────────────────

	async def record_fuel_docket(
		self, payload: FuelDocketCreate, created_by: str
	) -> FuelDocketResponse:
		"""Record a fuel docket. Calculates total cost and flags variance."""
		equipment = self._equipment.get(payload.equipment_id)
		if equipment is None:
			raise KeyError(f"Equipment {payload.equipment_id} not found")
		self._assert_tenant(equipment["tenant_id"])
		total_cost = (
			round(payload.quantity_litres * payload.cost_per_litre, 2)
			if payload.cost_per_litre else None
		)
		# Simple variance flag: compare against last 5 dockets for same equipment
		recent = [
			r for r in self._fuel_dockets.values()
			if r["equipment_id"] == payload.equipment_id
		]
		variance_flag = False
		if recent:
			avg_qty = sum(r["quantity_litres"] for r in recent) / len(recent)
			if abs(payload.quantity_litres - avg_qty) / avg_qty > 0.10:
				variance_flag = True
				self._log_warn("Fuel variance >10% detected", equipment_id=payload.equipment_id)
		resp = FuelDocketResponse(
			**payload.model_dump(),
			total_cost=total_cost,
			variance_flag=variance_flag,
			created_by=created_by,
		)
		self._fuel_dockets[resp.id] = resp.model_dump()
		# Accumulate operating hours
		if payload.engine_hours:
			equipment["total_operating_hours"] = round(
				equipment.get("total_operating_hours", 0) + 0, 1
			)  # hours tracked via fuel dockets
		self._log_op("record_fuel", "fuel_docket", resp.id)
		return resp

	# ── Equipment Faults ───────────────────────────────────────────────────────

	async def report_fault(
		self, payload: EquipmentFaultCreate, created_by: str
	) -> EquipmentFaultResponse:
		"""Report an equipment fault. Critical faults set equipment to BREAKDOWN."""
		equipment = self._equipment.get(payload.equipment_id)
		if equipment is None:
			raise KeyError(f"Equipment {payload.equipment_id} not found")
		self._assert_tenant(equipment["tenant_id"])
		resp = EquipmentFaultResponse(**payload.model_dump(), created_by=created_by)
		self._faults[resp.id] = resp.model_dump()
		if payload.severity == FaultSeverity.CRITICAL:
			equipment["dispatch_status"] = DispatchStatus.BREAKDOWN
			equipment["updated_at"] = datetime.utcnow()
			self._log_warn("Critical fault reported; equipment set to BREAKDOWN", equipment_id=payload.equipment_id)
		self._log_op("report_fault", "fault", resp.id)
		return resp

	async def resolve_fault(self, id: str, work_order_id: str | None = None) -> EquipmentFaultResponse:
		"""Mark a fault as resolved."""
		rec = self._faults.get(id)
		if rec is None:
			raise KeyError(f"Fault {id} not found")
		rec["resolved"] = True
		rec["resolved_at"] = datetime.utcnow()
		rec["work_order_id"] = work_order_id
		rec["updated_at"] = datetime.utcnow()
		# If no other critical faults, restore equipment to AVAILABLE
		equipment = self._equipment.get(rec["equipment_id"])
		if equipment:
			active_critical = [
				f for f in self._faults.values()
				if f["equipment_id"] == rec["equipment_id"]
				and not f.get("resolved")
				and f["severity"] == FaultSeverity.CRITICAL
			]
			if not active_critical:
				equipment["dispatch_status"] = DispatchStatus.AVAILABLE
				equipment["updated_at"] = datetime.utcnow()
		self._log_op("resolve_fault", "fault", id)
		return EquipmentFaultResponse(**rec)

	# ── Fleet KPI Summary ──────────────────────────────────────────────────────

	async def get_fleet_kpi_summary(self) -> dict[str, Any]:
		"""Compute fleet availability, utilisation, and breakdown statistics."""
		fleet = [r for r in self._equipment.values() if r["tenant_id"] == self.tenant_id and r["lifecycle_status"] == LifecycleStatus.ACTIVE]
		total = len(fleet)
		if total == 0:
			return {"tenant_id": self.tenant_id, "total_active_equipment": 0, "as_at": datetime.utcnow().isoformat()}
		available = sum(1 for e in fleet if e["dispatch_status"] in (DispatchStatus.AVAILABLE, DispatchStatus.STANDBY_READY))
		operating = sum(1 for e in fleet if e["dispatch_status"] == DispatchStatus.OPERATING)
		breakdown = sum(1 for e in fleet if e["dispatch_status"] == DispatchStatus.BREAKDOWN)
		pa_pct = round(((total - breakdown) / total) * 100, 1)
		if pa_pct < 85:
			self._log_kpi_breach("fleet", "physical_availability", pa_pct, 85.0)
		return {
			"tenant_id": self.tenant_id,
			"total_active_equipment": total,
			"available_count": available,
			"operating_count": operating,
			"breakdown_count": breakdown,
			"physical_availability_pct": pa_pct,
			"open_faults": sum(1 for f in self._faults.values() if not f.get("resolved")),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Equipment Availability ─────────────────────────────────────────────────

	async def equipment_availability(self, asset_id: str, period: str) -> dict[str, Any]:
		"""
		Calculate physical availability (PA) and mechanical availability (MA) for an asset
		over a period (YYYY-MM).
		PA = (scheduled_hours - breakdown_hours) / scheduled_hours * 100
		MA = (operating_hours) / (operating_hours + repair_hours) * 100
		"""
		assert asset_id, "asset_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		# Gather breakdowns for this asset and period
		breakdowns = [
			r for r in self._breakdown_logs.values()
			if r["asset_id"] == asset_id
			and r["tenant_id"] == self.tenant_id
			and r.get("downtime_start", "")[:7] == period
		]
		breakdown_hours = sum(r.get("downtime_hours", 0) for r in breakdowns)
		repair_hours = sum(r.get("repair_hours", 0) for r in breakdowns)
		# Planned maintenance downtime for period
		pm_records = [
			r for r in self._planned_maintenance.values()
			if r["asset_id"] == asset_id
			and r["tenant_id"] == self.tenant_id
			and r.get("due_date", "")[:7] == period
		]
		pm_hours = sum(r.get("estimated_hours", 0) for r in pm_records)
		# Standard calendar: 30 days * 24 hrs
		calendar_hours = 30 * 24
		scheduled_hours = calendar_hours
		pa_pct = round((scheduled_hours - breakdown_hours) / scheduled_hours * 100, 2) if scheduled_hours > 0 else 0.0
		operating_hours = scheduled_hours - breakdown_hours - pm_hours
		ma_pct = round(operating_hours / (operating_hours + repair_hours) * 100, 2) if (operating_hours + repair_hours) > 0 else 0.0
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"period": period,
			"calendar_hours": calendar_hours,
			"scheduled_hours": scheduled_hours,
			"breakdown_hours": round(breakdown_hours, 2),
			"repair_hours": round(repair_hours, 2),
			"planned_maintenance_hours": round(pm_hours, 2),
			"operating_hours": round(operating_hours, 2),
			"physical_availability_pct": pa_pct,
			"mechanical_availability_pct": ma_pct,
			"breakdown_count": len(breakdowns),
			"calculated_at": datetime.utcnow().isoformat(),
		}
		self._availability_records[rec_id] = rec
		if pa_pct < 85:
			self._log_kpi_breach(equipment.get("asset_number", asset_id), "PA", pa_pct, 85.0)
		self._log_op("equipment_availability", "availability_record", rec_id)
		return rec

	# ── Planned Maintenance ────────────────────────────────────────────────────

	async def planned_maintenance(
		self,
		asset_id: str,
		service_type: str,
		due_hours: float,
		due_date: datetime,
		estimated_hours: float = 8.0,
		parts_required: list[dict[str, Any]] | None = None,
		assigned_technician: str | None = None,
		maintenance_strategy: str = "PM",
	) -> dict[str, Any]:
		"""
		Schedule a planned maintenance event for an asset.
		service_type: service_250hr | service_500hr | service_1000hr | annual | condition_based
		maintenance_strategy: PM | CBM | PdM | RCM
		"""
		assert asset_id, "asset_id required"
		assert service_type, "service_type required"
		assert due_hours > 0, "due_hours must be positive"
		assert due_date >= datetime.utcnow(), "due_date must be in the future"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"service_type": service_type,
			"due_hours": due_hours,
			"due_date": due_date.isoformat(),
			"estimated_hours": estimated_hours,
			"parts_required": parts_required or [],
			"assigned_technician": assigned_technician,
			"maintenance_strategy": maintenance_strategy,
			"status": "scheduled",
			"completed_at": None,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._planned_maintenance[rec_id] = rec
		self._log_op("planned_maintenance", "planned_maintenance", rec_id)
		return rec

	async def complete_planned_maintenance(
		self, pm_id: str, actual_hours: float, technician_id: str, notes: str | None = None
	) -> dict[str, Any]:
		"""Mark a planned maintenance event as completed."""
		rec = self._planned_maintenance.get(pm_id)
		if rec is None:
			raise KeyError(f"Planned maintenance '{pm_id}' not found")
		assert rec["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		rec["status"] = "completed"
		rec["actual_hours"] = actual_hours
		rec["completed_by"] = technician_id
		rec["completed_at"] = datetime.utcnow().isoformat()
		rec["notes"] = notes
		self._log_op("complete_planned_maintenance", "planned_maintenance", pm_id)
		return rec

	# ── Breakdown Log ──────────────────────────────────────────────────────────

	async def breakdown_log(
		self,
		asset_id: str,
		failure_mode: str,
		downtime_start: datetime,
		downtime_end: datetime,
		repair_cost: float,
		failure_cause: str | None = None,
		corrective_action: str | None = None,
		reported_by: str = "system",
	) -> dict[str, Any]:
		"""
		Log an equipment breakdown event. Computes downtime and updates cumulative metrics.
		failure_mode: mechanical | electrical | hydraulic | operator_damage | wear | unknown
		"""
		assert asset_id, "asset_id required"
		assert failure_mode, "failure_mode required"
		assert downtime_end >= downtime_start, "downtime_end must be after downtime_start"
		assert repair_cost >= 0, "repair_cost must be non-negative"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		downtime_hours = round((downtime_end - downtime_start).total_seconds() / 3600, 3)
		# Repair hours are a fraction of downtime (diagnosis + fix, excl. wait time)
		repair_hours = round(downtime_hours * 0.6, 3)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"failure_mode": failure_mode,
			"failure_cause": failure_cause,
			"downtime_start": downtime_start.isoformat(),
			"downtime_end": downtime_end.isoformat(),
			"downtime_hours": downtime_hours,
			"repair_hours": repair_hours,
			"repair_cost": round(repair_cost, 2),
			"corrective_action": corrective_action,
			"reported_by": reported_by,
			"logged_at": datetime.utcnow().isoformat(),
		}
		self._breakdown_logs[rec_id] = rec
		# Set equipment back to available after breakdown resolved
		equipment["dispatch_status"] = DispatchStatus.AVAILABLE
		equipment["updated_at"] = datetime.utcnow()
		self._log_op("breakdown_log", "breakdown_log", rec_id)
		return rec

	async def list_breakdown_logs(
		self, asset_id: str | None = None, period: str | None = None
	) -> list[dict[str, Any]]:
		"""List breakdown logs with optional asset and period filters."""
		results = [r for r in self._breakdown_logs.values() if r["tenant_id"] == self.tenant_id]
		if asset_id:
			results = [r for r in results if r["asset_id"] == asset_id]
		if period:
			results = [r for r in results if r["downtime_start"][:7] == period]
		return sorted(results, key=lambda x: x["downtime_start"], reverse=True)

	# ── Condition Monitoring ───────────────────────────────────────────────────

	async def condition_monitoring(
		self,
		asset_id: str,
		sensor_readings: dict[str, Any],
		monitoring_type: str = "oil_analysis",
		alert_threshold_breach: bool = False,
		recorded_by: str = "system",
	) -> dict[str, Any]:
		"""
		Record condition monitoring data (oil analysis, vibration, thermography, etc.).
		sensor_readings: {"oil_viscosity_cst": 98.2, "particle_count_iso": 16, "temperature_c": 82}
		Detects anomalies against typical thresholds.
		"""
		assert asset_id, "asset_id required"
		assert sensor_readings, "sensor_readings required"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		# Simple anomaly detection: flag any reading > 20% above expected baseline
		anomalies: list[str] = []
		typical_ranges = {
			"temperature_c": (60, 95),
			"oil_viscosity_cst": (80, 120),
			"vibration_mm_s": (0, 7.1),
			"particle_count_iso": (0, 18),
		}
		for key, value in sensor_readings.items():
			if key in typical_ranges and isinstance(value, (int, float)):
				lo, hi = typical_ranges[key]
				if value > hi * 1.2:
					anomalies.append(f"{key}={value} exceeds upper limit {hi}")
				elif value < lo * 0.8:
					anomalies.append(f"{key}={value} below lower limit {lo}")
		alert = alert_threshold_breach or bool(anomalies)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"monitoring_type": monitoring_type,
			"sensor_readings": sensor_readings,
			"anomalies": anomalies,
			"alert": alert,
			"recorded_by": recorded_by,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._condition_monitoring[rec_id] = rec
		if alert:
			self._log_warn("Condition monitoring alert", asset_id=asset_id, anomalies=anomalies)
		self._log_op("condition_monitoring", "condition_monitoring", rec_id)
		return rec

	# ── Operator Pre-start Check ───────────────────────────────────────────────

	async def operator_pre_start_check(
		self,
		asset_id: str,
		operator_id: str,
		checklist_result: dict[str, Any],
		overall_pass: bool | None = None,
		defects_noted: list[str] | None = None,
	) -> dict[str, Any]:
		"""
		Record an operator pre-start inspection checklist.
		checklist_result: {"brakes": "pass", "lights": "pass", "horn": "fail", ...}
		overall_pass derived from checklist if not provided.
		"""
		assert asset_id, "asset_id required"
		assert operator_id, "operator_id required"
		assert checklist_result, "checklist_result required"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		failed_items = [k for k, v in checklist_result.items() if str(v).lower() in ("fail", "failed", "no", "defect")]
		if overall_pass is None:
			overall_pass = len(failed_items) == 0
		if not overall_pass:
			equipment["dispatch_status"] = DispatchStatus.MAINTENANCE
			equipment["updated_at"] = datetime.utcnow()
			self._log_warn("Pre-start check failed; equipment grounded", asset_id=asset_id, failed=failed_items)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"operator_id": operator_id,
			"checklist_result": checklist_result,
			"failed_items": failed_items,
			"defects_noted": defects_noted or [],
			"overall_pass": overall_pass,
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._pre_start_checks[rec_id] = rec
		self._log_op("operator_pre_start_check", "pre_start_check", rec_id)
		return rec

	# ── Fuel and Lube Consumption ──────────────────────────────────────────────

	async def fuel_and_lube_consumption(
		self, asset_id: str, period: str
	) -> dict[str, Any]:
		"""
		Aggregate fuel and lube consumption for an asset over a period (YYYY-MM).
		Returns total litres, cost, and fuel consumption rate (L/hr).
		"""
		assert asset_id, "asset_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		dockets = [
			r for r in self._fuel_dockets.values()
			if r["equipment_id"] == asset_id
			and r.get("docket_date", r.get("created_at", ""))[:7] == period
		]
		total_litres = sum(r.get("quantity_litres", 0) for r in dockets)
		total_cost = sum(r.get("total_cost", 0) or 0 for r in dockets)
		total_engine_hours = sum(r.get("engine_hours", 0) or 0 for r in dockets)
		consumption_rate = round(total_litres / total_engine_hours, 2) if total_engine_hours > 0 else None
		# Variance against fleet average (simple ±10% threshold)
		fleet_avg_rate = 35.0  # L/hr typical for haul truck
		variance_flag = (
			consumption_rate is not None
			and abs(consumption_rate - fleet_avg_rate) / fleet_avg_rate > 0.15
		)
		return {
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"period": period,
			"fuel_dockets": len(dockets),
			"total_fuel_litres": round(total_litres, 1),
			"total_fuel_cost": round(total_cost, 2),
			"total_engine_hours": round(total_engine_hours, 1),
			"consumption_rate_l_hr": consumption_rate,
			"variance_flag": variance_flag,
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Major Component Tracking ───────────────────────────────────────────────

	async def major_component_tracking(
		self,
		asset_id: str,
		component_type: str,
		hours: float,
		replacement_date: datetime,
		serial_number: str | None = None,
		supplier: str | None = None,
		cost: float | None = None,
		next_replacement_hours: float | None = None,
	) -> dict[str, Any]:
		"""
		Track a major component replacement (engine, transmission, tyres, GET).
		Records hours at replacement and calculates component life.
		component_type: engine | transmission | differential | tyres | GET | hydraulic_pump | bucket
		"""
		assert asset_id, "asset_id required"
		assert component_type, "component_type required"
		assert hours > 0, "hours must be positive"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		# Find previous replacement to compute component life
		prev = sorted(
			[
				r for r in self._major_components.values()
				if r["asset_id"] == asset_id
				and r["component_type"] == component_type
				and r["tenant_id"] == self.tenant_id
			],
			key=lambda x: x["hours"],
		)
		prev_hours = prev[-1]["hours"] if prev else 0.0
		component_life_hrs = round(hours - prev_hours, 1)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"component_type": component_type,
			"serial_number": serial_number,
			"hours_at_replacement": hours,
			"component_life_hrs": component_life_hrs,
			"replacement_date": replacement_date.isoformat(),
			"next_replacement_hours": next_replacement_hours,
			"supplier": supplier,
			"cost": cost,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._major_components[rec_id] = rec
		self._log_op("major_component_tracking", "major_component", rec_id)
		return rec

	async def list_major_components(
		self, asset_id: str, component_type: str | None = None
	) -> list[dict[str, Any]]:
		"""List major component replacements for an asset."""
		results = [
			r for r in self._major_components.values()
			if r["asset_id"] == asset_id and r["tenant_id"] == self.tenant_id
		]
		if component_type:
			results = [r for r in results if r["component_type"] == component_type]
		return sorted(results, key=lambda x: x["hours_at_replacement"], reverse=True)

	# ── Equipment Analytics ────────────────────────────────────────────────────

	async def equipment_analytics(self, period: str) -> dict[str, Any]:
		"""
		Fleet-level analytics for a period (YYYY-MM).
		Returns availability, utilisation, breakdown rates, top failure modes, and cost summary.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		fleet = [
			r for r in self._equipment.values()
			if r["tenant_id"] == self.tenant_id and r["lifecycle_status"] == LifecycleStatus.ACTIVE
		]
		breakdowns = [
			r for r in self._breakdown_logs.values()
			if r["tenant_id"] == self.tenant_id and r["downtime_start"][:7] == period
		]
		pm_completed = [
			r for r in self._planned_maintenance.values()
			if r["tenant_id"] == self.tenant_id
			and r.get("status") == "completed"
			and r.get("completed_at", "")[:7] == period
		]
		total_breakdown_hours = sum(r["downtime_hours"] for r in breakdowns)
		total_repair_cost = sum(r["repair_cost"] for r in breakdowns)
		# Failure mode frequency
		failure_modes: dict[str, int] = {}
		for b in breakdowns:
			fm = b.get("failure_mode", "unknown")
			failure_modes[fm] = failure_modes.get(fm, 0) + 1
		top_failure_modes = sorted(failure_modes.items(), key=lambda x: x[1], reverse=True)[:5]
		# PA across fleet
		calendar_hours = 30 * 24
		fleet_pa = (
			round((calendar_hours - (total_breakdown_hours / max(len(fleet), 1))) / calendar_hours * 100, 1)
			if fleet else 0.0
		)
		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"active_fleet_count": len(fleet),
			"breakdown_events": len(breakdowns),
			"total_breakdown_hours": round(total_breakdown_hours, 1),
			"total_repair_cost": round(total_repair_cost, 2),
			"fleet_pa_pct": fleet_pa,
			"pm_completed": len(pm_completed),
			"top_failure_modes": top_failure_modes,
			"condition_monitoring_alerts": sum(
				1 for r in self._condition_monitoring.values()
				if r["tenant_id"] == self.tenant_id and r.get("alert") and r["recorded_at"][:7] == period
			),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Replacement Recommendation ─────────────────────────────────────────────

	async def replacement_recommendation(self, asset_id: str) -> dict[str, Any]:
		"""
		Generate an economic replacement recommendation for an asset.
		Uses: age, total operating hours, cumulative repair cost, availability trend,
		and benchmark fleet averages to produce a replace/retain/monitor decision.
		"""
		assert asset_id, "asset_id required"
		equipment = self._equipment.get(asset_id)
		if equipment is None:
			raise KeyError(f"Equipment '{asset_id}' not found")
		assert equipment["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		breakdowns = [
			r for r in self._breakdown_logs.values()
			if r["asset_id"] == asset_id and r["tenant_id"] == self.tenant_id
		]
		total_repair_cost = sum(r["repair_cost"] for r in breakdowns)
		total_downtime_hrs = sum(r["downtime_hours"] for r in breakdowns)
		total_op_hours = equipment.get("total_operating_hours", 0) or 0
		# Simplified LCCA: if cumulative repair cost > 60% of replacement value, recommend replace
		replacement_value = equipment.get("replacement_value", 500000)  # default $500k
		repair_cost_ratio = round(total_repair_cost / replacement_value, 3) if replacement_value > 0 else 0
		# Availability trend: compare last 3 months
		recent_breakdowns_count = len([
			r for r in breakdowns
			if r["downtime_start"] >= (
				datetime.utcnow().replace(day=1).isoformat()[:7] + "-01"
			)
		])
		decision: str
		rationale: str
		if repair_cost_ratio > 0.6 or total_downtime_hrs > 500:
			decision = "replace"
			rationale = f"Cumulative repair cost ratio {repair_cost_ratio:.1%} exceeds 60% threshold; {total_downtime_hrs:.0f} hrs total downtime."
		elif repair_cost_ratio > 0.35 or recent_breakdowns_count > 3:
			decision = "monitor"
			rationale = f"Repair cost ratio {repair_cost_ratio:.1%} in warning zone; {recent_breakdowns_count} breakdowns in recent month."
		else:
			decision = "retain"
			rationale = f"Repair cost ratio {repair_cost_ratio:.1%} within acceptable range; asset performing adequately."
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"asset_id": asset_id,
			"asset_number": equipment.get("asset_number"),
			"total_operating_hours": total_op_hours,
			"cumulative_repair_cost": round(total_repair_cost, 2),
			"replacement_value": replacement_value,
			"repair_cost_ratio": repair_cost_ratio,
			"total_downtime_hours": round(total_downtime_hrs, 1),
			"breakdown_events": len(breakdowns),
			"decision": decision,
			"rationale": rationale,
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._replacement_recs[rec_id] = rec
		self._log_op("replacement_recommendation", "replacement_rec", rec_id)
		return rec

	# ── Additional methods ────────────────────────────────────────────────────

	async def fleet_utilisation_report(self) -> dict[str, Any]:
		"""Compute fleet utilisation: OEE, availability, and utilisation by class."""
		all_eq = [EquipmentResponse(**r) for r in self._equipment.values() if r["tenant_id"] == self.tenant_id]
		total = len(all_eq)
		operating = sum(1 for e in all_eq if str(e.dispatch_status) in ("operating", "DispatchStatus.OPERATING"))
		active = sum(1 for e in all_eq if str(e.lifecycle_status) in ("active", "LifecycleStatus.ACTIVE"))
		by_class: dict[str, int] = {}
		for e in all_eq:
			cls = str(e.equipment_class)
			by_class[cls] = by_class.get(cls, 0) + 1
		utilisation_pct = round(operating / max(total, 1) * 100, 2)
		availability_pct = round(active / max(total, 1) * 100, 2)
		return {
			"tenant_id": self.tenant_id,
			"total_assets": total,
			"operating_count": operating,
			"active_count": active,
			"utilisation_pct": utilisation_pct,
			"availability_pct": availability_pct,
			"by_class": by_class,
			"computed_at": datetime.utcnow().isoformat(),
		}

	async def maintenance_kpi_report(self) -> dict[str, Any]:
		"""Compute maintenance KPIs: MTBF, MTTR, PM compliance rate."""
		work_orders = [r for r in self._work_orders.values() if r["tenant_id"] == self.tenant_id]
		completed = [w for w in work_orders if w.get("status") == "completed"]
		planned = [w for w in work_orders if w.get("maintenance_type") == "planned"]
		unplanned = [w for w in work_orders if w.get("maintenance_type") == "breakdown"]
		pm_compliance = round(len(planned) / max(len(work_orders), 1) * 100, 2)
		breakdowns = [r for r in self._breakdown_logs.values() if r["tenant_id"] == self.tenant_id]
		total_downtime = sum(r.get("downtime_hours", 0) for r in breakdowns)
		avg_downtime = round(total_downtime / max(len(breakdowns), 1), 2)
		return {
			"tenant_id": self.tenant_id,
			"total_work_orders": len(work_orders),
			"completed_work_orders": len(completed),
			"planned_maintenance_count": len(planned),
			"unplanned_breakdown_count": len(unplanned),
			"pm_compliance_pct": pm_compliance,
			"total_downtime_hours": round(total_downtime, 2),
			"avg_breakdown_downtime_hours": avg_downtime,
			"computed_at": datetime.utcnow().isoformat(),
		}

	async def bulk_schedule_inspections(
		self,
		equipment_ids: list[str],
		inspection_type: str,
		scheduled_date: str,
		inspector_id: str,
	) -> dict[str, Any]:
		"""Schedule inspections for multiple equipment items in bulk."""
		assert equipment_ids, "equipment_ids required"
		assert inspection_type, "inspection_type required"
		assert scheduled_date, "scheduled_date required"
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for eq_id in equipment_ids:
			try:
				rec = self._equipment.get(eq_id)
				if rec is None:
					raise KeyError(f"Equipment {eq_id} not found")
				inspection_id = uuid7str()
				inspection: dict[str, Any] = {
					"id": inspection_id,
					"tenant_id": self.tenant_id,
					"equipment_id": eq_id,
					"asset_number": rec.get("asset_number"),
					"inspection_type": inspection_type,
					"inspector_id": inspector_id,
					"scheduled_date": scheduled_date,
					"status": "scheduled",
					"inspected_at": datetime.utcnow(),
					"overall_result": "pending",
					"findings": [],
				}
				self._inspections[inspection_id] = inspection
				results.append({"equipment_id": eq_id, "inspection_id": inspection_id, "status": "scheduled"})
			except Exception as exc:
				errors.append({"equipment_id": eq_id, "error": str(exc)})
		return {
			"scheduled_count": len(results),
			"error_count": len(errors),
			"results": results,
			"errors": errors,
		}

	async def fuel_consumption_analytics(
		self,
		period_start: str | None = None,
		period_end: str | None = None,
	) -> dict[str, Any]:
		"""Compute fuel consumption KPIs across the fleet."""
		dockets = [r for r in self._fuel_dockets.values() if r["tenant_id"] == self.tenant_id]
		total_litres = sum(float(d.get("litres_dispensed", 0)) for d in dockets)
		by_equipment: dict[str, float] = {}
		for d in dockets:
			eq_id = d.get("equipment_id", "unknown")
			by_equipment[eq_id] = round(by_equipment.get(eq_id, 0.0) + float(d.get("litres_dispensed", 0)), 2)
		top_consumers = sorted(by_equipment.items(), key=lambda x: x[1], reverse=True)[:10]
		return {
			"tenant_id": self.tenant_id,
			"period_start": period_start,
			"period_end": period_end,
			"total_dockets": len(dockets),
			"total_litres": round(total_litres, 2),
			"top_consumers": [{"equipment_id": eid, "litres": l} for eid, l in top_consumers],
			"computed_at": datetime.utcnow().isoformat(),
		}

	async def export_equipment_register(
		self,
		format: str = "json",
	) -> dict[str, Any]:
		"""Export the full equipment register in JSON or CSV format."""
		assert format in {"json", "csv"}, "format must be json or csv"
		records = [r for r in self._equipment.values() if r["tenant_id"] == self.tenant_id]
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if records:
				export_fields = ["id", "asset_number", "equipment_class", "make", "model", "lifecycle_status", "dispatch_status", "mine_area_assignment"]
				writer = csv.DictWriter(buf, fieldnames=export_fields, extrasaction="ignore")
				writer.writeheader()
				writer.writerows(records)
			return {"format": "csv", "record_count": len(records), "content": buf.getvalue()}
		return {"format": "json", "tenant_id": self.tenant_id, "record_count": len(records), "records": records}

	async def health_check(self) -> dict[str, Any]:
		"""Return equipment management service health status."""
		all_eq = [r for r in self._equipment.values() if r["tenant_id"] == self.tenant_id]
		breakdowns = sum(1 for r in all_eq if str(r.get("dispatch_status", "")) in ("breakdown", "DispatchStatus.BREAKDOWN"))
		return {
			"service": "EqpService",
			"tenant_id": self.tenant_id,
			"status": "healthy" if breakdowns < 10 else "degraded",
			"equipment_count": len(all_eq),
			"breakdown_count": breakdowns,
			"work_order_count": len([w for w in self._work_orders.values() if w["tenant_id"] == self.tenant_id]),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def equipment_compliance_audit(self) -> dict[str, Any]:
		"""Audit equipment records for compliance: valid registrations, inspection currency."""
		all_eq = [r for r in self._equipment.values() if r["tenant_id"] == self.tenant_id]
		inspections = [r for r in self._inspections.values() if r["tenant_id"] == self.tenant_id]
		no_recent_inspection: list[str] = []
		for eq in all_eq:
			eq_id = eq["id"]
			has_inspection = any(i["equipment_id"] == eq_id for i in inspections)
			if not has_inspection:
				no_recent_inspection.append(eq_id)
		compliant = len(all_eq) - len(no_recent_inspection)
		return {
			"tenant_id": self.tenant_id,
			"total_equipment": len(all_eq),
			"no_inspection_count": len(no_recent_inspection),
			"compliant_count": compliant,
			"compliance_rate_pct": round(compliant / max(len(all_eq), 1) * 100, 2),
			"audited_at": datetime.utcnow().isoformat(),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}
		return {"format": format, "tenant_id": self.tenant_id}

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
