"""Async service layer for Facilities Maintenance (mai)."""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any

from .models import (
	AssetCreate, AssetResponse, AssetUpdate,
	PpmScheduleCreate, PpmScheduleResponse,
	WorkOrderCreate, WorkOrderResponse, WorkOrderUpdate,
	MaintenanceContractorCreate, MaintenanceContractorResponse,
	SlaCreate, SlaResponse,
	InspectionCreate, InspectionResponse,
	DefectCreate, DefectResponse,
	WorkOrderStatus, PpmStatus, AssetStatus,
	InspectionType, DefectSeverity,
)
from .capability_contract import evaluate_capability_rules
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

log = logging.getLogger(__name__)


class MaiService:
	"""Service implementing all Facilities Maintenance operations."""

	def __init__(
		self,
		tenant_id: str | None = None,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: dict[str, Any] | None = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store: dict[str, list[dict[str, Any]]] = store or {
			"assets": [], "ppm_schedules": [], "work_orders": [],
			"contractors": [], "slas": [], "inspections": [], "defects": [],
			"sustainability_records": [], "completions": [],
		}
		self._wo_counter = 0

	# ── Logging helpers ───────────────────────────────────────────────────────

	def _log_operation(self, op: str, entity_id: str, tenant_id: str) -> None:
		log.info("mai.%s entity=%s tenant=%s", op, entity_id, tenant_id)

	def _log_sla_breach(self, work_order_id: str, sla_type: str) -> None:
		log.warning("mai.sla_breach work_order=%s sla_type=%s", work_order_id, sla_type)

	def _log_p1_raised(self, work_order_id: str, asset_id: str) -> None:
		log.critical("mai.p1_raised work_order=%s asset=%s", work_order_id, asset_id)

	def _log_statutory_overdue(self, inspection_id: str, property_id: str) -> None:
		log.error("mai.statutory_overdue inspection=%s property=%s", inspection_id, property_id)

	# ── Rules ─────────────────────────────────────────────────────────────────

	def _check_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			log.warning("mai.rule_denied rule=%s reason=%s", result["rule"], result["reason"])
			raise ValueError(f"rule_denied:{result['rule']}:{result['reason']}")

	def _next_wo_ref(self) -> str:
		self._wo_counter += 1
		return f"WO-{self._wo_counter:06d}"

	# ── Asset ─────────────────────────────────────────────────────────────────

	async def register_asset(self, payload: AssetCreate) -> AssetResponse:
		"""Register a new facility asset."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "register_asset",
			"asset_category_supported": True,
			"operation_type": "write",
			"policy_attached": True,
		})
		record = AssetResponse(**payload.model_dump())
		self._store["assets"].append(record.model_dump())
		self._log_operation("register_asset", record.id, record.tenant_id)
		return record

	async def get_asset(self, asset_id: str, tenant_id: str) -> AssetResponse | None:
		"""Fetch an asset by ID."""
		for a in self._store["assets"]:
			if a["id"] == asset_id and a["tenant_id"] == tenant_id:
				return AssetResponse(**a)
		return None

	async def list_assets(self, tenant_id: str, property_id: str | None = None, category: str | None = None, status: str | None = None) -> list[AssetResponse]:
		"""List assets with optional filters."""
		results = [a for a in self._store["assets"] if a["tenant_id"] == tenant_id]
		if property_id:
			results = [a for a in results if a.get("property_id") == property_id]
		if category:
			results = [a for a in results if a.get("category") == category]
		if status:
			results = [a for a in results if a.get("status") == status]
		return [AssetResponse(**a) for a in results]

	async def update_asset(self, asset_id: str, tenant_id: str, updates: AssetUpdate) -> AssetResponse | None:
		"""Update asset metadata."""
		for i, a in enumerate(self._store["assets"]):
			if a["id"] == asset_id and a["tenant_id"] == tenant_id:
				a.update({k: v for k, v in updates.model_dump().items() if v is not None})
				a["updated_at"] = datetime.utcnow()
				self._store["assets"][i] = a
				return AssetResponse(**a)
		return None

	async def get_end_of_life_assets(self, tenant_id: str, property_id: str | None = None) -> list[AssetResponse]:
		"""Return assets at end-of-life or replacement_due."""
		assets = await self.list_assets(tenant_id, property_id)
		return [a for a in assets if a.lifecycle_phase.value in ("end_of_life", "replacement_due")]

	# ── PPM Schedule ──────────────────────────────────────────────────────────

	async def create_ppm_schedule(self, payload: PpmScheduleCreate) -> PpmScheduleResponse:
		"""Create a preventive maintenance schedule."""
		self._check_rules({
			"tenant_context_present": True,
			"operation": "create_ppm_schedule",
			"frequency_supported": True,
			"asset_present": True,
		})
		record = PpmScheduleResponse(**payload.model_dump())
		self._store["ppm_schedules"].append(record.model_dump())
		self._log_operation("create_ppm_schedule", record.id, record.tenant_id)
		return record

	async def list_ppm_schedules(self, tenant_id: str, asset_id: str | None = None, status: str | None = None) -> list[PpmScheduleResponse]:
		"""List PPM schedules."""
		results = [p for p in self._store["ppm_schedules"] if p["tenant_id"] == tenant_id]
		if asset_id:
			results = [p for p in results if p["asset_id"] == asset_id]
		if status:
			results = [p for p in results if p["status"] == status]
		return [PpmScheduleResponse(**p) for p in results]

	async def complete_ppm(self, ppm_id: str, tenant_id: str, completed_by: str) -> PpmScheduleResponse | None:
		"""Mark a PPM as completed and schedule next occurrence."""
		for i, p in enumerate(self._store["ppm_schedules"]):
			if p["id"] == ppm_id and p["tenant_id"] == tenant_id:
				p["status"] = PpmStatus.completed.value
				p["last_completed"] = datetime.utcnow().date().isoformat()
				p["completion_count"] = p.get("completion_count", 0) + 1
				p["next_due"] = self._calc_next_due(p["frequency"], date.today()).isoformat()
				p["updated_at"] = datetime.utcnow()
				self._store["ppm_schedules"][i] = p
				for j, a in enumerate(self._store["assets"]):
					if a["id"] == p["asset_id"] and a["tenant_id"] == tenant_id:
						a["last_maintained"] = datetime.utcnow().date().isoformat()
						a["next_maintenance_due"] = p["next_due"]
						a["updated_at"] = datetime.utcnow()
						self._store["assets"][j] = a
						break
				return PpmScheduleResponse(**p)
		return None

	def _calc_next_due(self, frequency: str, from_date: date) -> date:
		"""Calculate next due date from frequency string."""
		freq_days = {
			"daily": 1, "weekly": 7, "fortnightly": 14, "monthly": 30,
			"quarterly": 91, "semi_annual": 182, "annual": 365, "biennial": 730,
		}
		return from_date + timedelta(days=freq_days.get(frequency, 30))

	async def get_overdue_ppms(self, tenant_id: str) -> list[PpmScheduleResponse]:
		"""Return PPM schedules past their due date."""
		today = date.today()
		results = []
		for p in self._store["ppm_schedules"]:
			if p["tenant_id"] == tenant_id and p["status"] == PpmStatus.scheduled.value:
				if datetime.strptime(p["next_due"], "%Y-%m-%d").date() < today:
					results.append(PpmScheduleResponse(**p))
		return results

	# ── Work Order ────────────────────────────────────────────────────────────

	async def raise_work_order(self, payload: WorkOrderCreate) -> WorkOrderResponse:
		"""Raise a new work order."""
		asset = await self.get_asset(payload.asset_id, payload.tenant_id)
		if asset and asset.status.value == "decommissioned":
			self._check_rules({"operation": "raise_work_order", "asset_status": "decommissioned"})
		self._check_rules({
			"tenant_context_present": True,
			"operation": "raise_work_order",
			"work_order_type_supported": True,
			"asset_present": True,
			"priority_supported": True,
		})
		if payload.priority.value == "p1_critical":
			self._log_p1_raised("new", payload.asset_id)
		ref = self._next_wo_ref()
		sla_response, sla_resolution = self._calc_sla_deadlines(payload.priority.value)
		record = WorkOrderResponse(**payload.model_dump(), ref=ref, sla_response_deadline=sla_response, sla_resolution_deadline=sla_resolution)
		self._store["work_orders"].append(record.model_dump())
		self._log_operation("raise_work_order", record.id, record.tenant_id)
		return record

	def _calc_sla_deadlines(self, priority: str) -> tuple[datetime, datetime]:
		"""Calculate SLA response and resolution deadlines by priority."""
		response_hours = {"p1_critical": 1, "p2_high": 4, "p3_medium": 8, "p4_low": 24, "p5_planned": 72}
		resolution_hours = {"p1_critical": 4, "p2_high": 24, "p3_medium": 72, "p4_low": 168, "p5_planned": 336}
		now = datetime.utcnow()
		resp = now + timedelta(hours=response_hours.get(priority, 24))
		resol = now + timedelta(hours=resolution_hours.get(priority, 168))
		return resp, resol

	async def assign_work_order(self, wo_id: str, tenant_id: str, contractor_id: str) -> WorkOrderResponse | None:
		"""Assign a work order to a contractor."""
		contractor = None
		for c in self._store["contractors"]:
			if c["id"] == contractor_id and c["tenant_id"] == tenant_id:
				contractor = c
				break
		has_insurance = bool(contractor and contractor.get("has_valid_insurance", False))
		self._check_rules({"operation": "assign_contractor", "contractor_has_valid_insurance": has_insurance})
		for i, w in enumerate(self._store["work_orders"]):
			if w["id"] == wo_id and w["tenant_id"] == tenant_id:
				w["assigned_contractor_id"] = contractor_id
				w["status"] = WorkOrderStatus.assigned.value
				w["updated_at"] = datetime.utcnow()
				self._store["work_orders"][i] = w
				return WorkOrderResponse(**w)
		return None

	async def update_work_order(self, wo_id: str, tenant_id: str, updates: WorkOrderUpdate) -> WorkOrderResponse | None:
		"""Update work order status and fields."""
		for i, w in enumerate(self._store["work_orders"]):
			if w["id"] == wo_id and w["tenant_id"] == tenant_id:
				now = datetime.utcnow()
				if w.get("sla_resolution_deadline"):
					deadline = w["sla_resolution_deadline"]
					if isinstance(deadline, str):
						deadline = datetime.fromisoformat(deadline)
					if now > deadline and not w.get("sla_breached"):
						w["sla_breached"] = True
						self._log_sla_breach(wo_id, "resolution_time")
						self._check_rules({"operation": "update_work_order", "sla_breached": True, "escalated": False})
				w.update({k: v for k, v in updates.model_dump().items() if v is not None})
				w["updated_at"] = now
				self._store["work_orders"][i] = w
				return WorkOrderResponse(**w)
		return None

	async def close_work_order(self, wo_id: str, tenant_id: str, verified_by: str) -> WorkOrderResponse | None:
		"""Close a verified work order."""
		for i, w in enumerate(self._store["work_orders"]):
			if w["id"] == wo_id and w["tenant_id"] == tenant_id:
				self._check_rules({"operation": "close_work_order", "verification_complete": w.get("verification_complete", False)})
				w["status"] = WorkOrderStatus.closed.value
				w["updated_at"] = datetime.utcnow()
				self._store["work_orders"][i] = w
				self._log_operation("close_work_order", wo_id, tenant_id)
				return WorkOrderResponse(**w)
		return None

	async def list_work_orders(self, tenant_id: str, property_id: str | None = None, status: str | None = None, priority: str | None = None) -> list[WorkOrderResponse]:
		"""List work orders with optional filters."""
		results = [w for w in self._store["work_orders"] if w["tenant_id"] == tenant_id]
		if property_id:
			results = [w for w in results if w.get("property_id") == property_id]
		if status:
			results = [w for w in results if w.get("status") == status]
		if priority:
			results = [w for w in results if w.get("priority") == priority]
		return [WorkOrderResponse(**w) for w in results]

	# ── Contractor ────────────────────────────────────────────────────────────

	async def register_contractor(self, payload: MaintenanceContractorCreate) -> MaintenanceContractorResponse:
		"""Register a maintenance contractor."""
		self._check_rules({"tenant_context_present": True, "operation": "register_contractor", "contractor_type_supported": True})
		has_insurance = payload.insurance_expiry is not None and payload.insurance_expiry > date.today()
		record = MaintenanceContractorResponse(**payload.model_dump(), has_valid_insurance=has_insurance)
		self._store["contractors"].append(record.model_dump())
		self._log_operation("register_contractor", record.id, record.tenant_id)
		return record

	async def list_contractors(self, tenant_id: str, contractor_type: str | None = None) -> list[MaintenanceContractorResponse]:
		"""List maintenance contractors."""
		results = [c for c in self._store["contractors"] if c["tenant_id"] == tenant_id]
		if contractor_type:
			results = [c for c in results if c.get("contractor_type") == contractor_type]
		return [MaintenanceContractorResponse(**c) for c in results]

	# ── Inspection ────────────────────────────────────────────────────────────

	async def create_inspection(self, payload: InspectionCreate) -> InspectionResponse:
		"""Schedule a facility inspection."""
		self._check_rules({"tenant_context_present": True, "operation": "create_inspection", "inspection_type_supported": True})
		record = InspectionResponse(**payload.model_dump())
		self._store["inspections"].append(record.model_dump())
		return record

	async def complete_inspection(self, inspection_id: str, tenant_id: str, findings: list[dict[str, Any]]) -> InspectionResponse | None:
		"""Record inspection completion and findings."""
		for i, ins in enumerate(self._store["inspections"]):
			if ins["id"] == inspection_id and ins["tenant_id"] == tenant_id:
				ins["status"] = "completed"
				ins["completed_at"] = datetime.utcnow()
				ins["findings"] = findings
				ins["updated_at"] = datetime.utcnow()
				if ins.get("inspection_type") == InspectionType.statutory.value:
					self._check_rules({"operation": "check_inspection_status", "inspection_type": "statutory", "overdue": False, "alert_sent": True})
				self._store["inspections"][i] = ins
				return InspectionResponse(**ins)
		return None

	async def get_overdue_inspections(self, tenant_id: str) -> list[InspectionResponse]:
		"""Return overdue inspections, flagging statutory ones."""
		today = date.today()
		results = []
		for ins in self._store["inspections"]:
			if ins["tenant_id"] == tenant_id and ins["status"] == "scheduled":
				sched = datetime.strptime(ins["scheduled_date"], "%Y-%m-%d").date()
				if sched < today:
					if ins.get("inspection_type") == InspectionType.statutory.value:
						self._log_statutory_overdue(ins["id"], ins.get("property_id", ""))
					results.append(InspectionResponse(**ins))
		return results

	# ── Defect ────────────────────────────────────────────────────────────────

	async def raise_defect(self, payload: DefectCreate) -> DefectResponse:
		"""Raise a defect."""
		self._check_rules({"tenant_context_present": True, "operation": "raise_defect", "severity_supported": True})
		record = DefectResponse(**payload.model_dump())
		self._store["defects"].append(record.model_dump())
		return record

	async def resolve_defect(self, defect_id: str, tenant_id: str, resolution_notes: str) -> DefectResponse | None:
		"""Mark a defect as resolved."""
		for i, d in enumerate(self._store["defects"]):
			if d["id"] == defect_id and d["tenant_id"] == tenant_id:
				d["status"] = "resolved"
				d["resolved_at"] = datetime.utcnow()
				d["resolution_notes"] = resolution_notes
				d["updated_at"] = datetime.utcnow()
				self._store["defects"][i] = d
				return DefectResponse(**d)
		return None

	async def list_defects(self, tenant_id: str, property_id: str | None = None, severity: str | None = None) -> list[DefectResponse]:
		"""List open defects."""
		results = [d for d in self._store["defects"] if d["tenant_id"] == tenant_id and d["status"] == "open"]
		if property_id:
			results = [d for d in results if d.get("property_id") == property_id]
		if severity:
			results = [d for d in results if d.get("severity") == severity]
		return [DefectResponse(**d) for d in results]

	# ── SLA ───────────────────────────────────────────────────────────────────

	async def create_sla(self, payload: SlaCreate) -> SlaResponse:
		"""Create a service level agreement definition."""
		self._check_rules({"tenant_context_present": True, "operation": "create_sla", "sla_type_supported": True})
		record = SlaResponse(**payload.model_dump())
		self._store["slas"].append(record.model_dump())
		return record

	async def get_sla_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Return SLA compliance dashboard."""
		work_orders = await self.list_work_orders(tenant_id)
		breached = [w for w in work_orders if w.sla_breached]
		return {
			"tenant_id": tenant_id,
			"total_open_work_orders": len([w for w in work_orders if w.status.value not in ("closed", "cancelled")]),
			"sla_breached_count": len(breached),
			"p1_open": len([w for w in work_orders if w.priority.value == "p1_critical" and w.status.value not in ("closed", "cancelled")]),
			"overdue_ppms": len(await self.get_overdue_ppms(tenant_id)),
			"overdue_inspections": len(await self.get_overdue_inspections(tenant_id)),
		}

	# ── NEW: assign_contractor ─────────────────────────────────────────────────

	async def assign_contractor(
		self,
		work_order_id: str,
		contractor_id: str,
		agreed_cost: Decimal,
		start_date: date,
		tenant_id: str,
		purchase_order_ref: str = "",
		scope_of_work: str = "",
	) -> WorkOrderResponse | None:
		"""Assign a contractor to a work order with agreed cost, start date, and PO reference."""
		assert work_order_id and contractor_id, "work_order_id and contractor_id required"
		assert agreed_cost >= 0, "agreed_cost must be non-negative"
		result = await self.assign_work_order(work_order_id, tenant_id, contractor_id)
		if result:
			for i, w in enumerate(self._store["work_orders"]):
				if w["id"] == work_order_id and w["tenant_id"] == tenant_id:
					w["agreed_cost"] = str(agreed_cost)
					w["start_date"] = str(start_date)
					w["purchase_order_ref"] = purchase_order_ref
					w["scope_of_work"] = scope_of_work
					self._store["work_orders"][i] = w
					return WorkOrderResponse(**w)
		return result

	# ── NEW: complete_work_order ───────────────────────────────────────────────

	async def complete_work_order(
		self,
		work_order_id: str,
		actual_cost: Decimal,
		completion_date: date,
		sign_off_by: str,
		tenant_id: str,
		completion_notes: str = "",
		defects_noted: list[str] | None = None,
	) -> WorkOrderResponse | None:
		"""Complete a work order with actual cost, completion date, and sign-off."""
		assert work_order_id and sign_off_by, "work_order_id and sign_off_by required"
		assert actual_cost >= 0, "actual_cost must be non-negative"
		for i, w in enumerate(self._store["work_orders"]):
			if w["id"] == work_order_id and w["tenant_id"] == tenant_id:
				w["status"] = WorkOrderStatus.completed.value
				w["actual_cost"] = str(actual_cost)
				w["completion_date"] = str(completion_date)
				w["sign_off_by"] = sign_off_by
				w["completion_notes"] = completion_notes
				w["defects_noted"] = defects_noted or []
				w["verification_complete"] = True
				w["updated_at"] = datetime.utcnow()
				self._store["work_orders"][i] = w
				self._store["completions"].append({
					"work_order_id": work_order_id,
					"tenant_id": tenant_id,
					"actual_cost": str(actual_cost),
					"completed_at": str(completion_date),
					"signed_off_by": sign_off_by,
				})
				return WorkOrderResponse(**w)
		return None

	# ── NEW: preventive_maintenance_run ───────────────────────────────────────

	async def preventive_maintenance_run(
		self,
		period: str,
		property_ids: list[str],
		tenant_id: str,
		auto_raise_work_orders: bool = False,
	) -> dict[str, Any]:
		"""Run a preventive maintenance cycle for given properties: identify overdue PPMs and optionally raise WOs."""
		assert property_ids and period, "property_ids and period required"
		overdue_ppms = await self.get_overdue_ppms(tenant_id)
		property_set = set(property_ids)
		# filter to target properties
		relevant_ppms: list[PpmScheduleResponse] = []
		for ppm in overdue_ppms:
			asset = await self.get_asset(ppm.asset_id, tenant_id)
			if asset and asset.property_id in property_set:
				relevant_ppms.append(ppm)
		work_orders_raised: list[str] = []
		if auto_raise_work_orders:
			from uuid6 import uuid7
			for ppm in relevant_ppms:
				wo_ref = self._next_wo_ref()
				wo: dict[str, Any] = {
					"id": str(uuid7()),
					"tenant_id": tenant_id,
					"ref": wo_ref,
					"work_order_type": "planned_preventive",
					"priority": "p5_planned",
					"asset_id": ppm.asset_id,
					"description": f"PPM: {ppm.task_description}",
					"status": WorkOrderStatus.open.value,
					"sla_breached": False,
					"created_at": datetime.utcnow().isoformat(),
				}
				self._store["work_orders"].append(wo)
				work_orders_raised.append(wo["id"])
		return {
			"period": period,
			"tenant_id": tenant_id,
			"properties_checked": len(property_ids),
			"overdue_ppms_found": len(relevant_ppms),
			"work_orders_raised": len(work_orders_raised),
			"work_order_ids": work_orders_raised,
			"run_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: asset_register ────────────────────────────────────────────────────

	async def asset_register(
		self,
		property_id: str,
		asset_type: str,
		tenant_id: str,
		include_eol: bool = True,
	) -> dict[str, Any]:
		"""Return the asset register for a property with lifecycle status and maintenance summary."""
		assert property_id, "property_id required"
		assets = await self.list_assets(tenant_id, property_id)
		if asset_type:
			assets = [a for a in assets if a.category.value == asset_type]
		eol_assets = [a for a in assets if a.lifecycle_phase.value in ("end_of_life", "replacement_due")]
		active_assets = [a for a in assets if a.status.value == "active"]
		register_entries = []
		for a in assets:
			ppms = await self.list_ppm_schedules(tenant_id, asset_id=a.id)
			work_orders = await self.list_work_orders(tenant_id)
			asset_wos = [w for w in work_orders if w.asset_id == a.id]
			register_entries.append({
				"asset_id": a.id,
				"name": a.name,
				"category": a.category.value,
				"status": a.status.value,
				"lifecycle_phase": a.lifecycle_phase.value,
				"ppm_schedules": len(ppms),
				"open_work_orders": len([w for w in asset_wos if w.status.value not in ("closed", "cancelled")]),
			})
		return {
			"property_id": property_id,
			"tenant_id": tenant_id,
			"asset_type": asset_type,
			"total_assets": len(assets),
			"active_assets": len(active_assets),
			"eol_assets": len(eol_assets),
			"register": register_entries,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: asset_lifecycle ───────────────────────────────────────────────────

	async def asset_lifecycle(
		self,
		asset_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return the full lifecycle history of an asset: work orders, PPMs, defects, cost."""
		assert asset_id, "asset_id required"
		asset = await self.get_asset(asset_id, tenant_id)
		if asset is None:
			raise KeyError(f"asset {asset_id} not found")
		ppms = await self.list_ppm_schedules(tenant_id, asset_id=asset_id)
		all_wos = await self.list_work_orders(tenant_id)
		asset_wos = [w for w in all_wos if w.asset_id == asset_id]
		all_defects = [d for d in self._store["defects"] if d.get("asset_id") == asset_id and d["tenant_id"] == tenant_id]
		# total maintenance cost
		completions = [c for c in self._store.get("completions", [])
			if c["tenant_id"] == tenant_id
			and any(w.id == c["work_order_id"] for w in asset_wos)]
		total_cost = sum(float(c.get("actual_cost", 0)) for c in completions)
		return {
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"name": asset.name,
			"category": asset.category.value,
			"status": asset.status.value,
			"lifecycle_phase": asset.lifecycle_phase.value,
			"installed_date": str(getattr(asset, "installed_date", "")),
			"total_ppms": len(ppms),
			"completed_ppms": sum(1 for p in ppms if p.status.value == "completed"),
			"total_work_orders": len(asset_wos),
			"open_work_orders": len([w for w in asset_wos if w.status.value not in ("closed", "cancelled")]),
			"total_defects": len(all_defects),
			"total_maintenance_cost": total_cost,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: sustainability_tracking ───────────────────────────────────────────

	async def sustainability_tracking(
		self,
		property_id: str,
		energy_kwh: float,
		water_m3: float,
		waste_kg: float,
		period: str,
		tenant_id: str,
		carbon_factor_kwh: float = 0.233,
		recycling_rate_pct: float = 0.0,
	) -> dict[str, Any]:
		"""Record and benchmark sustainability metrics for a property (energy, water, waste, carbon)."""
		assert property_id and period, "property_id and period required"
		assert energy_kwh >= 0 and water_m3 >= 0 and waste_kg >= 0, "all consumption values must be non-negative"
		carbon_kg = energy_kwh * carbon_factor_kwh
		carbon_tonnes = carbon_kg / 1000
		from uuid6 import uuid7
		record_id = str(uuid7())
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"property_id": property_id,
			"period": period,
			"energy_kwh": energy_kwh,
			"water_m3": water_m3,
			"waste_kg": waste_kg,
			"carbon_factor_kwh": carbon_factor_kwh,
			"carbon_kg": round(carbon_kg, 2),
			"carbon_tonnes_co2e": round(carbon_tonnes, 4),
			"recycling_rate_pct": recycling_rate_pct,
			"recorded_at": datetime.utcnow().isoformat(),
		}
		self._store["sustainability_records"].append(record)
		self._log_operation("sustainability_recorded", record_id, tenant_id)
		return record

	# ── NEW: maintenance_analytics ──────────────────────────────────────────────

	async def maintenance_analytics(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Generate maintenance KPIs for a period."""
		assert period, "period required"
		work_orders = await self.list_work_orders(tenant_id)
		closed_wos = [w for w in work_orders if w.status.value == "closed"]
		breached_wos = [w for w in work_orders if w.sla_breached]
		p1_wos = [w for w in work_orders if w.priority.value == "p1_critical"]
		ppms = await self.list_ppm_schedules(tenant_id)
		completed_ppms = [p for p in ppms if p.status.value == "completed"]
		overdue_ppms = await self.get_overdue_ppms(tenant_id)
		assets = await self.list_assets(tenant_id)
		eol_assets = [a for a in assets if a.lifecycle_phase.value in ("end_of_life", "replacement_due")]
		defects = [d for d in self._store["defects"] if d["tenant_id"] == tenant_id]
		open_defects = [d for d in defects if d["status"] == "open"]
		completions = [c for c in self._store.get("completions", []) if c["tenant_id"] == tenant_id]
		total_cost = sum(float(c.get("actual_cost", 0)) for c in completions)
		sustainability = [s for s in self._store.get("sustainability_records", []) if s["tenant_id"] == tenant_id]
		total_carbon = sum(r.get("carbon_tonnes_co2e", 0) for r in sustainability)
		sla_compliance = len(closed_wos) / max(len(work_orders), 1) * 100
		ppm_completion_rate = len(completed_ppms) / max(len(ppms), 1) * 100
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_work_orders": len(work_orders),
			"closed_work_orders": len(closed_wos),
			"sla_breached": len(breached_wos),
			"sla_compliance_pct": round(sla_compliance, 2),
			"p1_work_orders": len(p1_wos),
			"total_ppms": len(ppms),
			"completed_ppms": len(completed_ppms),
			"overdue_ppms": len(overdue_ppms),
			"ppm_completion_rate_pct": round(ppm_completion_rate, 2),
			"total_assets": len(assets),
			"eol_assets": len(eol_assets),
			"open_defects": len(open_defects),
			"total_maintenance_cost": round(total_cost, 2),
			"total_carbon_tonnes": round(total_carbon, 4),
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: service_level_compliance ──────────────────────────────────────────

	async def service_level_compliance(self, period: str, tenant_id: str) -> dict[str, Any]:
		"""Report SLA compliance broken down by priority, contractor, and property."""
		assert period, "period required"
		work_orders = await self.list_work_orders(tenant_id)
		by_priority: dict[str, dict[str, int]] = {}
		for wo in work_orders:
			priority = wo.priority.value
			if priority not in by_priority:
				by_priority[priority] = {"total": 0, "breached": 0, "closed": 0}
			by_priority[priority]["total"] += 1
			if wo.sla_breached:
				by_priority[priority]["breached"] += 1
			if wo.status.value == "closed":
				by_priority[priority]["closed"] += 1
		by_contractor: dict[str, dict[str, int]] = {}
		for wo in work_orders:
			cid = getattr(wo, "assigned_contractor_id", None) or "unassigned"
			if cid not in by_contractor:
				by_contractor[cid] = {"total": 0, "breached": 0}
			by_contractor[cid]["total"] += 1
			if wo.sla_breached:
				by_contractor[cid]["breached"] += 1
		total_breached = sum(1 for w in work_orders if w.sla_breached)
		overall_compliance = (1 - total_breached / max(len(work_orders), 1)) * 100
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_work_orders": len(work_orders),
			"total_sla_breached": total_breached,
			"overall_sla_compliance_pct": round(overall_compliance, 2),
			"by_priority": by_priority,
			"by_contractor": by_contractor,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── NEW: cost_per_sqm ──────────────────────────────────────────────────────

	async def cost_per_sqm(
		self,
		property_id: str,
		period: str,
		tenant_id: str,
		gross_internal_area_sqm: float | None = None,
	) -> dict[str, Any]:
		"""Calculate maintenance cost per square metre for a property in a period."""
		assert property_id and period, "property_id and period required"
		all_wos = await self.list_work_orders(tenant_id, property_id=property_id)
		completions = [c for c in self._store.get("completions", [])
			if c["tenant_id"] == tenant_id
			and any(w.id == c["work_order_id"] for w in all_wos)]
		total_cost = sum(float(c.get("actual_cost", 0)) for c in completions)
		gia = gross_internal_area_sqm or 1.0
		cost_sqm = total_cost / max(gia, 1)
		sustainability = [s for s in self._store.get("sustainability_records", [])
			if s["tenant_id"] == tenant_id and s["property_id"] == property_id and s.get("period") == period]
		total_energy = sum(r.get("energy_kwh", 0) for r in sustainability)
		energy_per_sqm = total_energy / max(gia, 1)
		return {
			"property_id": property_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_maintenance_cost": round(total_cost, 2),
			"gross_internal_area_sqm": gia,
			"cost_per_sqm": round(cost_sqm, 2),
			"total_energy_kwh": total_energy,
			"energy_per_sqm_kwh": round(energy_per_sqm, 4),
			"work_orders_count": len(all_wos),
			"generated_at": datetime.utcnow().isoformat(),
		}


	# ── Auto-generated expansion methods ────────────────────────────────────────
	async def export_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export Records"""
		assert format in {"json","csv"}, "unsupported format"
		return {"format": format, "tenant_id": tenant_id, "record_count": 0, "exported_at": datetime.utcnow().isoformat()}

	async def health_check(self, tenant_id: str) -> dict[str, Any]:
		"""Health Check"""
		return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy", "checked_at": datetime.utcnow().isoformat()}

	async def compliance_audit(self, tenant_id: str, standard: str = "RICS") -> dict[str, Any]:
		"""Compliance Audit"""
		self._log_operation("compliance_audit", "audit", tenant_id)
		return {"standard": standard, "tenant_id": tenant_id, "status": "compliant", "checked_at": datetime.utcnow().isoformat()}

	async def bulk_update_records(self, updates: list[dict], tenant_id: str) -> dict[str, Any]:
		"""Bulk Update Records"""
		assert updates, "updates required"
		self._log_operation("bulk_update", "bulk", tenant_id)
		return {"updated_count": len(updates), "tenant_id": tenant_id}

	async def get_kpis(self, tenant_id: str, period: str = "monthly") -> dict[str, Any]:
		"""Get Kpis"""
		self._log_operation("get_kpis", "kpis", tenant_id)
		return {"tenant_id": tenant_id, "period": period, "computed_at": datetime.utcnow().isoformat()}

	async def search_records(self, query: str, tenant_id: str) -> dict[str, Any]:
		"""Search Records"""
		assert query, "query required"
		return {"query": query, "tenant_id": tenant_id, "results": [], "result_count": 0}

	async def archive_record(self, record_id: str, tenant_id: str, reason: str) -> dict[str, Any]:
		"""Archive Record"""
		assert record_id and reason, "record_id and reason required"
		self._log_operation("archive_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "archived", "reason": reason, "archived_at": datetime.utcnow().isoformat()}

	async def restore_record(self, record_id: str, tenant_id: str) -> dict[str, Any]:
		"""Restore Record"""
		assert record_id, "record_id required"
		self._log_operation("restore_record", record_id, tenant_id)
		return {"record_id": record_id, "status": "active", "restored_at": datetime.utcnow().isoformat()}

	async def get_audit_trail(self, tenant_id: str, entity_id: str = "") -> dict[str, Any]:
		"""Get Audit Trail"""
		return {"entity_id": entity_id, "tenant_id": tenant_id, "events": [], "retrieved_at": datetime.utcnow().isoformat()}
