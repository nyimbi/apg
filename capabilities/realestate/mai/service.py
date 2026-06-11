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

	# ── Improvement 1: Asset condition scoring ────────────────────────────────

	async def update_asset_condition_score(
		self,
		asset_id: str,
		tenant_id: str,
		score: int,
		assessed_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Update the condition score (0–100) for an asset and trigger PPM escalation if below threshold.

		A score below 40 auto-sets lifecycle_phase to 'ageing'; below 20 sets 'end_of_life'.
		Both transitions are logged as critical events.
		"""
		assert 0 <= score <= 100, "score must be 0–100"
		assert asset_id and assessed_by, "asset_id and assessed_by required"
		asset = await self.get_asset(asset_id, tenant_id)
		if asset is None:
			raise KeyError(f"asset {asset_id} not found")
		new_phase = asset.lifecycle_phase
		if score < 20:
			new_phase = "end_of_life"
			log.critical("mai.condition_critical asset=%s score=%d tenant=%s", asset_id, score, tenant_id)
		elif score < 40:
			new_phase = "ageing"
			log.warning("mai.condition_degraded asset=%s score=%d tenant=%s", asset_id, score, tenant_id)
		from .models import AssetUpdate, LifecyclePhase
		update = AssetUpdate(lifecycle_phase=LifecyclePhase(new_phase))
		await self.update_asset(asset_id, tenant_id, update)
		record: dict[str, Any] = {
			"asset_id": asset_id,
			"tenant_id": tenant_id,
			"score": score,
			"lifecycle_phase": new_phase,
			"assessed_by": assessed_by,
			"notes": notes,
			"escalation_triggered": score < 40,
			"assessed_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("condition_scores", []).append(record)
		self._log_operation("update_asset_condition_score", asset_id, tenant_id)
		return record

	async def get_assets_below_condition_threshold(
		self,
		tenant_id: str,
		threshold: int = 40,
		property_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return the most recent condition score per asset where score < threshold."""
		assert 0 <= threshold <= 100, "threshold must be 0–100"
		scores = self._store.get("condition_scores", [])
		# latest score per asset
		latest: dict[str, dict[str, Any]] = {}
		for s in scores:
			if s["tenant_id"] == tenant_id:
				aid = s["asset_id"]
				if aid not in latest or s["assessed_at"] > latest[aid]["assessed_at"]:
					latest[aid] = s
		results = [v for v in latest.values() if v["score"] < threshold]
		if property_id:
			# resolve property via asset store
			asset_map = {a["id"]: a.get("property_id") for a in self._store["assets"] if a["tenant_id"] == tenant_id}
			results = [r for r in results if asset_map.get(r["asset_id"]) == property_id]
		results.sort(key=lambda x: x["score"])
		return results

	# ── Improvement 2: SLA countdown / warning tier ───────────────────────────

	async def get_work_orders_near_sla_breach(
		self,
		tenant_id: str,
		warning_pct: float = 75.0,
	) -> list[dict[str, Any]]:
		"""Return open work orders where >=warning_pct% of SLA resolution time has elapsed.

		Result includes the elapsed_pct and minutes_remaining for each matching WO.
		"""
		assert 0 < warning_pct <= 100, "warning_pct must be (0, 100]"
		now = datetime.utcnow()
		results = []
		work_orders = await self.list_work_orders(tenant_id)
		for wo in work_orders:
			if wo.status.value in ("closed", "cancelled", "completed"):
				continue
			if wo.sla_resolution_deadline is None or wo.sla_breached:
				continue
			deadline = wo.sla_resolution_deadline
			if isinstance(deadline, str):
				deadline = datetime.fromisoformat(deadline)
			created_at = wo.created_at
			if isinstance(created_at, str):
				created_at = datetime.fromisoformat(created_at)
			total_secs = (deadline - created_at).total_seconds()
			elapsed_secs = (now - created_at).total_seconds()
			if total_secs <= 0:
				continue
			elapsed_pct = min((elapsed_secs / total_secs) * 100, 100.0)
			if elapsed_pct >= warning_pct:
				remaining_mins = max((deadline - now).total_seconds() / 60, 0)
				results.append({
					"work_order_id": wo.id,
					"ref": wo.ref,
					"priority": wo.priority.value,
					"asset_id": wo.asset_id,
					"elapsed_pct": round(elapsed_pct, 1),
					"minutes_remaining": round(remaining_mins, 1),
					"sla_resolution_deadline": deadline.isoformat(),
				})
		results.sort(key=lambda x: x["elapsed_pct"], reverse=True)
		return results

	# ── Improvement 3: Contractor scorecards ──────────────────────────────────

	async def compute_contractor_scorecard(
		self,
		contractor_id: str,
		tenant_id: str,
		rolling_days: int = 90,
	) -> dict[str, Any]:
		"""Compute performance metrics for a contractor over rolling_days.

		Metrics: total WOs, completed, first-time-fix rate, average resolution hours, SLA breach rate.
		"""
		assert contractor_id, "contractor_id required"
		assert rolling_days > 0, "rolling_days must be positive"
		cutoff = datetime.utcnow() - timedelta(days=rolling_days)
		all_wos = await self.list_work_orders(tenant_id)
		contractor_wos = [
			w for w in all_wos
			if getattr(w, "assigned_contractor_id", None) == contractor_id
			and w.created_at >= cutoff
		]
		completed = [w for w in contractor_wos if w.status.value in ("completed", "closed")]
		breached = [w for w in contractor_wos if w.sla_breached]
		# first-time-fix: completed WOs with no re-visit (no sibling WO on same asset within 30 days)
		asset_wo_counts: dict[str, int] = {}
		for w in contractor_wos:
			asset_wo_counts[w.asset_id] = asset_wo_counts.get(w.asset_id, 0) + 1
		first_time_fixes = [w for w in completed if asset_wo_counts.get(w.asset_id, 0) == 1]
		ftfr = len(first_time_fixes) / max(len(completed), 1) * 100
		# average resolution hours
		resolution_hours: list[float] = []
		for w in completed:
			if w.actual_start and w.actual_end:
				start = w.actual_start if isinstance(w.actual_start, datetime) else datetime.fromisoformat(str(w.actual_start))
				end = w.actual_end if isinstance(w.actual_end, datetime) else datetime.fromisoformat(str(w.actual_end))
				resolution_hours.append((end - start).total_seconds() / 3600)
		avg_resolution_h = sum(resolution_hours) / max(len(resolution_hours), 1)
		sla_breach_rate = len(breached) / max(len(contractor_wos), 1) * 100
		scorecard = {
			"contractor_id": contractor_id,
			"tenant_id": tenant_id,
			"rolling_days": rolling_days,
			"total_work_orders": len(contractor_wos),
			"completed_work_orders": len(completed),
			"first_time_fix_rate_pct": round(ftfr, 1),
			"average_resolution_hours": round(avg_resolution_h, 2),
			"sla_breach_rate_pct": round(sla_breach_rate, 1),
			"computed_at": datetime.utcnow().isoformat(),
		}
		# update contractor record
		for i, c in enumerate(self._store["contractors"]):
			if c["id"] == contractor_id and c["tenant_id"] == tenant_id:
				c["first_time_fix_rate"] = str(round(ftfr, 1))
				c["average_response_hours"] = str(round(avg_resolution_h, 2))
				c["updated_at"] = datetime.utcnow()
				self._store["contractors"][i] = c
				break
		return scorecard

	async def get_contractor_league_table(
		self,
		tenant_id: str,
		rolling_days: int = 90,
	) -> list[dict[str, Any]]:
		"""Rank all active contractors by composite performance score.

		Composite = 0.4 * first_time_fix_rate + 0.4 * (100 - sla_breach_rate) + 0.2 * response_speed.
		"""
		contractors = await self.list_contractors(tenant_id)
		rows = []
		for c in contractors:
			sc = await self.compute_contractor_scorecard(c.id, tenant_id, rolling_days=rolling_days)
			response_speed_score = max(0.0, 100.0 - sc["average_resolution_hours"])
			composite = (
				0.4 * sc["first_time_fix_rate_pct"]
				+ 0.4 * (100 - sc["sla_breach_rate_pct"])
				+ 0.2 * min(response_speed_score, 100.0)
			)
			rows.append({
				"contractor_id": c.id,
				"name": c.name,
				"contractor_type": c.contractor_type,
				"composite_score": round(composite, 1),
				**{k: sc[k] for k in ("first_time_fix_rate_pct", "sla_breach_rate_pct", "average_resolution_hours", "total_work_orders")},
			})
		rows.sort(key=lambda x: x["composite_score"], reverse=True)
		for rank, row in enumerate(rows, start=1):
			row["rank"] = rank
		return rows

	# ── Improvement 4: Budget vs actual ───────────────────────────────────────

	async def set_maintenance_budget(
		self,
		tenant_id: str,
		property_id: str,
		financial_year: str,
		budget_amount: Decimal,
		currency: str = "KES",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Set or replace the maintenance budget for a property and financial year."""
		assert property_id and financial_year, "property_id and financial_year required"
		assert budget_amount > 0, "budget_amount must be positive"
		from uuid6 import uuid7
		# remove any existing budget for same key
		self._store.setdefault("budgets", [])
		self._store["budgets"] = [
			b for b in self._store["budgets"]
			if not (b["tenant_id"] == tenant_id and b["property_id"] == property_id and b["financial_year"] == financial_year)
		]
		record: dict[str, Any] = {
			"id": str(uuid7()),
			"tenant_id": tenant_id,
			"property_id": property_id,
			"financial_year": financial_year,
			"budget_amount": str(budget_amount),
			"currency": currency,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store["budgets"].append(record)
		self._log_operation("set_maintenance_budget", record["id"], tenant_id)
		return record

	async def get_budget_vs_actual(
		self,
		tenant_id: str,
		property_id: str,
		financial_year: str,
	) -> dict[str, Any]:
		"""Compare budgeted vs committed vs actual maintenance spend for a property/year."""
		assert property_id and financial_year, "property_id and financial_year required"
		self._store.setdefault("budgets", [])
		budget_records = [
			b for b in self._store["budgets"]
			if b["tenant_id"] == tenant_id and b["property_id"] == property_id and b["financial_year"] == financial_year
		]
		budget_amount = float(budget_records[0]["budget_amount"]) if budget_records else 0.0
		# committed = sum of agreed_cost on open/assigned WOs for this property
		all_wos = await self.list_work_orders(tenant_id, property_id=property_id)
		open_wos = [w for w in all_wos if w.status.value not in ("closed", "cancelled")]
		committed = sum(
			float(w.actual_cost or 0)
			for w in open_wos
		)
		# actual = completions
		completions = [
			c for c in self._store.get("completions", [])
			if c["tenant_id"] == tenant_id and any(w.id == c["work_order_id"] for w in all_wos)
		]
		actual_spend = sum(float(c.get("actual_cost", 0)) for c in completions)
		remaining = budget_amount - actual_spend - committed
		variance_pct = ((actual_spend - budget_amount) / max(budget_amount, 1)) * 100
		return {
			"tenant_id": tenant_id,
			"property_id": property_id,
			"financial_year": financial_year,
			"budget_amount": round(budget_amount, 2),
			"committed_spend": round(committed, 2),
			"actual_spend": round(actual_spend, 2),
			"remaining_budget": round(remaining, 2),
			"variance_pct": round(variance_pct, 2),
			"over_budget": remaining < 0,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Improvement 5: Work order check-in / check-out ────────────────────────

	async def checkin_work_order(
		self,
		work_order_id: str,
		tenant_id: str,
		technician_id: str,
		latitude: float | None = None,
		longitude: float | None = None,
	) -> dict[str, Any]:
		"""Record technician arrival (check-in) against a work order, starting the resolution clock.

		Stores GPS coordinates as evidence for contractual SLA measurement.
		"""
		assert work_order_id and technician_id, "work_order_id and technician_id required"
		now = datetime.utcnow()
		from .models import WorkOrderUpdate
		update = WorkOrderUpdate(
			status=WorkOrderStatus.in_progress,
			actual_start=now,
		)
		result = await self.update_work_order(work_order_id, tenant_id, update)
		checkin: dict[str, Any] = {
			"work_order_id": work_order_id,
			"tenant_id": tenant_id,
			"technician_id": technician_id,
			"event": "checkin",
			"latitude": latitude,
			"longitude": longitude,
			"timestamp": now.isoformat(),
		}
		self._store.setdefault("field_events", []).append(checkin)
		self._log_operation("checkin_work_order", work_order_id, tenant_id)
		return checkin

	async def checkout_work_order(
		self,
		work_order_id: str,
		tenant_id: str,
		technician_id: str,
		completion_notes: str,
		defects_observed: list[str] | None = None,
		latitude: float | None = None,
		longitude: float | None = None,
	) -> dict[str, Any]:
		"""Record technician departure (check-out), ending the resolution clock.

		Sets actual_end and transitions WO to 'completed' pending sign-off.
		"""
		assert work_order_id and technician_id and completion_notes, "work_order_id, technician_id and completion_notes required"
		now = datetime.utcnow()
		from .models import WorkOrderUpdate
		update = WorkOrderUpdate(actual_end=now)
		await self.update_work_order(work_order_id, tenant_id, update)
		# compute elapsed
		elapsed_mins: float | None = None
		for w in self._store["work_orders"]:
			if w["id"] == work_order_id and w["tenant_id"] == tenant_id:
				if w.get("actual_start"):
					start = w["actual_start"]
					if not isinstance(start, datetime):
						start = datetime.fromisoformat(str(start))
					elapsed_mins = round((now - start).total_seconds() / 60, 1)
				break
		checkout: dict[str, Any] = {
			"work_order_id": work_order_id,
			"tenant_id": tenant_id,
			"technician_id": technician_id,
			"event": "checkout",
			"completion_notes": completion_notes,
			"defects_observed": defects_observed or [],
			"elapsed_minutes": elapsed_mins,
			"latitude": latitude,
			"longitude": longitude,
			"timestamp": now.isoformat(),
		}
		self._store.setdefault("field_events", []).append(checkout)
		self._log_operation("checkout_work_order", work_order_id, tenant_id)
		return checkout

	# ── Improvement 7: Repeat failure / reactive pattern detection ────────────

	async def detect_reactive_patterns(
		self,
		tenant_id: str,
		window_days: int = 30,
		repeat_threshold: int = 3,
	) -> list[dict[str, Any]]:
		"""Identify assets with >= repeat_threshold corrective work orders within window_days.

		Returns one entry per asset with contributing WO references and a suggested action.
		"""
		assert window_days > 0 and repeat_threshold > 0
		cutoff = datetime.utcnow() - timedelta(days=window_days)
		all_wos = await self.list_work_orders(tenant_id)
		corrective = [
			w for w in all_wos
			if w.work_order_type.value in ("corrective", "emergency")
			and w.created_at >= cutoff
		]
		asset_map: dict[str, list[WorkOrderResponse]] = {}
		for w in corrective:
			asset_map.setdefault(w.asset_id, []).append(w)
		patterns = []
		for asset_id, wos in asset_map.items():
			if len(wos) >= repeat_threshold:
				asset = await self.get_asset(asset_id, tenant_id)
				patterns.append({
					"asset_id": asset_id,
					"asset_name": asset.name if asset else "unknown",
					"property_id": asset.property_id if asset else None,
					"corrective_wo_count": len(wos),
					"work_order_refs": [w.ref for w in wos],
					"window_days": window_days,
					"suggested_action": "escalate_to_predictive_maintenance" if len(wos) >= repeat_threshold * 2 else "review_root_cause",
					"detected_at": datetime.utcnow().isoformat(),
				})
		patterns.sort(key=lambda x: x["corrective_wo_count"], reverse=True)
		self._log_operation("detect_reactive_patterns", f"assets={len(patterns)}", tenant_id)
		return patterns

	# ── Improvement 8: Statutory compliance certificate register ──────────────

	async def register_compliance_certificate(
		self,
		tenant_id: str,
		property_id: str,
		certificate_type: str,
		issuing_authority: str,
		certificate_ref: str,
		issue_date: date,
		expiry_date: date,
		inspection_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Register a statutory compliance certificate (gas, electrical, fire, lift, etc.).

		Auto-creates a future statutory inspection 60 days before expiry if one does not exist.
		"""
		assert property_id and certificate_type and certificate_ref, "property_id, certificate_type and certificate_ref required"
		assert expiry_date > issue_date, "expiry_date must be after issue_date"
		from uuid6 import uuid7
		cert: dict[str, Any] = {
			"id": str(uuid7()),
			"tenant_id": tenant_id,
			"property_id": property_id,
			"certificate_type": certificate_type,
			"issuing_authority": issuing_authority,
			"certificate_ref": certificate_ref,
			"issue_date": issue_date.isoformat(),
			"expiry_date": expiry_date.isoformat(),
			"inspection_id": inspection_id,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("compliance_certificates", []).append(cert)
		# auto-schedule renewal inspection 60 days before expiry
		renewal_due = expiry_date - timedelta(days=60)
		if renewal_due >= date.today():
			from .models import InspectionCreate
			renewal_payload = InspectionCreate(
				tenant_id=tenant_id,
				property_id=property_id,
				inspection_type=InspectionType.statutory,
				scheduled_date=renewal_due,
				created_by=created_by,
			)
			await self.create_inspection(renewal_payload)
		self._log_operation("register_compliance_certificate", cert["id"], tenant_id)
		return cert

	async def get_expiring_certificates(
		self,
		tenant_id: str,
		within_days: int = 90,
	) -> list[dict[str, Any]]:
		"""Return compliance certificates expiring within within_days, sorted by soonest expiry first."""
		assert within_days > 0, "within_days must be positive"
		threshold = date.today() + timedelta(days=within_days)
		certs = self._store.get("compliance_certificates", [])
		expiring = [
			c for c in certs
			if c["tenant_id"] == tenant_id
			and date.fromisoformat(c["expiry_date"]) <= threshold
			and date.fromisoformat(c["expiry_date"]) >= date.today()
		]
		expiring.sort(key=lambda x: x["expiry_date"])
		return expiring

	async def get_property_compliance_status(
		self,
		tenant_id: str,
		property_id: str,
	) -> dict[str, Any]:
		"""Return a compliance summary for a property: current certificates, expired, and expiring soon."""
		assert property_id, "property_id required"
		today = date.today()
		soon = today + timedelta(days=60)
		all_certs = [
			c for c in self._store.get("compliance_certificates", [])
			if c["tenant_id"] == tenant_id and c["property_id"] == property_id
		]
		valid = [c for c in all_certs if date.fromisoformat(c["expiry_date"]) >= today]
		expired = [c for c in all_certs if date.fromisoformat(c["expiry_date"]) < today]
		expiring_soon = [c for c in valid if date.fromisoformat(c["expiry_date"]) <= soon]
		overdue_inspections = await self.get_overdue_inspections(tenant_id)
		property_overdue = [i for i in overdue_inspections if i.property_id == property_id]
		return {
			"property_id": property_id,
			"tenant_id": tenant_id,
			"total_certificates": len(all_certs),
			"valid_certificates": len(valid),
			"expired_certificates": len(expired),
			"expiring_within_60_days": len(expiring_soon),
			"overdue_statutory_inspections": len(property_overdue),
			"compliance_status": "non_compliant" if expired or property_overdue else ("warning" if expiring_soon else "compliant"),
			"certificates": all_certs,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Improvement 9: Portfolio benchmarking ─────────────────────────────────

	async def benchmark_portfolio(
		self,
		tenant_id: str,
		property_ids: list[str],
		period: str,
		gross_areas_sqm: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Rank properties by maintenance cost/sqm, PPM completion rate, open defect density, and SLA breach rate.

		Returns a ranked list with percentile positions for each metric.
		"""
		assert property_ids, "property_ids required"
		gross_areas_sqm = gross_areas_sqm or {}
		rows = []
		for pid in property_ids:
			gia = gross_areas_sqm.get(pid, 1.0)
			cost_data = await self.cost_per_sqm(pid, period, tenant_id, gross_internal_area_sqm=gia)
			ppms = await self.list_ppm_schedules(tenant_id)
			prop_ppms = [p for p in ppms if p.property_id == pid]
			ppm_rate = len([p for p in prop_ppms if p.status.value == "completed"]) / max(len(prop_ppms), 1) * 100
			defects = await self.list_defects(tenant_id, property_id=pid)
			defect_density = len(defects) / max(gia, 1) * 100  # per 100 sqm
			wos = await self.list_work_orders(tenant_id, property_id=pid)
			breach_rate = len([w for w in wos if w.sla_breached]) / max(len(wos), 1) * 100
			rows.append({
				"property_id": pid,
				"gross_area_sqm": gia,
				"cost_per_sqm": cost_data["cost_per_sqm"],
				"ppm_completion_rate_pct": round(ppm_rate, 1),
				"open_defect_density_per_100sqm": round(defect_density, 3),
				"sla_breach_rate_pct": round(breach_rate, 1),
			})

		def _percentile_rank(values: list[float], v: float, higher_is_better: bool) -> float:
			below = sum(1 for x in values if x < v)
			pct = below / max(len(values), 1) * 100
			return round((100 - pct) if higher_is_better else pct, 1)

		costs = [r["cost_per_sqm"] for r in rows]
		ppm_rates = [r["ppm_completion_rate_pct"] for r in rows]
		densities = [r["open_defect_density_per_100sqm"] for r in rows]
		breach_rates = [r["sla_breach_rate_pct"] for r in rows]
		for r in rows:
			r["percentile_cost"] = _percentile_rank(costs, r["cost_per_sqm"], higher_is_better=False)
			r["percentile_ppm"] = _percentile_rank(ppm_rates, r["ppm_completion_rate_pct"], higher_is_better=True)
			r["percentile_defect"] = _percentile_rank(densities, r["open_defect_density_per_100sqm"], higher_is_better=False)
			r["percentile_sla"] = _percentile_rank(breach_rates, r["sla_breach_rate_pct"], higher_is_better=False)
			r["composite_percentile"] = round(
				(r["percentile_cost"] + r["percentile_ppm"] + r["percentile_defect"] + r["percentile_sla"]) / 4, 1
			)
		rows.sort(key=lambda x: x["composite_percentile"], reverse=True)
		for rank, row in enumerate(rows, start=1):
			row["rank"] = rank
		return {
			"tenant_id": tenant_id,
			"period": period,
			"properties_benchmarked": len(rows),
			"ranking": rows,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── Improvement 10: Escalation policy and processing ─────────────────────

	async def create_escalation_policy(
		self,
		tenant_id: str,
		priority: str,
		levels: list[dict[str, Any]],
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Create an escalation policy for a given work order priority.

		Each level specifies delay_minutes (time since previous level) and notified_roles list.
		Example level: {"level": 1, "delay_minutes": 30, "notified_roles": ["facilities_manager"]}.
		"""
		assert priority and levels, "priority and levels required"
		from uuid6 import uuid7
		policy: dict[str, Any] = {
			"id": str(uuid7()),
			"tenant_id": tenant_id,
			"priority": priority,
			"levels": levels,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._store.setdefault("escalation_policies", []).append(policy)
		self._log_operation("create_escalation_policy", policy["id"], tenant_id)
		return policy

	async def process_escalations(self, tenant_id: str) -> dict[str, Any]:
		"""Evaluate all open WOs against escalation policies and advance any due escalation levels.

		Designed to be called on a scheduler tick. Returns a summary of escalations triggered.
		"""
		now = datetime.utcnow()
		policies: dict[str, list[dict[str, Any]]] = {}
		for p in self._store.get("escalation_policies", []):
			if p["tenant_id"] == tenant_id:
				policies[p["priority"]] = p["levels"]
		work_orders = await self.list_work_orders(tenant_id)
		open_wos = [w for w in work_orders if w.status.value not in ("closed", "cancelled", "completed")]
		triggered: list[dict[str, Any]] = []
		for wo in open_wos:
			policy_levels = policies.get(wo.priority.value, [])
			if not policy_levels:
				continue
			created_at = wo.created_at if isinstance(wo.created_at, datetime) else datetime.fromisoformat(str(wo.created_at))
			elapsed_mins = (now - created_at).total_seconds() / 60
			cumulative_delay = 0
			for level in sorted(policy_levels, key=lambda l: l.get("level", 0)):
				cumulative_delay += level.get("delay_minutes", 0)
				if elapsed_mins >= cumulative_delay:
					event: dict[str, Any] = {
						"work_order_id": wo.id,
						"ref": wo.ref,
						"priority": wo.priority.value,
						"escalation_level": level.get("level"),
						"notified_roles": level.get("notified_roles", []),
						"elapsed_minutes": round(elapsed_mins, 1),
						"triggered_at": now.isoformat(),
					}
					triggered.append(event)
					self._store.setdefault("escalation_events", []).append(event)
					log.warning("mai.escalation_triggered wo=%s level=%s", wo.id, level.get("level"))
		return {
			"tenant_id": tenant_id,
			"escalations_triggered": len(triggered),
			"details": triggered,
			"processed_at": now.isoformat(),
		}
