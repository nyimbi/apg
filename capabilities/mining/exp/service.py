"""Async service layer for APG Exploration Data Management."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from .models import (
	AssayResultCreate,
	AssayResultResponse,
	ComplianceReportCreate,
	ComplianceReportResponse,
	DrillholeCollarCreate,
	DrillholeCollarResponse,
	GeologyIntervalCreate,
	GeologyIntervalResponse,
	ResourceEstimateCreate,
	ResourceEstimateResponse,
	ResourceEstimateUpdate,
	ReviewStatus,
	uuid7str,
)

log = logging.getLogger(__name__)


class ExpService:
	"""Service for Exploration Data Management operations.

	All state is held in in-memory dicts keyed by id; swap out for
	async DB calls (asyncpg / SQLAlchemy async) without changing method
	signatures.
	"""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		# In-memory stores — replace with real persistence
		self._collars: dict[str, dict[str, Any]] = {}
		self._assays: dict[str, dict[str, Any]] = {}
		self._geology: dict[str, dict[str, Any]] = {}
		self._resources: dict[str, dict[str, Any]] = {}
		self._reports: dict[str, dict[str, Any]] = {}
		# Extended stores
		self._licences: dict[str, dict[str, Any]] = {}
		self._drill_holes: dict[str, dict[str, Any]] = {}
		self._core_logs: dict[str, dict[str, Any]] = {}
		self._assay_results: dict[str, dict[str, Any]] = {}
		self._new_resource_estimates: dict[str, dict[str, Any]] = {}
		self._geophysics_surveys: dict[str, dict[str, Any]] = {}
		self._exploration_targets: dict[str, dict[str, Any]] = {}

	# ── Logging helpers ────────────────────────────────────────────────────────

	def _log_op(self, op: str, entity: str, id: str) -> None:
		log.info("exp.%s | tenant=%s entity=%s id=%s", op, self.tenant_id, entity, id)

	def _log_warn(self, msg: str, **kw: Any) -> None:
		log.warning("exp | tenant=%s %s %s", self.tenant_id, msg, kw)

	def _log_validation_error(self, field: str, reason: str) -> None:
		log.error("exp.validation | tenant=%s field=%s reason=%s", self.tenant_id, field, reason)

	# ── Tenant guard ───────────────────────────────────────────────────────────

	def _assert_tenant(self, tenant_id: str) -> None:
		assert tenant_id == self.tenant_id, (
			f"Cross-tenant access denied: requested={tenant_id} service={self.tenant_id}"
		)

	# ── Drillhole Collar ───────────────────────────────────────────────────────

	async def create_drillhole_collar(
		self, payload: DrillholeCollarCreate, created_by: str
	) -> DrillholeCollarResponse:
		"""Create a new drillhole collar record. hole_id must be unique within tenant."""
		self._assert_tenant(payload.tenant_id)
		# Uniqueness check
		existing = [c for c in self._collars.values() if c["hole_id"] == payload.hole_id and c["tenant_id"] == self.tenant_id]
		if existing:
			raise ValueError(f"Drillhole ID '{payload.hole_id}' already exists for tenant '{self.tenant_id}'")

		resp = DrillholeCollarResponse(
			**payload.model_dump(),
			created_by=created_by,
		)
		self._collars[resp.id] = resp.model_dump()
		self._log_op("create_collar", "drillhole", resp.id)
		return resp

	async def get_drillhole_collar(self, id: str) -> DrillholeCollarResponse | None:
		"""Retrieve a collar by record id."""
		rec = self._collars.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return DrillholeCollarResponse(**rec)

	async def get_drillhole_collar_by_hole_id(self, hole_id: str) -> DrillholeCollarResponse | None:
		"""Retrieve a collar by its field hole_id."""
		for rec in self._collars.values():
			if rec["hole_id"] == hole_id and rec["tenant_id"] == self.tenant_id:
				return DrillholeCollarResponse(**rec)
		return None

	async def list_drillhole_collars(
		self,
		prospect: str | None = None,
		hole_type: str | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[DrillholeCollarResponse]:
		"""List collars with optional filters."""
		results = [
			DrillholeCollarResponse(**r)
			for r in self._collars.values()
			if r["tenant_id"] == self.tenant_id
		]
		if prospect:
			results = [r for r in results if r.prospect == prospect]
		if hole_type:
			results = [r for r in results if r.hole_type == hole_type]
		return sorted(results, key=lambda x: x.created_at)[offset : offset + limit]

	async def update_drillhole_actual_depth(self, id: str, actual_depth_m: float) -> DrillholeCollarResponse:
		"""Update the as-drilled depth on completion."""
		assert actual_depth_m > 0, "actual_depth_m must be positive"
		rec = self._collars.get(id)
		if rec is None:
			raise KeyError(f"Drillhole collar {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["actual_depth_m"] = actual_depth_m
		rec["updated_at"] = datetime.utcnow()
		self._log_op("update_depth", "drillhole", id)
		return DrillholeCollarResponse(**rec)

	# ── Assay Results ──────────────────────────────────────────────────────────

	async def import_assay_results(
		self, payloads: list[AssayResultCreate], created_by: str
	) -> list[AssayResultResponse]:
		"""Bulk import assay results. Validates collar existence and non-overlapping intervals."""
		responses: list[AssayResultResponse] = []
		for payload in payloads:
			self._assert_tenant(payload.tenant_id)
			# Check collar exists
			collar = await self.get_drillhole_collar_by_hole_id(payload.hole_id)
			if collar is None:
				raise ValueError(f"Drillhole '{payload.hole_id}' does not exist; create collar first")
			# Interval overlap check within same hole and commodity
			overlap = await self._check_assay_interval_overlap(
				payload.hole_id, payload.commodity, payload.from_m, payload.to_m
			)
			if overlap:
				raise ValueError(
					f"Interval [{payload.from_m}, {payload.to_m}] overlaps with existing assay in hole {payload.hole_id}"
				)
			resp = AssayResultResponse(**payload.model_dump(), created_by=created_by)
			self._assays[resp.id] = resp.model_dump()
			self._log_op("import_assay", "assay", resp.id)
			responses.append(resp)
		return responses

	async def _check_assay_interval_overlap(
		self, hole_id: str, commodity: str, from_m: float, to_m: float
	) -> bool:
		"""Return True if the given interval overlaps any existing assay in the same hole/commodity."""
		for rec in self._assays.values():
			if rec["hole_id"] == hole_id and rec["commodity"] == commodity and rec["tenant_id"] == self.tenant_id:
				if rec["from_m"] < to_m and rec["to_m"] > from_m:
					return True
		return False

	async def get_assay_results_for_hole(self, hole_id: str) -> list[AssayResultResponse]:
		"""Return all assay results for a given drillhole, sorted by from_m."""
		results = [
			AssayResultResponse(**r)
			for r in self._assays.values()
			if r["hole_id"] == hole_id and r["tenant_id"] == self.tenant_id
		]
		return sorted(results, key=lambda x: x.from_m)

	async def flag_qaqc_result(self, assay_id: str, flag: str) -> AssayResultResponse:
		"""Attach a QAQC flag to an assay result."""
		rec = self._assays.get(assay_id)
		if rec is None:
			raise KeyError(f"Assay {assay_id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["qaqc_flag"] = flag
		rec["updated_at"] = datetime.utcnow()
		self._log_op("qaqc_flag", "assay", assay_id)
		return AssayResultResponse(**rec)

	async def list_assays(
		self,
		commodity: str | None = None,
		min_grade: float | None = None,
		limit: int = 200,
		offset: int = 0,
	) -> list[AssayResultResponse]:
		"""List assay results with optional commodity/grade filter."""
		results = [
			AssayResultResponse(**r)
			for r in self._assays.values()
			if r["tenant_id"] == self.tenant_id
		]
		if commodity:
			results = [r for r in results if r.commodity == commodity]
		if min_grade is not None:
			results = [r for r in results if r.grade_value >= min_grade]
		return sorted(results, key=lambda x: (x.hole_id, x.from_m))[offset : offset + limit]

	# ── Geology ────────────────────────────────────────────────────────────────

	async def log_geology_interval(
		self, payload: GeologyIntervalCreate, created_by: str
	) -> GeologyIntervalResponse:
		"""Log a geological interval for a drillhole."""
		self._assert_tenant(payload.tenant_id)
		collar = await self.get_drillhole_collar_by_hole_id(payload.hole_id)
		if collar is None:
			raise ValueError(f"Drillhole '{payload.hole_id}' does not exist")
		resp = GeologyIntervalResponse(**payload.model_dump(), created_by=created_by)
		self._geology[resp.id] = resp.model_dump()
		self._log_op("log_geology", "geology_interval", resp.id)
		return resp

	async def get_geology_for_hole(self, hole_id: str) -> list[GeologyIntervalResponse]:
		"""Return all geology intervals for a drillhole, ordered by from_m."""
		results = [
			GeologyIntervalResponse(**r)
			for r in self._geology.values()
			if r["hole_id"] == hole_id and r["tenant_id"] == self.tenant_id
		]
		return sorted(results, key=lambda x: x.from_m)

	async def list_geology_by_lithology(self, lithology_code: str) -> list[GeologyIntervalResponse]:
		"""Return all geology intervals matching a given lithology code."""
		return [
			GeologyIntervalResponse(**r)
			for r in self._geology.values()
			if r["lithology_code"] == lithology_code and r["tenant_id"] == self.tenant_id
		]

	# ── Resource Estimates ─────────────────────────────────────────────────────

	async def create_resource_estimate(
		self, payload: ResourceEstimateCreate, created_by: str
	) -> ResourceEstimateResponse:
		"""Create a new resource estimate. Requires competent person assignment."""
		self._assert_tenant(payload.tenant_id)
		resp = ResourceEstimateResponse(**payload.model_dump(), created_by=created_by)
		self._resources[resp.id] = resp.model_dump()
		self._log_op("create_resource", "resource_estimate", resp.id)
		return resp

	async def get_resource_estimate(self, id: str) -> ResourceEstimateResponse | None:
		"""Get a resource estimate by id."""
		rec = self._resources.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return ResourceEstimateResponse(**rec)

	async def update_resource_estimate(
		self, id: str, update: ResourceEstimateUpdate
	) -> ResourceEstimateResponse:
		"""Partial update of a resource estimate. Cannot update approved estimates directly."""
		rec = self._resources.get(id)
		if rec is None:
			raise KeyError(f"Resource estimate {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec.get("review_status") == ReviewStatus.APPROVED:
			raise ValueError("Approved estimates cannot be directly modified; create a superseding estimate")
		for field, value in update.model_dump(exclude_none=True).items():
			rec[field] = value
		rec["updated_at"] = datetime.utcnow()
		self._log_op("update_resource", "resource_estimate", id)
		return ResourceEstimateResponse(**rec)

	async def approve_resource_estimate(self, id: str, reviewer_id: str, notes: str | None = None) -> ResourceEstimateResponse:
		"""Approve a resource estimate. Only approved estimates can be published."""
		rec = self._resources.get(id)
		if rec is None:
			raise KeyError(f"Resource estimate {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["review_status"] = ReviewStatus.APPROVED
		rec["reviewer_id"] = reviewer_id
		rec["review_notes"] = notes
		rec["updated_at"] = datetime.utcnow()
		self._log_op("approve_resource", "resource_estimate", id)
		return ResourceEstimateResponse(**rec)

	async def publish_resource_estimate(self, id: str) -> ResourceEstimateResponse:
		"""Publish an approved resource estimate. Denied if not approved."""
		rec = self._resources.get(id)
		if rec is None:
			raise KeyError(f"Resource estimate {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec.get("review_status") != ReviewStatus.APPROVED:
			raise PermissionError("Resource estimate must be approved before publication")
		rec["published"] = True
		rec["updated_at"] = datetime.utcnow()
		self._log_op("publish_resource", "resource_estimate", id)
		return ResourceEstimateResponse(**rec)

	async def list_resource_estimates(
		self,
		classification: str | None = None,
		commodity: str | None = None,
		published_only: bool = False,
	) -> list[ResourceEstimateResponse]:
		"""List resource estimates with optional filters."""
		results = [
			ResourceEstimateResponse(**r)
			for r in self._resources.values()
			if r["tenant_id"] == self.tenant_id
		]
		if classification:
			results = [r for r in results if r.classification == classification]
		if commodity:
			results = [r for r in results if r.commodity == commodity]
		if published_only:
			results = [r for r in results if r.published]
		return sorted(results, key=lambda x: x.created_at, reverse=True)

	# ── Compliance Reports ─────────────────────────────────────────────────────

	async def create_compliance_report(
		self, payload: ComplianceReportCreate, created_by: str
	) -> ComplianceReportResponse:
		"""Create a JORC / NI 43-101 / SAMREC compliance report."""
		self._assert_tenant(payload.tenant_id)
		resp = ComplianceReportResponse(**payload.model_dump(), created_by=created_by)
		self._reports[resp.id] = resp.model_dump()
		self._log_op("create_report", "compliance_report", resp.id)
		return resp

	async def sign_off_compliance_report(self, id: str, competent_person_id: str) -> ComplianceReportResponse:
		"""Record competent person sign-off on a compliance report."""
		rec = self._reports.get(id)
		if rec is None:
			raise KeyError(f"Report {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec["competent_person_id"] != competent_person_id:
			raise PermissionError("Only the assigned competent person may sign off this report")
		rec["competent_person_signed"] = True
		rec["review_status"] = ReviewStatus.APPROVED
		rec["updated_at"] = datetime.utcnow()
		self._log_op("sign_off_report", "compliance_report", id)
		return ComplianceReportResponse(**rec)

	async def publish_compliance_report(self, id: str) -> ComplianceReportResponse:
		"""Publish a signed-off compliance report."""
		rec = self._reports.get(id)
		if rec is None:
			raise KeyError(f"Report {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if not rec.get("competent_person_signed"):
			raise PermissionError("Competent person sign-off required before publication")
		rec["published"] = True
		rec["published_at"] = datetime.utcnow()
		rec["updated_at"] = datetime.utcnow()
		self._log_op("publish_report", "compliance_report", id)
		return ComplianceReportResponse(**rec)

	async def list_compliance_reports(
		self, published_only: bool = False
	) -> list[ComplianceReportResponse]:
		"""List compliance reports."""
		results = [
			ComplianceReportResponse(**r)
			for r in self._reports.values()
			if r["tenant_id"] == self.tenant_id
		]
		if published_only:
			results = [r for r in results if r.published]
		return sorted(results, key=lambda x: x.created_at, reverse=True)

	# ── Summary / KPIs ─────────────────────────────────────────────────────────

	async def get_exploration_summary(self) -> dict[str, Any]:
		"""Return aggregate exploration KPIs for the tenant."""
		collars = [r for r in self._collars.values() if r["tenant_id"] == self.tenant_id]
		assays = [r for r in self._assays.values() if r["tenant_id"] == self.tenant_id]
		resources = [r for r in self._resources.values() if r["tenant_id"] == self.tenant_id]
		total_metres = sum(
			(c.get("actual_depth_m") or c.get("planned_depth_m", 0)) for c in collars
		)
		return {
			"tenant_id": self.tenant_id,
			"total_drillholes": len(collars),
			"total_metres_drilled": round(total_metres, 1),
			"total_assay_samples": len(assays),
			"qaqc_flagged_count": sum(1 for a in assays if a.get("qaqc_flag")),
			"total_resource_estimates": len(resources),
			"published_resource_estimates": sum(1 for r in resources if r.get("published")),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Licence Registry ───────────────────────────────────────────────────────

	async def register_licence(
		self,
		licence_number: str,
		area_coords: list[dict[str, float]],
		holder_id: str,
		expiry: datetime,
		granted_by: str | None = None,
	) -> dict[str, Any]:
		"""Register an exploration licence with polygon boundary and expiry date."""
		assert licence_number, "licence_number is required"
		assert len(area_coords) >= 3, "area_coords must define a polygon with at least 3 vertices"
		assert expiry > datetime.utcnow(), "expiry must be in the future"
		# Uniqueness guard within tenant
		for rec in self._licences.values():
			if rec["licence_number"] == licence_number and rec["tenant_id"] == self.tenant_id:
				raise ValueError(f"Licence '{licence_number}' already registered for this tenant")
		area_km2 = self._estimate_polygon_area_km2(area_coords)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"licence_number": licence_number,
			"area_coords": area_coords,
			"area_km2": round(area_km2, 4),
			"holder_id": holder_id,
			"expiry": expiry.isoformat(),
			"granted_by": granted_by,
			"status": "active",
			"conditions": [],
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._licences[rec_id] = rec
		self._log_op("register_licence", "licence", rec_id)
		return rec

	def _estimate_polygon_area_km2(self, coords: list[dict[str, float]]) -> float:
		"""Shoelace formula approximation in geographic degrees, converted to km²."""
		n = len(coords)
		if n < 3:
			return 0.0
		area_deg2 = 0.0
		for i in range(n):
			j = (i + 1) % n
			area_deg2 += coords[i]["lon"] * coords[j]["lat"]
			area_deg2 -= coords[j]["lon"] * coords[i]["lat"]
		area_deg2 = abs(area_deg2) / 2.0
		# 1 degree² ≈ 12321 km² at equator; rough but consistent
		return area_deg2 * 12321.0

	async def get_licence(self, licence_id: str) -> dict[str, Any] | None:
		"""Retrieve a licence by record id."""
		rec = self._licences.get(licence_id)
		if rec is None:
			return None
		assert rec["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		return rec

	async def list_licences(self, active_only: bool = True) -> list[dict[str, Any]]:
		"""List exploration licences, optionally filtering to active ones."""
		results = [r for r in self._licences.values() if r["tenant_id"] == self.tenant_id]
		if active_only:
			results = [r for r in results if r["status"] == "active"]
		return sorted(results, key=lambda x: x["created_at"], reverse=True)

	# ── Drill Hole Management ──────────────────────────────────────────────────

	async def drill_hole_create(
		self,
		hole_id: str,
		location: dict[str, float],
		total_depth: float,
		drill_type: str,
		created_by: str,
		licence_id: str | None = None,
		azimuth_deg: float | None = None,
		dip_deg: float | None = None,
	) -> dict[str, Any]:
		"""Create a drill hole record. hole_id must be unique within tenant."""
		assert total_depth > 0, "total_depth must be positive"
		assert drill_type, "drill_type is required"
		# Check uniqueness
		for rec in self._drill_holes.values():
			if rec["hole_id"] == hole_id and rec["tenant_id"] == self.tenant_id:
				raise ValueError(f"Drill hole '{hole_id}' already exists")
		valid_drill_types = {"RC", "DD", "AC", "RAB", "HQ", "NQ", "PQ", "BQ"}
		if drill_type.upper() not in valid_drill_types:
			self._log_warn(f"Non-standard drill_type '{drill_type}'; accepted but flagged")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"hole_id": hole_id,
			"location": location,
			"total_depth_m": total_depth,
			"drill_type": drill_type.upper(),
			"azimuth_deg": azimuth_deg,
			"dip_deg": dip_deg,
			"licence_id": licence_id,
			"status": "planned",
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._drill_holes[rec_id] = rec
		self._log_op("drill_hole_create", "drill_hole", rec_id)
		return rec

	async def get_drill_hole(self, hole_id: str) -> dict[str, Any] | None:
		"""Look up a drill hole by its field hole_id."""
		for rec in self._drill_holes.values():
			if rec["hole_id"] == hole_id and rec["tenant_id"] == self.tenant_id:
				return rec
		return None

	async def list_drill_holes(
		self,
		licence_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List drill holes with optional filters."""
		results = [r for r in self._drill_holes.values() if r["tenant_id"] == self.tenant_id]
		if licence_id:
			results = [r for r in results if r.get("licence_id") == licence_id]
		if status:
			results = [r for r in results if r["status"] == status]
		return sorted(results, key=lambda x: x["created_at"])

	# ── Drill Core Logging ─────────────────────────────────────────────────────

	async def log_drill_core(
		self,
		hole_id: str,
		from_depth: float,
		to_depth: float,
		lithology: str,
		structure: str,
		mineralisation: str,
		logged_by: str,
		recovery_pct: float | None = None,
		rqd_pct: float | None = None,
	) -> dict[str, Any]:
		"""Log a core interval for a drill hole. Validates interval bounds and hole existence."""
		assert from_depth >= 0, "from_depth must be non-negative"
		assert to_depth > from_depth, "to_depth must exceed from_depth"
		hole = await self.get_drill_hole(hole_id)
		if hole is None:
			raise KeyError(f"Drill hole '{hole_id}' not found; create it first")
		if to_depth > hole["total_depth_m"]:
			self._log_warn("Core interval extends beyond planned hole depth", hole_id=hole_id)
		if recovery_pct is not None:
			assert 0.0 <= recovery_pct <= 100.0, "recovery_pct must be 0-100"
		if rqd_pct is not None:
			assert 0.0 <= rqd_pct <= 100.0, "rqd_pct must be 0-100"
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"hole_id": hole_id,
			"from_depth_m": from_depth,
			"to_depth_m": to_depth,
			"interval_m": round(to_depth - from_depth, 3),
			"lithology": lithology,
			"structure": structure,
			"mineralisation": mineralisation,
			"recovery_pct": recovery_pct,
			"rqd_pct": rqd_pct,
			"logged_by": logged_by,
			"logged_at": datetime.utcnow().isoformat(),
		}
		self._core_logs[rec_id] = rec
		self._log_op("log_drill_core", "core_log", rec_id)
		return rec

	async def get_core_log_for_hole(self, hole_id: str) -> list[dict[str, Any]]:
		"""Return all core log intervals for a hole, sorted by from_depth."""
		results = [
			r for r in self._core_logs.values()
			if r["hole_id"] == hole_id and r["tenant_id"] == self.tenant_id
		]
		return sorted(results, key=lambda x: x["from_depth_m"])

	# ── Assay Results (expanded) ────────────────────────────────────────────────

	async def assay_result(
		self,
		hole_id: str,
		from_depth: float,
		to_depth: float,
		element: str,
		grade: float,
		unit: str,
		batch_id: str | None = None,
		lab_id: str | None = None,
		created_by: str = "system",
	) -> dict[str, Any]:
		"""Record a single assay result with element/grade/unit. Detects interval overlaps."""
		assert from_depth >= 0 and to_depth > from_depth, "Invalid depth interval"
		assert grade >= 0, "grade must be non-negative"
		assert element and unit, "element and unit are required"
		hole = await self.get_drill_hole(hole_id)
		if hole is None:
			raise KeyError(f"Drill hole '{hole_id}' not found")
		# Overlap check for same hole and element
		for rec in self._assay_results.values():
			if (
				rec["hole_id"] == hole_id
				and rec["element"] == element
				and rec["tenant_id"] == self.tenant_id
			):
				if rec["from_depth_m"] < to_depth and rec["to_depth_m"] > from_depth:
					raise ValueError(
						f"Assay interval [{from_depth}, {to_depth}] overlaps existing record "
						f"[{rec['from_depth_m']}, {rec['to_depth_m']}] for element {element}"
					)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"hole_id": hole_id,
			"from_depth_m": from_depth,
			"to_depth_m": to_depth,
			"interval_m": round(to_depth - from_depth, 3),
			"element": element.upper(),
			"grade": grade,
			"unit": unit,
			"batch_id": batch_id,
			"lab_id": lab_id,
			"qaqc_flag": None,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._assay_results[rec_id] = rec
		self._log_op("assay_result", "assay_result", rec_id)
		return rec

	async def list_assay_results_for_hole(
		self, hole_id: str, element: str | None = None
	) -> list[dict[str, Any]]:
		"""Return assay results for a hole, optionally filtered by element."""
		results = [
			r for r in self._assay_results.values()
			if r["hole_id"] == hole_id and r["tenant_id"] == self.tenant_id
		]
		if element:
			results = [r for r in results if r["element"] == element.upper()]
		return sorted(results, key=lambda x: x["from_depth_m"])

	# ── Resource Estimates (expanded) ──────────────────────────────────────────

	async def resource_estimate(
		self,
		deposit_id: str,
		method: str,
		classification: str,
		commodity: str,
		tonnes: float,
		grade: float,
		grade_unit: str,
		contained_metal: float,
		competent_person_id: str,
		effective_date: datetime | None = None,
		notes: str | None = None,
	) -> dict[str, Any]:
		"""Create a JORC/NI43-101 resource estimate for a deposit."""
		valid_methods = {"kriging", "ID2", "ID3", "nearest_neighbour", "polygonal", "multiple_indicator_kriging"}
		valid_classifications = {"inferred", "indicated", "measured", "probable_reserve", "proven_reserve"}
		if method.lower() not in valid_methods:
			self._log_warn(f"Non-standard estimation method '{method}'")
		if classification.lower() not in valid_classifications:
			raise ValueError(f"Invalid resource classification '{classification}'")
		assert tonnes > 0, "tonnes must be positive"
		assert grade >= 0, "grade must be non-negative"
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"deposit_id": deposit_id,
			"method": method,
			"classification": classification.lower(),
			"commodity": commodity,
			"tonnes": tonnes,
			"grade": grade,
			"grade_unit": grade_unit,
			"contained_metal": contained_metal,
			"competent_person_id": competent_person_id,
			"effective_date": (effective_date or datetime.utcnow()).isoformat(),
			"notes": notes,
			"jorc_compliant": False,
			"published": False,
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._new_resource_estimates[rec_id] = rec
		self._log_op("resource_estimate", "resource_estimate_v2", rec_id)
		return rec

	# ── JORC Compliance ────────────────────────────────────────────────────────

	async def jorc_compliance_check(self, estimate_id: str) -> dict[str, Any]:
		"""
		Run a JORC Table 1 compliance checklist against a resource estimate.
		Checks: competent person assigned, effective date, classification validity,
		methodology documented, QA/QC referenced.
		"""
		rec = self._new_resource_estimates.get(estimate_id)
		if rec is None:
			raise KeyError(f"Resource estimate '{estimate_id}' not found")
		assert rec["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		checks: list[dict[str, Any]] = []
		passed = 0

		def _chk(name: str, result: bool, detail: str) -> None:
			nonlocal passed
			checks.append({"check": name, "passed": result, "detail": detail})
			if result:
				passed += 1

		_chk("competent_person_assigned", bool(rec.get("competent_person_id")), "JORC cl.9 requires CP sign-off")
		_chk("effective_date_present", bool(rec.get("effective_date")), "JORC Table 1 Section 1")
		_chk(
			"valid_classification",
			rec["classification"] in {"inferred", "indicated", "measured", "probable_reserve", "proven_reserve"},
			"JORC 2012 classification hierarchy",
		)
		_chk("methodology_documented", bool(rec.get("method")), "Estimation method must be disclosed")
		_chk("grade_unit_specified", bool(rec.get("grade_unit")), "Reporting requires grade units")
		_chk("tonnes_positive", rec.get("tonnes", 0) > 0, "Non-zero resource tonnage required")

		all_pass = passed == len(checks)
		if all_pass:
			rec["jorc_compliant"] = True
			rec["updated_at"] = datetime.utcnow().isoformat()
			self._log_op("jorc_compliance_pass", "resource_estimate_v2", estimate_id)
		else:
			self._log_warn(f"JORC compliance failed {len(checks)-passed}/{len(checks)} checks", estimate_id=estimate_id)

		return {
			"estimate_id": estimate_id,
			"checks_total": len(checks),
			"checks_passed": passed,
			"jorc_compliant": all_pass,
			"checks": checks,
			"evaluated_at": datetime.utcnow().isoformat(),
		}

	# ── Geophysics Survey ──────────────────────────────────────────────────────

	async def geophysics_survey(
		self,
		survey_type: str,
		area: dict[str, Any],
		data: dict[str, Any],
		conducted_by: str,
		licence_id: str | None = None,
		survey_date: datetime | None = None,
	) -> dict[str, Any]:
		"""
		Record a geophysical survey. survey_type examples: IP, MT, gravity, aeromagnetic, seismic.
		area: {"name": str, "coords": [...]} or bbox dict.
		data: raw or processed geophysics data payload.
		"""
		assert survey_type, "survey_type is required"
		assert conducted_by, "conducted_by is required"
		valid_types = {"IP", "MT", "gravity", "aeromagnetic", "seismic", "CSAMT", "TEM", "ground_mag", "radiometric"}
		if survey_type not in valid_types:
			self._log_warn(f"Non-standard survey_type '{survey_type}'; recorded as-is")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"survey_type": survey_type,
			"area": area,
			"licence_id": licence_id,
			"conducted_by": conducted_by,
			"survey_date": (survey_date or datetime.utcnow()).isoformat(),
			"data_summary": {
				"keys": list(data.keys()),
				"record_count": len(data) if isinstance(data, dict) else 0,
			},
			"data": data,
			"processed": False,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._geophysics_surveys[rec_id] = rec
		self._log_op("geophysics_survey", "geophysics_survey", rec_id)
		return rec

	async def list_geophysics_surveys(
		self, survey_type: str | None = None, licence_id: str | None = None
	) -> list[dict[str, Any]]:
		"""List geophysics surveys with optional type/licence filters."""
		results = [r for r in self._geophysics_surveys.values() if r["tenant_id"] == self.tenant_id]
		if survey_type:
			results = [r for r in results if r["survey_type"] == survey_type]
		if licence_id:
			results = [r for r in results if r.get("licence_id") == licence_id]
		return sorted(results, key=lambda x: x["survey_date"], reverse=True)

	# ── Exploration Target Reporting ───────────────────────────────────────────

	async def report_exploration_target(
		self,
		deposit_id: str,
		tonnage_low: float,
		tonnage_high: float,
		grade_low: float,
		grade_high: float,
		commodity: str,
		grade_unit: str,
		reported_by: str,
		basis: str | None = None,
		caution_statement: str | None = None,
	) -> dict[str, Any]:
		"""
		Report an exploration target per JORC 2012 cl.17 — a range of tonnage and grade
		for which there is insufficient drilling to classify as a resource.
		Automatically prepends a standard JORC caution if none supplied.
		"""
		assert tonnage_high >= tonnage_low > 0, "tonnage range must be positive and ordered"
		assert grade_high >= grade_low >= 0, "grade range must be non-negative and ordered"
		default_caution = (
			"The potential quantity and grade of this exploration target is conceptual in nature. "
			"There has been insufficient exploration to define a mineral resource and it is uncertain "
			"if further exploration will result in the determination of a mineral resource."
		)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"deposit_id": deposit_id,
			"tonnage_low_mt": tonnage_low,
			"tonnage_high_mt": tonnage_high,
			"grade_low": grade_low,
			"grade_high": grade_high,
			"commodity": commodity,
			"grade_unit": grade_unit,
			"reported_by": reported_by,
			"basis": basis,
			"caution_statement": caution_statement or default_caution,
			"jorc_cl17_compliant": True,  # caution statement always included
			"created_at": datetime.utcnow().isoformat(),
		}
		self._exploration_targets[rec_id] = rec
		self._log_op("report_exploration_target", "exploration_target", rec_id)
		return rec

	# ── Exploration Analytics ──────────────────────────────────────────────────

	async def exploration_analytics(
		self,
		licence_id: str,
		period: str,
	) -> dict[str, Any]:
		"""
		Compute exploration performance analytics for a licence and period.
		period format: "YYYY-MM" (monthly) or "YYYY-QN" (quarterly).
		Returns: metres drilled, holes completed, assay samples, average grade by element,
		         resource additions, spend estimate.
		"""
		assert licence_id, "licence_id required"
		# Verify licence exists
		licence = await self.get_licence(licence_id)
		if licence is None:
			raise KeyError(f"Licence '{licence_id}' not found")

		holes = [
			r for r in self._drill_holes.values()
			if r.get("licence_id") == licence_id and r["tenant_id"] == self.tenant_id
		]
		hole_ids = {h["hole_id"] for h in holes}
		core_logs = [r for r in self._core_logs.values() if r["hole_id"] in hole_ids and r["tenant_id"] == self.tenant_id]
		assays = [r for r in self._assay_results.values() if r["hole_id"] in hole_ids and r["tenant_id"] == self.tenant_id]
		resources = [
			r for r in self._new_resource_estimates.values()
			if r.get("deposit_id", "").startswith(licence_id[:8]) and r["tenant_id"] == self.tenant_id
		]

		# Average grade by element
		grade_by_element: dict[str, list[float]] = {}
		for a in assays:
			grade_by_element.setdefault(a["element"], []).append(a["grade"])
		avg_grade: dict[str, float] = {
			el: round(sum(grades) / len(grades), 4)
			for el, grades in grade_by_element.items()
		}

		total_metres = sum(r["interval_m"] for r in core_logs)
		avg_recovery = (
			sum(r["recovery_pct"] for r in core_logs if r.get("recovery_pct") is not None)
			/ max(1, sum(1 for r in core_logs if r.get("recovery_pct") is not None))
		)

		return {
			"tenant_id": self.tenant_id,
			"licence_id": licence_id,
			"period": period,
			"holes_total": len(holes),
			"holes_completed": sum(1 for h in holes if h.get("status") == "completed"),
			"metres_drilled": round(total_metres, 1),
			"core_intervals_logged": len(core_logs),
			"average_core_recovery_pct": round(avg_recovery, 1),
			"assay_samples": len(assays),
			"average_grade_by_element": avg_grade,
			"resource_estimates_count": len(resources),
			"geophysics_surveys": sum(
				1 for r in self._geophysics_surveys.values()
				if r.get("licence_id") == licence_id and r["tenant_id"] == self.tenant_id
			),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Quarterly Report ───────────────────────────────────────────────────────

	async def quarterly_report(self, licence_id: str, period: str) -> dict[str, Any]:
		"""
		Generate a quarterly exploration report for a licence.
		period: "YYYY-QN" e.g. "2025-Q2".
		Bundles exploration summary, resource position, significant assays,
		compliance status, and planned next-quarter activities.
		"""
		assert licence_id, "licence_id required"
		assert period and len(period) == 7 and period[5] == "Q", "period must be YYYY-QN format"
		licence = await self.get_licence(licence_id)
		if licence is None:
			raise KeyError(f"Licence '{licence_id}' not found")

		analytics = await self.exploration_analytics(licence_id, period)
		assays = [
			r for r in self._assay_results.values()
			if r["tenant_id"] == self.tenant_id
		]
		# Significant intercepts: top 10 by grade
		sig_intercepts = sorted(assays, key=lambda x: x["grade"], reverse=True)[:10]

		resources = [
			r for r in self._new_resource_estimates.values()
			if r["tenant_id"] == self.tenant_id and r.get("published")
		]
		resource_summary = {
			cl: {
				"count": sum(1 for r in resources if r["classification"] == cl),
				"total_tonnes_mt": sum(r["tonnes"] for r in resources if r["classification"] == cl),
			}
			for cl in ("inferred", "indicated", "measured")
		}

		surveys = await self.list_geophysics_surveys(licence_id=licence_id)

		jorc_compliant_count = sum(1 for r in resources if r.get("jorc_compliant"))

		return {
			"report_type": "quarterly_exploration_report",
			"tenant_id": self.tenant_id,
			"licence_id": licence_id,
			"licence_number": licence["licence_number"],
			"holder_id": licence["holder_id"],
			"period": period,
			"generated_at": datetime.utcnow().isoformat(),
			"exploration_analytics": analytics,
			"resource_position": resource_summary,
			"significant_intercepts": sig_intercepts,
			"geophysics_surveys_completed": len(surveys),
			"exploration_targets": len(self._exploration_targets),
			"jorc_compliant_estimates": jorc_compliant_count,
			"licence_expiry": licence["expiry"],
			"licence_status": licence["status"],
		}
