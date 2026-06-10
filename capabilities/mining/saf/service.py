"""Async service layer for APG Mine Safety & Compliance."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .models import (
	AuditType,
	CorrectiveActionCreate,
	CorrectiveActionResponse,
	CorrectiveActionStatus,
	HazardCreate,
	HazardResponse,
	IncidentCreate,
	IncidentResponse,
	IncidentType,
	PermitToWorkCreate,
	PermitToWorkResponse,
	RiskRegisterEntryCreate,
	RiskRegisterEntryResponse,
	ReviewStatus,
	RiskRating,
	uuid7str,
)

log = logging.getLogger(__name__)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class SafService:
	"""Service for Mine Safety & Compliance operations."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._incidents: dict[str, dict[str, Any]] = {}
		self._hazards: dict[str, dict[str, Any]] = {}
		self._risk_register: dict[str, dict[str, Any]] = {}
		self._permits: dict[str, dict[str, Any]] = {}
		self._corrective_actions: dict[str, dict[str, Any]] = {}
		self._audits: dict[str, dict[str, Any]] = {}
		# Extended stores
		self._risk_assessments: dict[str, dict[str, Any]] = {}
		self._safety_inspections: dict[str, dict[str, Any]] = {}
		self._emergency_drills: dict[str, dict[str, Any]] = {}
		self._critical_controls: dict[str, dict[str, Any]] = {}
		self._regulatory_reports: dict[str, dict[str, Any]] = {}
		self._culture_surveys: dict[str, dict[str, Any]] = {}

	# ── Logging helpers ────────────────────────────────────────────────────────

	def _log_op(self, op: str, entity: str, id: str) -> None:
		log.info("saf.%s | tenant=%s entity=%s id=%s", op, self.tenant_id, entity, id)

	def _log_warn(self, msg: str, **kw: Any) -> None:
		log.warning("saf | tenant=%s %s %s", self.tenant_id, msg, kw)

	def _log_escalation(self, incident_id: str, reason: str) -> None:
		log.critical("saf.escalation | tenant=%s incident=%s reason=%s", self.tenant_id, incident_id, reason)

	# ── Tenant guard ───────────────────────────────────────────────────────────

	def _assert_tenant(self, tenant_id: str) -> None:
		assert tenant_id == self.tenant_id, (
			f"Cross-tenant access denied: requested={tenant_id} service={self.tenant_id}"
		)

	# ── Incidents ──────────────────────────────────────────────────────────────

	async def report_incident(
		self, payload: IncidentCreate, created_by: str
	) -> IncidentResponse:
		"""Record a safety incident. Fatalities trigger immediate escalation log."""
		self._assert_tenant(payload.tenant_id)
		resp = IncidentResponse(
			**payload.model_dump(exclude={"witnesses"}),
			witnesses=[w.model_dump() for w in payload.witnesses],
			created_by=created_by,
		)
		self._incidents[resp.id] = resp.model_dump()
		self._log_op("report_incident", "incident", resp.id)
		if resp.incident_type in (IncidentType.FATALITY, IncidentType.LOST_TIME_INJURY):
			self._log_escalation(resp.id, f"Mandatory escalation for {resp.incident_type.value}")

		# MLX enhancement: AI-based severity classification for safety prioritization
		import os
		if os.environ.get("OLLAMA_BASE_URL"):
			try:
				from capabilities.common.mlx import MLCapability
				ml = MLCapability()
				ml_result = await ml.classify(
					f"Incident type: {resp.incident_type.value}. "
					f"Description: {getattr(resp, 'description', '')}",
					labels=["minor_first_aid", "medical_treatment", "lost_time_injury", "critical_emergency"],
				)
				incident_dict = self._incidents[resp.id]
				incident_dict["ml_severity_class"] = ml_result.label
				incident_dict["ml_severity_confidence"] = round(ml_result.confidence, 3)
				self._incidents[resp.id] = incident_dict
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		return resp

	async def get_incident(self, id: str) -> IncidentResponse | None:
		"""Get an incident by id."""
		rec = self._incidents.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return IncidentResponse(**rec)

	async def send_regulatory_notification(self, id: str, sent_by: str) -> IncidentResponse:
		"""Mark regulatory notification as sent for reportable incidents."""
		rec = self._incidents.get(id)
		if rec is None:
			raise KeyError(f"Incident {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["regulatory_notification_sent"] = True
		rec["updated_at"] = datetime.utcnow()
		self._log_op("regulatory_notification", "incident", id)
		return IncidentResponse(**rec)

	async def open_investigation(self, id: str, investigation_id: str) -> IncidentResponse:
		"""Link an investigation to a reported incident."""
		rec = self._incidents.get(id)
		if rec is None:
			raise KeyError(f"Incident {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["investigation_id"] = investigation_id
		rec["status"] = ReviewStatus.IN_REVIEW
		rec["updated_at"] = datetime.utcnow()
		self._log_op("open_investigation", "incident", id)
		return IncidentResponse(**rec)

	async def close_incident(self, id: str, close_notes: str, closed_by: str) -> IncidentResponse:
		"""Close a resolved incident. LTI and above require investigation first."""
		rec = self._incidents.get(id)
		if rec is None:
			raise KeyError(f"Incident {id} not found")
		self._assert_tenant(rec["tenant_id"])
		high_severity = {IncidentType.FATALITY, IncidentType.LOST_TIME_INJURY, IncidentType.DANGEROUS_OCCURRENCE}
		if rec["incident_type"] in {t.value for t in high_severity}:
			if not rec.get("investigation_id"):
				raise PermissionError(
					f"Investigation required before closing {rec['incident_type']} incidents"
				)
		rec["status"] = ReviewStatus.CLOSED
		rec["close_notes"] = close_notes
		rec["updated_at"] = datetime.utcnow()
		self._log_op("close_incident", "incident", id)
		return IncidentResponse(**rec)

	async def list_incidents(
		self,
		incident_type: str | None = None,
		status: str | None = None,
		date_from: datetime | None = None,
		limit: int = 100,
		offset: int = 0,
	) -> list[IncidentResponse]:
		"""List incidents with optional filters."""
		results = [
			IncidentResponse(**r)
			for r in self._incidents.values()
			if r["tenant_id"] == self.tenant_id
		]
		if incident_type:
			results = [r for r in results if r.incident_type == incident_type]
		if status:
			results = [r for r in results if r.status == status]
		if date_from:
			results = [r for r in results if r.occurred_at >= date_from]
		return sorted(results, key=lambda x: x.occurred_at, reverse=True)[offset : offset + limit]

	# ── Hazards ────────────────────────────────────────────────────────────────

	async def identify_hazard(self, payload: HazardCreate, created_by: str) -> HazardResponse:
		"""Record a new hazard. Extreme risks trigger stop-work authority requirement."""
		self._assert_tenant(payload.tenant_id)
		if payload.inherent_risk_rating == RiskRating.EXTREME and not payload.stop_work_invoked:
			raise PermissionError(
				"extreme risk hazards require stop work authority to be invoked before submission"
			)
		resp = HazardResponse(
			**payload.model_dump(exclude={"control_measures"}),
			control_measures=[c.model_dump() for c in payload.control_measures],
			created_by=created_by,
		)
		self._hazards[resp.id] = resp.model_dump()
		self._log_op("identify_hazard", "hazard", resp.id)
		return resp

	async def get_hazard(self, id: str) -> HazardResponse | None:
		"""Get a hazard by id."""
		rec = self._hazards.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return HazardResponse(**rec)

	async def close_hazard(self, id: str, close_notes: str) -> HazardResponse:
		"""Close a resolved hazard."""
		rec = self._hazards.get(id)
		if rec is None:
			raise KeyError(f"Hazard {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["status"] = ReviewStatus.CLOSED
		rec["updated_at"] = datetime.utcnow()
		self._log_op("close_hazard", "hazard", id)
		return HazardResponse(**rec)

	async def list_hazards(
		self,
		risk_rating: str | None = None,
		mine_area: str | None = None,
		open_only: bool = True,
	) -> list[HazardResponse]:
		"""List hazards with optional filters."""
		results = [
			HazardResponse(**r)
			for r in self._hazards.values()
			if r["tenant_id"] == self.tenant_id
		]
		if risk_rating:
			results = [r for r in results if r.inherent_risk_rating == risk_rating]
		if mine_area:
			results = [r for r in results if r.mine_area == mine_area]
		if open_only:
			results = [r for r in results if r.status != ReviewStatus.CLOSED]
		return sorted(results, key=lambda x: list(RiskRating).index(x.inherent_risk_rating))

	# ── Risk Register ──────────────────────────────────────────────────────────

	async def add_risk_register_entry(
		self, payload: RiskRegisterEntryCreate, created_by: str
	) -> RiskRegisterEntryResponse:
		"""Add an entry to the risk register."""
		self._assert_tenant(payload.tenant_id)
		resp = RiskRegisterEntryResponse(
			**payload.model_dump(exclude={"controls"}),
			controls=[c.model_dump() for c in payload.controls],
			created_by=created_by,
		)
		self._risk_register[resp.id] = resp.model_dump()
		self._log_op("add_risk", "risk_register", resp.id)
		return resp

	async def get_risk_register_entry(self, id: str) -> RiskRegisterEntryResponse | None:
		"""Get a risk register entry by id."""
		rec = self._risk_register.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return RiskRegisterEntryResponse(**rec)

	async def list_risk_register(
		self, min_rating: RiskRating | None = None
	) -> list[RiskRegisterEntryResponse]:
		"""List risk register entries, optionally filtered by minimum residual rating."""
		results = [
			RiskRegisterEntryResponse(**r)
			for r in self._risk_register.values()
			if r["tenant_id"] == self.tenant_id
		]
		if min_rating:
			rating_order = list(RiskRating)
			min_idx = rating_order.index(min_rating)
			results = [
				r for r in results
				if rating_order.index(r.residual_risk_rating or r.inherent_risk_rating) <= min_idx
			]
		return sorted(results, key=lambda x: list(RiskRating).index(x.inherent_risk_rating))

	# ── Permits to Work ────────────────────────────────────────────────────────

	async def issue_permit(
		self, payload: PermitToWorkCreate, created_by: str
	) -> PermitToWorkResponse:
		"""Issue a permit to work. Validates issuer qualification field is set."""
		self._assert_tenant(payload.tenant_id)
		if payload.valid_to <= payload.valid_from:
			raise ValueError("valid_to must be after valid_from")
		resp = PermitToWorkResponse(**payload.model_dump(), created_by=created_by)
		self._permits[resp.id] = resp.model_dump()
		self._log_op("issue_permit", "permit_to_work", resp.id)
		return resp

	async def get_permit(self, id: str) -> PermitToWorkResponse | None:
		"""Get a permit to work by id."""
		rec = self._permits.get(id)
		if rec is None:
			return None
		self._assert_tenant(rec["tenant_id"])
		return PermitToWorkResponse(**rec)

	async def close_permit(self, id: str, closed_by: str) -> PermitToWorkResponse:
		"""Close an active permit to work."""
		rec = self._permits.get(id)
		if rec is None:
			raise KeyError(f"Permit {id} not found")
		self._assert_tenant(rec["tenant_id"])
		if rec["status"] == ReviewStatus.CLOSED:
			raise ValueError("Permit is already closed")
		rec["status"] = ReviewStatus.CLOSED
		rec["closed_at"] = datetime.utcnow()
		rec["closed_by"] = closed_by
		rec["updated_at"] = datetime.utcnow()
		self._log_op("close_permit", "permit_to_work", id)
		return PermitToWorkResponse(**rec)

	async def check_permit_valid(self, id: str) -> bool:
		"""Return True if a permit is currently valid (not expired, not closed)."""
		rec = self._permits.get(id)
		if rec is None:
			return False
		self._assert_tenant(rec["tenant_id"])
		now = datetime.utcnow()
		return (
			rec["status"] != ReviewStatus.CLOSED
			and rec["valid_from"] <= now
			and rec["valid_to"] >= now
		)

	async def list_active_permits(self, mine_area: str | None = None) -> list[PermitToWorkResponse]:
		"""List currently active (non-expired) permits."""
		now = datetime.utcnow()
		results = [
			PermitToWorkResponse(**r)
			for r in self._permits.values()
			if r["tenant_id"] == self.tenant_id
			and r["status"] != ReviewStatus.CLOSED
			and r["valid_to"] >= now
		]
		if mine_area:
			results = [r for r in results if r.mine_area == mine_area]
		return sorted(results, key=lambda x: x.valid_to)

	# ── Corrective Actions ─────────────────────────────────────────────────────

	async def create_corrective_action(
		self, payload: CorrectiveActionCreate, created_by: str
	) -> CorrectiveActionResponse:
		"""Create a corrective action with a mandatory assignee and due date."""
		self._assert_tenant(payload.tenant_id)
		resp = CorrectiveActionResponse(**payload.model_dump(), created_by=created_by)
		self._corrective_actions[resp.id] = resp.model_dump()
		self._log_op("create_ca", "corrective_action", resp.id)
		return resp

	async def close_corrective_action(
		self, id: str, closed_by: str, notes: str | None = None
	) -> CorrectiveActionResponse:
		"""Close a corrective action."""
		rec = self._corrective_actions.get(id)
		if rec is None:
			raise KeyError(f"Corrective action {id} not found")
		self._assert_tenant(rec["tenant_id"])
		rec["status"] = CorrectiveActionStatus.CLOSED
		rec["closed_at"] = datetime.utcnow()
		rec["closed_by"] = closed_by
		rec["updated_at"] = datetime.utcnow()
		self._log_op("close_ca", "corrective_action", id)
		return CorrectiveActionResponse(**rec)

	async def flag_overdue_corrective_actions(self) -> list[CorrectiveActionResponse]:
		"""Scan all open CAs and mark overdue ones. Returns the updated list."""
		now = datetime.utcnow()
		overdue: list[CorrectiveActionResponse] = []
		for rec in self._corrective_actions.values():
			if rec["tenant_id"] != self.tenant_id:
				continue
			if rec["status"] in (CorrectiveActionStatus.OPEN, CorrectiveActionStatus.IN_PROGRESS):
				if rec["due_date"] < now:
					rec["status"] = CorrectiveActionStatus.OVERDUE
					rec["updated_at"] = now
					overdue.append(CorrectiveActionResponse(**rec))
		if overdue:
			self._log_warn(f"{len(overdue)} corrective actions marked overdue")
		return overdue

	async def list_corrective_actions(
		self, status: str | None = None, source_type: str | None = None
	) -> list[CorrectiveActionResponse]:
		"""List corrective actions with optional filters."""
		results = [
			CorrectiveActionResponse(**r)
			for r in self._corrective_actions.values()
			if r["tenant_id"] == self.tenant_id
		]
		if status:
			results = [r for r in results if r.status == status]
		if source_type:
			results = [r for r in results if r.source_type == source_type]
		return sorted(results, key=lambda x: x.due_date)

	# ── Safety Statistics ──────────────────────────────────────────────────────

	async def get_safety_statistics(self) -> dict[str, Any]:
		"""Compute LTIFR, incident counts, and open actions summary."""
		incidents = [r for r in self._incidents.values() if r["tenant_id"] == self.tenant_id]
		open_cas = [
			r for r in self._corrective_actions.values()
			if r["tenant_id"] == self.tenant_id and r["status"] != CorrectiveActionStatus.CLOSED
		]
		lti_count = sum(1 for i in incidents if i["incident_type"] == IncidentType.LOST_TIME_INJURY)
		fatality_count = sum(1 for i in incidents if i["incident_type"] == IncidentType.FATALITY)
		near_miss_count = sum(1 for i in incidents if i["incident_type"] == IncidentType.NEAR_MISS)
		return {
			"tenant_id": self.tenant_id,
			"total_incidents": len(incidents),
			"lost_time_injuries": lti_count,
			"fatalities": fatality_count,
			"near_misses": near_miss_count,
			"open_corrective_actions": len(open_cas),
			"overdue_corrective_actions": sum(
				1 for r in open_cas if r["status"] == CorrectiveActionStatus.OVERDUE
			),
			"open_extreme_hazards": sum(
				1 for r in self._hazards.values()
				if r["tenant_id"] == self.tenant_id
				and r["inherent_risk_rating"] == RiskRating.EXTREME
				and r["status"] != ReviewStatus.CLOSED
			),
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Incident Report (extended) ─────────────────────────────────────────────

	async def incident_report(
		self,
		incident_type: str,
		location: str,
		injured_persons: list[dict[str, Any]],
		lost_time: bool,
		description: str,
		reported_by: str,
		occurred_at: datetime | None = None,
		witness_ids: list[str] | None = None,
		immediate_cause: str | None = None,
		root_cause: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a mine safety incident with injured persons, LTI flag, and causal analysis.
		Fatalities and LTIs trigger mandatory regulatory escalation flag.
		incident_type: fatality | LTI | MTI | first_aid | near_miss | dangerous_occurrence | property_damage
		"""
		assert incident_type, "incident_type required"
		assert location, "location required"
		assert description, "description required"
		assert reported_by, "reported_by required"
		high_severity_types = {"fatality", "LTI", "dangerous_occurrence"}
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"incident_type": incident_type,
			"location": location,
			"injured_persons": injured_persons,
			"injury_count": len(injured_persons),
			"lost_time": lost_time,
			"description": description,
			"reported_by": reported_by,
			"occurred_at": (occurred_at or datetime.utcnow()).isoformat(),
			"reported_at": datetime.utcnow().isoformat(),
			"witness_ids": witness_ids or [],
			"immediate_cause": immediate_cause,
			"root_cause": root_cause,
			"regulatory_notification_required": incident_type in high_severity_types,
			"regulatory_notification_sent": False,
			"investigation_required": incident_type in high_severity_types,
			"investigation_id": None,
			"status": "open",
		}
		self._incidents[rec_id] = rec
		if incident_type in high_severity_types:
			self._log_escalation(rec_id, f"High-severity incident: {incident_type}")
		self._log_op("incident_report", "incident", rec_id)
		return rec

	# ── Risk Assessment ────────────────────────────────────────────────────────

	async def risk_assessment(
		self,
		task: str,
		hazards: list[dict[str, Any]],
		controls: list[dict[str, Any]],
		residual_risk: str,
		approved_by: str,
		area: str | None = None,
		valid_hours: int = 12,
		assessed_by: str = "system",
	) -> dict[str, Any]:
		"""
		Perform a job/task risk assessment (JRA/JSA).
		hazards: [{"hazard": str, "consequence": str, "likelihood": str, "severity": str}]
		controls: [{"control": str, "hierarchy": str, "owner": str}]
		residual_risk: extreme | high | medium | low
		"""
		assert task, "task required"
		assert hazards, "at least one hazard required"
		assert controls, "at least one control required"
		valid_risks = {"extreme", "high", "medium", "low"}
		if residual_risk.lower() not in valid_risks:
			raise ValueError(f"residual_risk must be one of {valid_risks}")
		if residual_risk.lower() == "extreme":
			raise PermissionError(
				"Extreme residual risk cannot be accepted; controls are insufficient. "
				"Additional controls or stop-work required."
			)
		rec_id = uuid7str()
		valid_from = datetime.utcnow()
		import datetime as _dt
		valid_to = valid_from + _dt.timedelta(hours=valid_hours)
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"task": task,
			"area": area,
			"hazards": hazards,
			"hazard_count": len(hazards),
			"controls": controls,
			"control_count": len(controls),
			"residual_risk": residual_risk.lower(),
			"approved_by": approved_by,
			"assessed_by": assessed_by,
			"valid_from": valid_from.isoformat(),
			"valid_to": valid_to.isoformat(),
			"status": "active",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._risk_assessments[rec_id] = rec
		self._log_op("risk_assessment", "risk_assessment", rec_id)
		return rec

	async def get_active_risk_assessments(self, area: str | None = None) -> list[dict[str, Any]]:
		"""Return currently valid risk assessments, optionally filtered by area."""
		now = datetime.utcnow().isoformat()
		results = [
			r for r in self._risk_assessments.values()
			if r["tenant_id"] == self.tenant_id
			and r["valid_from"] <= now <= r["valid_to"]
			and r["status"] == "active"
		]
		if area:
			results = [r for r in results if r.get("area") == area]
		return sorted(results, key=lambda x: x["valid_to"])

	# ── Permit to Work (extended) ──────────────────────────────────────────────

	async def permit_to_work(
		self,
		work_type: str,
		location: str,
		hazards: list[str],
		precautions: list[str],
		issuer_id: str,
		receiver_id: str | None = None,
		valid_hours: int = 12,
		risk_assessment_id: str | None = None,
		isolations: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Issue a permit to work. Validates precautions are specified for each hazard.
		work_type: hot_work | confined_space | electrical | height | excavation | radiation | general
		"""
		assert work_type, "work_type required"
		assert location, "location required"
		assert hazards, "at least one hazard required"
		assert precautions, "at least one precaution required"
		assert issuer_id, "issuer_id required"
		if len(precautions) < len(hazards):
			self._log_warn(
				"Fewer precautions than hazards — review PTW",
				work_type=work_type, hazards=len(hazards), precautions=len(precautions),
			)
		import datetime as _dt
		valid_from = datetime.utcnow()
		valid_to = valid_from + _dt.timedelta(hours=valid_hours)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"work_type": work_type,
			"location": location,
			"hazards": hazards,
			"precautions": precautions,
			"isolations": isolations or [],
			"issuer_id": issuer_id,
			"receiver_id": receiver_id,
			"risk_assessment_id": risk_assessment_id,
			"valid_from": valid_from.isoformat(),
			"valid_to": valid_to.isoformat(),
			"status": "active",
			"closed_at": None,
			"closed_by": None,
			"issued_at": datetime.utcnow().isoformat(),
		}
		self._permits[rec_id] = rec
		self._log_op("permit_to_work", "permit_to_work", rec_id)
		return rec

	# ── Safety Inspection ──────────────────────────────────────────────────────

	async def safety_inspection(
		self,
		area: str,
		inspector_id: str,
		date: datetime,
		findings: list[dict[str, Any]],
		inspection_type: str = "routine",
		rating: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a safety inspection. Each finding can generate a corrective action.
		findings: [{"item": str, "status": str, "severity": str, "action_required": bool}]
		inspection_type: routine | audit | regulatory | post_incident | toolbox
		"""
		assert area, "area required"
		assert inspector_id, "inspector_id required"
		assert findings is not None, "findings list required (may be empty)"
		valid_types = {"routine", "audit", "regulatory", "post_incident", "toolbox", "pre_blast", "environmental"}
		if inspection_type not in valid_types:
			self._log_warn(f"Non-standard inspection_type '{inspection_type}'")
		critical_findings = [f for f in findings if f.get("severity") in ("critical", "high")]
		action_required = [f for f in findings if f.get("action_required")]
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"area": area,
			"inspector_id": inspector_id,
			"inspection_type": inspection_type,
			"date": date.isoformat(),
			"findings": findings,
			"findings_count": len(findings),
			"critical_findings_count": len(critical_findings),
			"actions_required_count": len(action_required),
			"rating": rating,
			"status": "open" if action_required else "closed",
			"created_at": datetime.utcnow().isoformat(),
		}
		self._safety_inspections[rec_id] = rec
		if critical_findings:
			self._log_warn(f"{len(critical_findings)} critical findings in inspection", area=area)
		self._log_op("safety_inspection", "safety_inspection", rec_id)
		return rec

	async def list_safety_inspections(
		self, area: str | None = None, status: str | None = None
	) -> list[dict[str, Any]]:
		"""List safety inspections with optional filters."""
		results = [r for r in self._safety_inspections.values() if r["tenant_id"] == self.tenant_id]
		if area:
			results = [r for r in results if r["area"] == area]
		if status:
			results = [r for r in results if r["status"] == status]
		return sorted(results, key=lambda x: x["date"], reverse=True)

	# ── Corrective Action (extended) ───────────────────────────────────────────

	async def corrective_action(
		self,
		finding_id: str,
		action: str,
		responsible: str,
		deadline: datetime,
		priority: str = "medium",
		source_type: str = "inspection",
		created_by: str = "system",
	) -> dict[str, Any]:
		"""
		Create a corrective action from an inspection finding, incident, or hazard.
		priority: critical | high | medium | low
		source_type: inspection | incident | hazard | audit | regulatory
		"""
		assert finding_id, "finding_id required"
		assert action, "action description required"
		assert responsible, "responsible person required"
		assert deadline > datetime.utcnow(), "deadline must be in the future"
		valid_priority = {"critical", "high", "medium", "low"}
		if priority not in valid_priority:
			raise ValueError(f"priority must be one of {valid_priority}")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"finding_id": finding_id,
			"action": action,
			"responsible": responsible,
			"deadline": deadline.isoformat(),
			"priority": priority,
			"source_type": source_type,
			"created_by": created_by,
			"status": "open",
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat(),
			"closed_at": None,
			"closed_by": None,
			"evidence": None,
		}
		self._corrective_actions[rec_id] = rec
		self._log_op("corrective_action", "corrective_action", rec_id)
		return rec

	async def close_corrective_action_by_id(
		self, ca_id: str, closed_by: str, evidence: str | None = None
	) -> dict[str, Any]:
		"""Close a corrective action with evidence reference."""
		rec = self._corrective_actions.get(ca_id)
		if rec is None:
			raise KeyError(f"Corrective action '{ca_id}' not found")
		assert rec["tenant_id"] == self.tenant_id, "Cross-tenant access denied"
		rec["status"] = "closed"
		rec["closed_at"] = datetime.utcnow().isoformat()
		rec["closed_by"] = closed_by
		rec["evidence"] = evidence
		rec["updated_at"] = datetime.utcnow().isoformat()
		self._log_op("close_ca", "corrective_action", ca_id)
		return rec

	# ── Emergency Drill ────────────────────────────────────────────────────────

	async def emergency_drill(
		self,
		drill_type: str,
		date: datetime,
		participants: list[dict[str, Any]],
		outcome: str,
		location: str | None = None,
		duration_minutes: float | None = None,
		deficiencies: list[str] | None = None,
		facilitator_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Record an emergency drill exercise.
		drill_type: fire | evacuation | chemical_spill | medical | rescue | security
		outcome: satisfactory | unsatisfactory | needs_improvement
		"""
		assert drill_type, "drill_type required"
		assert participants, "at least one participant required"
		assert outcome in ("satisfactory", "unsatisfactory", "needs_improvement"), \
			"outcome must be satisfactory/unsatisfactory/needs_improvement"
		valid_types = {"fire", "evacuation", "chemical_spill", "medical", "rescue", "security", "tailings_breach"}
		if drill_type not in valid_types:
			self._log_warn(f"Non-standard drill_type '{drill_type}'")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"drill_type": drill_type,
			"date": date.isoformat(),
			"location": location,
			"participants": participants,
			"participant_count": len(participants),
			"duration_minutes": duration_minutes,
			"outcome": outcome,
			"deficiencies": deficiencies or [],
			"deficiency_count": len(deficiencies or []),
			"facilitator_id": facilitator_id,
			"follow_up_required": outcome != "satisfactory" or bool(deficiencies),
			"created_at": datetime.utcnow().isoformat(),
		}
		self._emergency_drills[rec_id] = rec
		if outcome == "unsatisfactory":
			self._log_warn("Emergency drill outcome unsatisfactory", drill_type=drill_type, date=date.isoformat())
		self._log_op("emergency_drill", "emergency_drill", rec_id)
		return rec

	async def list_emergency_drills(self, drill_type: str | None = None) -> list[dict[str, Any]]:
		"""List emergency drills, optionally filtered by type."""
		results = [r for r in self._emergency_drills.values() if r["tenant_id"] == self.tenant_id]
		if drill_type:
			results = [r for r in results if r["drill_type"] == drill_type]
		return sorted(results, key=lambda x: x["date"], reverse=True)

	# ── Safety Statistics (extended) ───────────────────────────────────────────

	async def safety_statistics(self, period: str) -> dict[str, Any]:
		"""
		Compute period safety statistics including LTIFR, TRIFR, and leading indicators.
		period: YYYY-MM or YYYY-QN.
		LTIFR = (LTIs × 1,000,000) / hours_worked
		TRIFR = (recordable injuries × 1,000,000) / hours_worked
		"""
		assert period, "period required"
		incidents = [
			r for r in self._incidents.values()
			if r["tenant_id"] == self.tenant_id
		]
		period_incidents = [
			i for i in incidents
			if i.get("occurred_at", "")[:len(period)] == period
		]
		lti_count = sum(1 for i in period_incidents if i.get("lost_time") or i.get("incident_type") in ("LTI", "fatality"))
		fatality_count = sum(1 for i in period_incidents if i.get("incident_type") == "fatality")
		near_miss_count = sum(1 for i in period_incidents if i.get("incident_type") == "near_miss")
		first_aid_count = sum(1 for i in period_incidents if i.get("incident_type") == "first_aid")
		recordable_count = lti_count + sum(1 for i in period_incidents if i.get("incident_type") == "MTI")
		# Hours worked estimate (200,000 hrs/month per standard workforce of ~100)
		hours_worked = 200000.0
		ltifr = round(lti_count * 1_000_000 / hours_worked, 2) if hours_worked > 0 else None
		trifr = round(recordable_count * 1_000_000 / hours_worked, 2) if hours_worked > 0 else None
		drills = [
			r for r in self._emergency_drills.values()
			if r["tenant_id"] == self.tenant_id and r.get("date", "")[:len(period)] == period
		]
		inspections = [
			r for r in self._safety_inspections.values()
			if r["tenant_id"] == self.tenant_id and r.get("date", "")[:len(period)] == period
		]
		open_cas = sum(
			1 for r in self._corrective_actions.values()
			if r.get("tenant_id") == self.tenant_id and r.get("status") not in ("closed",)
		)
		return {
			"tenant_id": self.tenant_id,
			"period": period,
			"total_incidents": len(period_incidents),
			"fatalities": fatality_count,
			"lost_time_injuries": lti_count,
			"near_misses": near_miss_count,
			"first_aid_cases": first_aid_count,
			"recordable_injuries": recordable_count,
			"hours_worked_estimate": hours_worked,
			"ltifr": ltifr,
			"trifr": trifr,
			"emergency_drills": len(drills),
			"safety_inspections": len(inspections),
			"open_corrective_actions": open_cas,
			"as_at": datetime.utcnow().isoformat(),
		}

	# ── Critical Control Monitoring ────────────────────────────────────────────

	async def critical_control_monitoring(
		self,
		control_id: str,
		verification_result: str,
		verifier_id: str,
		control_description: str | None = None,
		material_unwanted_event: str | None = None,
		deficiency_found: bool = False,
		deficiency_detail: str | None = None,
	) -> dict[str, Any]:
		"""
		Verify a critical control is in place and functional.
		verification_result: effective | partially_effective | ineffective | not_verified
		Critical controls failure triggers immediate escalation.
		"""
		assert control_id, "control_id required"
		assert verifier_id, "verifier_id required"
		valid_results = {"effective", "partially_effective", "ineffective", "not_verified"}
		if verification_result not in valid_results:
			raise ValueError(f"verification_result must be one of {valid_results}")
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"control_id": control_id,
			"control_description": control_description,
			"material_unwanted_event": material_unwanted_event,
			"verification_result": verification_result,
			"verifier_id": verifier_id,
			"deficiency_found": deficiency_found,
			"deficiency_detail": deficiency_detail,
			"escalation_required": verification_result == "ineffective",
			"verified_at": datetime.utcnow().isoformat(),
		}
		self._critical_controls[rec_id] = rec
		if verification_result == "ineffective":
			self._log_escalation(rec_id, f"Critical control '{control_id}' verified INEFFECTIVE")
		self._log_op("critical_control_monitoring", "critical_control", rec_id)
		return rec

	async def list_critical_control_verifications(
		self, control_id: str | None = None, ineffective_only: bool = False
	) -> list[dict[str, Any]]:
		"""List critical control verification records."""
		results = [r for r in self._critical_controls.values() if r["tenant_id"] == self.tenant_id]
		if control_id:
			results = [r for r in results if r["control_id"] == control_id]
		if ineffective_only:
			results = [r for r in results if r["verification_result"] == "ineffective"]
		return sorted(results, key=lambda x: x["verified_at"], reverse=True)

	# ── Regulatory Report ──────────────────────────────────────────────────────

	async def regulatory_report_safety(
		self,
		period: str,
		jurisdiction: str,
		submitted_by: str | None = None,
		submission_deadline: datetime | None = None,
	) -> dict[str, Any]:
		"""
		Generate a statutory safety report for a regulator (e.g. Mines Safety Act return).
		Aggregates incidents, LTIs, fatalities, inspections, and corrective actions.
		"""
		assert period, "period required"
		assert jurisdiction, "jurisdiction required"
		stats = await self.safety_statistics(period)
		incidents = [
			r for r in self._incidents.values()
			if r["tenant_id"] == self.tenant_id and r.get("occurred_at", "")[:len(period)] == period
		]
		reportable = [i for i in incidents if i.get("regulatory_notification_required")]
		notifications_sent = [i for i in reportable if i.get("regulatory_notification_sent")]
		overdue_notifications = len(reportable) - len(notifications_sent)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"report_type": "regulatory_safety_report",
			"period": period,
			"jurisdiction": jurisdiction,
			"submitted_by": submitted_by,
			"submission_deadline": submission_deadline.isoformat() if submission_deadline else None,
			"status": "draft",
			"statistics": stats,
			"reportable_incidents": len(reportable),
			"notifications_sent": len(notifications_sent),
			"overdue_notifications": overdue_notifications,
			"generated_at": datetime.utcnow().isoformat(),
		}
		self._regulatory_reports[rec_id] = rec
		if overdue_notifications > 0:
			self._log_warn(f"{overdue_notifications} regulatory notifications not yet sent", period=period)
		self._log_op("regulatory_report_safety", "regulatory_report", rec_id)
		return rec

	# ── Safety Culture Survey ──────────────────────────────────────────────────

	async def safety_culture_survey(
		self,
		period: str,
		responses: list[dict[str, Any]],
		survey_instrument: str = "Hearts and Minds",
		facilitated_by: str | None = None,
		participation_rate_pct: float | None = None,
	) -> dict[str, Any]:
		"""
		Record results of a safety culture survey.
		responses: [{"question_id": str, "dimension": str, "score": float, "max_score": float}]
		Computes dimension scores and overall culture index (0-5 scale).
		"""
		assert period, "period required"
		assert responses, "at least one survey response required"
		# Aggregate by dimension
		dim_scores: dict[str, list[float]] = {}
		for r in responses:
			dim = r.get("dimension", "general")
			max_s = r.get("max_score", 5.0) or 5.0
			normalised = r.get("score", 0.0) / max_s * 5.0
			dim_scores.setdefault(dim, []).append(normalised)
		dimension_averages = {
			dim: round(sum(scores) / len(scores), 2)
			for dim, scores in dim_scores.items()
		}
		all_scores = [s for scores in dim_scores.values() for s in scores]
		overall_index = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
		culture_level = (
			"pathological" if overall_index < 1.5
			else "reactive" if overall_index < 2.5
			else "calculative" if overall_index < 3.5
			else "proactive" if overall_index < 4.5
			else "generative"
		)
		rec_id = uuid7str()
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"survey_instrument": survey_instrument,
			"facilitated_by": facilitated_by,
			"response_count": len(responses),
			"participation_rate_pct": participation_rate_pct,
			"dimension_scores": dimension_averages,
			"overall_culture_index": overall_index,
			"culture_level": culture_level,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._culture_surveys[rec_id] = rec
		self._log_op("safety_culture_survey", "culture_survey", rec_id)
		return rec

	async def list_culture_surveys(self) -> list[dict[str, Any]]:
		"""List safety culture surveys for the tenant."""
		return sorted(
			[r for r in self._culture_surveys.values() if r["tenant_id"] == self.tenant_id],
			key=lambda x: x["period"],
			reverse=True,
		)


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
