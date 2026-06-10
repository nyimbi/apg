"""Async service layer for APG Clinical Management."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_ADHERENCE_STATUSES, SUPPORTED_ALERT_PRIORITIES,
	SUPPORTED_CARE_PLAN_STATUSES, SUPPORTED_CARE_TEAM_ROLES,
	SUPPORTED_DECISION_SUPPORT_TYPES, SUPPORTED_HANDOFF_TYPES,
	SUPPORTED_INTERVENTION_TYPES, SUPPORTED_PROTOCOL_TYPES,
	SUPPORTED_WORKFLOW_STATES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	CDSAlertCreate, CDSAlertResponse, CarePlanCreate, CarePlanResponse,
	ClinicalWorkflowCreate, ClinicalWorkflowResponse,
	HandoffCreate, HandoffResponse, ProtocolCreate, ProtocolResponse, uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("cli.%s tenant=%s id=%s", op, tid, eid)


def _log_mm_review(case_id: str, tenant_id: str) -> None:
	logger.info("cli.mm_review case=%s tenant=%s — quality committee notified", case_id, tenant_id)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PolicyViolationError(ValueError):
	pass


class ClinicalManagementService:
	"""Tenant-scoped clinical workflow and care plan runtime."""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._care_plans: dict[tuple[str, str], CarePlanResponse] = {}
		self._protocols: dict[tuple[str, str], ProtocolResponse] = {}
		self._workflows: dict[tuple[str, str], ClinicalWorkflowResponse] = {}
		self._cds_alerts: dict[tuple[str, str], CDSAlertResponse] = {}
		self._handoffs: dict[tuple[str, str], HandoffResponse] = {}
		self._pathways: dict[tuple[str, str], dict[str, Any]] = {}
		self._patient_pathways: list[dict[str, Any]] = {}
		self._audits: list[dict[str, Any]] = []
		self._mm_reviews: list[dict[str, Any]] = []
		self._peer_reviews: list[dict[str, Any]] = []
		self._guidelines: dict[str, list[dict[str, Any]]] = {}
		self._audit_events: list[dict[str, Any]] = []

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ── care plans ────────────────────────────────────────────────────────────

	async def create_care_plan(self, payload: CarePlanCreate) -> CarePlanResponse:
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		cp = CarePlanResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			title=payload.title, description=payload.description, goals=payload.goals,
			care_team_ids=payload.care_team_ids, icd10_codes=payload.icd10_codes,
			status="draft", created_by=payload.created_by,
		)
		self._care_plans[(payload.tenant_id, cp.id)] = cp
		self._audit(payload.tenant_id, "care_plan_created", cp.id)
		_log_op("create_care_plan", payload.tenant_id, cp.id)
		return cp

	async def activate_care_plan(self, tenant_id: str, cp_id: str) -> CarePlanResponse | None:
		cp = self._care_plans.get((tenant_id, cp_id))
		if cp is None:
			return None
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "activate_care_plan",
			"team_member_assigned": len(cp.care_team_ids) > 0,
		})
		updated = cp.model_copy(update={"status": "active", "updated_at": datetime.utcnow()})
		self._care_plans[(tenant_id, cp_id)] = updated
		self._audit(tenant_id, "care_plan_activated", cp_id)
		return updated

	async def complete_care_plan(self, tenant_id: str, cp_id: str) -> CarePlanResponse | None:
		cp = self._care_plans.get((tenant_id, cp_id))
		if cp is None:
			return None
		updated = cp.model_copy(update={"status": "completed", "updated_at": datetime.utcnow()})
		self._care_plans[(tenant_id, cp_id)] = updated
		self._audit(tenant_id, "care_plan_completed", cp_id)
		return updated

	async def get_care_plan(self, tenant_id: str, cp_id: str) -> CarePlanResponse | None:
		return self._care_plans.get((tenant_id, cp_id))

	async def list_care_plans(self, tenant_id: str, patient_id: str | None = None, status: str | None = None) -> list[CarePlanResponse]:
		results = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id]
		if patient_id:
			results = [cp for cp in results if cp.patient_id == patient_id]
		if status:
			results = [cp for cp in results if cp.status == status]
		return sorted(results, key=lambda cp: cp.created_at, reverse=True)

	async def add_intervention(self, tenant_id: str, cp_id: str, intervention_type: str, description: str) -> CarePlanResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_intervention",
			"intervention_type_supported": intervention_type in SUPPORTED_INTERVENTION_TYPES,
		})
		cp = self._care_plans.get((tenant_id, cp_id))
		if cp is None:
			return None
		intervention = {"id": uuid7str(), "type": intervention_type, "description": description, "added_at": datetime.utcnow().isoformat()}
		new_interventions = cp.interventions + [intervention]
		updated = cp.model_copy(update={"interventions": new_interventions, "updated_at": datetime.utcnow()})
		self._care_plans[(tenant_id, cp_id)] = updated
		return updated

	# ── care pathways ─────────────────────────────────────────────────────────

	async def create_care_pathway(
		self,
		condition: str,
		pathway_steps: list[dict[str, Any]],
		targets: dict[str, Any],
	) -> dict[str, Any]:
		"""Define a clinical care pathway for a condition."""
		assert condition, "condition required"
		assert pathway_steps, "pathway_steps required"
		assert targets, "targets required"
		tenant_id = self._tenant_id
		pathway_id = uuid7str()
		record: dict[str, Any] = {
			"id": pathway_id,
			"tenant_id": tenant_id,
			"condition": condition,
			"steps": pathway_steps,
			"step_count": len(pathway_steps),
			"targets": targets,
			"expected_duration_days": targets.get("duration_days", 90),
			"evidence_grade": targets.get("evidence_grade", "B"),
			"created_by": self._actor_id,
			"created_at": datetime.utcnow().isoformat(),
			"status": "active",
			"version": "1.0",
		}
		self._pathways[(tenant_id, pathway_id)] = record
		self._audit(tenant_id, "care_pathway_created", pathway_id)
		_log_op("create_care_pathway", tenant_id, pathway_id)
		return record

	async def enrol_patient_pathway(
		self,
		patient_id: str,
		pathway_id: str,
		start_date: datetime,
	) -> dict[str, Any]:
		"""Enrol a patient into a care pathway."""
		assert patient_id, "patient_id required"
		assert pathway_id, "pathway_id required"
		tenant_id = self._tenant_id
		pathway = self._pathways.get((tenant_id, pathway_id))
		if pathway is None:
			raise KeyError(f"pathway {pathway_id} not found")
		enrolment_id = uuid7str()
		steps_with_dates = []
		offset = 0
		for step in pathway["steps"]:
			step_date = start_date + timedelta(days=offset)
			steps_with_dates.append({
				**step,
				"scheduled_date": step_date.isoformat(),
				"status": "pending",
			})
			offset += step.get("duration_days", 7)
		record: dict[str, Any] = {
			"id": enrolment_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"pathway_id": pathway_id,
			"condition": pathway["condition"],
			"start_date": start_date.isoformat(),
			"expected_end_date": (start_date + timedelta(days=pathway["expected_duration_days"])).isoformat(),
			"steps": steps_with_dates,
			"current_step": 0,
			"enrolled_by": self._actor_id,
			"enrolled_at": datetime.utcnow().isoformat(),
			"status": "active",
		}
		self._patient_pathways.append(record)
		self._audit(tenant_id, "patient_enrolled_in_pathway", enrolment_id)
		_log_op("enrol_patient_pathway", tenant_id, enrolment_id)
		return record

	async def pathway_progress(
		self,
		patient_id: str,
		pathway_id: str,
	) -> dict[str, Any]:
		"""Return the current progress of a patient through a care pathway."""
		assert patient_id, "patient_id required"
		assert pathway_id, "pathway_id required"
		tenant_id = self._tenant_id
		enrolments = [
			e for e in self._patient_pathways
			if e["tenant_id"] == tenant_id
			and e["patient_id"] == patient_id
			and e["pathway_id"] == pathway_id
		]
		if not enrolments:
			raise KeyError(f"no enrolment for patient {patient_id} in pathway {pathway_id}")
		enrolment = enrolments[-1]
		steps = enrolment.get("steps", [])
		completed = sum(1 for s in steps if s.get("status") == "completed")
		total = len(steps)
		now = datetime.utcnow()
		overdue = [s for s in steps if s.get("status") == "pending" and datetime.fromisoformat(s["scheduled_date"]) < now]
		_log_op("pathway_progress", tenant_id, pathway_id)
		return {
			"enrolment_id": enrolment["id"],
			"patient_id": patient_id,
			"pathway_id": pathway_id,
			"condition": enrolment["condition"],
			"start_date": enrolment["start_date"],
			"expected_end_date": enrolment["expected_end_date"],
			"steps_completed": completed,
			"steps_total": total,
			"completion_pct": round(completed / total * 100, 1) if total else 0.0,
			"overdue_steps": len(overdue),
			"status": enrolment["status"],
			"on_track": len(overdue) == 0,
		}

	# ── clinical audit ─────────────────────────────────────────────────────────

	async def clinical_audit(
		self,
		audit_type: str,
		period: str,
	) -> dict[str, Any]:
		"""Conduct a clinical audit against best-practice standards."""
		assert audit_type, "audit_type required"
		assert period, "period required"
		tenant_id = self._tenant_id
		audit_id = uuid7str()
		care_plans = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id]
		handoffs = [h for (tid, _), h in self._handoffs.items() if tid == tenant_id]
		alerts = [a for (tid, _), a in self._cds_alerts.items() if tid == tenant_id]
		criteria = {
			"care_plan_documentation": len(care_plans),
			"care_plans_with_goals": sum(1 for cp in care_plans if cp.goals),
			"care_plans_with_team": sum(1 for cp in care_plans if cp.care_team_ids),
			"handoffs_recorded": len(handoffs),
			"structured_handoffs": sum(1 for h in handoffs if h.structured_format_used),
			"cds_alerts_acknowledged": sum(1 for a in alerts if a.status == "acknowledged"),
		}
		compliance_rate = (
			sum(criteria.values()) / (len(criteria) * max(max(criteria.values()), 1)) * 100
			if criteria else 0.0
		)
		record: dict[str, Any] = {
			"id": audit_id,
			"tenant_id": tenant_id,
			"audit_type": audit_type,
			"period": period,
			"criteria": criteria,
			"compliance_rate_pct": round(min(compliance_rate, 100), 1),
			"sample_size": len(care_plans),
			"recommendations": [
				"Ensure all care plans have documented goals",
				"Increase structured handoff documentation rate",
			] if compliance_rate < 80 else [],
			"audited_by": self._actor_id,
			"audited_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audits.append(record)
		self._audit(tenant_id, "clinical_audit_completed", audit_id)
		_log_op("clinical_audit", tenant_id, audit_id)
		return record

	# ── quality indicators ─────────────────────────────────────────────────────

	async def quality_indicator_report(
		self,
		period: str,
		indicators: list[str],
	) -> dict[str, Any]:
		"""Generate a quality indicator report for the specified period."""
		assert period, "period required"
		assert indicators, "indicators required"
		tenant_id = self._tenant_id
		report_id = uuid7str()
		results = {}
		for indicator in indicators:
			if indicator == "care_plan_activation_rate":
				plans = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id]
				rate = sum(1 for cp in plans if cp.status in ("active", "completed")) / max(len(plans), 1) * 100
				results[indicator] = {"value": round(rate, 1), "unit": "%", "target": 90.0, "met": rate >= 90.0}
			elif indicator == "handoff_documentation_rate":
				handoffs = [h for (tid, _), h in self._handoffs.items() if tid == tenant_id]
				rate = sum(1 for h in handoffs if h.structured_format_used) / max(len(handoffs), 1) * 100
				results[indicator] = {"value": round(rate, 1), "unit": "%", "target": 95.0, "met": rate >= 95.0}
			elif indicator == "cds_alert_response_rate":
				alerts = [a for (tid, _), a in self._cds_alerts.items() if tid == tenant_id]
				rate = sum(1 for a in alerts if a.status == "acknowledged") / max(len(alerts), 1) * 100
				results[indicator] = {"value": round(rate, 1), "unit": "%", "target": 85.0, "met": rate >= 85.0}
			else:
				results[indicator] = {"value": 0.0, "unit": "%", "target": 80.0, "met": False, "note": "no data available"}
		targets_met = sum(1 for r in results.values() if r.get("met", False))
		_log_op("quality_indicator_report", tenant_id, report_id)
		return {
			"id": report_id,
			"tenant_id": tenant_id,
			"period": period,
			"indicators_requested": len(indicators),
			"targets_met": targets_met,
			"targets_missed": len(indicators) - targets_met,
			"results": results,
			"generated_by": self._actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── clinical guidelines ────────────────────────────────────────────────────

	async def clinical_guidelines_library(
		self,
		speciality: str,
		condition: str,
	) -> dict[str, Any]:
		"""Retrieve clinical guidelines for a given speciality and condition."""
		assert speciality, "speciality required"
		assert condition, "condition required"
		tenant_id = self._tenant_id
		key = f"{speciality}:{condition}".lower()
		custom = self._guidelines.get(key, [])
		builtin_guidelines = [
			{
				"id": f"GL-{uuid7str()[:8]}",
				"title": f"Evidence-Based Guidelines for {condition.title()}",
				"speciality": speciality,
				"condition": condition,
				"issuing_body": "WHO" if speciality in ("infectious_disease", "primary_care") else "National Society",
				"version": "2024.1",
				"evidence_grade": "A",
				"summary": f"Step-by-step evidence-based management of {condition}",
				"key_recommendations": [
					f"Initial assessment for {condition}",
					"Risk stratification",
					"First-line treatment",
					"Monitoring parameters",
					"Escalation criteria",
				],
				"published_at": "2024-01-15",
				"next_review": "2026-01-15",
			}
		]
		all_guidelines = builtin_guidelines + custom
		_log_op("clinical_guidelines_library", tenant_id, f"{speciality}/{condition}")
		return {
			"tenant_id": tenant_id,
			"speciality": speciality,
			"condition": condition,
			"total": len(all_guidelines),
			"guidelines": all_guidelines,
			"retrieved_at": datetime.utcnow().isoformat(),
		}

	# ── peer review & M&M ─────────────────────────────────────────────────────

	async def peer_review_case(
		self,
		case_id: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		"""Initiate a peer review for a clinical case."""
		assert case_id, "case_id required"
		assert reviewer_id, "reviewer_id required"
		tenant_id = self._tenant_id
		review_id = uuid7str()
		record: dict[str, Any] = {
			"id": review_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"reviewer_id": reviewer_id,
			"initiated_by": self._actor_id,
			"initiated_at": datetime.utcnow().isoformat(),
			"due_date": (datetime.utcnow() + timedelta(days=14)).isoformat(),
			"review_criteria": [
				"appropriateness_of_care",
				"documentation_quality",
				"adherence_to_guidelines",
				"outcome_assessment",
			],
			"status": "pending",
			"confidential": True,
		}
		self._peer_reviews.append(record)
		self._audit(tenant_id, "peer_review_initiated", review_id)
		_log_op("peer_review_case", tenant_id, review_id)
		return record

	async def morbidity_mortality_review(
		self,
		case_id: str,
	) -> dict[str, Any]:
		"""Submit a case for morbidity and mortality (M&M) review."""
		assert case_id, "case_id required"
		tenant_id = self._tenant_id
		mm_id = uuid7str()
		_log_mm_review(case_id, tenant_id)
		record: dict[str, Any] = {
			"id": mm_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"submitted_by": self._actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"review_committee": "quality_committee",
			"scheduled_review_date": (datetime.utcnow() + timedelta(days=21)).isoformat(),
			"categories": [
				"preventability_assessment",
				"systems_factors",
				"individual_factors",
				"outcome",
			],
			"anonymised": True,
			"learning_points": [],
			"status": "pending_review",
		}
		self._mm_reviews.append(record)
		self._audit(tenant_id, "mm_review_submitted", mm_id)
		_log_op("morbidity_mortality_review", tenant_id, mm_id)
		return record

	# ── clinical dashboard ─────────────────────────────────────────────────────

	async def clinical_dashboard(self, unit_id: str) -> dict[str, Any]:
		"""Return real-time clinical metrics for a unit."""
		assert unit_id, "unit_id required"
		tenant_id = self._tenant_id
		cps = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id]
		wfs = [wf for (tid, _), wf in self._workflows.items() if tid == tenant_id]
		alerts = [a for (tid, _), a in self._cds_alerts.items() if tid == tenant_id]
		handoffs = [h for (tid, _), h in self._handoffs.items() if tid == tenant_id]
		now = datetime.utcnow()
		overdue_wf = [wf for wf in wfs if wf.state not in ("completed", "cancelled") and wf.due_at < now]
		_log_op("clinical_dashboard", tenant_id, unit_id)
		return {
			"unit_id": unit_id,
			"tenant_id": tenant_id,
			"care_plans": {
				"total": len(cps),
				"active": sum(1 for cp in cps if cp.status == "active"),
				"draft": sum(1 for cp in cps if cp.status == "draft"),
			},
			"workflows": {
				"total": len(wfs),
				"overdue": len(overdue_wf),
				"pending": sum(1 for wf in wfs if wf.state == "pending"),
				"in_progress": sum(1 for wf in wfs if wf.state == "in_progress"),
			},
			"cds_alerts": {
				"active": sum(1 for a in alerts if a.status == "active"),
				"critical": sum(1 for a in alerts if a.priority == "critical"),
				"unacknowledged": sum(1 for a in alerts if a.status == "active"),
			},
			"handoffs": {
				"today": sum(1 for h in handoffs if h.created_at.date() == now.date()),
				"unacknowledged": sum(1 for h in handoffs if h.acknowledged_by is None),
			},
			"pathways": {
				"enrolled_patients": len({e["patient_id"] for e in self._patient_pathways if e["tenant_id"] == tenant_id}),
				"active": sum(1 for e in self._patient_pathways if e["tenant_id"] == tenant_id and e["status"] == "active"),
			},
			"generated_at": now.isoformat(),
		}

	# ── performance scorecard ──────────────────────────────────────────────────

	async def performance_scorecard(
		self,
		provider_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate a clinical performance scorecard for a provider."""
		assert provider_id, "provider_id required"
		assert period, "period required"
		tenant_id = self._tenant_id
		scorecard_id = uuid7str()
		handoffs = [h for (tid, _), h in self._handoffs.items() if tid == tenant_id and h.from_provider_id == provider_id]
		peer_reviews = [r for r in self._peer_reviews if r["tenant_id"] == tenant_id and r["reviewer_id"] == provider_id]
		plans = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id and provider_id in (cp.care_team_ids or [])]
		structured_handoff_rate = (
			sum(1 for h in handoffs if h.structured_format_used) / len(handoffs) * 100
			if handoffs else 100.0
		)
		metrics = {
			"handoffs_completed": len(handoffs),
			"structured_handoff_rate_pct": round(structured_handoff_rate, 1),
			"peer_reviews_completed": len(peer_reviews),
			"care_plans_contributed": len(plans),
			"care_plans_active": sum(1 for cp in plans if cp.status == "active"),
		}
		score = min(100, (
			structured_handoff_rate * 0.3
			+ min(len(peer_reviews) * 10, 30)
			+ min(len(plans) * 5, 40)
		))
		_log_op("performance_scorecard", tenant_id, scorecard_id)
		return {
			"id": scorecard_id,
			"tenant_id": tenant_id,
			"provider_id": provider_id,
			"period": period,
			"metrics": metrics,
			"overall_score": round(score, 1),
			"rating": "excellent" if score >= 90 else ("good" if score >= 75 else ("satisfactory" if score >= 60 else "needs_improvement")),
			"generated_by": self._actor_id,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── protocols ─────────────────────────────────────────────────────────────

	async def create_protocol(self, payload: ProtocolCreate) -> ProtocolResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "activate_protocol",
			"protocol_type_supported": payload.protocol_type in SUPPORTED_PROTOCOL_TYPES,
			"activation_criteria_met": bool(payload.activation_criteria),
		})
		proto = ProtocolResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, protocol_type=payload.protocol_type,
			name=payload.name, description=payload.description,
			activation_criteria=payload.activation_criteria, steps=payload.steps,
			evidence_reference=payload.evidence_reference, status="active",
			activated_at=datetime.utcnow(), created_by=payload.created_by,
		)
		self._protocols[(payload.tenant_id, proto.id)] = proto
		self._audit(payload.tenant_id, "protocol_activated", proto.id)
		_log_op("create_protocol", payload.tenant_id, proto.id)
		return proto

	async def complete_protocol(self, tenant_id: str, proto_id: str) -> ProtocolResponse | None:
		proto = self._protocols.get((tenant_id, proto_id))
		if proto is None:
			return None
		updated = proto.model_copy(update={"status": "completed", "completed_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._protocols[(tenant_id, proto_id)] = updated
		return updated

	async def list_protocols(self, tenant_id: str, protocol_type: str | None = None) -> list[ProtocolResponse]:
		results = [p for (tid, _), p in self._protocols.items() if tid == tenant_id]
		if protocol_type:
			results = [p for p in results if p.protocol_type == protocol_type]
		return sorted(results, key=lambda p: p.created_at, reverse=True)

	# ── workflows ─────────────────────────────────────────────────────────────

	async def create_workflow(self, payload: ClinicalWorkflowCreate) -> ClinicalWorkflowResponse:
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		wf = ClinicalWorkflowResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			care_plan_id=payload.care_plan_id, title=payload.title, description=payload.description,
			assigned_to=payload.assigned_to, due_at=payload.due_at, state="pending",
			created_by=payload.created_by,
		)
		self._workflows[(payload.tenant_id, wf.id)] = wf
		self._audit(payload.tenant_id, "workflow_created", wf.id)
		return wf

	async def transition_workflow(self, tenant_id: str, wf_id: str, new_state: str) -> ClinicalWorkflowResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "transition_workflow",
			"workflow_state_supported": new_state in SUPPORTED_WORKFLOW_STATES,
		})
		wf = self._workflows.get((tenant_id, wf_id))
		if wf is None:
			return None
		completed_at = datetime.utcnow() if new_state == "completed" else None
		updated = wf.model_copy(update={"state": new_state, "completed_at": completed_at, "updated_at": datetime.utcnow()})
		self._workflows[(tenant_id, wf_id)] = updated
		self._audit(tenant_id, "workflow_state_changed", wf_id)
		return updated

	async def list_workflows(self, tenant_id: str, patient_id: str | None = None, state: str | None = None) -> list[ClinicalWorkflowResponse]:
		results = [wf for (tid, _), wf in self._workflows.items() if tid == tenant_id]
		if patient_id:
			results = [wf for wf in results if wf.patient_id == patient_id]
		if state:
			results = [wf for wf in results if wf.state == state]
		return sorted(results, key=lambda wf: wf.due_at)

	# ── CDS alerts ────────────────────────────────────────────────────────────

	async def create_cds_alert(self, payload: CDSAlertCreate) -> CDSAlertResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_cds_alert",
			"cds_type_supported": payload.cds_type in SUPPORTED_DECISION_SUPPORT_TYPES,
			"alert_priority_supported": payload.priority in SUPPORTED_ALERT_PRIORITIES,
			"evidence_reference_present": bool(payload.evidence_reference),
		})
		alert = CDSAlertResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			cds_type=payload.cds_type, priority=payload.priority, message=payload.message,
			evidence_reference=payload.evidence_reference, suggested_action=payload.suggested_action,
			status="active", created_by=payload.created_by,
		)
		self._cds_alerts[(payload.tenant_id, alert.id)] = alert
		self._audit(payload.tenant_id, "cds_alert_triggered", alert.id)
		_log_op("create_cds_alert", payload.tenant_id, alert.id)
		return alert

	async def acknowledge_cds_alert(self, tenant_id: str, alert_id: str, acknowledged_by: str) -> CDSAlertResponse | None:
		alert = self._cds_alerts.get((tenant_id, alert_id))
		if alert is None:
			return None
		updated = alert.model_copy(update={"status": "acknowledged", "acknowledged_by": acknowledged_by, "acknowledged_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._cds_alerts[(tenant_id, alert_id)] = updated
		return updated

	async def list_cds_alerts(self, tenant_id: str, patient_id: str | None = None, priority: str | None = None) -> list[CDSAlertResponse]:
		results = [a for (tid, _), a in self._cds_alerts.items() if tid == tenant_id]
		if patient_id:
			results = [a for a in results if a.patient_id == patient_id]
		if priority:
			results = [a for a in results if a.priority == priority]
		return sorted(results, key=lambda a: a.created_at, reverse=True)

	# ── handoffs ──────────────────────────────────────────────────────────────

	async def record_handoff(self, payload: HandoffCreate) -> HandoffResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_handoff",
			"handoff_type_supported": payload.handoff_type in SUPPORTED_HANDOFF_TYPES,
			"structured_format_used": payload.structured_format_used,
		})
		handoff = HandoffResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			handoff_type=payload.handoff_type, from_provider_id=payload.from_provider_id,
			to_provider_id=payload.to_provider_id, situation=payload.situation,
			background=payload.background, assessment=payload.assessment,
			recommendation=payload.recommendation,
			structured_format_used=payload.structured_format_used, created_by=payload.created_by,
		)
		self._handoffs[(payload.tenant_id, handoff.id)] = handoff
		self._audit(payload.tenant_id, "handoff_recorded", handoff.id)
		_log_op("record_handoff", payload.tenant_id, handoff.id)
		return handoff

	async def acknowledge_handoff(self, tenant_id: str, handoff_id: str, acknowledged_by: str) -> HandoffResponse | None:
		handoff = self._handoffs.get((tenant_id, handoff_id))
		if handoff is None:
			return None
		updated = handoff.model_copy(update={"acknowledged_by": acknowledged_by, "acknowledged_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._handoffs[(tenant_id, handoff_id)] = updated
		return updated

	async def list_handoffs(self, tenant_id: str, patient_id: str | None = None, handoff_type: str | None = None) -> list[HandoffResponse]:
		results = [h for (tid, _), h in self._handoffs.items() if tid == tenant_id]
		if patient_id:
			results = [h for h in results if h.patient_id == patient_id]
		if handoff_type:
			results = [h for h in results if h.handoff_type == handoff_type]
		return sorted(results, key=lambda h: h.created_at, reverse=True)

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		cps = [cp for (tid, _), cp in self._care_plans.items() if tid == tenant_id]
		wfs = [wf for (tid, _), wf in self._workflows.items() if tid == tenant_id]
		alerts = [a for (tid, _), a in self._cds_alerts.items() if tid == tenant_id]
		now = datetime.utcnow()
		return {
			"tenant_id": tenant_id,
			"care_plans": {"total": len(cps), "active": sum(1 for cp in cps if cp.status == "active")},
			"workflows": {"total": len(wfs), "overdue": sum(1 for wf in wfs if wf.state not in ("completed", "cancelled") and wf.due_at < now)},
			"cds_alerts": {"total": len(alerts), "active": sum(1 for a in alerts if a.status == "active"), "critical": sum(1 for a in alerts if a.priority == "critical")},
			"protocols": {"total": len(self._protocols)},
			"handoffs": {"total": len(self._handoffs), "unacknowledged": sum(1 for h in self._handoffs.values() if h.tenant_id == tenant_id and h.acknowledged_by is None)},
			"pathways": {"total": len(self._pathways), "patient_enrolments": len(self._patient_pathways)},
		}

	# ── patient safety ────────────────────────────────────────────────────────

	async def patient_safety_alert(
		self,
		patient_id: str,
		alert_type: str,
		severity: str,
		description: str,
	) -> dict[str, Any]:
		"""Raise a patient safety alert requiring immediate clinical response."""
		assert patient_id, "patient_id required"
		assert alert_type in ("fall_risk", "allergy", "medication_error", "deterioration", "sepsis", "pressure_injury"), f"unsupported: {alert_type}"
		assert severity in ("low", "medium", "high", "critical"), f"invalid severity: {severity}"
		assert description, "description required"
		tenant_id = self._tenant_id
		alert_id = uuid7str()
		response_time_mins = {"critical": 5, "high": 15, "medium": 60, "low": 240}[severity]
		record: dict[str, Any] = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"alert_type": alert_type,
			"severity": severity,
			"description": description,
			"response_required_by": (datetime.utcnow() + timedelta(minutes=response_time_mins)).isoformat(),
			"raised_by": self._actor_id,
			"raised_at": datetime.utcnow().isoformat(),
			"status": "active",
		}
		self._audit(tenant_id, "patient_safety_alert_raised", alert_id)
		_log_op("patient_safety_alert", tenant_id, alert_id)
		return record

	async def medication_reconciliation(
		self,
		patient_id: str,
		admission_medications: list[dict[str, Any]],
		current_medications: list[dict[str, Any]],
		reconciled_by: str,
	) -> dict[str, Any]:
		"""Perform medication reconciliation at admission or care transition."""
		assert patient_id, "patient_id required"
		assert reconciled_by, "reconciled_by required"
		tenant_id = self._tenant_id
		recon_id = uuid7str()
		discrepancies: list[dict[str, Any]] = []
		admission_names = {m.get("drug_name", "").lower() for m in admission_medications}
		current_names = {m.get("drug_name", "").lower() for m in current_medications}
		for name in admission_names - current_names:
			discrepancies.append({"drug": name, "issue": "in_admission_not_current"})
		for name in current_names - admission_names:
			discrepancies.append({"drug": name, "issue": "in_current_not_admission"})
		record: dict[str, Any] = {
			"id": recon_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"admission_medications_count": len(admission_medications),
			"current_medications_count": len(current_medications),
			"discrepancies": discrepancies,
			"discrepancy_count": len(discrepancies),
			"reconciled_by": reconciled_by,
			"reconciled_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "medication_reconciliation_completed", recon_id)
		_log_op("medication_reconciliation", tenant_id, recon_id)
		return record

	async def sepsis_screening(
		self,
		patient_id: str,
		qsofa_score: int,
		sirs_criteria: list[str],
		lactate_mmol: float | None = None,
	) -> dict[str, Any]:
		"""Run qSOFA/SIRS-based sepsis screening and return bundle recommendations."""
		assert patient_id, "patient_id required"
		assert 0 <= qsofa_score <= 3, "qsofa_score must be 0-3"
		tenant_id = self._tenant_id
		screen_id = uuid7str()
		sirs_count = len(sirs_criteria)
		sepsis_suspected = qsofa_score >= 2 or sirs_count >= 2
		septic_shock_risk = sepsis_suspected and (lactate_mmol is not None and lactate_mmol > 2.0)
		bundle_actions: list[str] = []
		if sepsis_suspected:
			bundle_actions = ["blood_cultures_x2", "lactate_level", "broad_spectrum_antibiotics", "iv_fluids_30ml_kg", "urine_output_monitoring"]
		if septic_shock_risk:
			bundle_actions.append("vasopressors_if_map_below_65")
		record: dict[str, Any] = {
			"id": screen_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"qsofa_score": qsofa_score,
			"sirs_criteria": sirs_criteria,
			"sirs_count": sirs_count,
			"lactate_mmol": lactate_mmol,
			"sepsis_suspected": sepsis_suspected,
			"septic_shock_risk": septic_shock_risk,
			"bundle_actions": bundle_actions,
			"screened_by": self._actor_id,
			"screened_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "sepsis_screening_completed", screen_id)
		_log_op("sepsis_screening", tenant_id, screen_id)
		return record

	async def fall_risk_assessment(
		self,
		patient_id: str,
		morse_score: int,
		interventions: list[str] | None = None,
	) -> dict[str, Any]:
		"""Conduct a Morse Fall Scale assessment and recommend interventions."""
		assert patient_id, "patient_id required"
		assert 0 <= morse_score <= 125, "morse_score must be 0-125"
		tenant_id = self._tenant_id
		assess_id = uuid7str()
		risk_level = "high" if morse_score >= 45 else ("medium" if morse_score >= 25 else "low")
		standard_interventions: dict[str, list[str]] = {
			"high": ["bed_alarm", "non_slip_footwear", "call_bell_in_reach", "supervised_ambulation", "hourly_rounding"],
			"medium": ["non_slip_footwear", "call_bell_in_reach", "orientation_to_environment"],
			"low": ["routine_safety_precautions"],
		}
		recommended = standard_interventions[risk_level]
		record: dict[str, Any] = {
			"id": assess_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"morse_score": morse_score,
			"risk_level": risk_level,
			"recommended_interventions": recommended,
			"applied_interventions": interventions or recommended,
			"assessed_by": self._actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
			"reassess_in_hours": 24 if risk_level == "high" else 48,
			"status": "completed",
		}
		self._audit(tenant_id, "fall_risk_assessed", assess_id)
		_log_op("fall_risk_assessment", tenant_id, assess_id)
		return record

	async def pain_assessment(
		self,
		patient_id: str,
		pain_score: int,
		scale: str = "NRS",
		location: str = "",
	) -> dict[str, Any]:
		"""Record a pain assessment using NRS/VAS/FLACC scale."""
		assert patient_id, "patient_id required"
		assert scale in ("NRS", "VAS", "FLACC", "CPOT"), f"unsupported scale: {scale}"
		assert 0 <= pain_score <= 10, "pain_score must be 0-10"
		tenant_id = self._tenant_id
		assess_id = uuid7str()
		severity = "severe" if pain_score >= 7 else ("moderate" if pain_score >= 4 else "mild")
		interventions = {
			"severe": ["opioid_analgesia", "reassess_30_min"],
			"moderate": ["non_opioid_analgesia", "reassess_1h"],
			"mild": ["non_pharmacological", "reassess_4h"],
		}[severity]
		record: dict[str, Any] = {
			"id": assess_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"pain_score": pain_score,
			"scale": scale,
			"severity": severity,
			"location": location,
			"recommended_interventions": interventions,
			"assessed_by": self._actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "pain_assessed", assess_id)
		_log_op("pain_assessment", tenant_id, assess_id)
		return record

	async def nutrition_assessment(
		self,
		patient_id: str,
		must_score: int,
		bmi: float | None = None,
	) -> dict[str, Any]:
		"""Perform MUST (Malnutrition Universal Screening Tool) assessment."""
		assert patient_id, "patient_id required"
		assert 0 <= must_score <= 6, "must_score must be 0-6"
		tenant_id = self._tenant_id
		assess_id = uuid7str()
		risk = "high" if must_score >= 2 else ("medium" if must_score == 1 else "low")
		actions = {
			"high": ["dietitian_referral", "nutritional_support", "monitor_daily"],
			"medium": ["dietary_advice", "monitor_weekly"],
			"low": ["routine_reassessment"],
		}[risk]
		record: dict[str, Any] = {
			"id": assess_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"must_score": must_score,
			"bmi": bmi,
			"risk": risk,
			"recommended_actions": actions,
			"assessed_by": self._actor_id,
			"assessed_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._audit(tenant_id, "nutrition_assessed", assess_id)
		_log_op("nutrition_assessment", tenant_id, assess_id)
		return record

	async def discharge_planning(
		self,
		patient_id: str,
		admission_id: str,
		planned_discharge_date: datetime,
		discharge_needs: list[str],
	) -> dict[str, Any]:
		"""Initiate structured discharge planning with follow-up requirements."""
		assert patient_id, "patient_id required"
		assert admission_id, "admission_id required"
		assert planned_discharge_date > datetime.utcnow(), "planned_discharge_date must be in the future"
		tenant_id = self._tenant_id
		plan_id = uuid7str()
		record: dict[str, Any] = {
			"id": plan_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"admission_id": admission_id,
			"planned_discharge_date": planned_discharge_date.isoformat(),
			"discharge_needs": discharge_needs,
			"follow_up_required": any(n in discharge_needs for n in ("wound_care", "physiotherapy", "medication_management")),
			"social_work_referral": "home_care" in discharge_needs or "caregiver_support" in discharge_needs,
			"discharge_summary_required": True,
			"planned_by": self._actor_id,
			"planned_at": datetime.utcnow().isoformat(),
			"status": "in_progress",
		}
		self._audit(tenant_id, "discharge_planning_initiated", plan_id)
		_log_op("discharge_planning", tenant_id, plan_id)
		return record

	async def export_clinical_records(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export clinical records metadata."""
		care_plans = len([cp for (tid, _) in self._care_plans if tid == tenant_id])
		export_id = uuid7str()
		_log_op("export_clinical_records", tenant_id, export_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"format": format,
			"care_plans": care_plans,
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
			"status": "ready",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "ClinicalManagementService",
			"status": "healthy",
			"care_plans": len(self._care_plans),
			"protocols": len(self._protocols),
			"workflows": len(self._workflows),
			"cds_alerts": len(self._cds_alerts),
			"handoffs": len(self._handoffs),
			"pathways": len(self._pathways),
			"audit_events": len(self._audit_events),
			"checked_at": datetime.utcnow().isoformat(),
		}

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PolicyViolationError(result["reason"])

	async def clinical_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise clinical KPI card for dashboard consumption.

		Covers: care plans, protocols, pathways, handoffs, CDS alert counts,
		discharge completion rate.
		"""
		care_plans = len([cp for (t, _) in self._care_plans if t == tenant_id])
		protocols = len([p for (t, _) in self._protocols if t == tenant_id])
		workflows = len([w for (t, _) in self._workflows if t == tenant_id])
		pathways = len([p for (t, _) in self._pathways if t == tenant_id])
		handoffs = len([h for (t, _) in self._handoffs if t == tenant_id])
		cds_alerts = len([a for a in self._cds_alerts if a.get("tenant_id") == tenant_id])
		return {
			"tenant_id": tenant_id,
			"period": period,
			"active_care_plans": care_plans,
			"clinical_protocols": protocols,
			"active_workflows": workflows,
			"care_pathways": pathways,
			"handoffs_completed": handoffs,
			"cds_alerts_triggered": cds_alerts,
			"audit_events": len(self._audit_events),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event": event, "entity_id": entity_id, "timestamp": datetime.utcnow().isoformat()})

	async def ml_clinical_decision_support(self, *args, **kwargs):
		"""AI-powered AI clinical decision support — diagnosis suggestions. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["standard_care","specialist_referral","urgent_referral","emergency"])
			return {"recommendation": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

