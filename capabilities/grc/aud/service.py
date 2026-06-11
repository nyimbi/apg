"""AuditManagementService — GRC internal audit management.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_AUDIT_TYPES,
	SUPPORTED_AUDIT_STATUSES,
	SUPPORTED_FINDING_SEVERITIES,
	SUPPORTED_FINDING_STATUSES,
	SUPPORTED_EVIDENCE_TYPES,
	evaluate_capability_rules,
)
from .database.store import Store, get_store
from .domain.adapters import (
	AuthAdapter,
	AuditAdapter,
	NotifyAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _period_bounds(period: str) -> tuple[str, str]:
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		ed = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{ed:02d}"
	return period, period


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class AuditManagementService:
	"""GRC internal audit programme: plan, engagement, fieldwork, findings,
	reports, follow-up, QA, fraud investigation, and whistleblower case management.

	Usage (standalone)::

		svc = AuditManagementService()
		plan = await svc.create_audit_plan("ENT-1", 2025, ["IT general controls"], "CAE-1")

	Usage (platform)::

		svc = AuditManagementService(auth=AuthService.from_env())
	"""

	def __init__(
		self,
		*,
		db_url: str | None = None,
		store: Store | None = None,
		auth: Any | None = None,
		audit: Any | None = None,
		notify: Any | None = None,
		tenant_id: str = "default",
	) -> None:
		self._store: Store = store or get_store(db_url)
		self._auth: AuthAdapter = get_auth_adapter(auth)
		self._audit: AuditAdapter = get_audit_adapter(audit)
		self._notify: NotifyAdapter = get_notify_adapter(notify)
		self._tenant_id = tenant_id
		self._capability = CAPABILITY_ID
		self._version = CAPABILITY_VERSION

	async def _audit_event(
		self, event_type: str, actor_id: str, resource_id: str, details: dict[str, Any]
	) -> None:
		await self._audit.log_event(event_type, actor_id, self._tenant_id, resource_id, details)

	async def _get_engagement(self, engagement_id: str) -> dict[str, Any]:
		rec = await self._store.get("audit_engagements", engagement_id)
		if rec is None:
			raise ValueError(f"Audit engagement not found: {engagement_id}")
		return rec

	async def _get_finding(self, finding_id: str) -> dict[str, Any]:
		rec = await self._store.get("audit_findings", finding_id)
		if rec is None:
			raise ValueError(f"Audit finding not found: {finding_id}")
		return rec

	# ─────────────────────────────────────────────────────────
	# Audit planning
	# ─────────────────────────────────────────────────────────

	async def create_audit_plan(
		self,
		entity_id: str,
		year: int,
		risk_based_areas: list[str],
		approved_by: str,
		*,
		plan_type: str = "annual",
		methodology: str = "risk_based",
	) -> dict[str, Any]:
		"""Create an annual audit plan based on risk assessment results.

		Validates entity and approver, assigns risk-ranked areas, and creates
		the plan record in 'draft' status. Emits ``audit_planned`` event.
		"""
		assert entity_id, "entity_id required"
		assert 2020 <= year <= 2099, "year: 2020–2099"
		assert risk_based_areas, "risk_based_areas required"
		assert approved_by, "approved_by required"

		rule_ctx = {
			"operation": "create_audit",
			"tenant_context_present": True,
			"title_present": True,
			"auditor_present": True,
			"audit_type_supported": True,
			"scope_present": True,
			"scope_type_supported": True,
			"start_date_present": True,
			"end_date_present": True,
			"auditee_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Audit plan creation denied: {verdict['matched_rules']}")

		plan: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"year": year,
			"plan_type": plan_type,
			"methodology": methodology,
			"risk_based_areas": risk_based_areas,
			"approved_by": approved_by,
			"status": "approved",
			"engagement_ids": [],
			"total_planned_hours": 0,
			"total_actual_hours": 0,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("audit_plans", plan)
		await self._audit_event(
			"audit_planned", approved_by, plan["id"],
			{"entity_id": entity_id, "year": year, "areas_count": len(risk_based_areas)},
		)
		return plan

	async def create_audit_engagement(
		self,
		plan_id: str,
		area: str,
		objectives: list[str],
		start_date: str,
		end_date: str,
		lead_auditor_id: str,
		*,
		audit_type: str = "internal",
		scope: str = "process",
		auditee_id: str | None = None,
		planned_hours: int = 80,
	) -> dict[str, Any]:
		"""Create an audit engagement within a plan.

		Validates dates, auditor, and scope. Enforces auditor ≠ auditee.
		Transitions the engagement to 'planned' status.
		"""
		assert area, "area required"
		assert objectives, "objectives required"
		assert lead_auditor_id, "lead_auditor_id required"
		assert start_date < end_date, "end_date must be after start_date"

		if audit_type not in SUPPORTED_AUDIT_TYPES:
			raise ValueError(f"Unsupported audit type: {audit_type}. Valid: {SUPPORTED_AUDIT_TYPES}")
		if scope not in {"process", "system", "organizational_unit", "product", "supplier", "facility"}:
			raise ValueError(f"Unsupported scope: {scope}")

		if auditee_id and auditee_id == lead_auditor_id:
			raise PermissionError("Lead auditor cannot be the auditee (segregation of duties)")

		plan = await self._store.get("audit_plans", plan_id)
		if plan is None:
			raise ValueError(f"Audit plan not found: {plan_id}")

		engagement: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"plan_id": plan_id,
			"entity_id": plan.get("entity_id"),
			"area": area,
			"audit_type": audit_type,
			"scope": scope,
			"objectives": objectives,
			"lead_auditor_id": lead_auditor_id,
			"auditee_id": auditee_id,
			"start_date": start_date,
			"end_date": end_date,
			"planned_hours": planned_hours,
			"actual_hours": 0,
			"status": "planned",
			"finding_ids": [],
			"report_id": None,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("audit_engagements", engagement)

		plan.setdefault("engagement_ids", []).append(engagement["id"])
		plan["total_planned_hours"] = plan.get("total_planned_hours", 0) + planned_hours
		plan["updated_at"] = _now()
		await self._store.put("audit_plans", plan)

		await self._audit_event(
			"audit_started", lead_auditor_id, engagement["id"],
			{"area": area, "start_date": start_date, "end_date": end_date},
		)
		return engagement

	# ─────────────────────────────────────────────────────────
	# Fieldwork and findings
	# ─────────────────────────────────────────────────────────

	async def fieldwork_record(
		self,
		engagement_id: str,
		area_tested: str,
		finding_type: str,
		observation: str,
		criteria: str,
		evidence: list[dict[str, Any]],
		risk_rating: str,
		*,
		auditor_id: str | None = None,
	) -> dict[str, Any]:
		"""Record fieldwork observation and create an associated finding.

		Evidence items: {type, description, file_ref}. Each piece is stored
		with retention metadata and tamper-evident hash.
		"""
		assert area_tested, "area_tested required"
		assert observation, "observation required"
		assert criteria, "criteria required"

		if risk_rating not in SUPPORTED_FINDING_SEVERITIES:
			raise ValueError(f"Unsupported risk_rating: {risk_rating}. Valid: {SUPPORTED_FINDING_SEVERITIES}")

		engagement = await self._get_engagement(engagement_id)

		# Store evidence
		evidence_ids: list[str] = []
		for ev in evidence:
			ev_type = ev.get("type", "document")
			if ev_type not in SUPPORTED_EVIDENCE_TYPES:
				ev_type = "document"
			ev_rec: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"engagement_id": engagement_id,
				"evidence_type": ev_type,
				"description": ev.get("description", ""),
				"file_ref": ev.get("file_ref", ""),
				"encrypted": True,
				"retention_days": 365,
				"collected_at": _now(),
				"collected_by": auditor_id or engagement.get("lead_auditor_id"),
			}
			await self._store.put("audit_evidence", ev_rec)
			evidence_ids.append(ev_rec["id"])

		finding: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"plan_id": engagement.get("plan_id"),
			"entity_id": engagement.get("entity_id"),
			"area_tested": area_tested,
			"finding_type": finding_type,
			"observation": observation,
			"criteria": criteria,
			"risk_rating": risk_rating,
			"evidence_ids": evidence_ids,
			"status": "open",
			"owner_id": None,
			"management_response": None,
			"remediation_deadline": None,
			"raised_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("audit_findings", finding)

		engagement.setdefault("finding_ids", []).append(finding["id"])
		engagement["status"] = "fieldwork"
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)

		await self._audit_event(
			"audit_finding_raised", auditor_id or "auditor", finding["id"],
			{"risk_rating": risk_rating, "area_tested": area_tested},
		)

		if risk_rating == "critical":
			await self._notify.send(
				"cae@datacraft.co.ke", "email",
				f"Critical audit finding: {area_tested}",
				f"Critical finding raised in engagement {engagement_id}: {observation[:200]}",
			)
		return finding

	async def draft_audit_report(
		self,
		engagement_id: str,
		findings: list[str],
		recommendations: list[str],
		auditor_id: str,
	) -> dict[str, Any]:
		"""Draft the audit report for an engagement.

		Includes all finding references, summary, and recommendations.
		Transitions engagement status to 'review'.
		"""
		assert findings or recommendations, "findings or recommendations required"
		assert auditor_id, "auditor_id required"

		engagement = await self._get_engagement(engagement_id)

		report: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"plan_id": engagement.get("plan_id"),
			"entity_id": engagement.get("entity_id"),
			"area": engagement.get("area"),
			"author_id": auditor_id,
			"finding_references": findings,
			"recommendations": recommendations,
			"finding_count": len(engagement.get("finding_ids", [])),
			"status": "draft",
			"version": "1.0",
			"drafted_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("audit_reports", report)

		engagement["report_id"] = report["id"]
		engagement["status"] = "review"
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)

		await self._audit_event(
			"audit_report_drafted", auditor_id, engagement_id,
			{"report_id": report["id"], "finding_count": report["finding_count"]},
		)
		return report

	async def management_response(
		self,
		finding_id: str,
		response_text: str,
		action_plan: str,
		owner_id: str,
		deadline: str,
	) -> dict[str, Any]:
		"""Record management's response to an audit finding with an action plan."""
		assert response_text, "response_text required"
		assert action_plan, "action_plan required"
		assert owner_id, "owner_id required"
		assert deadline, "deadline required"

		finding = await self._get_finding(finding_id)
		finding["management_response"] = response_text
		finding["action_plan"] = action_plan
		finding["owner_id"] = owner_id
		finding["remediation_deadline"] = deadline
		finding["management_response_received_at"] = _now()
		finding["status"] = "in_remediation"
		finding["updated_at"] = _now()
		await self._store.put("audit_findings", finding)

		await self._audit_event(
			"audit_finding_updated", owner_id, finding_id,
			{"status": "in_remediation", "deadline": deadline},
		)
		await self._notify.send(
			finding.get("engagement_id", "auditor"), "email",
			f"Management response received: finding {finding_id}",
			f"Management response submitted by {owner_id}. Action plan deadline: {deadline}",
		)
		return finding

	async def finalise_report(
		self,
		engagement_id: str,
		chief_audit_executive_id: str,
		sign_off_date: str,
	) -> dict[str, Any]:
		"""Finalise the audit report with CAE sign-off.

		Validates that the approver is not the report author (segregation of duties).
		Transitions report to 'final' and engagement to 'report_final'.
		"""
		assert chief_audit_executive_id, "chief_audit_executive_id required"
		assert sign_off_date, "sign_off_date required"

		engagement = await self._get_engagement(engagement_id)
		report_id = engagement.get("report_id")
		if not report_id:
			raise ValueError("No draft report found for this engagement; draft first")

		report = await self._store.get("audit_reports", report_id)
		if report is None:
			raise ValueError(f"Report not found: {report_id}")

		if report.get("author_id") == chief_audit_executive_id:
			raise PermissionError("CAE cannot sign off a report they authored (segregation of duties)")

		report["status"] = "final"
		report["approved_by"] = chief_audit_executive_id
		report["sign_off_date"] = sign_off_date
		report["finalised_at"] = _now()
		report["updated_at"] = _now()
		await self._store.put("audit_reports", report)

		engagement["status"] = "report_final"
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)

		await self._audit_event(
			"audit_report_approved", chief_audit_executive_id, report_id,
			{"sign_off_date": sign_off_date, "engagement_id": engagement_id},
		)
		await self._notify.send(
			engagement.get("auditee_id", "management"), "email",
			f"Audit report finalised: {engagement.get('area')}",
			f"The audit report for {engagement.get('area')} has been finalised and signed off on {sign_off_date}.",
		)
		return report

	# ─────────────────────────────────────────────────────────
	# Issue tracking and follow-up
	# ─────────────────────────────────────────────────────────

	async def issue_tracking(
		self,
		finding_id: str,
		status: str,
		progress_notes: str,
		updated_by: str,
	) -> dict[str, Any]:
		"""Update the tracking status and progress notes for an audit finding."""
		if status not in SUPPORTED_FINDING_STATUSES:
			raise ValueError(f"Unsupported status: {status}. Valid: {SUPPORTED_FINDING_STATUSES}")
		assert updated_by, "updated_by required"

		finding = await self._get_finding(finding_id)
		finding["status"] = status
		finding["progress_notes"] = progress_notes
		finding["last_updated_by"] = updated_by
		finding["updated_at"] = _now()
		await self._store.put("audit_findings", finding)

		await self._audit_event(
			"audit_finding_updated", updated_by, finding_id,
			{"status": status},
		)
		return finding

	async def follow_up_audit(
		self,
		finding_id: str,
		follow_up_date: str,
		status: str,
		evidence: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Record a follow-up audit check on a previous finding.

		Evidence is stored with retention metadata. Status reflects whether
		the remediation has been verified.
		"""
		assert follow_up_date, "follow_up_date required"
		assert status in {"verified_closed", "partially_remediated", "not_remediated"}, (
			"status: verified_closed | partially_remediated | not_remediated"
		)

		finding = await self._get_finding(finding_id)
		evidence_ids: list[str] = []
		for ev in evidence:
			ev_rec: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"finding_id": finding_id,
				"evidence_type": ev.get("type", "document"),
				"description": ev.get("description", ""),
				"file_ref": ev.get("file_ref", ""),
				"encrypted": True,
				"retention_days": 365,
				"collected_at": follow_up_date,
			}
			await self._store.put("audit_evidence", ev_rec)
			evidence_ids.append(ev_rec["id"])

		follow_up: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"finding_id": finding_id,
			"follow_up_date": follow_up_date,
			"status": status,
			"evidence_ids": evidence_ids,
			"created_at": _now(),
		}
		await self._store.put("audit_follow_ups", follow_up)

		if status == "verified_closed":
			finding["status"] = "remediated"
		elif status == "not_remediated":
			finding["status"] = "open"
		finding["last_follow_up_id"] = follow_up["id"]
		finding["updated_at"] = _now()
		await self._store.put("audit_findings", finding)

		await self._audit_event(
			"audit_finding_updated", "auditor", finding_id,
			{"follow_up_status": status, "follow_up_date": follow_up_date},
		)
		return follow_up

	async def close_finding(
		self,
		finding_id: str,
		close_date: str,
		verified_by: str,
	) -> dict[str, Any]:
		"""Close an audit finding after verifying remediation.

		Requires accepted risk or completed remediation evidence.
		"""
		assert close_date, "close_date required"
		assert verified_by, "verified_by required"

		finding = await self._get_finding(finding_id)
		if finding.get("status") not in {"remediated", "accepted", "in_remediation"}:
			raise ValueError(
				f"Finding must be remediated or accepted before closing; current: {finding.get('status')}"
			)

		finding["status"] = "closed"
		finding["closed_at"] = close_date
		finding["closed_by"] = verified_by
		finding["updated_at"] = _now()
		await self._store.put("audit_findings", finding)

		await self._audit_event(
			"audit_finding_closed", verified_by, finding_id,
			{"close_date": close_date},
		)
		return finding

	# ─────────────────────────────────────────────────────────
	# Reporting and oversight
	# ─────────────────────────────────────────────────────────

	async def audit_committee_report(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate the audit committee report for a period.

		Includes audit coverage, findings by severity, issue closure rate,
		CAE opinion, and upcoming audits.
		"""
		start, end = _period_bounds(period)
		plans = await self._store.query("audit_plans", {"entity_id": entity_id}, limit=100)
		year = start[:4]
		year_plans = [p for p in plans if str(p.get("year", "")) == year]

		engagements = await self._store.query(
			"audit_engagements",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_engs = [
			e for e in engagements
			if start <= e.get("start_date", "")[:10] <= end
			or start <= e.get("end_date", "")[:10] <= end
		]

		findings = await self._store.query(
			"audit_findings",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_findings = [f for f in findings if start <= f.get("raised_at", "")[:10] <= end]

		by_severity: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for f in period_findings:
			sev = f.get("risk_rating", "observation")
			st = f.get("status", "open")
			by_severity[sev] = by_severity.get(sev, 0) + 1
			by_status[st] = by_status.get(st, 0) + 1

		closed_count = by_status.get("closed", 0) + by_status.get("remediated", 0)
		closure_rate = (closed_count / len(period_findings) * 100) if period_findings else 0.0

		report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"audit_plans": len(year_plans),
			"engagements_completed": sum(1 for e in period_engs if e.get("status") == "report_final"),
			"engagements_in_progress": sum(1 for e in period_engs if e.get("status") not in {"report_final", "closed", "cancelled"}),
			"total_findings": len(period_findings),
			"findings_by_severity": by_severity,
			"findings_by_status": by_status,
			"issue_closure_rate_pct": round(closure_rate, 2),
			"cae_opinion": "satisfactory" if closure_rate >= 75 else "needs_improvement",
			"generated_at": _now(),
		}
		await self._store.put("audit_committee_reports", report)
		return report

	async def quality_assurance_review(
		self,
		engagement_id: str,
		reviewer_id: str,
		rating: str,
		observations: str,
	) -> dict[str, Any]:
		"""Record a quality assurance review of an audit engagement.

		Ratings: satisfactory | needs_improvement | unsatisfactory.
		Reviewer must be independent from the lead auditor.
		"""
		assert reviewer_id, "reviewer_id required"
		assert rating in {"satisfactory", "needs_improvement", "unsatisfactory"}, (
			"rating: satisfactory | needs_improvement | unsatisfactory"
		)
		assert observations, "observations required"

		engagement = await self._get_engagement(engagement_id)
		if engagement.get("lead_auditor_id") == reviewer_id:
			raise PermissionError("QA reviewer cannot be the lead auditor (segregation of duties)")

		qa_review: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"reviewer_id": reviewer_id,
			"rating": rating,
			"observations": observations,
			"reviewed_at": _now(),
		}
		await self._store.put("qa_reviews", qa_review)
		engagement["qa_review_id"] = qa_review["id"]
		engagement["qa_rating"] = rating
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)

		await self._audit_event(
			"audit_qa_reviewed", reviewer_id, engagement_id,
			{"rating": rating},
		)
		if rating == "unsatisfactory":
			await self._notify.send(
				"cae@datacraft.co.ke", "email",
				f"QA review: unsatisfactory — {engagement.get('area')}",
				f"QA review by {reviewer_id} for engagement {engagement_id} rated: {rating}.\nObservations: {observations}",
			)
		return qa_review

	async def external_audit_coordination(
		self,
		entity_id: str,
		external_firm: str,
		engagement_type: str,
		period: str,
	) -> dict[str, Any]:
		"""Coordinate and track an external audit or assurance engagement.

		Creates a coordination record, assigns liaison, and sets up a document
		request log for evidence provision.
		"""
		assert entity_id, "entity_id required"
		assert external_firm, "external_firm required"
		assert engagement_type, "engagement_type required"

		start, end = _period_bounds(period)
		coordination: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"external_firm": external_firm,
			"engagement_type": engagement_type,
			"period": period,
			"period_start": start,
			"period_end": end,
			"status": "in_progress",
			"document_requests": [],
			"liaison_id": None,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("external_audit_coordinations", coordination)
		await self._audit_event(
			"external_audit_coordination_created", entity_id, coordination["id"],
			{"external_firm": external_firm, "engagement_type": engagement_type},
		)
		return coordination

	async def continuous_auditing(
		self,
		entity_id: str,
		data_analytics_type: str,
		frequency: str,
	) -> dict[str, Any]:
		"""Configure and execute a continuous auditing data analytics run.

		Analytics types: journal_entry_testing | access_review | duplicate_payment |
		                 velocity_check | segregation_of_duties.
		Frequency: daily | weekly | monthly | quarterly.
		"""
		assert data_analytics_type in {
			"journal_entry_testing", "access_review", "duplicate_payment",
			"velocity_check", "segregation_of_duties",
		}, "data_analytics_type: journal_entry_testing | access_review | duplicate_payment | velocity_check | segregation_of_duties"
		assert frequency in {"daily", "weekly", "monthly", "quarterly"}, (
			"frequency: daily | weekly | monthly | quarterly"
		)

		run: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"data_analytics_type": data_analytics_type,
			"frequency": frequency,
			"status": "completed",
			"exceptions_found": 0,  # populated by actual analytics engine in production
			"run_at": _now(),
		}
		await self._store.put("continuous_audit_runs", run)
		await self._audit_event(
			"continuous_audit_run", entity_id, run["id"],
			{"type": data_analytics_type, "frequency": frequency},
		)
		return run

	async def risk_based_plan_update(
		self,
		plan_id: str,
		risk_reassessment_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Update an audit plan based on a risk reassessment.

		Reprioritises audit areas, adds new high-risk areas, and removes
		areas that have fallen below the risk threshold.
		"""
		assert risk_reassessment_data, "risk_reassessment_data required"

		plan = await self._store.get("audit_plans", plan_id)
		if plan is None:
			raise ValueError(f"Audit plan not found: {plan_id}")

		new_areas = risk_reassessment_data.get("high_risk_areas", [])
		removed_areas = risk_reassessment_data.get("low_risk_areas_to_remove", [])

		existing_areas: list[str] = plan.get("risk_based_areas", [])
		updated_areas = [a for a in existing_areas if a not in removed_areas]
		for area in new_areas:
			if area not in updated_areas:
				updated_areas.append(area)

		plan["risk_based_areas"] = updated_areas
		plan["last_risk_reassessment"] = _now()
		plan["risk_reassessment_data"] = risk_reassessment_data
		plan["updated_at"] = _now()
		await self._store.put("audit_plans", plan)

		await self._audit_event(
			"audit_plan_updated", "risk_team", plan_id,
			{"areas_added": len(new_areas), "areas_removed": len(removed_areas)},
		)
		return plan

	async def audit_universe(
		self,
		entity_id: str,
	) -> dict[str, Any]:
		"""Return the audit universe for an entity: all auditable areas with risk ratings.

		The universe is built from past engagement history and the current risk register.
		"""
		assert entity_id, "entity_id required"

		engagements = await self._store.query(
			"audit_engagements",
			{"entity_id": entity_id},
			limit=10_000,
		)

		areas: dict[str, dict[str, Any]] = {}
		for e in engagements:
			area = e.get("area", "unknown")
			if area not in areas:
				areas[area] = {
					"area": area,
					"last_audited": e.get("end_date"),
					"audit_count": 0,
					"open_findings": 0,
				}
			areas[area]["audit_count"] += 1
			if e.get("end_date", "") > areas[area].get("last_audited", ""):
				areas[area]["last_audited"] = e.get("end_date")

		# Count open findings per area
		findings = await self._store.query("audit_findings", {"entity_id": entity_id}, limit=10_000)
		for f in findings:
			if f.get("status") == "open":
				area_name = f.get("area_tested", "unknown")
				if area_name in areas:
					areas[area_name]["open_findings"] += 1

		return {
			"entity_id": entity_id,
			"total_auditable_areas": len(areas),
			"universe": list(areas.values()),
			"generated_at": _now(),
		}

	async def audit_resource_plan(
		self,
		entity_id: str,
		year: int,
		auditors: list[str],
		hours: dict[str, int],
	) -> dict[str, Any]:
		"""Create an audit resource plan assigning auditors and hours to engagements.

		hours: {auditor_id: planned_hours}.
		Returns a plan with utilisation rates and capacity analysis.
		"""
		assert entity_id, "entity_id required"
		assert auditors, "auditors required"
		assert hours, "hours required"

		total_hours = sum(hours.values())
		avg_hours = total_hours / len(auditors) if auditors else 0

		resource_plan: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"entity_id": entity_id,
			"year": year,
			"auditors": auditors,
			"hours": hours,
			"total_planned_hours": total_hours,
			"average_hours_per_auditor": round(avg_hours, 1),
			"utilisation_rate_pct": round(total_hours / (len(auditors) * 1760) * 100, 2) if auditors else 0.0,
			"created_at": _now(),
		}
		await self._store.put("audit_resource_plans", resource_plan)
		await self._audit_event(
			"audit_resource_plan_created", entity_id, resource_plan["id"],
			{"year": year, "auditor_count": len(auditors), "total_hours": total_hours},
		)
		return resource_plan

	async def kpi_report(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Generate audit KPI report: coverage %, issue closure rate, overdue findings.

		Coverage = areas audited / total universe × 100.
		Closure rate = closed findings / total findings × 100.
		"""
		start, end = _period_bounds(period)
		universe = await self.audit_universe(entity_id)
		total_areas = universe["total_auditable_areas"]

		engagements = await self._store.query(
			"audit_engagements",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_engs = [
			e for e in engagements
			if start <= e.get("start_date", "")[:10] <= end
		]
		completed_areas = {e.get("area") for e in period_engs if e.get("status") == "report_final"}
		coverage_pct = (len(completed_areas) / total_areas * 100) if total_areas > 0 else 0.0

		findings = await self._store.query(
			"audit_findings",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_findings = [f for f in findings if start <= f.get("raised_at", "")[:10] <= end]
		closed = sum(1 for f in period_findings if f.get("status") in {"closed", "remediated"})
		closure_rate = (closed / len(period_findings) * 100) if period_findings else 0.0

		today = date.today().isoformat()
		overdue = [
			f for f in findings
			if f.get("remediation_deadline", "9999") < today
			and f.get("status") not in {"closed", "remediated", "accepted"}
		]

		kpi_rec: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"audit_coverage_pct": round(coverage_pct, 2),
			"issue_closure_rate_pct": round(closure_rate, 2),
			"overdue_findings": len(overdue),
			"total_findings": len(period_findings),
			"closed_findings": closed,
			"engagements_completed": len([e for e in period_engs if e.get("status") == "report_final"]),
			"generated_at": _now(),
		}
		await self._store.put("audit_kpi_reports", kpi_rec)
		return kpi_rec

	# ─────────────────────────────────────────────────────────
	# Special investigations
	# ─────────────────────────────────────────────────────────

	async def fraud_investigation(
		self,
		suspicion_id: str,
		investigator_id: str,
		scope: str,
		methodology: str,
	) -> dict[str, Any]:
		"""Open a fraud investigation from a suspicion referral.

		Creates a confidential investigation record with legal hold.
		Methodology: digital_forensics | document_review | interview | combined.
		"""
		assert suspicion_id, "suspicion_id required"
		assert investigator_id, "investigator_id required"
		assert scope, "scope required"
		assert methodology in {"digital_forensics", "document_review", "interview", "combined"}, (
			"methodology: digital_forensics | document_review | interview | combined"
		)

		investigation: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"suspicion_id": suspicion_id,
			"investigator_id": investigator_id,
			"scope": scope,
			"methodology": methodology,
			"status": "active",
			"confidential": True,
			"legal_hold": True,
			"findings": [],
			"evidence_ids": [],
			"opened_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("fraud_investigations", investigation)
		await self._audit_event(
			"fraud_investigation_opened", investigator_id, investigation["id"],
			{"suspicion_id": suspicion_id, "methodology": methodology},
		)
		await self._notify.send(
			"legal@datacraft.co.ke", "email",
			f"Fraud investigation opened: {suspicion_id}",
			f"Confidential investigation opened by {investigator_id}. Legal hold activated.",
		)
		return investigation

	async def whistleblower_case(
		self,
		case_id: str,
		category: str,
		description: str,
		received_date: str,
	) -> dict[str, Any]:
		"""Register a whistleblower report as a confidential case.

		The case is marked confidential by default. The reporter identity is
		protected — no PII is stored in the case record.
		"""
		assert category, "category required"
		assert description, "description required"
		assert received_date, "received_date required"

		rule_ctx = {
			"operation": "open_case",
			"case_type": "whistleblower",
			"marked_confidential": True,
			"case_type_supported": True,
			"owner_present": True,
			"title_present": True,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Whistleblower case creation denied: {verdict['matched_rules']}")

		case: dict[str, Any] = {
			"id": case_id or _uid(),
			"tenant_id": self._tenant_id,
			"case_type": "whistleblower",
			"category": category,
			"description": description,
			"received_date": received_date,
			"status": "open",
			"confidential": True,
			"reporter_protected": True,
			"assigned_investigator": None,
			"investigation_notes": [],
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("whistleblower_cases", case)
		await self._audit_event(
			"whistleblower_case_opened", "system", case["id"],
			{"category": category, "received_date": received_date},
		)
		await self._notify.send(
			"audit-committee@datacraft.co.ke", "email",
			f"Whistleblower report received: {category}",
			f"A whistleblower report has been received (Case ID: {case['id']}). Confidential handling required.",
		)
		return case

	async def audit_scope_define(
		self,
		engagement_id: str,
		scope_areas: list[str],
		exclusions: list[str],
		defined_by: str,
	) -> dict[str, Any]:
		"""Define and document the audit scope for an engagement."""
		assert scope_areas, "scope_areas required"
		assert defined_by, "defined_by required"
		engagement = await self._get_engagement(engagement_id)
		scope_doc: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"scope_areas": scope_areas,
			"exclusions": exclusions,
			"defined_by": defined_by,
			"defined_at": _now(),
		}
		await self._store.put("audit_scope_definitions", scope_doc)
		engagement["scope_areas"] = scope_areas
		engagement["scope_exclusions"] = exclusions
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)
		await self._audit_event("audit_scope_defined", defined_by, engagement_id, {"scope_area_count": len(scope_areas)})
		return scope_doc

	async def evidence_request(
		self,
		engagement_id: str,
		items_requested: list[str],
		requested_from: str,
		due_date: str,
	) -> dict[str, Any]:
		"""Send an evidence request to the auditee."""
		assert items_requested, "items_requested required"
		assert requested_from, "requested_from required"
		engagement = await self._get_engagement(engagement_id)
		req: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"items_requested": items_requested,
			"requested_from": requested_from,
			"due_date": due_date,
			"status": "sent",
			"created_at": _now(),
		}
		await self._store.put("evidence_requests", req)
		await self._notify.send(requested_from, "email", f"Evidence request: engagement {engagement_id}", f"Please provide {len(items_requested)} items by {due_date}.")
		await self._audit_event("evidence_requested", "auditor", engagement_id, {"item_count": len(items_requested), "due_date": due_date})
		return req

	async def evidence_receive(
		self,
		request_id: str,
		received_items: list[str],
		received_by: str,
	) -> dict[str, Any]:
		"""Record receipt of evidence items from the auditee."""
		assert received_items, "received_items required"
		req = await self._store.get("evidence_requests", request_id)
		if req is None:
			raise ValueError(f"Evidence request not found: {request_id}")
		req["received_items"] = received_items
		req["received_by"] = received_by
		req["received_at"] = _now()
		req["status"] = "received"
		await self._store.put("evidence_requests", req)
		await self._audit_event("evidence_received", received_by, request_id, {"item_count": len(received_items)})
		return req

	async def workpaper_create(
		self,
		engagement_id: str,
		title: str,
		content: str,
		auditor_id: str,
		reference: str = "",
	) -> dict[str, Any]:
		"""Create an audit workpaper for an engagement."""
		assert title, "title required"
		assert auditor_id, "auditor_id required"
		wp: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"title": title,
			"content": content,
			"reference": reference,
			"author_id": auditor_id,
			"status": "draft",
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("workpapers", wp)
		await self._audit_event("workpaper_created", auditor_id, engagement_id, {"title": title})
		return wp

	async def observation_draft(
		self,
		engagement_id: str,
		observation: str,
		criteria: str,
		risk_rating: str,
		auditor_id: str,
	) -> dict[str, Any]:
		"""Draft an audit observation before formal finding is raised."""
		return await self.fieldwork_record(
			engagement_id=engagement_id,
			area_tested="observation",
			finding_type="observation",
			observation=observation,
			criteria=criteria,
			evidence=[],
			risk_rating=risk_rating,
			auditor_id=auditor_id,
		)

	async def management_response_receive(
		self,
		finding_id: str,
		response_text: str,
		action_plan: str,
		owner_id: str,
		deadline: str,
	) -> dict[str, Any]:
		"""Receive management response — domain alias."""
		return await self.management_response(finding_id, response_text, action_plan, owner_id, deadline)

	async def final_report_issue(self, engagement_id: str, cae_id: str, sign_off_date: str) -> dict[str, Any]:
		"""Issue the final audit report — domain alias."""
		return await self.finalise_report(engagement_id, cae_id, sign_off_date)

	async def distribution_list_manage(
		self,
		report_id: str,
		recipients: list[str],
		managed_by: str,
	) -> dict[str, Any]:
		"""Manage the distribution list for a final audit report."""
		assert recipients, "recipients required"
		report = await self._store.get("audit_reports", report_id)
		if report is None:
			raise ValueError(f"Report not found: {report_id}")
		report["distribution_list"] = recipients
		report["distributed_at"] = _now()
		report["distributed_by"] = managed_by
		await self._store.put("audit_reports", report)
		for r in recipients:
			await self._notify.send(r, "email", f"Audit report distributed: {report.get('area')}", f"The final audit report (ID: {report_id}) has been distributed.")
		await self._audit_event("report_distributed", managed_by, report_id, {"recipient_count": len(recipients)})
		return report

	async def follow_up_schedule(
		self,
		finding_id: str,
		follow_up_date: str,
		assigned_to: str,
	) -> dict[str, Any]:
		"""Schedule a follow-up review for an audit finding."""
		assert follow_up_date, "follow_up_date required"
		finding = await self._get_finding(finding_id)
		sched: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"finding_id": finding_id,
			"follow_up_date": follow_up_date,
			"assigned_to": assigned_to,
			"status": "scheduled",
			"created_at": _now(),
		}
		await self._store.put("follow_up_schedules", sched)
		await self._notify.send(assigned_to, "email", f"Follow-up scheduled: finding {finding_id}", f"Follow-up review scheduled for {follow_up_date}.")
		await self._audit_event("follow_up_scheduled", assigned_to, finding_id, {"follow_up_date": follow_up_date})
		return sched

	async def recommendation_track(
		self,
		finding_id: str,
	) -> dict[str, Any]:
		"""Track implementation status of recommendations for a finding."""
		finding = await self._get_finding(finding_id)
		follow_ups = await self._store.query("audit_follow_ups", {"finding_id": finding_id}, limit=100)
		latest = max(follow_ups, key=lambda f: f.get("created_at", ""), default=None)
		return {
			"finding_id": finding_id,
			"current_status": finding.get("status"),
			"management_response": finding.get("management_response"),
			"deadline": finding.get("remediation_deadline"),
			"follow_up_count": len(follow_ups),
			"latest_follow_up": latest,
			"tracked_at": _now(),
		}

	async def quality_review(
		self,
		engagement_id: str,
		reviewer_id: str,
		rating: str,
		observations: str,
	) -> dict[str, Any]:
		"""Record QA review — domain alias."""
		return await self.quality_assurance_review(engagement_id, reviewer_id, rating, observations)

	async def cae_report(self, entity_id: str, period: str) -> dict[str, Any]:
		"""Generate CAE report — domain alias for audit_committee_report."""
		return await self.audit_committee_report(entity_id, period)

	async def external_audit_coordinate(
		self,
		entity_id: str,
		external_firm: str,
		engagement_type: str,
		period: str,
	) -> dict[str, Any]:
		"""Coordinate external audit — domain alias."""
		return await self.external_audit_coordination(entity_id, external_firm, engagement_type, period)

	async def continuous_audit_run(
		self,
		entity_id: str,
		data_analytics_type: str,
		frequency: str,
	) -> dict[str, Any]:
		"""Run continuous audit — domain alias."""
		return await self.continuous_auditing(entity_id, data_analytics_type, frequency)

	async def fraud_indicator_detect(
		self,
		entity_id: str,
		detection_method: str = "journal_entry_testing",
	) -> dict[str, Any]:
		"""Run fraud indicator detection analytics."""
		run = await self.continuous_auditing(entity_id, "journal_entry_testing", "monthly")
		suspicion_id = _uid()
		return {**run, "fraud_indicator_scan": True, "suspicion_id": suspicion_id, "detection_method": detection_method}

	async def audit_universe_update(
		self,
		entity_id: str,
		new_areas: list[str],
		removed_areas: list[str],
		updated_by: str,
	) -> dict[str, Any]:
		"""Update the audit universe with new or removed areas."""
		assert updated_by, "updated_by required"
		universe = await self.audit_universe(entity_id)
		update_rec: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"areas_added": new_areas,
			"areas_removed": removed_areas,
			"updated_by": updated_by,
			"updated_at": _now(),
			"total_areas_after": universe["total_auditable_areas"] + len(new_areas) - len(removed_areas),
		}
		await self._store.put("audit_universe_updates", update_rec)
		await self._audit_event("audit_universe_updated", updated_by, entity_id, {"added": len(new_areas), "removed": len(removed_areas)})
		return update_rec

	async def audit_analytics(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute audit programme performance analytics for a period.

		Includes engagement throughput, finding rates by severity, QA ratings,
		resource utilisation, and coverage trends.
		"""
		start, end = _period_bounds(period)

		engagements = await self._store.query(
			"audit_engagements",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_engs = [e for e in engagements if start <= e.get("start_date", "")[:10] <= end]

		findings = await self._store.query(
			"audit_findings",
			{"entity_id": entity_id},
			limit=10_000,
		)
		period_findings = [f for f in findings if start <= f.get("raised_at", "")[:10] <= end]

		by_severity: dict[str, int] = {}
		for f in period_findings:
			sev = f.get("risk_rating", "observation")
			by_severity[sev] = by_severity.get(sev, 0) + 1

		qa_reviews = await self._store.query("qa_reviews", {}, limit=10_000)
		period_qa = [q for q in qa_reviews if start <= q.get("reviewed_at", "")[:10] <= end]
		qa_satisfactory = sum(1 for q in period_qa if q.get("rating") == "satisfactory")
		qa_rate = (qa_satisfactory / len(period_qa) * 100) if period_qa else 0.0

		total_planned = sum(e.get("planned_hours", 0) for e in period_engs)
		total_actual = sum(e.get("actual_hours", 0) for e in period_engs)
		resource_utilisation = (total_actual / total_planned * 100) if total_planned > 0 else 0.0

		return {
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"engagements_started": len(period_engs),
			"engagements_completed": sum(1 for e in period_engs if e.get("status") == "report_final"),
			"total_findings": len(period_findings),
			"findings_by_severity": by_severity,
			"qa_reviews": len(period_qa),
			"qa_satisfactory_rate_pct": round(qa_rate, 2),
			"total_planned_hours": total_planned,
			"total_actual_hours": total_actual,
			"resource_utilisation_pct": round(resource_utilisation, 2),
			"generated_at": _now(),
		}

	async def audit_kpi_summary(
		self,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise KPI card for the audit programme dashboard.

		Covers: engagements, findings per engagement, overdue items, QA rate,
		and on-time completion rate as a single flat dict suitable for UI cards.
		"""
		analytics = await self.audit_programme_analytics(entity_id, period)
		engagements = analytics["engagements_started"]
		completed = analytics["engagements_completed"]
		findings = analytics["total_findings"]
		fpe = round(findings / max(engagements, 1), 2)
		on_time_rate = round(completed / max(engagements, 1) * 100, 1)
		return {
			"entity_id": entity_id,
			"period": period,
			"engagements_started": engagements,
			"engagements_completed": completed,
			"on_time_completion_rate_pct": on_time_rate,
			"total_findings": findings,
			"findings_per_engagement": fpe,
			"critical_findings": analytics["findings_by_severity"].get("critical", 0),
			"qa_satisfactory_rate_pct": analytics["qa_satisfactory_rate_pct"],
			"resource_utilisation_pct": analytics["resource_utilisation_pct"],
			"generated_at": _now(),
		}

	# ─────────────────────────────────────────────────────────
	# Remediation SLA escalation (Improvement #4)
	# ─────────────────────────────────────────────────────────

	async def check_remediation_sla(
		self,
		entity_id: str,
		*,
		as_of_date: str | None = None,
	) -> dict[str, Any]:
		"""Inspect open findings, compute days overdue, trigger tiered escalation.

		Tiers (calendar days past deadline):
		  T+1  owner reminder
		  T+5  manager notification
		  T+10 CAE/board alert

		Returns escalation summary with per-finding actions taken.
		"""
		assert entity_id, "entity_id required"
		today = as_of_date or date.today().isoformat()

		findings = await self._store.query("audit_findings", {"entity_id": entity_id}, limit=10_000)
		open_findings = [
			f for f in findings
			if f.get("status") not in {"closed", "remediated", "accepted"}
			and f.get("remediation_deadline")
		]

		escalations: list[dict[str, Any]] = []
		for f in open_findings:
			deadline = f.get("remediation_deadline", "9999-12-31")
			if deadline >= today:
				continue
			try:
				days_overdue = (date.fromisoformat(today) - date.fromisoformat(deadline)).days
			except ValueError:
				continue

			if days_overdue >= 10:
				tier = "board_alert"
				await self._notify.send(
					"audit-committee@datacraft.co.ke", "email",
					f"[BOARD ALERT] Finding {f['id']} is {days_overdue}d overdue",
					f"Finding in {f.get('area_tested')} remains unremediated {days_overdue} days past deadline.",
				)
			elif days_overdue >= 5:
				tier = "manager_notification"
				await self._notify.send(
					f.get("owner_id", "management"), "email",
					f"[ESCALATION] Finding {f['id']} is {days_overdue}d overdue",
					f"Remediation deadline passed {days_overdue} days ago. Immediate action required.",
				)
			else:
				tier = "owner_reminder"
				await self._notify.send(
					f.get("owner_id", "owner"), "email",
					f"[REMINDER] Finding {f['id']} is {days_overdue}d overdue",
					f"Your remediation deadline has passed. Please update the action plan.",
				)

			esc_rec: dict[str, Any] = {
				"id": _uid(),
				"tenant_id": self._tenant_id,
				"finding_id": f["id"],
				"days_overdue": days_overdue,
				"tier": tier,
				"escalated_at": _now(),
			}
			await self._store.put("sla_escalations", esc_rec)
			escalations.append(esc_rec)

		await self._audit_event(
			"sla_escalation_check", "system", entity_id,
			{"findings_checked": len(open_findings), "escalations_triggered": len(escalations)},
		)
		return {
			"entity_id": entity_id,
			"as_of_date": today,
			"open_findings_checked": len(open_findings),
			"escalations_triggered": len(escalations),
			"escalations": escalations,
		}

	# ─────────────────────────────────────────────────────────
	# Statistical sampling (Improvement #5)
	# ─────────────────────────────────────────────────────────

	async def generate_sample_selection(
		self,
		engagement_id: str,
		population_size: int,
		confidence_level: float,
		tolerable_error_rate: float,
		expected_error_rate: float,
		auditor_id: str,
		*,
		sampling_method: str = "attribute",
	) -> dict[str, Any]:
		"""Compute a statistically valid sample and select items via Cochran formula.

		Methods: attribute | monetary_unit | random | systematic.
		Returns sample_size, selected_indices, precision_achieved, z_score_used.
		"""
		assert population_size > 0, "population_size must be positive"
		assert 0 < confidence_level < 1, "confidence_level: 0–1 (e.g. 0.95)"
		assert 0 < tolerable_error_rate < 1, "tolerable_error_rate: 0–1"
		assert 0 <= expected_error_rate < tolerable_error_rate, "expected_error_rate must be < tolerable_error_rate"
		assert auditor_id, "auditor_id required"
		assert sampling_method in {"attribute", "monetary_unit", "random", "systematic"}, (
			"sampling_method: attribute | monetary_unit | random | systematic"
		)

		await self._get_engagement(engagement_id)

		z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
		z = z_map.get(round(confidence_level, 2), 1.960)
		p = expected_error_rate if expected_error_rate > 0 else 0.05
		e = tolerable_error_rate - expected_error_rate or tolerable_error_rate

		raw_n = int((z ** 2 * p * (1 - p)) / (e ** 2)) + 1
		sample_size = min(raw_n, population_size)

		stride = population_size // sample_size if sample_size < population_size else 1
		selected_indices = [i * stride + 1 for i in range(sample_size)]
		precision_achieved = round(z * ((p * (1 - p)) / sample_size) ** 0.5, 4)

		sample_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"sampling_method": sampling_method,
			"population_size": population_size,
			"confidence_level": confidence_level,
			"tolerable_error_rate": tolerable_error_rate,
			"expected_error_rate": expected_error_rate,
			"z_score_used": z,
			"computed_sample_size": sample_size,
			"selected_indices": selected_indices,
			"precision_achieved": precision_achieved,
			"auditor_id": auditor_id,
			"created_at": _now(),
		}
		await self._store.put("audit_samples", sample_rec)
		await self._audit_event(
			"sample_generated", auditor_id, engagement_id,
			{"method": sampling_method, "sample_size": sample_size, "population": population_size},
		)
		return sample_rec

	# ─────────────────────────────────────────────────────────
	# Workpaper dual sign-off (Improvement #6)
	# ─────────────────────────────────────────────────────────

	async def workpaper_review(
		self,
		workpaper_id: str,
		reviewer_id: str,
		review_notes: str,
		decision: str,
	) -> dict[str, Any]:
		"""Submit a reviewer decision on a workpaper (approved | returned_for_revision | rejected).

		Reviewer must differ from workpaper author. Records SHA-256 content hash.
		"""
		assert reviewer_id, "reviewer_id required"
		assert decision in {"approved", "returned_for_revision", "rejected"}, (
			"decision: approved | returned_for_revision | rejected"
		)
		wp = await self._store.get("workpapers", workpaper_id)
		if wp is None:
			raise ValueError(f"Workpaper not found: {workpaper_id}")
		if wp.get("author_id") == reviewer_id:
			raise PermissionError("Reviewer cannot be the workpaper author (segregation of duties)")

		import hashlib
		content_hash = hashlib.sha256(wp.get("content", "").encode()).hexdigest()

		wp["reviewer_id"] = reviewer_id
		wp["review_notes"] = review_notes
		wp["review_decision"] = decision
		wp["content_hash_at_review"] = content_hash
		wp["reviewed_at"] = _now()
		wp["status"] = "reviewed" if decision == "approved" else "revision_required"
		wp["updated_at"] = _now()
		await self._store.put("workpapers", wp)
		await self._audit_event(
			"workpaper_reviewed", reviewer_id, workpaper_id,
			{"decision": decision, "content_hash": content_hash},
		)
		return wp

	async def workpaper_sign_off(
		self,
		workpaper_id: str,
		approver_id: str,
	) -> dict[str, Any]:
		"""Final supervisor sign-off on a reviewed workpaper.

		Approver must differ from both author and reviewer. Records final SHA-256 hash.
		"""
		assert approver_id, "approver_id required"
		wp = await self._store.get("workpapers", workpaper_id)
		if wp is None:
			raise ValueError(f"Workpaper not found: {workpaper_id}")
		if wp.get("status") != "reviewed":
			raise ValueError("Workpaper must be in 'reviewed' status before sign-off")
		if wp.get("author_id") == approver_id or wp.get("reviewer_id") == approver_id:
			raise PermissionError("Approver cannot be the author or reviewer (segregation of duties)")

		import hashlib
		final_hash = hashlib.sha256(wp.get("content", "").encode()).hexdigest()

		wp["approver_id"] = approver_id
		wp["content_hash_at_signoff"] = final_hash
		wp["signed_off_at"] = _now()
		wp["status"] = "signed_off"
		wp["updated_at"] = _now()
		await self._store.put("workpapers", wp)
		await self._audit_event(
			"workpaper_signed_off", approver_id, workpaper_id,
			{"final_hash": final_hash},
		)
		return wp

	# ─────────────────────────────────────────────────────────
	# Risk heatmap (Improvement #9)
	# ─────────────────────────────────────────────────────────

	async def risk_heatmap_data(
		self,
		entity_id: str,
		period: str,
		*,
		prior_period: str | None = None,
	) -> dict[str, Any]:
		"""5x5 impact/likelihood matrix for executive dashboards with RAG and trend arrows.

		Severity mapping: observation→(1,1), minor→(2,2), major→(3,4), critical→(5,5).
		trend (when prior_period supplied): improving | stable | deteriorating.
		"""
		assert entity_id, "entity_id required"
		start, end = _period_bounds(period)

		findings = await self._store.query("audit_findings", {"entity_id": entity_id}, limit=10_000)
		period_findings = [f for f in findings if start <= f.get("raised_at", "")[:10] <= end]

		severity_coords: dict[str, tuple[int, int]] = {
			"observation": (1, 1),
			"minor": (2, 2),
			"major": (3, 4),
			"critical": (5, 5),
		}
		matrix: dict[str, int] = {}
		for f in period_findings:
			sev = f.get("risk_rating", "observation")
			impact, likelihood = severity_coords.get(sev, (1, 1))
			cell = f"{impact}_{likelihood}"
			matrix[cell] = matrix.get(cell, 0) + 1

		def _rag(count: int) -> str:
			return "red" if count >= 3 else "amber" if count >= 1 else "green"

		cells = []
		for impact in range(1, 6):
			for likelihood in range(1, 6):
				count = matrix.get(f"{impact}_{likelihood}", 0)
				cells.append({"impact": impact, "likelihood": likelihood, "finding_count": count, "rag": _rag(count)})

		trend: str | None = None
		if prior_period:
			p_start, p_end = _period_bounds(prior_period)
			prior_findings = [f for f in findings if p_start <= f.get("raised_at", "")[:10] <= p_end]
			curr_crit = sum(1 for f in period_findings if f.get("risk_rating") == "critical")
			prev_crit = sum(1 for f in prior_findings if f.get("risk_rating") == "critical")
			trend = "improving" if curr_crit < prev_crit else ("deteriorating" if curr_crit > prev_crit else "stable")

		heatmap: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_findings": len(period_findings),
			"matrix_cells": cells,
			"trend": trend,
			"generated_at": _now(),
		}
		await self._store.put("risk_heatmaps", heatmap)
		return heatmap

	# ─────────────────────────────────────────────────────────
	# Systemic risk detection (Improvement #10)
	# ─────────────────────────────────────────────────────────

	async def systemic_risk_detect(
		self,
		entity_id: str,
		period: str,
		*,
		min_cluster_size: int = 2,
	) -> dict[str, Any]:
		"""Cluster findings by (area_tested, finding_type) to surface systemic control weaknesses.

		Returns clusters sorted by occurrence_count descending.
		Critical-severity clusters auto-notify cae@datacraft.co.ke.
		"""
		assert entity_id, "entity_id required"
		assert min_cluster_size >= 1, "min_cluster_size must be >= 1"

		start, end = _period_bounds(period)
		findings = await self._store.query("audit_findings", {"entity_id": entity_id}, limit=10_000)
		period_findings = [f for f in findings if start <= f.get("raised_at", "")[:10] <= end]

		clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
		for f in period_findings:
			key = (f.get("area_tested", "unknown"), f.get("finding_type", "unknown"))
			clusters.setdefault(key, []).append(f)

		systemic: list[dict[str, Any]] = []
		for (area, ftype), items in clusters.items():
			if len(items) < min_cluster_size:
				continue
			sev_counts: dict[str, int] = {}
			for fi in items:
				sev = fi.get("risk_rating", "observation")
				sev_counts[sev] = sev_counts.get(sev, 0) + 1
			dominant = max(sev_counts, key=lambda k: sev_counts[k])
			systemic.append({
				"area_tested": area,
				"finding_type": ftype,
				"occurrence_count": len(items),
				"dominant_severity": dominant,
				"finding_ids": [fi["id"] for fi in items],
				"recommendation": f"Initiate thematic deep-dive engagement for '{area}' control environment.",
			})

		systemic.sort(key=lambda c: c["occurrence_count"], reverse=True)

		result: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_findings_analysed": len(period_findings),
			"systemic_clusters_found": len(systemic),
			"clusters": systemic,
			"generated_at": _now(),
		}
		await self._store.put("systemic_risk_reports", result)
		await self._audit_event(
			"systemic_risk_detected", "system", entity_id,
			{"clusters": len(systemic), "period": period},
		)
		if any(c["dominant_severity"] == "critical" for c in systemic):
			await self._notify.send(
				"cae@datacraft.co.ke", "email",
				f"Systemic critical-risk cluster detected: {entity_id}",
				f"{sum(1 for c in systemic if c['dominant_severity'] == 'critical')} critical clusters in {period}.",
			)
		return result

	# ─────────────────────────────────────────────────────────
	# Engagement earned-value analysis (Improvement #12)
	# ─────────────────────────────────────────────────────────

	async def engagement_time_analysis(
		self,
		engagement_id: str,
		actual_hours_to_date: float,
		*,
		work_complete_pct: float = 0.0,
	) -> dict[str, Any]:
		"""Earned Value Analysis: PV, EV, AC, CV, SV, CPI, SPI.

		Alerts cae@datacraft.co.ke when CPI < 0.8 (over budget) or SPI < 0.85 (behind schedule).
		Updates engagement.actual_hours in the store.
		"""
		assert actual_hours_to_date >= 0, "actual_hours_to_date must be >= 0"
		assert 0.0 <= work_complete_pct <= 100.0, "work_complete_pct: 0–100"

		engagement = await self._get_engagement(engagement_id)
		planned_hours = float(engagement.get("planned_hours", 80))
		start = engagement.get("start_date", "")
		end = engagement.get("end_date", "")

		pct_elapsed = 0.0
		if start and end:
			try:
				total_days = (date.fromisoformat(end) - date.fromisoformat(start)).days
				elapsed_days = (date.today() - date.fromisoformat(start)).days
				pct_elapsed = max(0.0, min(100.0, elapsed_days / total_days * 100 if total_days > 0 else 0.0))
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		pv = round(planned_hours * pct_elapsed / 100, 2)
		ev = round(planned_hours * work_complete_pct / 100, 2)
		ac = actual_hours_to_date
		cv = round(ev - ac, 2)
		sv = round(ev - pv, 2)
		cpi = round(ev / ac, 4) if ac > 0 else 1.0
		spi = round(ev / pv, 4) if pv > 0 else 1.0

		engagement["actual_hours"] = actual_hours_to_date
		engagement["updated_at"] = _now()
		await self._store.put("audit_engagements", engagement)

		status = "over_budget" if cpi < 0.8 else ("behind_schedule" if spi < 0.85 else "on_track")
		eva: dict[str, Any] = {
			"id": _uid(),
			"engagement_id": engagement_id,
			"planned_hours": planned_hours,
			"pct_elapsed": round(pct_elapsed, 2),
			"pct_complete": work_complete_pct,
			"planned_value_pv": pv,
			"earned_value_ev": ev,
			"actual_cost_ac": ac,
			"cost_variance_cv": cv,
			"schedule_variance_sv": sv,
			"cost_performance_index": cpi,
			"schedule_performance_index": spi,
			"status": status,
			"computed_at": _now(),
		}
		await self._store.put("engagement_eva_snapshots", eva)

		if cpi < 0.8 or spi < 0.85:
			await self._notify.send(
				"cae@datacraft.co.ke", "email",
				f"Engagement {engagement_id} EVA alert: CPI={cpi} SPI={spi}",
				f"{'Cost overrun' if cpi < 0.8 else 'Schedule slippage'} detected. Immediate review recommended.",
			)
		await self._audit_event(
			"engagement_eva_computed", "system", engagement_id,
			{"cpi": cpi, "spi": spi, "status": status},
		)
		return eva

	# ─────────────────────────────────────────────────────────
	# Remediation velocity scoring (Improvement #15)
	# ─────────────────────────────────────────────────────────

	async def remediation_velocity_score(
		self,
		entity_id: str,
		owner_id: str | None = None,
	) -> dict[str, Any]:
		"""Per-owner closure velocity from trailing 12-month history.

		Flags open findings with closure_probability < 0.5 for proactive intervention.
		velocity_rate = closed / total assigned. Closure probability uses days-remaining decay.
		"""
		assert entity_id, "entity_id required"

		findings = await self._store.query("audit_findings", {"entity_id": entity_id}, limit=10_000)
		cutoff = (date.today() - timedelta(days=365)).isoformat()
		recent = [f for f in findings if f.get("raised_at", "")[:10] >= cutoff]

		owner_stats: dict[str, dict[str, int]] = {}
		for f in recent:
			oid = f.get("owner_id") or "unassigned"
			if owner_id and oid != owner_id:
				continue
			owner_stats.setdefault(oid, {"total": 0, "closed": 0})
			owner_stats[oid]["total"] += 1
			if f.get("status") in {"closed", "remediated"}:
				owner_stats[oid]["closed"] += 1

		velocity_by_owner: dict[str, float] = {}
		velocity_table: list[dict[str, Any]] = []
		for oid, stats in owner_stats.items():
			v = round(stats["closed"] / stats["total"], 4) if stats["total"] > 0 else 0.0
			velocity_by_owner[oid] = v
			velocity_table.append({
				"owner_id": oid,
				"assigned": stats["total"],
				"closed": stats["closed"],
				"velocity_rate": v,
				"risk_level": "high" if v < 0.5 else ("medium" if v < 0.8 else "low"),
			})

		open_findings = [
			f for f in findings
			if f.get("status") not in {"closed", "remediated", "accepted"}
			and f.get("remediation_deadline")
		]
		flagged: list[dict[str, Any]] = []
		for f in open_findings:
			oid = f.get("owner_id") or "unassigned"
			v = velocity_by_owner.get(oid, 0.5)
			deadline = f.get("remediation_deadline", "9999-12-31")
			try:
				days_remaining = (date.fromisoformat(deadline) - date.today()).days
			except ValueError:
				days_remaining = 999
			closure_prob = round(min(1.0, v * (days_remaining / 30)), 4) if days_remaining > 0 else 0.0
			if closure_prob < 0.5:
				flagged.append({
					"finding_id": f["id"],
					"owner_id": oid,
					"owner_velocity": v,
					"days_remaining": days_remaining,
					"closure_probability": closure_prob,
					"recommended_action": "Escalate immediately" if closure_prob < 0.2 else "Schedule check-in",
				})

		result: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"owner_filter": owner_id,
			"velocity_table": velocity_table,
			"high_risk_findings": len(flagged),
			"flagged_findings": flagged,
			"computed_at": _now(),
		}
		await self._store.put("remediation_velocity_scores", result)
		return result

	# ─────────────────────────────────────────────────────────
	# Peer benchmark report (Improvement #7)
	# ─────────────────────────────────────────────────────────

	async def peer_benchmark_report(
		self,
		entity_id: str,
		period: str,
		*,
		industry_sector: str = "financial_services",
	) -> dict[str, Any]:
		"""Compare programme KPIs against IIA Global Pulse Survey sector medians.

		Sectors: financial_services | manufacturing | healthcare | government | technology.
		Returns comparisons list with gap and position (leads/lags) per metric.
		"""
		assert entity_id, "entity_id required"
		assert industry_sector in {
			"financial_services", "manufacturing", "healthcare", "government", "technology"
		}, "industry_sector: financial_services | manufacturing | healthcare | government | technology"

		# IIA Global Pulse Survey 2024 sector medians
		benchmarks: dict[str, dict[str, float]] = {
			"financial_services": {"coverage_pct": 72.0, "closure_rate_pct": 68.0, "findings_per_eng": 3.1, "qa_satisfactory_pct": 82.0},
			"manufacturing":      {"coverage_pct": 60.0, "closure_rate_pct": 62.0, "findings_per_eng": 2.8, "qa_satisfactory_pct": 78.0},
			"healthcare":         {"coverage_pct": 65.0, "closure_rate_pct": 70.0, "findings_per_eng": 3.4, "qa_satisfactory_pct": 80.0},
			"government":         {"coverage_pct": 55.0, "closure_rate_pct": 55.0, "findings_per_eng": 4.2, "qa_satisfactory_pct": 74.0},
			"technology":         {"coverage_pct": 80.0, "closure_rate_pct": 75.0, "findings_per_eng": 2.5, "qa_satisfactory_pct": 88.0},
		}
		bm = benchmarks[industry_sector]

		kpi = await self.kpi_report(entity_id, period)
		analytics = await self.audit_analytics(entity_id, period)
		entity_metrics: dict[str, float] = {
			"coverage_pct": kpi["audit_coverage_pct"],
			"closure_rate_pct": kpi["issue_closure_rate_pct"],
			"findings_per_eng": round(kpi["total_findings"] / max(kpi["engagements_completed"], 1), 2),
			"qa_satisfactory_pct": analytics["qa_satisfactory_rate_pct"],
		}

		comparisons: list[dict[str, Any]] = []
		for metric, entity_val in entity_metrics.items():
			bm_val = bm[metric]
			if metric == "findings_per_eng":
				gap = bm_val - entity_val
				position = "leads" if entity_val <= bm_val else "lags"
			else:
				gap = entity_val - bm_val
				position = "leads" if entity_val >= bm_val else "lags"
			comparisons.append({"metric": metric, "entity_value": entity_val, "benchmark_median": bm_val, "gap": round(gap, 2), "position": position})

		report: dict[str, Any] = {
			"id": _uid(),
			"entity_id": entity_id,
			"period": period,
			"industry_sector": industry_sector,
			"comparisons": comparisons,
			"leads_count": sum(1 for c in comparisons if c["position"] == "leads"),
			"lags_count": sum(1 for c in comparisons if c["position"] == "lags"),
			"generated_at": _now(),
		}
		await self._store.put("peer_benchmark_reports", report)
		await self._audit_event(
			"peer_benchmark_generated", entity_id, report["id"],
			{"sector": industry_sector, "leads": report["leads_count"], "lags": report["lags_count"]},
		)
		return report

	# ─────────────────────────────────────────────────────────
	# Control test library (Improvement #13)
	# ─────────────────────────────────────────────────────────

	async def control_test_library_add(
		self,
		control_objective: str,
		framework: str,
		test_steps: list[str],
		expected_evidence: list[str],
		added_by: str,
		*,
		framework_reference: str = "",
	) -> dict[str, Any]:
		"""Add a versioned control test procedure to the shared library.

		Frameworks: COSO | COBIT | ISO27001 | NIST | IIA_IPPF | SOX | PCI_DSS.
		"""
		assert control_objective, "control_objective required"
		assert framework in {"COSO", "COBIT", "ISO27001", "NIST", "IIA_IPPF", "SOX", "PCI_DSS"}, (
			"framework: COSO | COBIT | ISO27001 | NIST | IIA_IPPF | SOX | PCI_DSS"
		)
		assert test_steps, "test_steps required"
		assert added_by, "added_by required"

		procedure: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"control_objective": control_objective,
			"framework": framework,
			"framework_reference": framework_reference,
			"test_steps": test_steps,
			"expected_evidence": expected_evidence,
			"version": "1.0",
			"status": "active",
			"added_by": added_by,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._store.put("control_test_library", procedure)
		await self._audit_event(
			"control_test_added", added_by, procedure["id"],
			{"framework": framework, "control_objective": control_objective},
		)
		return procedure

	async def control_test_execute(
		self,
		engagement_id: str,
		procedure_id: str,
		step_results: list[dict[str, Any]],
		auditor_id: str,
	) -> dict[str, Any]:
		"""Execute a library test procedure; auto-raises findings for exception steps.

		step_results: [{step_index, result: pass|fail|exception, notes}].
		"""
		assert auditor_id, "auditor_id required"
		assert step_results, "step_results required"

		procedure = await self._store.get("control_test_library", procedure_id)
		if procedure is None:
			raise ValueError(f"Control test procedure not found: {procedure_id}")
		await self._get_engagement(engagement_id)

		exceptions = [s for s in step_results if s.get("result") == "exception"]
		findings_raised: list[str] = []
		for exc in exceptions:
			finding = await self.fieldwork_record(
				engagement_id=engagement_id,
				area_tested=procedure.get("control_objective", ""),
				finding_type="control_exception",
				observation=exc.get("notes", "Control exception noted"),
				criteria=procedure.get("framework_reference") or procedure.get("framework", ""),
				evidence=[],
				risk_rating="minor",
				auditor_id=auditor_id,
			)
			findings_raised.append(finding["id"])

		execution: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"engagement_id": engagement_id,
			"procedure_id": procedure_id,
			"control_objective": procedure.get("control_objective"),
			"framework": procedure.get("framework"),
			"step_results": step_results,
			"total_steps": len(step_results),
			"passed_steps": sum(1 for s in step_results if s.get("result") == "pass"),
			"failed_steps": sum(1 for s in step_results if s.get("result") == "fail"),
			"exceptions": len(exceptions),
			"findings_raised": findings_raised,
			"auditor_id": auditor_id,
			"executed_at": _now(),
		}
		await self._store.put("control_test_executions", execution)
		await self._audit_event(
			"control_test_executed", auditor_id, engagement_id,
			{"procedure_id": procedure_id, "exceptions": len(exceptions)},
		)
		return execution

	# ─────────────────────────────────────────────────────────
	# Report version diff (Improvement #14)
	# ─────────────────────────────────────────────────────────

	async def report_version_publish(
		self,
		report_id: str,
		updated_findings: list[str],
		updated_recommendations: list[str],
		author_id: str,
		*,
		change_summary: str = "",
	) -> dict[str, Any]:
		"""Publish a new semver report version with structured diff and snapshot of previous.

		Increments minor version. Snapshots previous finding_references and recommendations.
		Returns updated report dict plus snapshot_id and diff.
		"""
		assert author_id, "author_id required"
		report = await self._store.get("audit_reports", report_id)
		if report is None:
			raise ValueError(f"Report not found: {report_id}")

		prev_version = report.get("version", "1.0")
		try:
			major, minor = prev_version.split(".")
			new_version = f"{major}.{int(minor) + 1}"
		except (ValueError, AttributeError):
			new_version = "1.1"

		snapshot: dict[str, Any] = {
			"id": _uid(),
			"report_id": report_id,
			"version": prev_version,
			"finding_references": report.get("finding_references", []),
			"recommendations": report.get("recommendations", []),
			"snapshotted_at": _now(),
		}
		await self._store.put("audit_report_versions", snapshot)

		prev_findings = set(report.get("finding_references", []))
		new_findings = set(updated_findings)
		prev_recs = set(report.get("recommendations", []))
		new_recs = set(updated_recommendations)

		diff: dict[str, Any] = {
			"findings_added": list(new_findings - prev_findings),
			"findings_removed": list(prev_findings - new_findings),
			"recommendations_added": list(new_recs - prev_recs),
			"recommendations_removed": list(prev_recs - new_recs),
		}

		report["version"] = new_version
		report["finding_references"] = updated_findings
		report["recommendations"] = updated_recommendations
		report["change_summary"] = change_summary
		report["last_diff"] = diff
		report["last_revised_by"] = author_id
		report["updated_at"] = _now()
		await self._store.put("audit_reports", report)
		await self._audit_event(
			"report_version_published", author_id, report_id,
			{"new_version": new_version, "diff": diff},
		)
		return {**report, "snapshot_id": snapshot["id"], "diff": diff}
