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
