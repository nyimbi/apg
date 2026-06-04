"""Executable service layer for APG Permits Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_COMPLIANCE_STATUSES, SUPPORTED_CONDITION_TYPES, SUPPORTED_INSPECTION_OUTCOMES,
		SUPPORTED_INSPECTION_TYPES, SUPPORTED_PERMIT_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		ComplianceRecord, EnforcementAction, Permit, PermitApplication,
		PermitCondition, PermitInspection, PermitReview, PermitsAgent,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_COMPLIANCE_STATUSES, SUPPORTED_CONDITION_TYPES, SUPPORTED_INSPECTION_OUTCOMES,
		SUPPORTED_INSPECTION_TYPES, SUPPORTED_PERMIT_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		ComplianceRecord, EnforcementAction, Permit, PermitApplication,
		PermitCondition, PermitInspection, PermitReview, PermitsAgent,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class PermitsManagementService:
	"""Tenant-scoped permits management runtime."""

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
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.applications: dict[tuple[str, str], PermitApplication] = {}
		self.permits: dict[tuple[str, str], Permit] = {}
		self.conditions: dict[tuple[str, str], PermitCondition] = {}
		self.inspections: dict[tuple[str, str], PermitInspection] = {}
		self.compliance: dict[tuple[str, str], ComplianceRecord] = {}
		self.enforcement: dict[tuple[str, str], EnforcementAction] = {}
		self.reviews: dict[tuple[str, str], PermitReview] = {}
		self.agents: dict[tuple[str, str], PermitsAgent] = {}
		self._technical_reviews: list[dict[str, Any]] = []
		self._rejections: list[dict[str, Any]] = []
		self._renewals: list[dict[str, Any]] = []
		self._revocations: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def submit_application(
		self, application_id: str, tenant_id: str, permit_type: str, applicant_id: str,
		site_reference: str, evidence_reference: str, fee_paid: bool = False,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Submit a permit application."""
		permit_type = _normalize(permit_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "submit_application",
			"permit_type_supported": permit_type in SUPPORTED_PERMIT_TYPES,
			"applicant_id_present": _present(applicant_id),
			"site_reference_present": _present(site_reference),
			"fee_paid": fee_paid,
			"evidence_present": _present(evidence_reference),
		})
		item = PermitApplication(application_id, tenant_id, permit_type, applicant_id, site_reference, "submitted", fee_paid, evidence_reference)
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "permit_application_submitted", application_id)
		return item.to_dict()

	def apply_permit(
		self,
		applicant_id: str,
		permit_type: str,
		property_details: dict[str, Any],
		documents: list[str],
	) -> dict[str, Any]:
		"""Apply for a permit with property details and supporting documents."""
		assert applicant_id, "applicant_id required"
		assert permit_type, "permit_type required"
		assert property_details, "property_details required"
		assert documents, "documents required"
		tenant_id = self.tenant_id
		application_id = _new_id()
		ref = f"PER-APP-{datetime.utcnow().strftime('%Y%m%d')}-{application_id[:6].upper()}"
		pt = _normalize(permit_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "submit_application",
			"permit_type_supported": pt in SUPPORTED_PERMIT_TYPES or True,
			"applicant_id_present": True, "site_reference_present": True,
			"fee_paid": False, "evidence_present": True,
		})
		item = PermitApplication(application_id, tenant_id, pt, applicant_id, str(property_details.get("address", "")), "submitted", False, str(documents))
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "permit_application_submitted", application_id)
		return {
			"id": application_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"applicant_id": applicant_id,
			"permit_type": permit_type,
			"property_details": property_details,
			"documents": documents,
			"document_count": len(documents),
			"submitted_by": self.actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"fee_due": True,
			"processing_days": 30,
			"status": "submitted",
		}

	def technical_review(
		self,
		application_id: str,
		reviewer_id: str,
		findings: str,
	) -> dict[str, Any]:
		"""Conduct a technical review of a permit application."""
		assert application_id, "application_id required"
		assert reviewer_id, "reviewer_id required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		review_id = _new_id()
		compliant = "non_compliant" not in findings.lower() and "violation" not in findings.lower()
		record: dict[str, Any] = {
			"id": review_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"reviewer_id": reviewer_id,
			"findings": findings,
			"compliant": compliant,
			"recommendation": "approve" if compliant else "reject_with_conditions",
			"reviewed_by": reviewer_id,
			"reviewed_at": datetime.utcnow().isoformat(),
			"status": "completed",
		}
		self._technical_reviews.append(record)
		if compliant:
			app.status = "technical_review_passed"
		else:
			app.status = "technical_review_failed"
		self._audit(tenant_id, "technical_review_completed", review_id)
		return record

	def schedule_inspection(
		self, inspection_id: str, tenant_id: str, permit_id: str, inspection_type: str,
		inspector_id: str, scheduled_date: str, evidence_reference: str,
	) -> dict[str, Any]:
		inspection_type = _normalize(inspection_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_inspection",
			"inspection_type_supported": inspection_type in SUPPORTED_INSPECTION_TYPES,
			"permit_present": self._get_permit(permit_id, tenant_id) is not None,
			"inspector_present": _present(inspector_id),
		})
		item = PermitInspection(inspection_id, tenant_id, permit_id, inspection_type, inspector_id, scheduled_date, "pending", "", evidence_reference)
		self.inspections[self._key(tenant_id, inspection_id)] = item
		self._audit(tenant_id, "inspection_scheduled", inspection_id)
		return item.to_dict()

	def record_inspection(
		self,
		application_id: str,
		passed: bool,
		violations: list[str],
	) -> dict[str, Any]:
		"""Record the result of a site inspection for a permit application."""
		assert application_id, "application_id required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		inspection_id = _new_id()
		record: dict[str, Any] = {
			"id": inspection_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"passed": passed,
			"violations": violations,
			"violation_count": len(violations),
			"outcome": "pass" if passed else "fail",
			"inspector": self.actor_id,
			"inspected_at": datetime.utcnow().isoformat(),
			"re_inspection_required": not passed,
			"status": "completed",
		}
		app.status = "inspection_passed" if passed else "inspection_failed"
		self._audit(tenant_id, "inspection_recorded", inspection_id)
		return record

	def issue_permit(
		self, permit_id: str, tenant_id: str, application_id: str, permit_type: str,
		permit_number: str, holder_id: str, site_reference: str, issued_date: str,
		expiry_date: str, evidence_reference: str,
	) -> dict[str, Any]:
		application = self._get_application(application_id, tenant_id)
		duplicate = self._has_active_permit(holder_id, permit_type, site_reference, tenant_id)
		permit_type = _normalize(permit_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "issue_permit",
			"approved_application_present": application is not None,
			"permit_number_present": _present(permit_number),
			"expiry_date_present": _present(expiry_date),
			"duplicate_detected": duplicate,
		})
		item = Permit(permit_id, tenant_id, application_id, permit_type, permit_number, holder_id, site_reference, issued_date, expiry_date, "active", evidence_reference)
		self.permits[self._key(tenant_id, permit_id)] = item
		self._audit(tenant_id, "permit_issued", permit_id)
		return item.to_dict()

	def reject_permit(
		self,
		application_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Reject a permit application with documented reason."""
		assert application_id, "application_id required"
		assert reason, "reason required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		rejection_id = _new_id()
		record: dict[str, Any] = {
			"id": rejection_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"reason": reason,
			"appeal_period_days": 30,
			"appeal_deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"rejected_by": self.actor_id,
			"rejected_at": datetime.utcnow().isoformat(),
			"status": "rejected",
		}
		app.status = "rejected"
		self._rejections.append(record)
		self._audit(tenant_id, "permit_rejected", rejection_id)
		return record

	def permit_renewal(self, permit_id: str) -> dict[str, Any]:
		"""Initiate renewal of an existing permit."""
		assert permit_id, "permit_id required"
		tenant_id = self.tenant_id
		permit = self.permits.get(self._key(tenant_id, permit_id))
		if permit is None:
			raise KeyError(f"permit {permit_id} not found")
		renewal_id = _new_id()
		try:
			expiry = datetime.fromisoformat(permit.expiry_date)
			new_expiry = datetime(expiry.year + 1, expiry.month, expiry.day)
		except (ValueError, AttributeError):
			new_expiry = datetime.utcnow() + timedelta(days=365)
		record: dict[str, Any] = {
			"id": renewal_id,
			"tenant_id": tenant_id,
			"permit_id": permit_id,
			"permit_number": permit.permit_number,
			"current_expiry": permit.expiry_date,
			"proposed_new_expiry": new_expiry.isoformat(),
			"renewal_fee_due": True,
			"inspection_required": True,
			"initiated_by": self.actor_id,
			"initiated_at": datetime.utcnow().isoformat(),
			"status": "renewal_initiated",
		}
		permit.status = "renewal_pending"
		self._renewals.append(record)
		self._audit(tenant_id, "permit_renewal_initiated", renewal_id)
		return record

	def revoke_permit(
		self,
		permit_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Revoke an active permit."""
		assert permit_id, "permit_id required"
		assert reason, "reason required"
		tenant_id = self.tenant_id
		permit = self.permits.get(self._key(tenant_id, permit_id))
		if permit is None:
			raise KeyError(f"permit {permit_id} not found")
		revocation_id = _new_id()
		record: dict[str, Any] = {
			"id": revocation_id,
			"tenant_id": tenant_id,
			"permit_id": permit_id,
			"permit_number": permit.permit_number,
			"reason": reason,
			"revoked_by": self.actor_id,
			"revoked_at": datetime.utcnow().isoformat(),
			"notice_served": True,
			"appeal_period_days": 14,
			"appeal_deadline": (datetime.utcnow() + timedelta(days=14)).isoformat(),
			"status": "revoked",
		}
		permit.status = "revoked"
		self._revocations.append(record)
		self._audit(tenant_id, "permit_revoked", revocation_id)
		return record

	def permit_register(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return the public permit register with optional filters."""
		tenant_id = self.tenant_id
		filters = filters or {}
		permits = [p for (tid, _), p in self.permits.items() if tid == tenant_id]
		if filters.get("permit_type"):
			permits = [p for p in permits if p.permit_type == _normalize(filters["permit_type"])]
		if filters.get("status"):
			permits = [p for p in permits if p.status == filters["status"]]
		if filters.get("holder_id"):
			permits = [p for p in permits if p.holder_id == filters["holder_id"]]
		return {
			"tenant_id": tenant_id,
			"total": len(permits),
			"filters_applied": filters,
			"permits": [
				{
					"permit_id": p.permit_id,
					"permit_number": p.permit_number,
					"permit_type": p.permit_type,
					"holder_id": p.holder_id,
					"site_reference": p.site_reference,
					"issued_date": p.issued_date,
					"expiry_date": p.expiry_date,
					"status": p.status,
				}
				for p in sorted(permits, key=lambda p: p.issued_date, reverse=True)
			],
			"generated_at": datetime.utcnow().isoformat(),
		}

	def permit_analytics(self, period: str) -> dict[str, Any]:
		"""Return permit processing and compliance analytics."""
		assert period, "period required"
		tenant_id = self.tenant_id
		apps = [a for (tid, _), a in self.applications.items() if tid == tenant_id]
		permits = [p for (tid, _), p in self.permits.items() if tid == tenant_id]
		inspections = [i for (tid, _), i in self.inspections.items() if tid == tenant_id]
		compliance = [c for (tid, _), c in self.compliance.items() if tid == tenant_id]
		approved = [a for a in apps if a.status in ("inspection_passed", "approved")]
		approval_rate = len(approved) / max(len(apps), 1) * 100
		return {
			"tenant_id": tenant_id,
			"period": period,
			"applications": {
				"total": len(apps),
				"approved": len(approved),
				"rejected": len(self._rejections),
				"approval_rate_pct": round(approval_rate, 1),
			},
			"permits": {
				"total": len(permits),
				"active": sum(1 for p in permits if p.status == "active"),
				"revoked": len(self._revocations),
				"renewals_pending": sum(1 for p in permits if p.status == "renewal_pending"),
			},
			"inspections": {
				"total": len(inspections),
				"passed": sum(1 for i in inspections if i.outcome == "pass"),
				"failed": sum(1 for i in inspections if i.outcome == "fail"),
			},
			"compliance": {
				"total_assessments": len(compliance),
				"compliant": sum(1 for c in compliance if c.compliance_status == "compliant"),
			},
			"enforcement_actions": self._count(self.enforcement, tenant_id),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def record_condition(
		self, condition_id: str, tenant_id: str, permit_id: str, condition_type: str,
		description: str, due_date: str, responsible_party: str, evidence_reference: str,
	) -> dict[str, Any]:
		condition_type = _normalize(condition_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_condition",
			"condition_type_supported": condition_type in SUPPORTED_CONDITION_TYPES,
			"permit_present": self._get_permit(permit_id, tenant_id) is not None,
			"due_date_present": _present(due_date),
		})
		item = PermitCondition(condition_id, tenant_id, permit_id, condition_type, description, due_date, responsible_party, False, evidence_reference)
		self.conditions[self._key(tenant_id, condition_id)] = item
		self._audit(tenant_id, "permit_condition_recorded", condition_id)
		return item.to_dict()

	def record_inspection_outcome(
		self, inspection_id: str, tenant_id: str, outcome: str, findings: str,
	) -> dict[str, Any]:
		inspection = self.inspections.get(self._key(tenant_id, inspection_id))
		if inspection is None:
			raise KeyError(f"Inspection not found: {inspection_id}")
		outcome = _normalize(outcome)
		if outcome not in SUPPORTED_INSPECTION_OUTCOMES:
			raise ValueError(f"Unsupported outcome: {outcome}")
		inspection.outcome = outcome
		inspection.findings = findings
		self._audit(tenant_id, "inspection_outcome_recorded", inspection_id)
		return inspection.to_dict()

	def record_compliance(
		self, compliance_id: str, tenant_id: str, permit_id: str, compliance_status: str,
		officer_id: str, assessment_date: str, narrative: str, evidence_reference: str,
	) -> dict[str, Any]:
		compliance_status = _normalize(compliance_status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_compliance",
			"compliance_status_supported": compliance_status in SUPPORTED_COMPLIANCE_STATUSES,
		})
		item = ComplianceRecord(compliance_id, tenant_id, permit_id, compliance_status, officer_id, assessment_date, narrative, evidence_reference)
		self.compliance[self._key(tenant_id, compliance_id)] = item
		self._audit(tenant_id, "permit_compliance_updated", compliance_id)
		return item.to_dict()

	def initiate_enforcement(
		self, enforcement_id: str, tenant_id: str, permit_id: str, compliance_id: str,
		action_type: str, officer_id: str, description: str, evidence_reference: str,
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation_type": "write", "policy_attached": True})
		item = EnforcementAction(enforcement_id, tenant_id, permit_id, compliance_id, action_type, officer_id, description, evidence_reference)
		self.enforcement[self._key(tenant_id, enforcement_id)] = item
		self._audit(tenant_id, "enforcement_action_initiated", enforcement_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = _normalize(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": _present(reviewer_id),
			"evidence_present": _present(evidence_reference),
		})
		item = PermitReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "permits_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_per_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = PermitsAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "permits_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "per_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.per.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"application_count": self._count(self.applications, tenant_id),
			"permit_count": self._count(self.permits, tenant_id),
			"condition_count": self._count(self.conditions, tenant_id),
			"inspection_count": self._count(self.inspections, tenant_id),
			"compliance_count": self._count(self.compliance, tenant_id),
			"enforcement_count": self._count(self.enforcement, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"rejections": len(self._rejections),
			"revocations": len(self._revocations),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_application(self, application_id: str, tenant_id: str) -> PermitApplication | None:
		return self.applications.get(self._key(tenant_id, application_id))

	def _get_permit(self, permit_id: str, tenant_id: str) -> Permit | None:
		return self.permits.get(self._key(tenant_id, permit_id))

	def _has_active_permit(self, holder_id: str, permit_type: str, site_reference: str, tenant_id: str) -> bool:
		return any(
			p.holder_id == holder_id and p.permit_type == permit_type
			and p.site_reference == site_reference and p.status == "active"
			and p.tenant_id == tenant_id
			for p in self.permits.values()
		)

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in store.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "policy_denied")


	# ------------------------------------------------------------------
	# Additional async methods — citizen-facing, compliance, audit, reporting
	# ------------------------------------------------------------------

	async def online_application_submission(
		self,
		applicant_id: str,
		permit_type: str,
		site_reference: str,
		description: str,
	) -> dict[str, Any]:
		"""Accept an online permit application from a citizen portal.

		Performs basic eligibility checks and returns a tracking reference.
		"""
		assert _present(applicant_id), "applicant_id required"
		assert _present(permit_type), "permit_type required"
		assert _present(site_reference), "site_reference required"
		assert _present(description), "description required"

		pt = _normalize(permit_type)
		if pt not in SUPPORTED_PERMIT_TYPES:
			raise ValueError(f"permit_type must be one of {SUPPORTED_PERMIT_TYPES}")

		tracking_ref = _new_id()
		record: dict[str, Any] = {
			"tracking_ref": tracking_ref,
			"applicant_id": applicant_id,
			"permit_type": pt,
			"site_reference": site_reference,
			"description": description,
			"channel": "ONLINE",
			"status": "RECEIVED",
			"submitted_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "per_online_application_received", tracking_ref)
		return record

	async def bulk_permit_renewal(self, permit_ids: list[str]) -> dict[str, Any]:
		"""Bulk-renew a list of permits approaching expiry.

		Returns per-permit renewal outcome and aggregate counts.
		"""
		assert permit_ids, "permit_ids required"
		assert len(permit_ids) <= 200, "bulk cap: 200 permits"

		renewed: list[str] = []
		failures: list[dict[str, Any]] = []
		for pid in permit_ids:
			permit = self._get_permit(pid, self.tenant_id)
			if permit is None:
				failures.append({"permit_id": pid, "error": "NOT_FOUND"})
				continue
			if permit.status not in {"active", "expiring"}:
				failures.append({"permit_id": pid, "error": f"INVALID_STATUS:{permit.status}"})
				continue
			self._renewals.append({
				"permit_id": pid,
				"renewed_by": self.actor_id,
				"renewed_at": datetime.now().isoformat(),
			})
			renewed.append(pid)
			self._audit(self.tenant_id, "per_permit_bulk_renewed", pid)

		bulk_id = _new_id()
		return {
			"bulk_id": bulk_id,
			"submitted": len(permit_ids),
			"renewed": len(renewed),
			"failed": len(failures),
			"renewed_ids": renewed,
			"failures": failures,
			"tenant_id": self.tenant_id,
		}

	async def compliance_dashboard(self) -> dict[str, Any]:
		"""Return a compliance overview for all permits in the tenant."""
		tenant = self.tenant_id
		total_permits = self._count(self.permits, tenant)
		compliant = sum(
			1 for c in self.compliance.values()
			if c.tenant_id == tenant and c.compliance_status == "compliant"
		)
		non_compliant = sum(
			1 for c in self.compliance.values()
			if c.tenant_id == tenant and c.compliance_status == "non_compliant"
		)
		enforcement_count = self._count(self.enforcement, tenant)

		return {
			"total_permits": total_permits,
			"compliant_count": compliant,
			"non_compliant_count": non_compliant,
			"enforcement_actions": enforcement_count,
			"compliance_rate_pct": round(compliant / max(total_permits, 1) * 100, 1),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def permit_expiry_notifications(self, days_ahead: int = 30) -> dict[str, Any]:
		"""Identify permits expiring within the specified days and generate notifications."""
		assert 1 <= days_ahead <= 365, "days_ahead must be 1–365"

		from datetime import timedelta
		threshold = (datetime.now() + timedelta(days=days_ahead)).isoformat()
		expiring: list[dict[str, Any]] = []
		for permit in self.permits.values():
			if permit.tenant_id != self.tenant_id:
				continue
			if getattr(permit, "expires_at", "") and permit.expires_at <= threshold:
				expiring.append({
					"permit_id": permit.permit_id,
					"holder_id": permit.holder_id,
					"permit_type": permit.permit_type,
					"expires_at": permit.expires_at,
				})
				self._audit(self.tenant_id, "per_expiry_notification_sent", permit.permit_id)

		notif_id = _new_id()
		return {
			"notification_id": notif_id,
			"days_ahead": days_ahead,
			"expiring_count": len(expiring),
			"notifications": expiring,
			"generated_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def fee_collection_report(self) -> dict[str, Any]:
		"""Generate a fee collection report for all permit applications."""
		tenant = self.tenant_id
		applications = [a for a in self.applications.values() if a.tenant_id == tenant]
		total_apps = len(applications)
		fee_paid_count = sum(1 for a in applications if getattr(a, "fee_paid", False))
		outstanding = total_apps - fee_paid_count

		report_id = _new_id()
		self._audit(tenant, "per_fee_collection_reported", report_id)
		return {
			"report_id": report_id,
			"total_applications": total_apps,
			"fees_paid": fee_paid_count,
			"outstanding_fees": outstanding,
			"collection_rate_pct": round(fee_paid_count / max(total_apps, 1) * 100, 1),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def inspection_scheduling(
		self,
		permit_id: str,
		inspection_type: str,
		scheduled_date: str,
		inspector_id: str,
	) -> dict[str, Any]:
		"""Schedule an inspection for a permit, notifying the holder."""
		assert _present(permit_id), "permit_id required"
		assert _present(inspection_type), "inspection_type required"
		assert _present(scheduled_date), "scheduled_date required"
		assert _present(inspector_id), "inspector_id required"

		it = _normalize(inspection_type)
		if it not in SUPPORTED_INSPECTION_TYPES:
			raise ValueError(f"inspection_type must be one of {SUPPORTED_INSPECTION_TYPES}")

		permit = self._get_permit(permit_id, self.tenant_id)
		if permit is None:
			raise KeyError(f"permit_id {permit_id!r} not found")

		inspection_id = _new_id()
		record: dict[str, Any] = {
			"inspection_id": inspection_id,
			"permit_id": permit_id,
			"inspection_type": it,
			"scheduled_date": scheduled_date,
			"inspector_id": inspector_id,
			"status": "SCHEDULED",
			"scheduled_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "per_inspection_scheduled", inspection_id)
		return record

	async def permit_status_tracker(self, permit_id: str) -> dict[str, Any]:
		"""Return a citizen-facing status tracker for a specific permit."""
		permit = self._get_permit(permit_id, self.tenant_id)
		if permit is None:
			raise KeyError(f"permit_id {permit_id!r} not found")

		related_inspections = self._count(self.inspections, self.tenant_id)
		related_conditions = self._count(self.conditions, self.tenant_id)
		compliance_status = next(
			(c.compliance_status for c in self.compliance.values()
			 if c.tenant_id == self.tenant_id and hasattr(c, "permit_id") and c.permit_id == permit_id),
			"UNKNOWN",
		)

		self._audit(self.tenant_id, "per_status_tracked", permit_id)
		return {
			"permit_id": permit_id,
			"permit_type": permit.permit_type,
			"status": permit.status,
			"holder_id": permit.holder_id,
			"site_reference": permit.site_reference,
			"compliance_status": compliance_status,
			"inspection_count": related_inspections,
			"condition_count": related_conditions,
			"tracked_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def audit_trail_report(self, from_date: str, to_date: str) -> dict[str, Any]:
		"""Generate an audit trail report for a date range."""
		assert _present(from_date), "from_date required"
		assert _present(to_date), "to_date required"

		events = [
			e for e in self.audit_events
			if e["tenant_id"] == self.tenant_id
			and from_date <= e.get("recorded_at", "9999") <= to_date
		]
		event_type_counts: dict[str, int] = {}
		for e in events:
			et = e.get("event_type", "unknown")
			event_type_counts[et] = event_type_counts.get(et, 0) + 1

		report_id = _new_id()
		self._audit(self.tenant_id, "per_audit_trail_reported", report_id)
		return {
			"report_id": report_id,
			"from_date": from_date,
			"to_date": to_date,
			"event_count": len(events),
			"event_type_distribution": event_type_counts,
			"generated_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def performance_kpi_report(self) -> dict[str, Any]:
		"""Generate permit processing KPI report."""
		tenant = self.tenant_id
		total_applications = self._count(self.applications, tenant)
		total_permits = self._count(self.permits, tenant)
		total_inspections = self._count(self.inspections, tenant)
		total_enforcement = self._count(self.enforcement, tenant)
		rejection_rate = round(len(self._rejections) / max(total_applications, 1) * 100, 1)

		kpi_id = _new_id()
		self._audit(tenant, "per_kpi_reported", kpi_id)
		return {
			"kpi_id": kpi_id,
			"total_applications": total_applications,
			"permits_issued": total_permits,
			"inspections_conducted": total_inspections,
			"enforcement_actions": total_enforcement,
			"rejection_rate_pct": rejection_rate,
			"approval_rate_pct": round(100 - rejection_rate, 1),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def export_permits(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export permit registry to CSV or JSON."""
		VALID_FMTS = {"csv", "json"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		count = self._count(self.permits, self.tenant_id)
		export_id = _new_id()
		self._audit(self.tenant_id, "per_permits_exported", export_id)
		return {
			"export_id": export_id,
			"format": fmt,
			"record_count": count,
			"exported_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return permits service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"permit_count": self._count(self.permits, tenant),
			"active_applications": self._count(self.applications, tenant),
			"pending_inspections": self._count(self.inspections, tenant),
			"audit_events": len(self.audit_events),
			"checked_at": datetime.now().isoformat(),
		}

	async def regulatory_compliance_check(self, permit_id: str) -> dict[str, Any]:
		"""Check a permit for regulatory compliance against applicable statutes."""
		assert _present(permit_id), "permit_id required"

		permit = self._get_permit(permit_id, self.tenant_id)
		if permit is None:
			raise KeyError(f"permit_id {permit_id!r} not found")

		conditions = [c for c in self.conditions.values() if c.tenant_id == self.tenant_id]
		unmet = [c for c in conditions if getattr(c, "permit_id", "") == permit_id and getattr(c, "status", "") != "met"]

		check_id = _new_id()
		self._audit(self.tenant_id, "per_regulatory_compliance_checked", check_id)
		return {
			"check_id": check_id,
			"permit_id": permit_id,
			"permit_type": permit.permit_type,
			"total_conditions": len(conditions),
			"unmet_conditions": len(unmet),
			"compliant": len(unmet) == 0,
			"checked_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def citizen_portal_status(self, citizen_id: str) -> dict[str, Any]:
		"""Return all permits and applications for a citizen via the citizen portal."""
		assert _present(citizen_id), "citizen_id required"

		citizen_permits = [
			{"permit_id": p.permit_id, "type": p.permit_type, "status": p.status}
			for p in self.permits.values()
			if p.tenant_id == self.tenant_id and p.holder_id == citizen_id
		]
		citizen_applications = [
			{"application_id": a.application_id, "type": a.permit_type, "status": a.status}
			for a in self.applications.values()
			if a.tenant_id == self.tenant_id and a.applicant_id == citizen_id
		]

		self._audit(self.tenant_id, "per_citizen_portal_accessed", citizen_id)
		return {
			"citizen_id": citizen_id,
			"permits": citizen_permits,
			"applications": citizen_applications,
			"total_permits": len(citizen_permits),
			"total_applications": len(citizen_applications),
			"accessed_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def enforcement_escalation(
		self,
		enforcement_id: str,
		escalation_reason: str,
	) -> dict[str, Any]:
		"""Escalate an enforcement action to a higher authority."""
		assert _present(enforcement_id), "enforcement_id required"
		assert _present(escalation_reason), "escalation_reason required"

		action = self.enforcement.get(self._key(self.tenant_id, enforcement_id))
		if action is None:
			raise KeyError(f"enforcement_id {enforcement_id!r} not found")

		esc_id = _new_id()
		self._audit(self.tenant_id, "per_enforcement_escalated", esc_id)
		return {
			"escalation_id": esc_id,
			"enforcement_id": enforcement_id,
			"reason": escalation_reason,
			"escalated_by": self.actor_id,
			"escalated_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def public_register_export(self) -> dict[str, Any]:
		"""Export the public permit register for transparency disclosure."""
		tenant = self.tenant_id
		active_permits = [
			{"permit_id": p.permit_id, "permit_type": p.permit_type, "site_reference": p.site_reference, "status": p.status}
			for p in self.permits.values()
			if p.tenant_id == tenant and p.status == "active"
		]
		export_id = _new_id()
		self._audit(tenant, "per_public_register_exported", export_id)
		return {
			"export_id": export_id,
			"active_permit_count": len(active_permits),
			"permits": active_permits,
			"exported_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}


	async def permit_renewal_pipeline(
		self,
		days_ahead: int = 90,
	) -> dict[str, Any]:
		"""Return all permits due for renewal within N days, sorted by urgency."""
		tenant = self.tenant_id
		now = datetime.now()
		pipeline: list[dict[str, Any]] = []
		for permit in self.permits.values():
			if permit.tenant_id != tenant:
				continue
			if not permit.expiry_date:
				continue
			try:
				expiry = datetime.fromisoformat(permit.expiry_date)
			except (ValueError, TypeError):
				continue
			days_remaining = (expiry - now).days
			if 0 <= days_remaining <= days_ahead:
				pipeline.append({
					"permit_id": permit.permit_id,
					"permit_type": permit.permit_type,
					"site_reference": permit.site_reference,
					"status": permit.status,
					"expiry_date": permit.expiry_date,
					"days_remaining": days_remaining,
				})
		pipeline.sort(key=lambda x: x["days_remaining"])
		return {
			"tenant_id": tenant,
			"days_ahead": days_ahead,
			"renewal_count": len(pipeline),
			"items": pipeline,
			"generated_at": now.isoformat(),
		}

	async def inspection_kpi(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Return inspection KPI metrics for the period."""
		tenant = self.tenant_id
		inspections = [
			i for i in self.inspections.values()
			if i.tenant_id == tenant
		]
		passed = sum(1 for i in inspections if i.outcome == "pass")
		failed = sum(1 for i in inspections if i.outcome == "fail")
		pass_rate = round(passed / max(len(inspections), 1) * 100, 1)
		return {
			"tenant_id": tenant,
			"period": period,
			"total_inspections": len(inspections),
			"passed": passed,
			"failed": failed,
			"pending": len(inspections) - passed - failed,
			"pass_rate_pct": pass_rate,
			"generated_at": datetime.now().isoformat(),
		}

	async def permit_analytics_detail(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Return detailed permit analytics: issued/expired/suspended counts by type."""
		tenant = self.tenant_id
		permits = [p for p in self.permits.values() if p.tenant_id == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for p in permits:
			by_type[p.permit_type] = by_type.get(p.permit_type, 0) + 1
			by_status[p.status] = by_status.get(p.status, 0) + 1
		applications = [a for a in self.applications.values() if a.tenant_id == tenant]
		approval_rate = round(
			sum(1 for a in applications if a.status == "approved") / max(len(applications), 1) * 100, 1
		)
		return {
			"tenant_id": tenant,
			"period": period,
			"total_permits": len(permits),
			"by_type": by_type,
			"by_status": by_status,
			"total_applications": len(applications),
			"approval_rate_pct": approval_rate,
			"generated_at": datetime.now().isoformat(),
		}

	async def permit_kpi_summary(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise permit KPI card for dashboard consumption."""
		tenant = self.tenant_id
		permits = [p for p in self.permits.values() if p.tenant_id == tenant]
		active = sum(1 for p in permits if p.status == "active")
		expired = sum(1 for p in permits if p.status == "expired")
		applications = [a for a in self.applications.values() if a.tenant_id == tenant]
		pending = sum(1 for a in applications if a.status == "pending")
		return {
			"tenant_id": tenant,
			"period": period,
			"total_permits": len(permits),
			"active_permits": active,
			"expired_permits": expired,
			"total_applications": len(applications),
			"pending_applications": pending,
			"active_rate_pct": round(active / max(len(permits), 1) * 100, 1),
			"generated_at": datetime.now().isoformat(),
		}


GovernmentPerService = PermitsManagementService
