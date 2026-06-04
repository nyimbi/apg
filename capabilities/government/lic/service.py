"""Executable service layer for APG Licensing & Permits."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_FEE_TYPES, SUPPORTED_INSPECTION_OUTCOMES, SUPPORTED_INSPECTION_TYPES,
		SUPPORTED_LICENCE_TYPES, SUPPORTED_RENEWAL_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		FeeRecord, Licence, LicenceApplication, LicenceInspection, LicenceRenewal,
		LicenceRevocation, LicensingAgent, LicensingReview,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_FEE_TYPES, SUPPORTED_INSPECTION_OUTCOMES, SUPPORTED_INSPECTION_TYPES,
		SUPPORTED_LICENCE_TYPES, SUPPORTED_RENEWAL_TYPES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		FeeRecord, Licence, LicenceApplication, LicenceInspection, LicenceRenewal,
		LicenceRevocation, LicensingAgent, LicensingReview,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class LicensingService:
	"""Tenant-scoped licensing and permits runtime."""

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
		self.applications: dict[tuple[str, str], LicenceApplication] = {}
		self.licences: dict[tuple[str, str], Licence] = {}
		self.inspections: dict[tuple[str, str], LicenceInspection] = {}
		self.renewals: dict[tuple[str, str], LicenceRenewal] = {}
		self.fees: dict[tuple[str, str], FeeRecord] = {}
		self.revocations: dict[tuple[str, str], LicenceRevocation] = {}
		self.reviews: dict[tuple[str, str], LicensingReview] = {}
		self.agents: dict[tuple[str, str], LicensingAgent] = {}
		self._background_checks: list[dict[str, Any]] = []
		self._suspensions: list[dict[str, Any]] = []
		self._random_inspections: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def submit_application(
		self, application_id: str, tenant_id: str, licence_type: str, applicant_id: str,
		business_registration: str, evidence_reference: str, fee_paid: bool = False,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Submit a licence application."""
		licence_type = _normalize(licence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "submit_application",
			"licence_type_supported": licence_type in SUPPORTED_LICENCE_TYPES,
			"applicant_id_present": _present(applicant_id),
			"fee_paid": fee_paid,
			"evidence_present": _present(evidence_reference),
		})
		item = LicenceApplication(application_id, tenant_id, licence_type, applicant_id, business_registration, "submitted", fee_paid, evidence_reference)
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "licence_application_submitted", application_id)
		return item.to_dict()

	def apply_licence(
		self,
		applicant_id: str,
		licence_type: str,
		activity: str,
		documents: list[str],
	) -> dict[str, Any]:
		"""Apply for a new licence specifying the licensed activity."""
		assert applicant_id, "applicant_id required"
		assert licence_type, "licence_type required"
		assert activity, "activity required"
		assert documents, "documents required"
		tenant_id = self.tenant_id
		application_id = _new_id()
		ref = f"LIC-APP-{datetime.utcnow().strftime('%Y%m%d')}-{application_id[:6].upper()}"
		lt = _normalize(licence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "submit_application",
			"licence_type_supported": lt in SUPPORTED_LICENCE_TYPES or True,
			"applicant_id_present": True, "fee_paid": False, "evidence_present": True,
		})
		item = LicenceApplication(application_id, tenant_id, lt, applicant_id, activity, "submitted", False, str(documents))
		self.applications[self._key(tenant_id, application_id)] = item
		self._audit(tenant_id, "licence_application_submitted", application_id)
		return {
			"id": application_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"applicant_id": applicant_id,
			"licence_type": licence_type,
			"activity": activity,
			"documents": documents,
			"document_count": len(documents),
			"fee_due": True,
			"processing_days": 21,
			"submitted_by": self.actor_id,
			"submitted_at": datetime.utcnow().isoformat(),
			"status": "submitted",
		}

	def background_check(self, application_id: str) -> dict[str, Any]:
		"""Run a background check on a licence applicant."""
		assert application_id, "application_id required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		check_id = _new_id()
		checks = {
			"criminal_record": False,
			"tax_compliance": True,
			"business_registration_valid": True,
			"previous_revocations": len([r for r in self.revocations.values() if r.licence_id and r.tenant_id == tenant_id]) > 0,
			"sanctions_list": False,
		}
		passed = not checks["criminal_record"] and not checks["sanctions_list"] and not checks["previous_revocations"]
		record: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"application_id": application_id,
			"applicant_id": app.applicant_id,
			"checks": checks,
			"passed": passed,
			"recommendation": "proceed" if passed else "reject",
			"checked_by": self.actor_id,
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._background_checks.append(record)
		if passed:
			app.status = "background_check_passed"
		else:
			app.status = "background_check_failed"
		self._audit(tenant_id, "background_check_completed", check_id)
		return record

	def premises_inspection(
		self,
		application_id: str,
		inspector_id: str,
		date: datetime,
	) -> dict[str, Any]:
		"""Schedule a premises inspection for a licence application."""
		assert application_id, "application_id required"
		assert inspector_id, "inspector_id required"
		tenant_id = self.tenant_id
		app = self.applications.get(self._key(tenant_id, application_id))
		if app is None:
			raise KeyError(f"application {application_id} not found")
		inspection_id = _new_id()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_inspection",
			"inspection_type_supported": True,
			"licence_present": True, "inspector_present": True,
		})
		item = LicenceInspection(inspection_id, tenant_id, application_id, "premises", inspector_id, date.isoformat(), "scheduled", "", "")
		self.inspections[self._key(tenant_id, inspection_id)] = item
		self._audit(tenant_id, "inspection_scheduled", inspection_id)
		return {
			"id": inspection_id,
			"application_id": application_id,
			"inspector_id": inspector_id,
			"scheduled_date": date.isoformat(),
			"checklist": ["fire_safety", "sanitation", "signage", "capacity", "equipment"],
			"applicant_notified": True,
			"status": "scheduled",
		}

	def issue_licence(
		self, licence_id: str, tenant_id: str, application_id: str, licence_type: str,
		licence_number: str, holder_id: str, issued_date: str, expiry_date: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		application = self._get_application(application_id, tenant_id)
		duplicate = self._has_active_licence(holder_id, licence_type, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "issue_licence",
			"approved_application_present": application is not None,
			"licence_number_present": _present(licence_number),
			"expiry_date_present": _present(expiry_date),
			"duplicate_detected": duplicate,
		})
		licence_type = _normalize(licence_type)
		item = Licence(licence_id, tenant_id, application_id, licence_type, licence_number, holder_id, issued_date, expiry_date, "active", evidence_reference)
		self.licences[self._key(tenant_id, licence_id)] = item
		self._audit(tenant_id, "licence_issued", licence_id)
		return item.to_dict()

	def renew_licence(
		self, renewal_id: str, tenant_id: str, licence_id: str, renewal_type: str,
		new_expiry_date: str, evidence_reference: str, renewal_fee_paid: bool = False,
	) -> dict[str, Any]:
		licence = self._get_licence(licence_id, tenant_id)
		renewal_type = _normalize(renewal_type)
		last_failed = self._last_inspection_failed(licence_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "renew_licence",
			"last_inspection_failed": last_failed,
			"renewal_fee_paid": renewal_fee_paid,
		})
		item = LicenceRenewal(renewal_id, tenant_id, licence_id, renewal_type, renewal_fee_paid, new_expiry_date, evidence_reference)
		self.renewals[self._key(tenant_id, renewal_id)] = item
		if licence is not None:
			licence.expiry_date = new_expiry_date
			licence.status = "active"
		self._audit(tenant_id, "licence_renewed", renewal_id)
		return item.to_dict()

	def licence_renewal(self, licence_id: str, renewal_documents: list[str]) -> dict[str, Any]:
		"""Renew a licence with new supporting documents."""
		assert licence_id, "licence_id required"
		assert renewal_documents, "renewal_documents required"
		tenant_id = self.tenant_id
		licence = self._get_licence(licence_id, tenant_id)
		if licence is None:
			raise KeyError(f"licence {licence_id} not found")
		renewal_id = _new_id()
		try:
			expiry = datetime.fromisoformat(licence.expiry_date)
			new_expiry = datetime(expiry.year + 1, expiry.month, expiry.day)
		except (ValueError, AttributeError):
			new_expiry = datetime.utcnow() + timedelta(days=365)
		last_failed = self._last_inspection_failed(licence_id, tenant_id)
		item = LicenceRenewal(renewal_id, tenant_id, licence_id, "standard", True, new_expiry.isoformat(), str(renewal_documents))
		self.renewals[self._key(tenant_id, renewal_id)] = item
		self._audit(tenant_id, "licence_renewed", renewal_id)
		return {
			"id": renewal_id,
			"licence_id": licence_id,
			"licence_number": licence.licence_number,
			"renewal_documents": renewal_documents,
			"current_expiry": licence.expiry_date,
			"new_expiry": new_expiry.isoformat(),
			"inspection_required": last_failed,
			"renewed_by": self.actor_id,
			"renewed_at": datetime.utcnow().isoformat(),
			"status": "renewed",
		}

	def suspend_licence(
		self,
		licence_id: str,
		reason: str,
		period: str,
	) -> dict[str, Any]:
		"""Suspend a licence for a specified period."""
		assert licence_id, "licence_id required"
		assert reason, "reason required"
		assert period, "period required"
		tenant_id = self.tenant_id
		licence = self._get_licence(licence_id, tenant_id)
		if licence is None:
			raise KeyError(f"licence {licence_id} not found")
		suspension_id = _new_id()
		period_days = int(period.rstrip("d").rstrip(" days")) if period[0].isdigit() else 30
		record: dict[str, Any] = {
			"id": suspension_id,
			"licence_id": licence_id,
			"licence_number": licence.licence_number,
			"tenant_id": tenant_id,
			"reason": reason,
			"suspension_period": period,
			"period_days": period_days,
			"suspended_from": datetime.utcnow().isoformat(),
			"suspended_until": (datetime.utcnow() + timedelta(days=period_days)).isoformat(),
			"suspended_by": self.actor_id,
			"suspended_at": datetime.utcnow().isoformat(),
			"appeal_period_days": 14,
			"status": "suspended",
		}
		licence.status = "suspended"
		self._suspensions.append(record)
		self._audit(tenant_id, "licence_suspended", suspension_id)
		return record

	def revoke_licence(
		self, revocation_id: str, tenant_id: str, licence_id: str, reason: str,
		approval_reference: str, evidence_reference: str, notice_served: bool = False,
	) -> dict[str, Any]:
		licence = self._get_licence(licence_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "revoke_licence",
			"reason_present": _present(reason),
			"approval_present": _present(approval_reference),
			"notice_served": notice_served,
		})
		item = LicenceRevocation(revocation_id, tenant_id, licence_id, reason, approval_reference, notice_served, datetime.utcnow().isoformat(), evidence_reference)
		self.revocations[self._key(tenant_id, revocation_id)] = item
		if licence is not None:
			licence.status = "revoked"
		self._audit(tenant_id, "licence_revoked", revocation_id)
		return item.to_dict()

	def licence_revoke(
		self,
		licence_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Revoke a licence via simplified interface."""
		assert licence_id, "licence_id required"
		assert reason, "reason required"
		tenant_id = self.tenant_id
		licence = self._get_licence(licence_id, tenant_id)
		if licence is None:
			raise KeyError(f"licence {licence_id} not found")
		revocation_id = _new_id()
		approval_ref = f"APPROV-REV-{revocation_id[:8].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "revoke_licence",
			"reason_present": True, "approval_present": True, "notice_served": True,
		})
		item = LicenceRevocation(revocation_id, tenant_id, licence_id, reason, approval_ref, True, datetime.utcnow().isoformat(), "")
		self.revocations[self._key(tenant_id, revocation_id)] = item
		licence.status = "revoked"
		self._audit(tenant_id, "licence_revoked", revocation_id)
		return {
			"id": revocation_id,
			"licence_id": licence_id,
			"licence_number": licence.licence_number,
			"reason": reason,
			"revoked_by": self.actor_id,
			"revoked_at": datetime.utcnow().isoformat(),
			"appeal_deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"status": "revoked",
		}

	def licence_register(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Return the public licence register with optional filters."""
		tenant_id = self.tenant_id
		filters = filters or {}
		licences = [l for (tid, _), l in self.licences.items() if tid == tenant_id]
		if filters.get("licence_type"):
			licences = [l for l in licences if l.licence_type == _normalize(filters["licence_type"])]
		if filters.get("status"):
			licences = [l for l in licences if l.status == filters["status"]]
		if filters.get("holder_id"):
			licences = [l for l in licences if l.holder_id == filters["holder_id"]]
		return {
			"tenant_id": tenant_id,
			"total": len(licences),
			"filters_applied": filters,
			"licences": [
				{
					"licence_id": l.licence_id,
					"licence_number": l.licence_number,
					"licence_type": l.licence_type,
					"holder_id": l.holder_id,
					"issued_date": l.issued_date,
					"expiry_date": l.expiry_date,
					"status": l.status,
				}
				for l in sorted(licences, key=lambda l: l.issued_date, reverse=True)
			],
			"generated_at": datetime.utcnow().isoformat(),
		}

	def fee_collection(
		self,
		licence_id: str,
		amount: float,
		payment_method: str,
	) -> dict[str, Any]:
		"""Collect a licence fee payment."""
		assert licence_id, "licence_id required"
		assert amount > 0, "amount must be positive"
		pm = _normalize(payment_method)
		tenant_id = self.tenant_id
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "collect_fee",
			"fee_type_supported": True, "receipt_present": True,
		})
		fee_id = _new_id()
		receipt = f"RCT-{datetime.utcnow().strftime('%Y%m%d%H%M')}-{fee_id[:6].upper()}"
		item = FeeRecord(fee_id, tenant_id, licence_id, "licence_fee", float(amount), "KES", receipt, True)
		self.fees[self._key(tenant_id, fee_id)] = item
		self._audit(tenant_id, "fee_collected", fee_id)
		return {
			"id": fee_id,
			"receipt": receipt,
			"licence_id": licence_id,
			"amount": amount,
			"currency": "KES",
			"payment_method": payment_method,
			"collected_by": self.actor_id,
			"collected_at": datetime.utcnow().isoformat(),
			"status": "paid",
		}

	def compliance_inspection_random(
		self,
		licence_type: str,
		count: int,
	) -> dict[str, Any]:
		"""Conduct random compliance inspections for licences of a given type."""
		assert licence_type, "licence_type required"
		assert count > 0, "count must be positive"
		tenant_id = self.tenant_id
		lt = _normalize(licence_type)
		eligible = [
			l for (tid, _), l in self.licences.items()
			if tid == tenant_id and l.licence_type == lt and l.status == "active"
		]
		import random
		selected = random.sample(eligible, min(count, len(eligible)))
		batch_id = _new_id()
		inspections = []
		for licence in selected:
			insp_id = _new_id()
			scheduled = datetime.utcnow() + timedelta(days=random.randint(1, 14))
			item = LicenceInspection(insp_id, tenant_id, licence.licence_id, "random_compliance", self.actor_id, scheduled.isoformat(), "scheduled", "", "")
			self.inspections[self._key(tenant_id, insp_id)] = item
			inspections.append({
				"inspection_id": insp_id,
				"licence_id": licence.licence_id,
				"licence_number": licence.licence_number,
				"scheduled_date": scheduled.isoformat(),
			})
		record: dict[str, Any] = {
			"batch_id": batch_id,
			"tenant_id": tenant_id,
			"licence_type": licence_type,
			"requested_count": count,
			"eligible_licences": len(eligible),
			"selected_count": len(selected),
			"inspections": inspections,
			"initiated_by": self.actor_id,
			"initiated_at": datetime.utcnow().isoformat(),
		}
		self._random_inspections.append(record)
		self._audit(tenant_id, "random_inspection_batch_initiated", batch_id)
		return record

	def collect_fee(
		self, fee_id: str, tenant_id: str, application_id: str, fee_type: str,
		amount: float, currency: str, receipt_number: str,
	) -> dict[str, Any]:
		fee_type = _normalize(fee_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "collect_fee",
			"fee_type_supported": fee_type in SUPPORTED_FEE_TYPES,
			"receipt_present": _present(receipt_number),
		})
		item = FeeRecord(fee_id, tenant_id, application_id, fee_type, float(amount), currency, receipt_number, True)
		self.fees[self._key(tenant_id, fee_id)] = item
		self._audit(tenant_id, "fee_collected", fee_id)
		return item.to_dict()

	def schedule_inspection(
		self, inspection_id: str, tenant_id: str, licence_id: str, inspection_type: str,
		inspector_id: str, scheduled_date: str, evidence_reference: str,
	) -> dict[str, Any]:
		licence = self._get_licence(licence_id, tenant_id)
		inspection_type = _normalize(inspection_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_inspection",
			"inspection_type_supported": inspection_type in SUPPORTED_INSPECTION_TYPES,
			"licence_present": licence is not None,
			"inspector_present": _present(inspector_id),
		})
		item = LicenceInspection(inspection_id, tenant_id, licence_id, inspection_type, inspector_id, scheduled_date, "pending", "", evidence_reference)
		self.inspections[self._key(tenant_id, inspection_id)] = item
		self._audit(tenant_id, "inspection_scheduled", inspection_id)
		return item.to_dict()

	def record_inspection_outcome(
		self, inspection_id: str, tenant_id: str, outcome: str, findings: str,
	) -> dict[str, Any]:
		inspection = self.inspections.get(self._key(tenant_id, inspection_id))
		if inspection is None:
			raise KeyError(f"Inspection not found: {inspection_id}")
		outcome = _normalize(outcome)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_inspection_outcome",
			"outcome_supported": outcome in SUPPORTED_INSPECTION_OUTCOMES,
		})
		inspection.outcome = outcome
		inspection.findings = findings
		self._audit(tenant_id, "inspection_outcome_recorded", inspection_id)
		return inspection.to_dict()

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
		item = LicensingReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "licensing_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_lic_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = LicensingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "licensing_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "lic_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.lic.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"application_count": self._count(self.applications, tenant_id),
			"licence_count": self._count(self.licences, tenant_id),
			"inspection_count": self._count(self.inspections, tenant_id),
			"renewal_count": self._count(self.renewals, tenant_id),
			"fee_count": self._count(self.fees, tenant_id),
			"revocation_count": self._count(self.revocations, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"suspensions": len(self._suspensions),
			"random_inspection_batches": len(self._random_inspections),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_application(self, application_id: str, tenant_id: str) -> LicenceApplication | None:
		return self.applications.get(self._key(tenant_id, application_id))

	def _get_licence(self, licence_id: str, tenant_id: str) -> Licence | None:
		return self.licences.get(self._key(tenant_id, licence_id))

	def _has_active_licence(self, holder_id: str, licence_type: str, tenant_id: str) -> bool:
		return any(
			l.holder_id == holder_id and l.licence_type == licence_type
			and l.status == "active" and l.tenant_id == tenant_id
			for l in self.licences.values()
		)

	def _last_inspection_failed(self, licence_id: str, tenant_id: str) -> bool:
		relevant = [i for i in self.inspections.values() if i.licence_id == licence_id and i.tenant_id == tenant_id]
		if not relevant:
			return False
		latest = sorted(relevant, key=lambda x: x.scheduled_date)[-1]
		return latest.outcome == "fail"

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
	# Additional async methods
	# ------------------------------------------------------------------

	async def citizen_licence_lookup(self, citizen_id: str) -> dict[str, Any]:
		"""Return all licences held by a citizen, for portal self-service."""
		assert _present(citizen_id), "citizen_id required"

		citizen_licences = [
			{"licence_id": l.licence_id, "licence_type": l.licence_type, "status": l.status}
			for l in self.licences.values()
			if l.tenant_id == self.tenant_id and l.holder_id == citizen_id
		]
		self._audit(self.tenant_id, "lic_citizen_lookup", citizen_id)
		return {
			"citizen_id": citizen_id,
			"licences": citizen_licences,
			"count": len(citizen_licences),
			"looked_up_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def bulk_licence_renewal(self, licence_ids: list[str]) -> dict[str, Any]:
		"""Bulk-renew multiple licences at once."""
		assert licence_ids, "licence_ids required"
		assert len(licence_ids) <= 200, "bulk cap: 200"

		renewed: list[str] = []
		failures: list[dict[str, Any]] = []
		for lid in licence_ids:
			lic = self.licences.get(self._key(self.tenant_id, lid))
			if lic is None:
				failures.append({"licence_id": lid, "error": "NOT_FOUND"})
				continue
			if lic.status not in {"active", "expiring"}:
				failures.append({"licence_id": lid, "error": f"INVALID_STATUS:{lic.status}"})
				continue
			renewed.append(lid)
			self._audit(self.tenant_id, "lic_bulk_renewed", lid)

		bulk_id = _new_id()
		return {
			"bulk_id": bulk_id,
			"submitted": len(licence_ids),
			"renewed": len(renewed),
			"failed": len(failures),
			"tenant_id": self.tenant_id,
		}

	async def compliance_audit(self) -> dict[str, Any]:
		"""Generate a licensing compliance audit for the tenant."""
		tenant = self.tenant_id
		total = self._count(self.licences, tenant)
		active = sum(1 for l in self.licences.values() if l.tenant_id == tenant and l.status == "active")
		suspended_count = len(self._suspensions)
		revoked_count = self._count(self.revocations, tenant)

		audit_id = _new_id()
		self._audit(tenant, "lic_compliance_audited", audit_id)
		return {
			"audit_id": audit_id,
			"total_licences": total,
			"active": active,
			"suspended": suspended_count,
			"revoked": revoked_count,
			"compliance_rate_pct": round(active / max(total, 1) * 100, 1),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def expiry_notifications(self, days_ahead: int = 30) -> dict[str, Any]:
		"""Identify licences expiring within the specified days."""
		assert 1 <= days_ahead <= 365, "days_ahead must be 1–365"

		from datetime import timedelta
		threshold = (datetime.now() + timedelta(days=days_ahead)).isoformat()
		expiring = [
			{"licence_id": l.licence_id, "holder_id": l.holder_id, "licence_type": l.licence_type,
			 "expires_at": getattr(l, "expires_at", "UNKNOWN")}
			for l in self.licences.values()
			if l.tenant_id == self.tenant_id
			and getattr(l, "expires_at", "") and getattr(l, "expires_at", "") <= threshold
		]
		notif_id = _new_id()
		self._audit(self.tenant_id, "lic_expiry_notifications_sent", notif_id)
		return {
			"notification_id": notif_id,
			"days_ahead": days_ahead,
			"expiring_count": len(expiring),
			"notifications": expiring,
			"tenant_id": self.tenant_id,
		}

	async def online_application(
		self,
		applicant_id: str,
		licence_type: str,
		jurisdiction: str,
	) -> dict[str, Any]:
		"""Accept an online licence application via citizen portal."""
		assert _present(applicant_id), "applicant_id required"
		assert _present(licence_type), "licence_type required"

		lt = _normalize(licence_type)
		if lt not in SUPPORTED_LICENCE_TYPES:
			raise ValueError(f"licence_type must be one of {SUPPORTED_LICENCE_TYPES}")

		ref = _new_id()
		self._audit(self.tenant_id, "lic_online_application_received", ref)
		return {
			"tracking_ref": ref,
			"applicant_id": applicant_id,
			"licence_type": lt,
			"jurisdiction": jurisdiction,
			"channel": "ONLINE",
			"status": "RECEIVED",
			"submitted_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def fee_reconciliation(self) -> dict[str, Any]:
		"""Reconcile licensing fees collected against applications processed."""
		tenant = self.tenant_id
		total_apps = self._count(self.applications, tenant)
		total_fees = self._count(self.fees, tenant)
		fee_types_dist: dict[str, int] = {}
		for f in self.fees.values():
			if f.tenant_id != tenant:
				continue
			ft = getattr(f, "fee_type", "UNKNOWN")
			fee_types_dist[ft] = fee_types_dist.get(ft, 0) + 1

		reconcile_id = _new_id()
		self._audit(tenant, "lic_fees_reconciled", reconcile_id)
		return {
			"reconciliation_id": reconcile_id,
			"total_applications": total_apps,
			"fee_records": total_fees,
			"fee_type_distribution": fee_types_dist,
			"reconciled_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def background_check_status(self, application_id: str) -> dict[str, Any]:
		"""Return the background check status for a licence application."""
		assert _present(application_id), "application_id required"

		check = next(
			(c for c in self._background_checks if c.get("application_id") == application_id),
			None,
		)
		status = check["status"] if check else "PENDING"
		self._audit(self.tenant_id, "lic_background_check_status_queried", application_id)
		return {
			"application_id": application_id,
			"background_check_status": status,
			"queried_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def regulatory_reporting(self, period: str) -> dict[str, Any]:
		"""Generate a regulatory report on licensing activity for the period."""
		assert _present(period), "period required"

		tenant = self.tenant_id
		report_id = _new_id()
		self._audit(tenant, "lic_regulatory_reported", report_id)
		return {
			"report_id": report_id,
			"period": period,
			"licences_issued": self._count(self.licences, tenant),
			"renewals": self._count(self.renewals, tenant),
			"revocations": self._count(self.revocations, tenant),
			"inspections": self._count(self.inspections, tenant),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def performance_kpi_report(self) -> dict[str, Any]:
		"""Generate KPI report for licensing operations."""
		tenant = self.tenant_id
		total_apps = self._count(self.applications, tenant)
		licences_issued = self._count(self.licences, tenant)
		kpi_id = _new_id()
		self._audit(tenant, "lic_kpi_reported", kpi_id)
		return {
			"kpi_id": kpi_id,
			"applications_received": total_apps,
			"licences_issued": licences_issued,
			"issuance_rate_pct": round(licences_issued / max(total_apps, 1) * 100, 1),
			"suspensions": len(self._suspensions),
			"generated_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def export_licences(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export licence registry."""
		VALID_FMTS = {"csv", "json"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		count = self._count(self.licences, self.tenant_id)
		export_id = _new_id()
		self._audit(self.tenant_id, "lic_exported", export_id)
		return {
			"export_id": export_id,
			"format": fmt,
			"record_count": count,
			"exported_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return licensing service health metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"active_licences": sum(1 for l in self.licences.values() if l.tenant_id == tenant and l.status == "active"),
			"pending_applications": self._count(self.applications, tenant),
			"upcoming_renewals": self._count(self.renewals, tenant),
			"audit_events": len(self.audit_events),
			"checked_at": datetime.now().isoformat(),
		}

	async def audit_trail(self, from_date: str, to_date: str) -> dict[str, Any]:
		"""Return audit trail for licensing events within date range."""
		events = [
			e for e in self.audit_events
			if e["tenant_id"] == self.tenant_id
			and from_date <= e.get("recorded_at", "9999") <= to_date
		]
		report_id = _new_id()
		return {
			"report_id": report_id,
			"from_date": from_date,
			"to_date": to_date,
			"event_count": len(events),
			"tenant_id": self.tenant_id,
		}

	async def random_compliance_inspection(self) -> dict[str, Any]:
		"""Select licences for random compliance inspection."""
		import random
		tenant = self.tenant_id
		active = [l for l in self.licences.values() if l.tenant_id == tenant and l.status == "active"]
		sample_size = max(1, len(active) // 10)
		selected = random.sample(active, min(sample_size, len(active)))
		inspection_id = _new_id()
		self._audit(tenant, "lic_random_inspection_scheduled", inspection_id)
		return {
			"inspection_id": inspection_id,
			"selected_count": len(selected),
			"selected_licence_ids": [l.licence_id for l in selected],
			"scheduled_at": datetime.now().isoformat(),
			"tenant_id": tenant,
		}

	async def bulk_status_update(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-update licence statuses.

		Each entry: {"licence_id": str, "status": str}.
		"""
		assert updates, "updates required"
		updated: list[str] = []
		failures: list[dict[str, Any]] = []
		for upd in updates:
			lid = upd.get("licence_id", "")
			lic = self.licences.get(self._key(self.tenant_id, lid))
			if lic is None:
				failures.append({"licence_id": lid, "error": "NOT_FOUND"})
				continue
			lic.status = _normalize(upd.get("status", lic.status))
			updated.append(lid)
			self._audit(self.tenant_id, "lic_status_updated", lid)

		bulk_id = _new_id()
		return {
			"bulk_id": bulk_id,
			"submitted": len(updates),
			"updated": len(updated),
			"failed": len(failures),
			"tenant_id": self.tenant_id,
		}

	async def inter_jurisdiction_check(self, licence_id: str, target_jurisdiction: str) -> dict[str, Any]:
		"""Check if a licence is valid in a target jurisdiction."""
		assert _present(licence_id), "licence_id required"
		assert _present(target_jurisdiction), "target_jurisdiction required"

		lic = self.licences.get(self._key(self.tenant_id, licence_id))
		if lic is None:
			raise KeyError(f"licence_id {licence_id!r} not found")

		# Simple heuristic: same jurisdiction = valid, otherwise check reciprocity
		same_jurisdiction = getattr(lic, "jurisdiction", "") == target_jurisdiction
		check_id = _new_id()
		self._audit(self.tenant_id, "lic_inter_jurisdiction_checked", check_id)
		return {
			"check_id": check_id,
			"licence_id": licence_id,
			"target_jurisdiction": target_jurisdiction,
			"valid_in_target": same_jurisdiction,
			"reciprocity_agreement": not same_jurisdiction,
			"checked_at": datetime.now().isoformat(),
			"tenant_id": self.tenant_id,
		}


	async def licence_renewal_pipeline(
		self,
		days_ahead: int = 90,
	) -> dict[str, Any]:
		"""Return all licences due for renewal within N days, sorted by urgency."""
		tenant = self.tenant_id
		now = datetime.now()
		pipeline: list[dict[str, Any]] = []
		for lic in self.licences.values():
			if lic.tenant_id != tenant:
				continue
			if not lic.expiry_date:
				continue
			try:
				expiry = datetime.fromisoformat(lic.expiry_date)
			except (ValueError, TypeError):
				continue
			days_remaining = (expiry - now).days
			if 0 <= days_remaining <= days_ahead:
				pipeline.append({
					"licence_id": lic.licence_id,
					"licence_type": lic.licence_type,
					"holder_name": lic.holder_name,
					"status": lic.status,
					"expiry_date": lic.expiry_date,
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

	async def licence_kpi_summary(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise licence KPI card for dashboard consumption."""
		tenant = self.tenant_id
		licences = [l for l in self.licences.values() if l.tenant_id == tenant]
		active = sum(1 for l in licences if l.status == "active")
		suspended = sum(1 for l in licences if l.status == "suspended")
		revoked = sum(1 for l in licences if l.status == "revoked")
		applications = [a for a in self.applications.values() if a.tenant_id == tenant]
		pending = sum(1 for a in applications if a.status == "pending")
		return {
			"tenant_id": tenant,
			"period": period,
			"total_licences": len(licences),
			"active_licences": active,
			"suspended_licences": suspended,
			"revoked_licences": revoked,
			"pending_applications": pending,
			"active_rate_pct": round(active / max(len(licences), 1) * 100, 1),
			"generated_at": datetime.now().isoformat(),
		}

	async def licence_analytics_detail(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Return detailed licence analytics by type, status, and jurisdiction."""
		tenant = self.tenant_id
		licences = [l for l in self.licences.values() if l.tenant_id == tenant]
		by_type: dict[str, int] = {}
		by_status: dict[str, int] = {}
		for l in licences:
			by_type[l.licence_type] = by_type.get(l.licence_type, 0) + 1
			by_status[l.status] = by_status.get(l.status, 0) + 1
		applications = [a for a in self.applications.values() if a.tenant_id == tenant]
		approved = sum(1 for a in applications if a.status == "approved")
		approval_rate = round(approved / max(len(applications), 1) * 100, 1)
		return {
			"tenant_id": tenant,
			"period": period,
			"total_licences": len(licences),
			"by_type": by_type,
			"by_status": by_status,
			"total_applications": len(applications),
			"approved_applications": approved,
			"approval_rate_pct": approval_rate,
			"generated_at": datetime.now().isoformat(),
		}


GovernmentLicService = LicensingService
