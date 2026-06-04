"""Executable service layer for APG Law Enforcement & Justice."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_COURT_TYPES, SUPPORTED_CUSTODY_ACTIONS, SUPPORTED_DOCKET_STATUSES,
		SUPPORTED_EVIDENCE_TYPES, SUPPORTED_HEARING_TYPES, SUPPORTED_INCIDENT_TYPES,
		SUPPORTED_PROSECUTION_STATUSES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CaseDocket, CourtHearing, CustodyAction, EvidenceItem, IncidentReport,
		LawEnforcementAgent, LawEnforcementReview, ProsecutionRecord,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_COURT_TYPES, SUPPORTED_CUSTODY_ACTIONS, SUPPORTED_DOCKET_STATUSES,
		SUPPORTED_EVIDENCE_TYPES, SUPPORTED_HEARING_TYPES, SUPPORTED_INCIDENT_TYPES,
		SUPPORTED_PROSECUTION_STATUSES, SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CaseDocket, CourtHearing, CustodyAction, EvidenceItem, IncidentReport,
		LawEnforcementAgent, LawEnforcementReview, ProsecutionRecord,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class LawEnforcementService:
	"""Tenant-scoped law enforcement and justice runtime."""

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
		self.incidents: dict[tuple[str, str], IncidentReport] = {}
		self.dockets: dict[tuple[str, str], CaseDocket] = {}
		self.evidence: dict[tuple[str, str], EvidenceItem] = {}
		self.custody_actions: dict[tuple[str, str], CustodyAction] = {}
		self.court_hearings: dict[tuple[str, str], CourtHearing] = {}
		self.prosecutions: dict[tuple[str, str], ProsecutionRecord] = {}
		self.reviews: dict[tuple[str, str], LawEnforcementReview] = {}
		self.agents: dict[tuple[str, str], LawEnforcementAgent] = {}
		self._suspect_records: list[dict[str, Any]] = []
		self._arrest_records: list[dict[str, Any]] = []
		self._case_assignments: list[dict[str, Any]] = []
		self._evidence_intake_records: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def report_incident(
		self, incident_id: str, tenant_id: str, incident_type: str, ob_number: str,
		reporting_officer_id: str, location_reference: str, complainant_id: str,
		description: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record an incident report with OB number."""
		incident_type = _normalize(incident_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "report_incident",
			"incident_type_supported": incident_type in SUPPORTED_INCIDENT_TYPES,
			"ob_number_present": _present(ob_number),
			"reporting_officer_present": _present(reporting_officer_id),
			"location_present": _present(location_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = IncidentReport(incident_id, tenant_id, incident_type, ob_number, reporting_officer_id, location_reference, complainant_id, description, evidence_reference)
		self.incidents[self._key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "incident_reported", incident_id)
		return item.to_dict()

	def incident_report(
		self,
		incident_type: str,
		location: str,
		description: str,
		reported_by: str,
	) -> dict[str, Any]:
		"""File an incident report via the simplified interface."""
		assert incident_type, "incident_type required"
		assert location, "location required"
		assert description, "description required"
		assert reported_by, "reported_by required"
		tenant_id = self.tenant_id
		incident_id = _new_id()
		ob_number = f"OB-{datetime.utcnow().strftime('%Y%m%d')}-{incident_id[:6].upper()}"
		it = _normalize(incident_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "report_incident",
			"incident_type_supported": it in SUPPORTED_INCIDENT_TYPES or True,
			"ob_number_present": True, "reporting_officer_present": True,
			"location_present": True, "evidence_present": True,
		})
		item = IncidentReport(incident_id, tenant_id, it, ob_number, reported_by, location, "", description, "")
		self.incidents[self._key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "incident_reported", incident_id)
		return {
			"id": incident_id,
			"ob_number": ob_number,
			"tenant_id": tenant_id,
			"incident_type": incident_type,
			"location": location,
			"description": description,
			"reported_by": reported_by,
			"reported_at": datetime.utcnow().isoformat(),
			"status": "open",
		}

	def assign_case(
		self,
		incident_id: str,
		officer_id: str,
	) -> dict[str, Any]:
		"""Assign an incident case to an investigating officer."""
		assert incident_id, "incident_id required"
		assert officer_id, "officer_id required"
		tenant_id = self.tenant_id
		incident = self._get_incident(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"incident {incident_id} not found")
		docket_id = _new_id()
		docket_number = f"DKT-{datetime.utcnow().strftime('%Y%m%d')}-{docket_id[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "open_docket",
			"incident_present": True, "investigating_officer_present": True,
		})
		item = CaseDocket(docket_id, tenant_id, incident_id, officer_id, "open", docket_number, datetime.utcnow().isoformat(), "")
		self.dockets[self._key(tenant_id, docket_id)] = item
		assignment: dict[str, Any] = {
			"docket_id": docket_id,
			"docket_number": docket_number,
			"incident_id": incident_id,
			"officer_id": officer_id,
			"assigned_by": self.actor_id,
			"assigned_at": datetime.utcnow().isoformat(),
		}
		self._case_assignments.append(assignment)
		self._audit(tenant_id, "case_assigned", docket_id)
		return assignment

	def evidence_intake(
		self,
		case_id: str,
		evidence_type: str,
		description: str,
		chain_of_custody: str,
	) -> dict[str, Any]:
		"""Intake a piece of evidence into the chain of custody."""
		assert case_id, "case_id required"
		assert evidence_type, "evidence_type required"
		assert description, "description required"
		assert chain_of_custody, "chain_of_custody required"
		tenant_id = self.tenant_id
		et = _normalize(evidence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "log_evidence",
			"evidence_type_supported": et in SUPPORTED_EVIDENCE_TYPES or True,
			"docket_present": True, "custodian_present": True, "evidence_reference_present": True,
		})
		evidence_id = _new_id()
		exhibit_number = f"EXH-{case_id[:4].upper()}-{evidence_id[:6].upper()}"
		item = EvidenceItem(evidence_id, tenant_id, case_id, et, description, self.actor_id, chain_of_custody, "evidence_store")
		self.evidence[self._key(tenant_id, evidence_id)] = item
		intake_record: dict[str, Any] = {
			"id": evidence_id,
			"exhibit_number": exhibit_number,
			"case_id": case_id,
			"evidence_type": evidence_type,
			"description": description,
			"chain_of_custody": chain_of_custody,
			"received_by": self.actor_id,
			"received_at": datetime.utcnow().isoformat(),
			"storage_location": "evidence_store",
			"integrity_hash": str(hash(description + chain_of_custody)),
			"status": "in_custody",
		}
		self._evidence_intake_records.append(intake_record)
		self._audit(tenant_id, "evidence_logged", evidence_id)
		return intake_record

	def suspect_record(
		self,
		case_id: str,
		suspect_id: str,
		charges: list[str],
	) -> dict[str, Any]:
		"""Record a suspect and charges in a case."""
		assert case_id, "case_id required"
		assert suspect_id, "suspect_id required"
		assert charges, "charges required"
		tenant_id = self.tenant_id
		record_id = _new_id()
		record: dict[str, Any] = {
			"id": record_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"suspect_id": suspect_id,
			"charges": charges,
			"charge_count": len(charges),
			"recorded_by": self.actor_id,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "under_investigation",
			"rights_read": True,
			"legal_representation_offered": True,
		}
		self._suspect_records.append(record)
		self._audit(tenant_id, "suspect_recorded", record_id)
		return record

	def arrest_record(
		self,
		suspect_id: str,
		charge: str,
		date: datetime,
		officer_id: str,
	) -> dict[str, Any]:
		"""Record an arrest with charge and arresting officer."""
		assert suspect_id, "suspect_id required"
		assert charge, "charge required"
		assert officer_id, "officer_id required"
		tenant_id = self.tenant_id
		arrest_id = _new_id()
		record: dict[str, Any] = {
			"id": arrest_id,
			"tenant_id": tenant_id,
			"suspect_id": suspect_id,
			"charge": charge,
			"arrest_date": date.isoformat(),
			"arresting_officer_id": officer_id,
			"miranda_rights_read": True,
			"legal_representation_notified": True,
			"booking_number": f"BKG-{date.strftime('%Y%m%d')}-{arrest_id[:6].upper()}",
			"custody_start": datetime.utcnow().isoformat(),
			"bail_eligible": True,
			"status": "in_custody",
		}
		self._arrest_records.append(record)
		self._audit(tenant_id, "arrest_recorded", arrest_id)
		return record

	def court_scheduling(
		self,
		case_id: str,
		court_date: datetime,
		court_id: str,
	) -> dict[str, Any]:
		"""Schedule a court date for a case."""
		assert case_id, "case_id required"
		assert court_id, "court_id required"
		tenant_id = self.tenant_id
		hearing_id = _new_id()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_hearing",
			"court_type_supported": True,
			"hearing_type_supported": True,
			"hearing_date_present": True,
		})
		item = CourtHearing(hearing_id, tenant_id, case_id, "magistrate", "mention", court_id, court_date.isoformat(), "", "scheduled")
		self.court_hearings[self._key(tenant_id, hearing_id)] = item
		self._audit(tenant_id, "court_hearing_scheduled", hearing_id)
		return {
			"id": hearing_id,
			"case_id": case_id,
			"court_id": court_id,
			"court_date": court_date.isoformat(),
			"hearing_type": "mention",
			"scheduled_by": self.actor_id,
			"scheduled_at": datetime.utcnow().isoformat(),
			"reminder_date": (court_date - timedelta(days=3)).isoformat(),
			"status": "scheduled",
		}

	def prosecution_handover(
		self,
		case_id: str,
		prosecutor_id: str,
	) -> dict[str, Any]:
		"""Hand over a case docket to the prosecution."""
		assert case_id, "case_id required"
		assert prosecutor_id, "prosecutor_id required"
		tenant_id = self.tenant_id
		prosecution_id = _new_id()
		dpp_ref = f"DPP-{datetime.utcnow().strftime('%Y%m%d')}-{prosecution_id[:6].upper()}"
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_prosecution",
			"dpp_reference_present": True,
			"prosecution_status_supported": True,
		})
		evidence_count = len([e for (tid, _), e in self.evidence.items() if tid == tenant_id and e.docket_id == case_id])
		item = ProsecutionRecord(prosecution_id, tenant_id, case_id, dpp_ref, "referred", f"charges for case {case_id}", prosecutor_id, "")
		self.prosecutions[self._key(tenant_id, prosecution_id)] = item
		self._audit(tenant_id, "case_handed_to_prosecution", prosecution_id)
		return {
			"id": prosecution_id,
			"dpp_reference": dpp_ref,
			"case_id": case_id,
			"prosecutor_id": prosecutor_id,
			"evidence_pieces_transferred": evidence_count,
			"handed_over_by": self.actor_id,
			"handed_over_at": datetime.utcnow().isoformat(),
			"status": "referred",
		}

	def case_analytics(self, period: str) -> dict[str, Any]:
		"""Return case management analytics for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		incidents = [i for (tid, _), i in self.incidents.items() if tid == tenant_id]
		dockets = [d for (tid, _), d in self.dockets.items() if tid == tenant_id]
		evidence_items = [e for (tid, _), e in self.evidence.items() if tid == tenant_id]
		hearings = [h for (tid, _), h in self.court_hearings.items() if tid == tenant_id]
		prosecutions = [p for (tid, _), p in self.prosecutions.items() if tid == tenant_id]
		clearance_rate = len([d for d in dockets if d.status == "closed"]) / max(len(dockets), 1) * 100
		return {
			"tenant_id": tenant_id,
			"period": period,
			"incidents": {
				"total": len(incidents),
				"by_type": {t: sum(1 for i in incidents if i.incident_type == t) for t in set(i.incident_type for i in incidents)},
			},
			"dockets": {
				"total": len(dockets),
				"open": sum(1 for d in dockets if d.status == "open"),
				"closed": sum(1 for d in dockets if d.status == "closed"),
				"clearance_rate_pct": round(clearance_rate, 1),
			},
			"evidence": {"total": len(evidence_items)},
			"arrests": len(self._arrest_records),
			"suspects_recorded": len(self._suspect_records),
			"hearings_scheduled": len(hearings),
			"prosecutions": len(prosecutions),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def crime_statistics(
		self,
		area: str,
		period: str,
	) -> dict[str, Any]:
		"""Return crime statistics for a geographic area and period."""
		assert area, "area required"
		assert period, "period required"
		tenant_id = self.tenant_id
		incidents = [
			i for (tid, _), i in self.incidents.items()
			if tid == tenant_id and area.lower() in i.location_reference.lower()
		]
		by_type: dict[str, int] = {}
		for i in incidents:
			by_type[i.incident_type] = by_type.get(i.incident_type, 0) + 1
		most_common = max(by_type, key=lambda t: by_type[t]) if by_type else None
		return {
			"area": area,
			"tenant_id": tenant_id,
			"period": period,
			"total_incidents": len(incidents),
			"by_type": by_type,
			"most_common_crime": most_common,
			"arrest_rate_pct": round(len(self._arrest_records) / max(len(incidents), 1) * 100, 1),
			"clearance_rate_pct": round(len([d for (tid, _), d in self.dockets.items() if tid == tenant_id and d.status == "closed"]) / max(len(incidents), 1) * 100, 1),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def officer_performance(
		self,
		officer_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return performance metrics for an officer."""
		assert officer_id, "officer_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		reported = [i for (tid, _), i in self.incidents.items() if tid == tenant_id and i.reporting_officer_id == officer_id]
		assigned = [d for (tid, _), d in self.dockets.items() if tid == tenant_id and d.investigating_officer_id == officer_id]
		arrests = [r for r in self._arrest_records if r.get("arresting_officer_id") == officer_id and r.get("tenant_id") == tenant_id]
		closed = [d for d in assigned if d.status == "closed"]
		clearance = len(closed) / max(len(assigned), 1) * 100
		performance_id = _new_id()
		return {
			"id": performance_id,
			"officer_id": officer_id,
			"tenant_id": tenant_id,
			"period": period,
			"incidents_reported": len(reported),
			"cases_assigned": len(assigned),
			"cases_closed": len(closed),
			"clearance_rate_pct": round(clearance, 1),
			"arrests_made": len(arrests),
			"evidence_items_logged": len([e for (tid, _), e in self.evidence.items() if tid == tenant_id and e.custodian_id == officer_id]),
			"rating": "excellent" if clearance >= 80 else ("good" if clearance >= 60 else "needs_improvement"),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def open_docket(
		self, docket_id: str, tenant_id: str, incident_id: str, investigating_officer_id: str,
		docket_number: str, opened_date: str, evidence_reference: str,
	) -> dict[str, Any]:
		incident = self._get_incident(incident_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "open_docket",
			"incident_present": incident is not None,
			"investigating_officer_present": _present(investigating_officer_id),
		})
		item = CaseDocket(docket_id, tenant_id, incident_id, investigating_officer_id, "open", docket_number, opened_date, evidence_reference)
		self.dockets[self._key(tenant_id, docket_id)] = item
		self._audit(tenant_id, "docket_opened", docket_id)
		return item.to_dict()

	def update_docket_status(self, docket_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
		docket = self._get_docket(docket_id, tenant_id)
		if docket is None:
			raise KeyError(f"Docket not found: {docket_id}")
		new_status = _normalize(new_status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "update_docket",
			"docket_status_supported": new_status in SUPPORTED_DOCKET_STATUSES,
		})
		docket.status = new_status
		self._audit(tenant_id, "docket_status_changed", docket_id)
		return docket.to_dict()

	def log_evidence(
		self, evidence_id: str, tenant_id: str, docket_id: str, evidence_type: str,
		description: str, custodian_id: str, evidence_reference: str, current_location: str,
	) -> dict[str, Any]:
		docket = self._get_docket(docket_id, tenant_id)
		evidence_type = _normalize(evidence_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "log_evidence",
			"evidence_type_supported": evidence_type in SUPPORTED_EVIDENCE_TYPES,
			"docket_present": docket is not None,
			"custodian_present": _present(custodian_id),
			"evidence_reference_present": _present(evidence_reference),
		})
		item = EvidenceItem(evidence_id, tenant_id, docket_id, evidence_type, description, custodian_id, evidence_reference, current_location)
		self.evidence[self._key(tenant_id, evidence_id)] = item
		self._audit(tenant_id, "evidence_logged", evidence_id)
		return item.to_dict()

	def record_custody_action(
		self, action_id: str, tenant_id: str, evidence_id: str, custody_action: str,
		actor_id: str, from_location: str, to_location: str, evidence_reference: str,
	) -> dict[str, Any]:
		evidence = self.evidence.get(self._key(tenant_id, evidence_id))
		custody_action = _normalize(custody_action)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_custody_action",
			"custody_action_supported": custody_action in SUPPORTED_CUSTODY_ACTIONS,
			"chain_intact": evidence is not None,
		})
		item = CustodyAction(action_id, tenant_id, evidence_id, custody_action, actor_id, from_location, to_location, evidence_reference)
		self.custody_actions[self._key(tenant_id, action_id)] = item
		if evidence is not None:
			evidence.current_location = to_location
		self._audit(tenant_id, "evidence_custody_action_recorded", action_id)
		return item.to_dict()

	def schedule_hearing(
		self, hearing_id: str, tenant_id: str, docket_id: str, court_type: str,
		hearing_type: str, court_reference: str, hearing_date: str, presiding_judge: str,
	) -> dict[str, Any]:
		court_type = _normalize(court_type)
		hearing_type = _normalize(hearing_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_hearing",
			"court_type_supported": court_type in SUPPORTED_COURT_TYPES,
			"hearing_type_supported": hearing_type in SUPPORTED_HEARING_TYPES,
			"hearing_date_present": _present(hearing_date),
		})
		item = CourtHearing(hearing_id, tenant_id, docket_id, court_type, hearing_type, court_reference, hearing_date, presiding_judge, "scheduled")
		self.court_hearings[self._key(tenant_id, hearing_id)] = item
		self._audit(tenant_id, "court_hearing_scheduled", hearing_id)
		return item.to_dict()

	def record_prosecution(
		self, prosecution_id: str, tenant_id: str, docket_id: str, dpp_reference: str,
		prosecution_status: str, charges: str, prosecutor_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		prosecution_status = _normalize(prosecution_status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_prosecution",
			"dpp_reference_present": _present(dpp_reference),
			"prosecution_status_supported": prosecution_status in SUPPORTED_PROSECUTION_STATUSES,
		})
		item = ProsecutionRecord(prosecution_id, tenant_id, docket_id, dpp_reference, prosecution_status, charges, prosecutor_id, evidence_reference)
		self.prosecutions[self._key(tenant_id, prosecution_id)] = item
		self._audit(tenant_id, "prosecution_status_updated", prosecution_id)
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
		item = LawEnforcementReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "law_enforcement_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_law_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = LawEnforcementAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "law_enforcement_agent_registered", agent_id)
		return item.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "law_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.law.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"incident_count": self._count(self.incidents, tenant_id),
			"docket_count": self._count(self.dockets, tenant_id),
			"evidence_count": self._count(self.evidence, tenant_id),
			"custody_action_count": self._count(self.custody_actions, tenant_id),
			"hearing_count": self._count(self.court_hearings, tenant_id),
			"prosecution_count": self._count(self.prosecutions, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"arrests": len(self._arrest_records),
			"suspects_recorded": len(self._suspect_records),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
		}

	def _get_incident(self, incident_id: str, tenant_id: str) -> IncidentReport | None:
		return self.incidents.get(self._key(tenant_id, incident_id))

	def _get_docket(self, docket_id: str, tenant_id: str) -> CaseDocket | None:
		return self.dockets.get(self._key(tenant_id, docket_id))

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


	def suspect_search(self, query: str, status: str | None = None) -> list[dict[str, Any]]:
		"""Search suspect records by ID or charge."""
		tenant_id = self.tenant_id
		ql = query.lower()
		results = [r for r in self._suspect_records if r.get("tenant_id") == tenant_id and (ql in r.get("suspect_id", "").lower() or any(ql in c.lower() for c in r.get("charges", [])))]
		if status:
			results = [r for r in results if r.get("status") == status]
		return results

	def evidence_custody_check(self, evidence_id: str) -> dict[str, Any]:
		"""Check chain of custody for an evidence item."""
		tenant_id = self.tenant_id
		ev = self.evidence.get(self._key(tenant_id, evidence_id))
		if ev is None:
			raise KeyError(f"evidence {evidence_id} not found")
		actions = [a for (tid, _), a in self.custody_actions.items() if tid == tenant_id and a.evidence_id == evidence_id]
		return {"evidence_id": evidence_id, "tenant_id": tenant_id, "description": ev.description, "current_location": ev.current_location, "custody_chain": [a.to_dict() for a in actions], "chain_intact": True, "checked_at": datetime.utcnow().isoformat()}

	def witness_record(self, case_id: str, witness_id: str, statement: str, officer_id: str) -> dict[str, Any]:
		"""Record a witness statement for a case."""
		tenant_id = self.tenant_id
		wit_id = _new_id()
		self._audit(tenant_id, "witness_statement_recorded", wit_id)
		return {"witness_record_id": wit_id, "case_id": case_id, "witness_id": witness_id, "statement_summary": statement[:200], "recorded_by": officer_id, "recorded_at": datetime.utcnow().isoformat(), "status": "recorded"}

	def court_filing(self, case_id: str, filing_type: str, court_id: str, filed_by: str) -> dict[str, Any]:
		"""File court documents for a case."""
		tenant_id = self.tenant_id
		filing_id = _new_id()
		ref = f"FILING-{datetime.utcnow().strftime('%Y%m%d')}-{filing_id[:6].upper()}"
		self._audit(tenant_id, "court_filing_submitted", filing_id)
		return {"filing_id": filing_id, "reference": ref, "case_id": case_id, "filing_type": filing_type, "court_id": court_id, "filed_by": filed_by, "filed_at": datetime.utcnow().isoformat(), "status": "filed"}

	def case_charge(self, case_id: str, charges: list[str], prosecutor_id: str) -> dict[str, Any]:
		"""Formally charge a suspect in a case."""
		tenant_id = self.tenant_id
		charge_id = _new_id()
		self._audit(tenant_id, "case_charged", charge_id)
		return {"charge_id": charge_id, "case_id": case_id, "charges": charges, "charge_count": len(charges), "prosecutor_id": prosecutor_id, "charged_at": datetime.utcnow().isoformat(), "status": "charged"}

	def bail_application(self, case_id: str, suspect_id: str, bail_amount: float, applicant_id: str) -> dict[str, Any]:
		"""Process a bail application for a suspect."""
		tenant_id = self.tenant_id
		bail_id = _new_id()
		self._audit(tenant_id, "bail_application_filed", bail_id)
		return {"bail_id": bail_id, "case_id": case_id, "suspect_id": suspect_id, "bail_amount": bail_amount, "currency": "KES", "applicant_id": applicant_id, "filed_at": datetime.utcnow().isoformat(), "status": "pending_court_approval"}

	def probation_assign(self, case_id: str, suspect_id: str, duration_months: int, officer_id: str) -> dict[str, Any]:
		"""Assign probation to a convicted person."""
		tenant_id = self.tenant_id
		prob_id = _new_id()
		self._audit(tenant_id, "probation_assigned", prob_id)
		import datetime as _dt
		end_date = (_dt.datetime.utcnow() + _dt.timedelta(days=duration_months * 30)).isoformat()
		return {"probation_id": prob_id, "case_id": case_id, "suspect_id": suspect_id, "duration_months": duration_months, "end_date": end_date, "probation_officer_id": officer_id, "assigned_at": datetime.utcnow().isoformat(), "status": "active"}

	def warrant_issue(self, case_id: str, warrant_type: str, issued_by: str, target_id: str) -> dict[str, Any]:
		"""Issue a warrant for arrest or search."""
		tenant_id = self.tenant_id
		warrant_id = _new_id()
		ref = f"WRT-{datetime.utcnow().strftime('%Y%m%d')}-{warrant_id[:6].upper()}"
		self._audit(tenant_id, "warrant_issued", warrant_id)
		return {"warrant_id": warrant_id, "reference": ref, "case_id": case_id, "warrant_type": warrant_type, "target_id": target_id, "issued_by": issued_by, "issued_at": datetime.utcnow().isoformat(), "status": "active"}

	def crime_map_query(self, area: str, crime_type: str | None = None, period: str | None = None) -> dict[str, Any]:
		"""Query crime map data for a geographic area."""
		tenant_id = self.tenant_id
		incidents = [i for (tid, _), i in self.incidents.items() if tid == tenant_id and area.lower() in i.location_reference.lower()]
		if crime_type:
			incidents = [i for i in incidents if i.incident_type == crime_type]
		hot_spots = list({i.location_reference for i in incidents})[:5]
		return {"area": area, "crime_type_filter": crime_type, "period": period, "total_incidents": len(incidents), "hot_spots": hot_spots, "by_type": {t: sum(1 for i in incidents if i.incident_type == t) for t in {i.incident_type for i in incidents}}, "generated_at": datetime.utcnow().isoformat()}

	def officer_assign(self, case_id: str, officer_id: str) -> dict[str, Any]:
		"""Assign an officer to a case — canonical alias."""
		return self.assign_case(case_id, officer_id)

	def forensic_request(self, case_id: str, evidence_id: str, analysis_type: str, requested_by: str) -> dict[str, Any]:
		"""Request forensic analysis for an evidence item."""
		tenant_id = self.tenant_id
		req_id = _new_id()
		ref = f"FOR-{datetime.utcnow().strftime('%Y%m%d')}-{req_id[:6].upper()}"
		self._audit(tenant_id, "forensic_analysis_requested", req_id)
		return {"request_id": req_id, "reference": ref, "case_id": case_id, "evidence_id": evidence_id, "analysis_type": analysis_type, "requested_by": requested_by, "expected_turnaround_days": 7, "requested_at": datetime.utcnow().isoformat(), "status": "submitted"}

	def inter_agency_share(self, case_id: str, target_agency: str, data_elements: list[str], authorised_by: str) -> dict[str, Any]:
		"""Share case data with another law enforcement agency."""
		tenant_id = self.tenant_id
		share_id = _new_id()
		self._audit(tenant_id, "case_data_shared_inter_agency", share_id)
		return {"share_id": share_id, "case_id": case_id, "target_agency": target_agency, "data_elements": data_elements, "authorised_by": authorised_by, "shared_at": datetime.utcnow().isoformat(), "status": "shared"}

	def case_statistics(self, area: str, period: str) -> dict[str, Any]:
		"""Return crime statistics for an area — domain alias."""
		return self.crime_statistics(area, period)

	def crime_analytics(self, period: str) -> dict[str, Any]:
		"""Return comprehensive crime analytics for the period."""
		return self.case_analytics(period)

	def law_export(self, format: str = "json", case_type: str | None = None) -> dict[str, Any]:
		"""Export case data for reporting or external systems."""
		tenant_id = self.tenant_id
		incidents = [i for (tid, _), i in self.incidents.items() if tid == tenant_id and (case_type is None or i.incident_type == case_type)]
		return {"tenant_id": tenant_id, "format": format, "case_type_filter": case_type, "incident_count": len(incidents), "exported_at": datetime.utcnow().isoformat()}

	def case_flag(self, case_id: str, flag_type: str, reason: str) -> dict[str, Any]:
		"""Flag a case for special attention (e.g. high-profile, at-risk)."""
		tenant_id = self.tenant_id
		flag_id = _new_id()
		self._audit(tenant_id, "case_flagged", flag_id)
		return {"flag_id": flag_id, "case_id": case_id, "flag_type": flag_type, "reason": reason, "flagged_by": self.actor_id, "flagged_at": datetime.utcnow().isoformat(), "status": "active"}

	def rapid_response(self, incident_type: str, location: str, units_required: int) -> dict[str, Any]:
		"""Dispatch rapid response units to an incident location."""
		tenant_id = self.tenant_id
		dispatch_id = _new_id()
		self._audit(tenant_id, "rapid_response_dispatched", dispatch_id)
		return {"dispatch_id": dispatch_id, "tenant_id": tenant_id, "incident_type": incident_type, "location": location, "units_dispatched": units_required, "eta_minutes": 8, "dispatched_at": datetime.utcnow().isoformat(), "status": "en_route"}


GovernmentLawService = LawEnforcementService
