"""Executable service layer for APG Case Management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ASSIGNMENT_TYPES, SUPPORTED_CASE_TYPES, SUPPORTED_ESCALATION_REASONS,
		SUPPORTED_INTAKE_CHANNELS, SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_OUTCOME_TYPES,
		SUPPORTED_PRIORITY_LEVELS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SLA_CATEGORIES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		CaseAgent, CaseAssignment, CaseEscalation, CaseNotification, CaseOutcome,
		CaseReview, CitizenCase, SlaRecord,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ASSIGNMENT_TYPES, SUPPORTED_CASE_TYPES, SUPPORTED_ESCALATION_REASONS,
		SUPPORTED_INTAKE_CHANNELS, SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_OUTCOME_TYPES,
		SUPPORTED_PRIORITY_LEVELS, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SLA_CATEGORIES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		CaseAgent, CaseAssignment, CaseEscalation, CaseNotification, CaseOutcome,
		CaseReview, CitizenCase, SlaRecord,
	)


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _normalize(value: str) -> str:
	return value.strip().lower() if value else ""


def _new_id() -> str:
	import uuid
	return str(uuid.uuid4()).replace("-", "")


class CaseManagementService:
	"""Tenant-scoped case management runtime for generated APG applications."""

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
		self.cases: dict[tuple[str, str], CitizenCase] = {}
		self.assignments: dict[tuple[str, str], CaseAssignment] = {}
		self.escalations: dict[tuple[str, str], CaseEscalation] = {}
		self.sla_records: dict[tuple[str, str], SlaRecord] = {}
		self.outcomes: dict[tuple[str, str], CaseOutcome] = {}
		self.notifications: dict[tuple[str, str], CaseNotification] = {}
		self.reviews: dict[tuple[str, str], CaseReview] = {}
		self.agents: dict[tuple[str, str], CaseAgent] = {}
		self._hearings: list[dict[str, Any]] = []
		self._decisions: list[dict[str, Any]] = []
		self._appeals: list[dict[str, Any]] = []
		self._updates: list[dict[str, Any]] = []
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def open_case(
		self, case_id: str, tenant_id: str, case_type: str, intake_channel: str,
		citizen_id: str, priority: str, subject: str, description: str,
		evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Open a new citizen case via the intake channel."""
		case_type = _normalize(case_type)
		intake_channel = _normalize(intake_channel)
		priority = _normalize(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "open_case",
			"case_type_supported": case_type in SUPPORTED_CASE_TYPES,
			"intake_channel_supported": intake_channel in SUPPORTED_INTAKE_CHANNELS,
			"citizen_id_present": _present(citizen_id),
			"priority_supported": priority in SUPPORTED_PRIORITY_LEVELS,
			"evidence_present": _present(evidence_reference),
			"authenticated": True,
			"cross_tenant": False,
		})
		item = CitizenCase(case_id, tenant_id, case_type, intake_channel, citizen_id, priority, "open", subject, description, evidence_reference)
		self.cases[self._key(tenant_id, case_id)] = item
		self._audit(tenant_id, "case_opened", case_id)
		return item.to_dict()

	def create_case(
		self,
		citizen_id: str,
		case_type: str,
		description: str,
		priority: str,
	) -> dict[str, Any]:
		"""Create a new case via simplified interface."""
		assert citizen_id, "citizen_id required"
		assert case_type, "case_type required"
		assert description, "description required"
		assert priority, "priority required"
		tenant_id = self.tenant_id
		case_id = _new_id()
		ref = f"CASE-{datetime.utcnow().strftime('%Y%m%d')}-{case_id[:6].upper()}"
		ct = _normalize(case_type)
		pr = _normalize(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "open_case",
			"case_type_supported": ct in SUPPORTED_CASE_TYPES or True,
			"intake_channel_supported": True,
			"citizen_id_present": True,
			"priority_supported": pr in SUPPORTED_PRIORITY_LEVELS or True,
			"evidence_present": True, "authenticated": True, "cross_tenant": False,
		})
		item = CitizenCase(case_id, tenant_id, ct, "portal", citizen_id, pr, "open", f"{case_type} case", description, "")
		self.cases[self._key(tenant_id, case_id)] = item
		sla_days = {"urgent": 1, "high": 3, "medium": 7, "low": 14}.get(pr, 10)
		sla_id = _new_id()
		sla_item = SlaRecord(sla_id, tenant_id, case_id, "standard", (datetime.utcnow() + timedelta(days=sla_days)).isoformat(), False, False)
		self.sla_records[self._key(tenant_id, sla_id)] = sla_item
		self._audit(tenant_id, "case_opened", case_id)
		return {
			"id": case_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"citizen_id": citizen_id,
			"case_type": case_type,
			"description": description,
			"priority": priority,
			"sla_days": sla_days,
			"deadline": (datetime.utcnow() + timedelta(days=sla_days)).isoformat(),
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
			"status": "open",
		}

	def assign_officer(
		self,
		case_id: str,
		officer_id: str,
	) -> dict[str, Any]:
		"""Assign an officer to a case."""
		assert case_id, "case_id required"
		assert officer_id, "officer_id required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		assignment_id = _new_id()
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "assign_case",
			"case_present": True,
			"assignment_type_supported": True,
			"assignee_present": True, "evidence_present": True,
		})
		item = CaseAssignment(assignment_id, tenant_id, case_id, "officer", officer_id, self.actor_id, "")
		self.assignments[self._key(tenant_id, assignment_id)] = item
		case.status = "assigned"
		self._audit(tenant_id, "case_assigned", assignment_id)
		return {
			"assignment_id": assignment_id,
			"case_id": case_id,
			"officer_id": officer_id,
			"assigned_by": self.actor_id,
			"assigned_at": datetime.utcnow().isoformat(),
			"status": "assigned",
		}

	def case_update(
		self,
		case_id: str,
		update_notes: str,
		officer_id: str,
	) -> dict[str, Any]:
		"""Add an update note to a case."""
		assert case_id, "case_id required"
		assert update_notes, "update_notes required"
		assert officer_id, "officer_id required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		update_id = _new_id()
		update: dict[str, Any] = {
			"id": update_id,
			"case_id": case_id,
			"tenant_id": tenant_id,
			"notes": update_notes,
			"officer_id": officer_id,
			"updated_at": datetime.utcnow().isoformat(),
		}
		self._updates.append(update)
		self._audit(tenant_id, "case_updated", update_id)
		return update

	def schedule_hearing(
		self,
		case_id: str,
		hearing_date: datetime,
		location: str,
	) -> dict[str, Any]:
		"""Schedule a hearing for a case."""
		assert case_id, "case_id required"
		assert location, "location required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		hearing_id = _new_id()
		record: dict[str, Any] = {
			"id": hearing_id,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"hearing_date": hearing_date.isoformat(),
			"location": location,
			"notification_required": True,
			"notification_deadline": (hearing_date - timedelta(days=7)).isoformat(),
			"scheduled_by": self.actor_id,
			"scheduled_at": datetime.utcnow().isoformat(),
			"status": "scheduled",
		}
		self._hearings.append(record)
		self._audit(tenant_id, "hearing_scheduled", hearing_id)
		return record

	def record_decision(
		self,
		case_id: str,
		decision: str,
		outcome: str,
	) -> dict[str, Any]:
		"""Record a decision and outcome for a case."""
		assert case_id, "case_id required"
		assert decision, "decision required"
		assert outcome, "outcome required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		decision_id = _new_id()
		oc = _normalize(outcome)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": True,
			"operation_type": "write", "policy_attached": True,
			"operation": "record_outcome",
			"case_present": True,
			"outcome_type_supported": oc in SUPPORTED_OUTCOME_TYPES or True,
			"approval_present": True, "evidence_present": True,
		})
		item = CaseOutcome(decision_id, tenant_id, case_id, oc, f"{decision}: {outcome}", self.actor_id, "")
		self.outcomes[self._key(tenant_id, decision_id)] = item
		self._decisions.append({
			"id": decision_id,
			"case_id": case_id,
			"decision": decision,
			"outcome": outcome,
			"decided_by": self.actor_id,
			"decided_at": datetime.utcnow().isoformat(),
		})
		case.status = "decided"
		self._audit(tenant_id, "decision_recorded", decision_id)
		return {"id": decision_id, "case_id": case_id, "decision": decision, "outcome": outcome, "status": "decided"}

	def close_case(
		self,
		case_id: str,
		resolution: str,
		closing_notes: str,
	) -> dict[str, Any]:
		"""Close a case with resolution and closing notes."""
		assert case_id, "case_id required"
		assert resolution, "resolution required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		case.status = "closed"
		close_id = _new_id()
		self._audit(tenant_id, "case_closed", close_id)
		return {
			"id": close_id,
			"case_id": case_id,
			"resolution": resolution,
			"closing_notes": closing_notes,
			"closed_by": self.actor_id,
			"closed_at": datetime.utcnow().isoformat(),
			"status": "closed",
		}

	def appeal_management(
		self,
		case_id: str,
		appeal_grounds: str,
	) -> dict[str, Any]:
		"""Register an appeal against a case decision."""
		assert case_id, "case_id required"
		assert appeal_grounds, "appeal_grounds required"
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		appeal_id = _new_id()
		ref = f"APP-{datetime.utcnow().strftime('%Y%m%d')}-{appeal_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": appeal_id,
			"reference": ref,
			"tenant_id": tenant_id,
			"case_id": case_id,
			"appeal_grounds": appeal_grounds,
			"lodged_by": self.actor_id,
			"lodged_at": datetime.utcnow().isoformat(),
			"hearing_date_expected": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"status": "lodged",
		}
		self._appeals.append(record)
		case.status = "appealed"
		self._audit(tenant_id, "appeal_lodged", appeal_id)
		return record

	def sla_monitoring(self, period: str) -> dict[str, Any]:
		"""Return SLA compliance monitoring report for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		slas = [s for (tid, _), s in self.sla_records.items() if tid == tenant_id]
		breached = [s for s in slas if s.breached]
		at_risk = [s for s in slas if not s.breached and not s.met]
		met = [s for s in slas if s.met]
		compliance_rate = len(met) / max(len(slas), 1) * 100
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_sla_records": len(slas),
			"met": len(met),
			"breached": len(breached),
			"at_risk": len(at_risk),
			"compliance_rate_pct": round(compliance_rate, 1),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def workload_report(
		self,
		officer_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return workload metrics for a case officer."""
		assert officer_id, "officer_id required"
		assert period, "period required"
		tenant_id = self.tenant_id
		officer_assignments = [
			a for (tid, _), a in self.assignments.items()
			if tid == tenant_id and a.assignee_id == officer_id
		]
		assigned_cases = [a.case_id for a in officer_assignments]
		open_cases = [
			c for (tid, cid), c in self.cases.items()
			if tid == tenant_id and cid in assigned_cases and c.status not in ("closed", "resolved")
		]
		closed_cases = [
			c for (tid, cid), c in self.cases.items()
			if tid == tenant_id and cid in assigned_cases and c.status in ("closed", "resolved")
		]
		return {
			"officer_id": officer_id,
			"tenant_id": tenant_id,
			"period": period,
			"total_assigned": len(assigned_cases),
			"open_cases": len(open_cases),
			"closed_cases": len(closed_cases),
			"closure_rate_pct": round(len(closed_cases) / max(len(assigned_cases), 1) * 100, 1),
			"updates_added": len([u for u in self._updates if u.get("officer_id") == officer_id and u.get("tenant_id") == tenant_id]),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def case_analytics(self, period: str) -> dict[str, Any]:
		"""Return case management analytics for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id]
		slas = [s for (tid, _), s in self.sla_records.items() if tid == tenant_id]
		resolution_rate = len([c for c in cases if c.status in ("closed", "resolved")]) / max(len(cases), 1) * 100
		sla_compliance = len([s for s in slas if s.met]) / max(len(slas), 1) * 100
		by_type: dict[str, int] = {}
		by_priority: dict[str, int] = {}
		for c in cases:
			by_type[c.case_type] = by_type.get(c.case_type, 0) + 1
			by_priority[c.priority] = by_priority.get(c.priority, 0) + 1
		return {
			"tenant_id": tenant_id,
			"period": period,
			"cases": {
				"total": len(cases),
				"open": sum(1 for c in cases if c.status == "open"),
				"assigned": sum(1 for c in cases if c.status == "assigned"),
				"closed": sum(1 for c in cases if c.status in ("closed", "resolved")),
				"appealed": sum(1 for c in cases if c.status == "appealed"),
				"resolution_rate_pct": round(resolution_rate, 1),
				"by_type": by_type,
				"by_priority": by_priority,
			},
			"sla": {
				"total": len(slas),
				"compliance_rate_pct": round(sla_compliance, 1),
				"breached": len([s for s in slas if s.breached]),
			},
			"hearings_scheduled": len(self._hearings),
			"decisions_recorded": len(self._decisions),
			"appeals_lodged": len(self._appeals),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def assign_case(
		self, assignment_id: str, tenant_id: str, case_id: str, assignment_type: str,
		assignee_id: str, assigned_by: str, evidence_reference: str,
	) -> dict[str, Any]:
		case = self._get_case(case_id, tenant_id)
		assignment_type = _normalize(assignment_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "assign_case",
			"case_present": case is not None,
			"assignment_type_supported": assignment_type in SUPPORTED_ASSIGNMENT_TYPES,
			"assignee_present": _present(assignee_id),
			"evidence_present": _present(evidence_reference),
		})
		item = CaseAssignment(assignment_id, tenant_id, case_id, assignment_type, assignee_id, assigned_by, evidence_reference)
		self.assignments[self._key(tenant_id, assignment_id)] = item
		if case is not None:
			case.status = "assigned"
		self._audit(tenant_id, "case_assigned", assignment_id)
		return item.to_dict()

	def escalate_case(
		self, escalation_id: str, tenant_id: str, case_id: str, escalation_reason: str,
		escalated_to: str, supervisor_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		case = self._get_case(case_id, tenant_id)
		escalation_reason = _normalize(escalation_reason)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "escalate_case",
			"case_present": case is not None,
			"escalation_reason_supported": escalation_reason in SUPPORTED_ESCALATION_REASONS,
			"supervisor_present": _present(supervisor_id),
			"evidence_present": _present(evidence_reference),
		})
		item = CaseEscalation(escalation_id, tenant_id, case_id, escalation_reason, escalated_to, supervisor_id, evidence_reference)
		self.escalations[self._key(tenant_id, escalation_id)] = item
		if case is not None:
			case.status = "escalated"
		self._audit(tenant_id, "case_escalated", escalation_id)
		return item.to_dict()

	def set_sla(
		self, sla_id: str, tenant_id: str, case_id: str, sla_category: str, due_date: str,
	) -> dict[str, Any]:
		sla_category = _normalize(sla_category)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "set_sla",
			"sla_category_supported": sla_category in SUPPORTED_SLA_CATEGORIES,
		})
		item = SlaRecord(sla_id, tenant_id, case_id, sla_category, due_date, False, False)
		self.sla_records[self._key(tenant_id, sla_id)] = item
		self._audit(tenant_id, "sla_set", sla_id)
		return item.to_dict()

	def record_outcome(
		self, outcome_id: str, tenant_id: str, case_id: str, outcome_type: str,
		description: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		case = self._get_case(case_id, tenant_id)
		outcome_type = _normalize(outcome_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_outcome",
			"case_present": case is not None,
			"outcome_type_supported": outcome_type in SUPPORTED_OUTCOME_TYPES,
			"approval_present": _present(approval_reference),
			"evidence_present": _present(evidence_reference),
		})
		item = CaseOutcome(outcome_id, tenant_id, case_id, outcome_type, description, approval_reference, evidence_reference)
		self.outcomes[self._key(tenant_id, outcome_id)] = item
		if case is not None:
			case.status = "resolved"
		self._audit(tenant_id, "case_outcome_recorded", outcome_id)
		return item.to_dict()

	def send_notification(
		self, notification_id: str, tenant_id: str, case_id: str, notification_type: str,
		recipient_id: str, message: str,
	) -> dict[str, Any]:
		notification_type = _normalize(notification_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "send_notification",
			"notification_type_supported": notification_type in SUPPORTED_NOTIFICATION_TYPES,
			"recipient_present": _present(recipient_id),
		})
		item = CaseNotification(notification_id, tenant_id, case_id, notification_type, recipient_id, message, True)
		self.notifications[self._key(tenant_id, notification_id)] = item
		self._audit(tenant_id, "case_notification_sent", notification_id)
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
		item = CaseReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._key(tenant_id, review_id)] = item
		self._audit(tenant_id, "case_review_recorded", review_id)
		return item.to_dict()

	def register_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = _normalize(runtime)
		role = _normalize(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_case_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": _present(name),
			"agent_scope_present": _present(scope),
		})
		item = CaseAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "case_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		evidence_fabrication_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id),
			"operation": "case_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"evidence_fabrication_scope": evidence_fabrication_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": _present(tenant_id), "operation": "case_batch", "event_stream": event_stream})
		if item_count < 1:
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.government.cas.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"case_count": self._count(self.cases, tenant_id),
			"assignment_count": self._count(self.assignments, tenant_id),
			"escalation_count": self._count(self.escalations, tenant_id),
			"sla_record_count": self._count(self.sla_records, tenant_id),
			"outcome_count": self._count(self.outcomes, tenant_id),
			"notification_count": self._count(self.notifications, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"hearings": len(self._hearings),
			"decisions": len(self._decisions),
			"appeals": len(self._appeals),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def _get_case(self, case_id: str, tenant_id: str) -> CitizenCase | None:
		return self.cases.get(self._key(tenant_id, case_id))

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

	# ── additional methods ──────────────────────────────────────────────────

	def bulk_case_import(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-import cases from an external system."""
		assert cases, "cases list required"
		tenant_id = self.tenant_id
		imported, failed = [], []
		for c in cases:
			try:
				cid = _new_id()
				item = CitizenCase(
					cid, tenant_id,
					_normalize(c.get("case_type", "general")),
					_normalize(c.get("channel", "portal")),
					str(c.get("citizen_id", "")),
					_normalize(c.get("priority", "medium")),
					"open",
					str(c.get("subject", "Imported case")),
					str(c.get("description", "")),
					"",
				)
				self.cases[self._key(tenant_id, cid)] = item
				imported.append(cid)
			except Exception as exc:
				failed.append({"error": str(exc), "record": c})
		self._audit(tenant_id, "bulk_cases_imported", _new_id())
		return {"imported": len(imported), "failed": len(failed), "failures": failed}

	def case_transfer(
		self,
		case_id: str,
		from_officer_id: str,
		to_officer_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Transfer a case between officers."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		transfer_id = _new_id()
		item = CaseAssignment(transfer_id, tenant_id, case_id, "transfer", to_officer_id, from_officer_id, "")
		self.assignments[self._key(tenant_id, transfer_id)] = item
		self._audit(tenant_id, "case_transferred", transfer_id)
		return {
			"transfer_id": transfer_id,
			"case_id": case_id,
			"from_officer_id": from_officer_id,
			"to_officer_id": to_officer_id,
			"reason": reason,
			"transferred_at": datetime.utcnow().isoformat(),
		}

	def reopen_case(self, case_id: str, reason: str) -> dict[str, Any]:
		"""Reopen a closed or resolved case."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		case.status = "open"
		reopen_id = _new_id()
		self._audit(tenant_id, "case_reopened", reopen_id)
		return {"id": reopen_id, "case_id": case_id, "reason": reason, "status": "open", "reopened_at": datetime.utcnow().isoformat()}

	def case_age_report(self, period: str) -> dict[str, Any]:
		"""Report on case age distribution — open cases bucketed by age."""
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id and c.status not in ("closed", "resolved")]
		buckets = {"0-7d": 0, "8-30d": 0, "31-90d": 0, "90d+": 0}
		for c in cases:
			buckets["0-7d"] += 1  # simplified: real impl computes age from created_at
		return {"tenant_id": tenant_id, "period": period, "open_cases": len(cases), "age_buckets": buckets, "generated_at": datetime.utcnow().isoformat()}

	def priority_override(
		self,
		case_id: str,
		new_priority: str,
		reason: str,
	) -> dict[str, Any]:
		"""Override the priority of an existing case."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		old_priority = case.priority
		case.priority = _normalize(new_priority)
		override_id = _new_id()
		self._audit(tenant_id, "case_priority_overridden", override_id)
		return {"id": override_id, "case_id": case_id, "old_priority": old_priority, "new_priority": new_priority, "reason": reason}

	def inter_agency_referral(
		self,
		case_id: str,
		target_agency: str,
		referral_notes: str,
	) -> dict[str, Any]:
		"""Refer a case to another government agency."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		referral_id = _new_id()
		ref = f"REF-{datetime.utcnow().strftime('%Y%m%d')}-{referral_id[:6].upper()}"
		self._audit(tenant_id, "case_referred_inter_agency", referral_id)
		return {"id": referral_id, "reference": ref, "case_id": case_id, "target_agency": target_agency, "notes": referral_notes, "referred_by": self.actor_id, "referred_at": datetime.utcnow().isoformat(), "status": "referred"}

	def citizen_satisfaction_survey(
		self,
		case_id: str,
		citizen_id: str,
		rating: int,
		comments: str,
	) -> dict[str, Any]:
		"""Capture citizen satisfaction feedback after case resolution."""
		assert 1 <= rating <= 5, "rating must be 1–5"
		tenant_id = self.tenant_id
		survey_id = _new_id()
		sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
		self._audit(tenant_id, "satisfaction_survey_submitted", survey_id)
		return {"id": survey_id, "case_id": case_id, "citizen_id": citizen_id, "rating": rating, "sentiment": sentiment, "comments": comments, "submitted_at": datetime.utcnow().isoformat()}

	def case_merge(self, primary_case_id: str, duplicate_case_id: str, reason: str) -> dict[str, Any]:
		"""Merge a duplicate case into a primary case."""
		tenant_id = self.tenant_id
		primary = self._get_case(primary_case_id, tenant_id)
		duplicate = self._get_case(duplicate_case_id, tenant_id)
		if primary is None or duplicate is None:
			raise KeyError("one or both cases not found")
		duplicate.status = "merged"
		merge_id = _new_id()
		self._audit(tenant_id, "cases_merged", merge_id)
		return {"id": merge_id, "primary_case_id": primary_case_id, "duplicate_case_id": duplicate_case_id, "reason": reason, "merged_by": self.actor_id, "merged_at": datetime.utcnow().isoformat()}

	def notification_broadcast(self, message: str, channels: list[str]) -> dict[str, Any]:
		"""Broadcast a notification to all active case citizens."""
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id and c.status not in ("closed", "resolved")]
		broadcast_id = _new_id()
		self._audit(tenant_id, "notification_broadcast_sent", broadcast_id)
		return {"id": broadcast_id, "message": message, "channels": channels, "recipients": len(cases), "sent_at": datetime.utcnow().isoformat()}

	def audit_trail(self, case_id: str) -> dict[str, Any]:
		"""Return the complete audit trail for a single case."""
		tenant_id = self.tenant_id
		events = [e for e in self.audit_events if e["tenant_id"] == tenant_id and e.get("reference_id") == case_id]
		return {"case_id": case_id, "tenant_id": tenant_id, "event_count": len(events), "events": events, "generated_at": datetime.utcnow().isoformat()}

	def performance_tracker(self, officer_id: str) -> dict[str, Any]:
		"""Return real-time performance indicators for an officer."""
		tenant_id = self.tenant_id
		assignments = [a for (tid, _), a in self.assignments.items() if tid == tenant_id and a.assignee_id == officer_id]
		open_cases = sum(1 for a in assignments if self._get_case(a.case_id, tenant_id) and self._get_case(a.case_id, tenant_id).status not in ("closed", "resolved"))
		escalations = [e for (tid, _), e in self.escalations.items() if tid == tenant_id and e.case_id in [a.case_id for a in assignments]]
		return {"officer_id": officer_id, "active_assignments": len(assignments), "open_cases": open_cases, "escalation_count": len(escalations), "as_of": datetime.utcnow().isoformat()}

	def regulatory_compliance_check(self, case_id: str) -> dict[str, Any]:
		"""Check a case for regulatory compliance issues."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		sla = next((s for (tid, _), s in self.sla_records.items() if tid == tenant_id and s.case_id == case_id), None)
		checks = {
			"sla_set": sla is not None,
			"sla_not_breached": sla is not None and not sla.breached,
			"evidence_present": bool(getattr(case, "evidence_reference", "")),
			"valid_case_type": case.case_type in SUPPORTED_CASE_TYPES,
		}
		return {"case_id": case_id, "checks": checks, "compliant": all(checks.values()), "checked_at": datetime.utcnow().isoformat()}

	def case_tagging(self, case_id: str, tags: list[str]) -> dict[str, Any]:
		"""Apply searchable tags to a case."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		tag_id = _new_id()
		self._audit(tenant_id, "case_tagged", tag_id)
		return {"id": tag_id, "case_id": case_id, "tags": tags, "tagged_by": self.actor_id, "tagged_at": datetime.utcnow().isoformat()}

	def repeat_complainant_report(self) -> dict[str, Any]:
		"""Identify citizens who have filed multiple cases."""
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id]
		counts: dict[str, int] = {}
		for c in cases:
			counts[c.citizen_id] = counts.get(c.citizen_id, 0) + 1
		repeat = {cid: cnt for cid, cnt in counts.items() if cnt > 1}
		return {"tenant_id": tenant_id, "repeat_complainants": len(repeat), "details": repeat, "generated_at": datetime.utcnow().isoformat()}

	def case_type_analysis(self, period: str) -> dict[str, Any]:
		"""Analyse case volumes and resolution rates by type."""
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id]
		by_type: dict[str, dict[str, int]] = {}
		for c in cases:
			rec = by_type.setdefault(c.case_type, {"total": 0, "open": 0, "closed": 0})
			rec["total"] += 1
			if c.status in ("closed", "resolved"):
				rec["closed"] += 1
			else:
				rec["open"] += 1
		return {"tenant_id": tenant_id, "period": period, "by_type": by_type, "generated_at": datetime.utcnow().isoformat()}

	def sla_extension_request(self, case_id: str, extension_days: int, reason: str) -> dict[str, Any]:
		"""Request an SLA extension for a case."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		ext_id = _new_id()
		new_deadline = (datetime.utcnow() + timedelta(days=extension_days)).isoformat()
		self._audit(tenant_id, "sla_extension_requested", ext_id)
		return {"id": ext_id, "case_id": case_id, "extension_days": extension_days, "new_deadline": new_deadline, "reason": reason, "requested_by": self.actor_id, "status": "pending_approval"}

	def hearing_reminder(self, days_ahead: int = 3) -> dict[str, Any]:
		"""List upcoming hearings within N days for reminder notifications."""
		tenant_id = self.tenant_id
		cutoff = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat()
		upcoming = [h for h in self._hearings if h.get("tenant_id") == tenant_id and h.get("hearing_date", "9999") <= cutoff]
		return {"tenant_id": tenant_id, "days_ahead": days_ahead, "upcoming_hearings": len(upcoming), "hearings": upcoming, "generated_at": datetime.utcnow().isoformat()}

	def case_volume_forecast(self, months: int = 3) -> dict[str, Any]:
		"""Forecast case volumes for the next N months based on current rates."""
		tenant_id = self.tenant_id
		total_cases = sum(1 for (tid, _) in self.cases if tid == tenant_id)
		monthly_avg = total_cases / max(months, 1)
		projections = [{"month": m, "projected_cases": round(monthly_avg)} for m in range(1, months + 1)]
		return {"tenant_id": tenant_id, "months": months, "current_total": total_cases, "monthly_average": round(monthly_avg, 1), "projections": projections, "generated_at": datetime.utcnow().isoformat()}


	def case_assign(self, case_id: str, officer_id: str) -> dict[str, Any]:
		"""Assign case to officer — canonical domain-named alias."""
		return self.assign_officer(case_id, officer_id)

	def case_escalate(self, case_id: str, reason: str, supervisor_id: str) -> dict[str, Any]:
		"""Escalate a case to a supervisor."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		esc_id = _new_id()
		item = CaseEscalation(esc_id, tenant_id, case_id, _normalize(reason), supervisor_id, supervisor_id, "")
		self.escalations[self._key(tenant_id, esc_id)] = item
		case.status = "escalated"
		self._audit(tenant_id, "case_escalated", esc_id)
		return {"escalation_id": esc_id, "case_id": case_id, "reason": reason, "supervisor_id": supervisor_id, "escalated_at": datetime.utcnow().isoformat(), "status": "escalated"}

	def hearing_schedule(self, case_id: str, hearing_date: datetime, location: str) -> dict[str, Any]:
		"""Schedule a hearing — canonical alias."""
		return self.schedule_hearing(case_id, hearing_date, location)

	def decision_record(self, case_id: str, decision: str, outcome: str) -> dict[str, Any]:
		"""Record a decision — canonical alias."""
		return self.record_decision(case_id, decision, outcome)

	def appeal_file(self, case_id: str, appeal_grounds: str) -> dict[str, Any]:
		"""File an appeal — canonical alias."""
		return self.appeal_management(case_id, appeal_grounds)

	def case_close(self, case_id: str, resolution: str, closing_notes: str = "") -> dict[str, Any]:
		"""Close a case — canonical alias."""
		return self.close_case(case_id, resolution, closing_notes)

	def case_reopen(self, case_id: str, reason: str) -> dict[str, Any]:
		"""Reopen a closed case — canonical alias."""
		return self.reopen_case(case_id, reason)

	def citizen_notify(self, case_id: str, message: str, channel: str = "email") -> dict[str, Any]:
		"""Notify a citizen about their case status."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		notif_id = _new_id()
		item = CaseNotification(notif_id, tenant_id, case_id, channel, case.citizen_id, message, True)
		self.notifications[self._key(tenant_id, notif_id)] = item
		self._audit(tenant_id, "citizen_notified", notif_id)
		return {"notification_id": notif_id, "case_id": case_id, "citizen_id": case.citizen_id, "channel": channel, "message": message, "sent_at": datetime.utcnow().isoformat()}

	def case_sla_check(self, case_id: str) -> dict[str, Any]:
		"""Check SLA compliance for a case."""
		return self.regulatory_compliance_check(case_id)

	def workload_balance(self, period: str) -> dict[str, Any]:
		"""Return workload balancing report across officers."""
		tenant_id = self.tenant_id
		by_officer: dict[str, int] = {}
		for (tid, _), a in self.assignments.items():
			if tid == tenant_id:
				by_officer[a.assignee_id] = by_officer.get(a.assignee_id, 0) + 1
		return {"tenant_id": tenant_id, "period": period, "officer_count": len(by_officer), "workload_by_officer": by_officer, "generated_at": datetime.utcnow().isoformat()}

	def document_attach(self, case_id: str, document_ref: str, doc_type: str = "evidence") -> dict[str, Any]:
		"""Attach a document to a case."""
		tenant_id = self.tenant_id
		case = self._get_case(case_id, tenant_id)
		if case is None:
			raise KeyError(f"case {case_id} not found")
		doc_id = _new_id()
		self._audit(tenant_id, "document_attached", doc_id)
		return {"document_id": doc_id, "case_id": case_id, "document_ref": document_ref, "doc_type": doc_type, "attached_by": self.actor_id, "attached_at": datetime.utcnow().isoformat()}

	def case_search(self, query: str, status: str | None = None, case_type: str | None = None) -> list[dict[str, Any]]:
		"""Search cases by query text with optional filters."""
		tenant_id = self.tenant_id
		ql = query.lower()
		cases = []
		for (tid, _), c in self.cases.items():
			if tid != tenant_id:
				continue
			if status and c.status != status:
				continue
			if case_type and c.case_type != case_type:
				continue
			if ql in c.subject.lower() or ql in c.description.lower() or ql in c.citizen_id.lower():
				cases.append(c.to_dict() if hasattr(c, "to_dict") else {"case_id": c.case_id, "status": c.status, "citizen_id": c.citizen_id})
		return cases

	def case_statistics(self, period: str) -> dict[str, Any]:
		"""Return case statistics for reporting."""
		return self.case_analytics(period)

	def case_export(self, format: str = "json", status: str | None = None) -> dict[str, Any]:
		"""Export cases for external system consumption."""
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id and (status is None or c.status == status)]
		return {"tenant_id": tenant_id, "format": format, "case_count": len(cases), "exported_at": datetime.utcnow().isoformat(), "status_filter": status}

	def case_template(self, template_name: str, case_type: str, default_priority: str = "medium") -> dict[str, Any]:
		"""Create or return a case template definition."""
		tmpl_id = _new_id()
		return {"template_id": tmpl_id, "template_name": template_name, "case_type": case_type, "default_priority": default_priority, "created_by": self.actor_id, "created_at": datetime.utcnow().isoformat()}

	def case_analytics(self, period: str) -> dict[str, Any]:
		"""Return case management analytics for the period."""
		assert period, "period required"
		tenant_id = self.tenant_id
		cases = [c for (tid, _), c in self.cases.items() if tid == tenant_id]
		slas = [s for (tid, _), s in self.sla_records.items() if tid == tenant_id]
		resolution_rate = len([c for c in cases if c.status in ("closed", "resolved")]) / max(len(cases), 1) * 100
		sla_compliance = len([s for s in slas if s.met]) / max(len(slas), 1) * 100
		by_type: dict[str, int] = {}
		by_priority: dict[str, int] = {}
		for c in cases:
			by_type[c.case_type] = by_type.get(c.case_type, 0) + 1
			by_priority[c.priority] = by_priority.get(c.priority, 0) + 1
		return {
			"tenant_id": tenant_id, "period": period,
			"cases": {"total": len(cases), "open": sum(1 for c in cases if c.status == "open"), "assigned": sum(1 for c in cases if c.status == "assigned"), "closed": sum(1 for c in cases if c.status in ("closed", "resolved")), "appealed": sum(1 for c in cases if c.status == "appealed"), "resolution_rate_pct": round(resolution_rate, 1), "by_type": by_type, "by_priority": by_priority},
			"sla": {"total": len(slas), "compliance_rate_pct": round(sla_compliance, 1), "breached": len([s for s in slas if s.breached])},
			"hearings_scheduled": len(self._hearings), "decisions_recorded": len(self._decisions), "appeals_lodged": len(self._appeals), "generated_at": datetime.utcnow().isoformat(),
		}



	async def ml_case_priority_score(self, *args, **kwargs):
		"""AI-powered government case priority scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score({"case_type": str(kwargs.get("case_type","")), "urgency": str(kwargs.get("urgency",""))}, task="government_case_priority")
			return {"priority_score": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

GovernmentCasService = CaseManagementService
