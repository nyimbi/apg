"""Executable service layer for APG ITSM Incident Management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CATEGORIES, SUPPORTED_ESCALATION_LEVELS,
		SUPPORTED_IMPACT_LEVELS, SUPPORTED_PRIORITIES,
		SUPPORTED_RESOLUTION_CODES, SUPPORTED_STATUSES,
		SUPPORTED_UPDATE_TYPES, SUPPORTED_URGENCY_LEVELS,
		SLA_MINUTES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import ItIncident, ItIncidentSLA, ItIncidentUpdate, ItMajorIncident
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_CATEGORIES, SUPPORTED_ESCALATION_LEVELS,
		SUPPORTED_IMPACT_LEVELS, SUPPORTED_PRIORITIES,
		SUPPORTED_RESOLUTION_CODES, SUPPORTED_STATUSES,
		SUPPORTED_UPDATE_TYPES, SUPPORTED_URGENCY_LEVELS,
		SLA_MINUTES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import ItIncident, ItIncidentSLA, ItIncidentUpdate, ItMajorIncident  # type: ignore

try:
	from uuid6 import uuid7
	def _uuid7() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid
	def _uuid7() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
	return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
	if not ts:
		return None
	try:
		dt = datetime.fromisoformat(ts)
		return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
	except (ValueError, TypeError):
		return None


def _present(v: Any) -> bool:
	if v is None:
		return False
	if isinstance(v, str):
		return bool(v.strip())
	return True


def _sla_due(created_at: str, priority: str) -> str:
	minutes = SLA_MINUTES.get(priority, 1440)
	dt = _parse_iso(created_at) or _now_dt()
	return (dt + timedelta(minutes=minutes)).isoformat()


# Priority numeric rank for sorting
_PRIORITY_RANK = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}


class IncidentManagementService:
	"""Tenant-scoped ITIL v4 Incident Management runtime for APG ITSM."""

	def __init__(self) -> None:
		self._incidents: dict[tuple[str, str], ItIncident] = {}
		self._updates: dict[tuple[str, str], ItIncidentUpdate] = {}
		self._slas: dict[tuple[str, str], ItIncidentSLA] = {}
		self._major_incidents: dict[tuple[str, str], ItMajorIncident] = {}
		# Timeline: (tenant_id, incident_id) -> list of timeline entries
		self._timelines: dict[tuple[str, str], list[dict[str, Any]]] = {}
		# Escalation records: (tenant_id, incident_id) -> list
		self._escalations: dict[tuple[str, str], list[dict[str, Any]]] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / evaluation
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Incident Creation
	# ------------------------------------------------------------------

	def create_incident(
		self,
		tenant_id: str,
		title: str,
		category: str,
		priority: str,
		*,
		inc_id: str | None = None,
		description: str = "",
		subcategory: str | None = None,
		impact: str = "medium",
		urgency: str = "medium",
		assigned_to: str | None = None,
		assigned_team: str | None = None,
		reported_by: str = "system",
		affected_ci_id: str | None = None,
		affected_ci_name: str | None = None,
		affected_service: str | None = None,
		source_alert_id: str | None = None,
		tags: list[str] | None = None,
		custom_fields: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create a new ITIL v4 incident and initialize its SLA clock."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "create_incident",
			"title_present": _present(title),
			"priority_supported": priority in SUPPORTED_PRIORITIES,
			"category_supported": category in SUPPORTED_CATEGORIES,
		})
		inc = ItIncident(
			id=inc_id or _uuid7(),
			tenant_id=tenant_id,
			title=title,
			description=description,
			category=category,
			subcategory=subcategory,
			priority=priority,
			impact=impact,
			urgency=urgency,
			status="new",
			assigned_to=assigned_to,
			assigned_team=assigned_team,
			reported_by=reported_by,
			affected_ci_id=affected_ci_id,
			affected_ci_name=affected_ci_name,
			affected_service=affected_service,
			source_alert_id=source_alert_id,
			tags=tags or [],
			custom_fields=custom_fields or {},
			sla_due_at=_sla_due(_now(), priority),
			resolve_sla_minutes=SLA_MINUTES.get(priority, 1440),
		)
		key = (tenant_id, inc.id)
		self._incidents[key] = inc
		self._timelines[key] = [{"ts": inc.created_at, "event": "created", "actor": reported_by, "notes": ""}]
		self._escalations[key] = []
		# Initialise resolve SLA record
		self._init_sla(tenant_id, inc)
		self._audit(tenant_id, "incident_created", inc.id)
		return inc.model_dump()

	# ------------------------------------------------------------------
	# Lifecycle Transitions
	# ------------------------------------------------------------------

	def acknowledge_incident(
		self,
		tenant_id: str,
		incident_id: str,
		acknowledged_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Transition new → acknowledged. Starts the clock for acknowledgement SLA."""
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		assert inc.status == "new", f"acknowledge requires status=new, got {inc.status!r}"
		assert _present(acknowledged_by), "acknowledged_by required"
		inc.status = "acknowledged"
		inc.acknowledged_at = _now()
		inc.version += 1
		self._add_timeline(tenant_id, incident_id, "acknowledged", acknowledged_by, notes)
		self._add_update(tenant_id, incident_id, "status_change", acknowledged_by, f"Acknowledged. {notes}", previous_status="new", new_status="acknowledged")
		self._audit(tenant_id, "incident_acknowledged", incident_id)
		return {"incident_id": incident_id, "status": "acknowledged", "acknowledged_at": inc.acknowledged_at}

	def start_incident(
		self,
		tenant_id: str,
		incident_id: str,
		started_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Transition acknowledged → in_progress."""
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		assert inc.status in ("new", "acknowledged"), f"start requires new/acknowledged, got {inc.status!r}"
		inc.status = "in_progress"
		inc.in_progress_at = _now()
		if not inc.acknowledged_at:
			inc.acknowledged_at = _now()
		inc.version += 1
		self._add_timeline(tenant_id, incident_id, "in_progress", started_by, notes)
		self._audit(tenant_id, "incident_in_progress", incident_id)
		return {"incident_id": incident_id, "status": "in_progress"}

	def resolve_incident(
		self,
		tenant_id: str,
		incident_id: str,
		resolved_by: str,
		resolution_code: str,
		resolution_notes: str,
		workaround: str | None = None,
		root_cause_summary: str | None = None,
	) -> dict[str, Any]:
		"""Transition → resolved. Closes the SLA clock."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "resolve_incident",
			"resolution_code_present": resolution_code in SUPPORTED_RESOLUTION_CODES,
		})
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		assert inc.status not in ("resolved", "closed"), f"already {inc.status!r}"
		inc.status = "resolved"
		inc.resolved_at = _now()
		inc.resolution_code = resolution_code
		inc.resolution_notes = resolution_notes
		inc.workaround = workaround
		inc.root_cause_summary = root_cause_summary
		inc.version += 1
		# Evaluate SLA
		self._evaluate_sla(tenant_id, incident_id, inc)
		self._add_timeline(tenant_id, incident_id, "resolved", resolved_by, resolution_notes)
		self._add_update(tenant_id, incident_id, "resolution", resolved_by, resolution_notes, previous_status="in_progress", new_status="resolved")
		self._audit(tenant_id, "incident_resolved", incident_id)
		return inc.model_dump()

	def close_incident(
		self,
		tenant_id: str,
		incident_id: str,
		closed_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Transition resolved → closed."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "close_incident",
			"is_resolved": self._get_inc_or_raise(tenant_id, incident_id).status == "resolved",
		})
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		inc.status = "closed"
		inc.closed_at = _now()
		inc.version += 1
		self._add_timeline(tenant_id, incident_id, "closed", closed_by, notes)
		self._audit(tenant_id, "incident_closed", incident_id)
		return {"incident_id": incident_id, "status": "closed", "closed_at": inc.closed_at}

	def reopen_incident(
		self,
		tenant_id: str,
		incident_id: str,
		reopened_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Reopen a resolved or closed incident."""
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		prev = inc.status
		inc.status = "in_progress"
		inc.resolved_at = None
		inc.closed_at = None
		inc.version += 1
		self._add_timeline(tenant_id, incident_id, "reopened", reopened_by, f"reason={reason} prev={prev}")
		self._audit(tenant_id, "incident_reopened", incident_id)
		return {"incident_id": incident_id, "status": "in_progress", "previous_status": prev}

	# ------------------------------------------------------------------
	# Updates
	# ------------------------------------------------------------------

	def add_update(
		self,
		tenant_id: str,
		incident_id: str,
		update_type: str,
		author_id: str,
		content: str,
		*,
		internal_only: bool = False,
		attachments: list[str] | None = None,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "add_update",
			"author_present": _present(author_id),
		})
		_ = self._get_inc_or_raise(tenant_id, incident_id)
		update = ItIncidentUpdate(
			tenant_id=tenant_id,
			incident_id=incident_id,
			update_type=update_type if update_type in SUPPORTED_UPDATE_TYPES else "note",
			author_id=author_id,
			content=content,
			internal_only=internal_only,
			attachments=attachments or [],
		)
		self._updates[(tenant_id, update.id)] = update
		self._add_timeline(tenant_id, incident_id, f"update:{update_type}", author_id, content[:120])
		self._audit(tenant_id, "incident_update_added", update.id)
		return update.model_dump()

	def get_updates(self, tenant_id: str, incident_id: str, include_internal: bool = True) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), upd in self._updates.items():
			if tid != tenant_id or upd.incident_id != incident_id:
				continue
			if not include_internal and upd.internal_only:
				continue
			results.append(upd.model_dump())
		return sorted(results, key=lambda r: r["created_at"])

	# ------------------------------------------------------------------
	# Assignment
	# ------------------------------------------------------------------

	def assign_incident(
		self,
		tenant_id: str,
		incident_id: str,
		assigned_to: str | None,
		assigned_team: str | None,
		assigned_by: str,
		notes: str = "",
	) -> dict[str, Any]:
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		prev_assignee = inc.assigned_to
		inc.assigned_to = assigned_to
		inc.assigned_team = assigned_team
		inc.version += 1
		self._add_update(tenant_id, incident_id, "assignment_change", assigned_by,
			f"Reassigned from {prev_assignee!r} to {assigned_to!r}. {notes}")
		self._audit(tenant_id, "incident_assigned", incident_id)
		return {"incident_id": incident_id, "assigned_to": assigned_to, "assigned_team": assigned_team}

	# ------------------------------------------------------------------
	# Escalation
	# ------------------------------------------------------------------

	def escalate_incident(
		self,
		tenant_id: str,
		incident_id: str,
		escalation_level: str,
		escalated_by: str,
		target_id: str,
		reason: str,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "escalate_incident",
			"escalation_level_supported": escalation_level in SUPPORTED_ESCALATION_LEVELS,
		})
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		record: dict[str, Any] = {
			"escalation_id": _uuid7(),
			"incident_id": incident_id,
			"escalation_level": escalation_level,
			"escalated_by": escalated_by,
			"target_id": target_id,
			"reason": reason,
			"escalated_at": _now(),
		}
		self._escalations[(tenant_id, incident_id)].append(record)
		self._add_timeline(tenant_id, incident_id, f"escalated:{escalation_level}", escalated_by, reason)
		self._add_update(tenant_id, incident_id, "escalation", escalated_by, f"Escalated to {escalation_level}: {reason}")
		self._audit(tenant_id, "incident_escalated", incident_id)
		return record

	def get_escalations(self, tenant_id: str, incident_id: str) -> list[dict[str, Any]]:
		return list(self._escalations.get((tenant_id, incident_id), []))

	# ------------------------------------------------------------------
	# Priority / SLA
	# ------------------------------------------------------------------

	def reprioritize(
		self,
		tenant_id: str,
		incident_id: str,
		new_priority: str,
		changed_by: str,
		justification: str,
	) -> dict[str, Any]:
		assert new_priority in SUPPORTED_PRIORITIES, f"unsupported priority {new_priority!r}"
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		old_priority = inc.priority
		inc.priority = new_priority
		inc.resolve_sla_minutes = SLA_MINUTES[new_priority]
		inc.sla_due_at = _sla_due(inc.created_at, new_priority)
		inc.version += 1
		self._add_update(tenant_id, incident_id, "note", changed_by,
			f"Priority changed {old_priority} → {new_priority}. Justification: {justification}")
		self._audit(tenant_id, "incident_reprioritized", incident_id)
		return {"incident_id": incident_id, "old_priority": old_priority, "new_priority": new_priority, "sla_due_at": inc.sla_due_at}

	def check_sla_breaches(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Scan all open incidents, flag SLA breaches, return breached list."""
		breached: list[dict[str, Any]] = []
		now = _now_dt()
		for (tid, iid), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			if inc.status in ("resolved", "closed", "cancelled"):
				continue
			if inc.sla_due_at and not inc.sla_breached:
				due_dt = _parse_iso(inc.sla_due_at)
				if due_dt and now > due_dt:
					inc.sla_breached = True
					inc.sla_breached_at = now.isoformat()
					self._audit(tenant_id, "sla_breach_detected", iid)
					breached.append({
						"incident_id": iid,
						"priority": inc.priority,
						"status": inc.status,
						"sla_due_at": inc.sla_due_at,
						"breached_at": inc.sla_breached_at,
						"overdue_minutes": round((now - due_dt).total_seconds() / 60.0, 1),
					})
		return breached

	def sla_status(self, tenant_id: str, incident_id: str) -> dict[str, Any]:
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		now = _now_dt()
		created_dt = _parse_iso(inc.created_at) or now
		elapsed_min = (now - created_dt).total_seconds() / 60.0
		target_min = SLA_MINUTES.get(inc.priority, 1440)
		remaining_min = max(0.0, target_min - elapsed_min)
		return {
			"incident_id": incident_id,
			"priority": inc.priority,
			"status": inc.status,
			"sla_due_at": inc.sla_due_at,
			"sla_breached": inc.sla_breached,
			"elapsed_minutes": round(elapsed_min, 1),
			"remaining_minutes": round(remaining_min, 1),
			"target_minutes": target_min,
			"as_of": _now(),
		}

	# ------------------------------------------------------------------
	# Major Incident
	# ------------------------------------------------------------------

	def declare_major_incident(
		self,
		tenant_id: str,
		incident_id: str,
		incident_commander_id: str,
		declared_by: str,
		affected_services: list[str],
		customer_impact_statement: str,
		linked_incident_ids: list[str] | None = None,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "declare_major_incident",
			"commander_present": _present(incident_commander_id),
		})
		inc = self._get_inc_or_raise(tenant_id, incident_id)
		if inc.is_major:
			raise ValueError(f"incident {incident_id!r} is already a major incident")
		inc.is_major = True
		inc.major_declared_at = _now()
		inc.incident_commander_id = incident_commander_id
		inc.version += 1
		major = ItMajorIncident(
			tenant_id=tenant_id,
			incident_id=incident_id,
			incident_commander_id=incident_commander_id,
			declared_by=declared_by,
			affected_services=affected_services,
			customer_impact_statement=customer_impact_statement,
			linked_incident_ids=linked_incident_ids or [],
		)
		self._major_incidents[(tenant_id, major.id)] = major
		self._add_timeline(tenant_id, incident_id, "major_declared", declared_by, customer_impact_statement[:120])
		self._audit(tenant_id, "incident_major_declared", major.id)
		return major.model_dump()

	def update_major_incident(
		self,
		tenant_id: str,
		major_id: str,
		update_type: str,
		content: str,
		updated_by: str,
		external: bool = False,
	) -> dict[str, Any]:
		major = self._major_incidents.get((tenant_id, major_id))
		if major is None:
			raise KeyError(f"major incident {major_id!r} not found")
		entry = {"ts": _now(), "type": update_type, "content": content, "by": updated_by}
		if external:
			major.external_communications.append(entry)
		else:
			major.internal_status_updates.append(entry)
		return major.model_dump()

	def complete_pir(
		self,
		tenant_id: str,
		major_id: str,
		pir_summary: str,
		lessons_learned: list[str],
		completed_by: str,
	) -> dict[str, Any]:
		major = self._major_incidents.get((tenant_id, major_id))
		if major is None:
			raise KeyError(f"major incident {major_id!r} not found")
		major.pir_completed_at = _now()
		major.pir_summary = pir_summary
		major.lessons_learned = lessons_learned
		self._audit(tenant_id, "incident_pir_completed", major_id)
		return major.model_dump()

	# ------------------------------------------------------------------
	# Querying
	# ------------------------------------------------------------------

	def get_incident(self, tenant_id: str, incident_id: str) -> dict[str, Any]:
		return self._get_inc_or_raise(tenant_id, incident_id).model_dump()

	def list_incidents(
		self,
		tenant_id: str,
		*,
		status: str | None = None,
		priority: str | None = None,
		category: str | None = None,
		assigned_to: str | None = None,
		assigned_team: str | None = None,
		sla_breached: bool | None = None,
		is_major: bool | None = None,
	) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			if status and inc.status != status:
				continue
			if priority and inc.priority != priority:
				continue
			if category and inc.category != category:
				continue
			if assigned_to and inc.assigned_to != assigned_to:
				continue
			if assigned_team and inc.assigned_team != assigned_team:
				continue
			if sla_breached is not None and inc.sla_breached != sla_breached:
				continue
			if is_major is not None and inc.is_major != is_major:
				continue
			results.append(inc.model_dump())
		# Sort: priority desc, created_at asc
		results.sort(key=lambda r: (-_PRIORITY_RANK.get(r["priority"], 0), r["created_at"]))
		return results

	def incident_timeline(self, tenant_id: str, incident_id: str) -> list[dict[str, Any]]:
		_ = self._get_inc_or_raise(tenant_id, incident_id)
		return list(self._timelines.get((tenant_id, incident_id), []))

	def incident_queue(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Priority-sorted open incident queue with SLA remaining."""
		now = _now_dt()
		queue: list[dict[str, Any]] = []
		for (tid, iid), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			if inc.status in ("resolved", "closed", "cancelled"):
				continue
			created_dt = _parse_iso(inc.created_at) or now
			elapsed_min = (now - created_dt).total_seconds() / 60.0
			target_min = SLA_MINUTES.get(inc.priority, 1440)
			remaining_min = max(0.0, target_min - elapsed_min)
			queue.append({
				"incident_id": iid,
				"title": inc.title,
				"priority": inc.priority,
				"status": inc.status,
				"category": inc.category,
				"assigned_to": inc.assigned_to,
				"sla_remaining_minutes": round(remaining_min, 1),
				"sla_breached": inc.sla_breached,
				"is_major": inc.is_major,
				"elapsed_minutes": round(elapsed_min, 1),
				"_rank": _PRIORITY_RANK.get(inc.priority, 0),
			})
		queue.sort(key=lambda r: (-r["_rank"], r["sla_remaining_minutes"]))
		for row in queue:
			del row["_rank"]
		return queue

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		by_status: dict[str, int] = {s: 0 for s in SUPPORTED_STATUSES}
		by_priority: dict[str, int] = {p: 0 for p in SUPPORTED_PRIORITIES}
		total_sla_breached = 0
		total_major = 0
		total = 0
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			total += 1
			by_status[inc.status] = by_status.get(inc.status, 0) + 1
			by_priority[inc.priority] = by_priority.get(inc.priority, 0) + 1
			if inc.sla_breached:
				total_sla_breached += 1
			if inc.is_major:
				total_major += 1
		return {
			"tenant_id": tenant_id,
			"total": total,
			"by_status": by_status,
			"by_priority": by_priority,
			"sla_breached": total_sla_breached,
			"major_incidents": total_major,
			"as_of": _now(),
		}

	def mean_time_to_resolve(self, tenant_id: str, priority: str | None = None) -> dict[str, Any]:
		durations: list[float] = []
		by_priority: dict[str, list[float]] = {p: [] for p in SUPPORTED_PRIORITIES}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			if inc.status not in ("resolved", "closed"):
				continue
			if priority and inc.priority != priority:
				continue
			created_dt = _parse_iso(inc.created_at)
			resolved_dt = _parse_iso(inc.resolved_at)
			if created_dt and resolved_dt:
				minutes = (resolved_dt - created_dt).total_seconds() / 60.0
				durations.append(minutes)
				by_priority[inc.priority].append(minutes)
		overall_mttr = round(sum(durations) / len(durations), 2) if durations else 0.0
		mttr_by_priority = {
			p: round(sum(v) / len(v), 2) if v else 0.0
			for p, v in by_priority.items()
		}
		return {
			"tenant_id": tenant_id,
			"overall_mttr_minutes": overall_mttr,
			"sample_size": len(durations),
			"mttr_by_priority": mttr_by_priority,
			"as_of": _now(),
		}

	def sla_compliance_report(self, tenant_id: str) -> dict[str, Any]:
		totals: dict[str, int] = {p: 0 for p in SUPPORTED_PRIORITIES}
		within_sla: dict[str, int] = {p: 0 for p in SUPPORTED_PRIORITIES}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			if inc.status not in ("resolved", "closed"):
				continue
			created_dt = _parse_iso(inc.created_at)
			resolved_dt = _parse_iso(inc.resolved_at)
			if not created_dt or not resolved_dt:
				continue
			p = inc.priority
			totals[p] = totals.get(p, 0) + 1
			elapsed_min = (resolved_dt - created_dt).total_seconds() / 60.0
			if elapsed_min <= SLA_MINUTES.get(p, 1440):
				within_sla[p] = within_sla.get(p, 0) + 1
		compliance = {}
		for p in SUPPORTED_PRIORITIES:
			t = totals[p]
			w = within_sla[p]
			compliance[p] = {
				"total_resolved": t,
				"within_sla": w,
				"compliance_pct": round((w / t * 100) if t else 0.0, 2),
				"target_minutes": SLA_MINUTES[p],
			}
		return {"tenant_id": tenant_id, "compliance": compliance, "as_of": _now()}

	def workload_by_team(self, tenant_id: str) -> dict[str, Any]:
		teams: dict[str, dict[str, int]] = {}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			team = inc.assigned_team or "unassigned"
			if team not in teams:
				teams[team] = {"open": 0, "resolved": 0, "total": 0, "sla_breached": 0}
			teams[team]["total"] += 1
			if inc.status in ("resolved", "closed"):
				teams[team]["resolved"] += 1
			else:
				teams[team]["open"] += 1
			if inc.sla_breached:
				teams[team]["sla_breached"] += 1
		return {"tenant_id": tenant_id, "teams": teams, "as_of": _now()}

	def incident_trend(self, tenant_id: str, days: int = 30) -> dict[str, Any]:
		"""Count created/resolved per day bucket over last `days` days."""
		now = _now_dt()
		cutoff = now - timedelta(days=days)
		daily_created: dict[str, int] = {}
		daily_resolved: dict[str, int] = {}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			created_dt = _parse_iso(inc.created_at)
			if created_dt and created_dt >= cutoff:
				day = created_dt.strftime("%Y-%m-%d")
				daily_created[day] = daily_created.get(day, 0) + 1
			resolved_dt = _parse_iso(inc.resolved_at)
			if resolved_dt and resolved_dt >= cutoff:
				day = resolved_dt.strftime("%Y-%m-%d")
				daily_resolved[day] = daily_resolved.get(day, 0) + 1
		all_days = sorted(set(list(daily_created.keys()) + list(daily_resolved.keys())))
		trend = [{"date": d, "created": daily_created.get(d, 0), "resolved": daily_resolved.get(d, 0)} for d in all_days]
		return {"tenant_id": tenant_id, "days": days, "trend": trend, "as_of": _now()}

	def top_recurring_categories(self, tenant_id: str, limit: int = 10) -> list[dict[str, Any]]:
		counts: dict[str, int] = {}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			counts[inc.category] = counts.get(inc.category, 0) + 1
		ranked = sorted(counts.items(), key=lambda x: -x[1])[:limit]
		return [{"category": cat, "count": cnt} for cat, cnt in ranked]

	def analyst_workload(self, tenant_id: str) -> list[dict[str, Any]]:
		analysts: dict[str, dict[str, int]] = {}
		for (tid, _), inc in self._incidents.items():
			if tid != tenant_id:
				continue
			aid = inc.assigned_to or "unassigned"
			if aid not in analysts:
				analysts[aid] = {"analyst_id": aid, "open": 0, "resolved": 0, "p1": 0, "p2": 0}
			if inc.status in ("resolved", "closed"):
				analysts[aid]["resolved"] += 1
			else:
				analysts[aid]["open"] += 1
			if inc.priority == "P1":
				analysts[aid]["p1"] += 1
			elif inc.priority == "P2":
				analysts[aid]["p2"] += 1
		return sorted(analysts.values(), key=lambda r: -(r["open"] + r["p1"] * 2))

	# ------------------------------------------------------------------
	# NATS integration hooks
	# ------------------------------------------------------------------

	def handle_ci_failure_event(
		self,
		tenant_id: str,
		ci_id: str,
		ci_name: str,
		failure_description: str,
		source_alert_id: str | None = None,
	) -> dict[str, Any]:
		"""Create P2 incident from a CMDB CI failure event (itsm_cmdb subscriber)."""
		return self.create_incident(
			tenant_id=tenant_id,
			title=f"CI Failure: {ci_name}",
			category="hardware",
			priority="P2",
			description=failure_description,
			affected_ci_id=ci_id,
			affected_ci_name=ci_name,
			reported_by="itsm_cmdb",
			source_alert_id=source_alert_id,
		)

	def handle_intel_alert_event(
		self,
		tenant_id: str,
		alert_id: str,
		alert_type: str,
		severity: str,
		alert_reference: str,
	) -> dict[str, Any]:
		"""Create incident from an intel_alerts event (intel subscriber)."""
		priority_map = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}
		priority = priority_map.get(severity, "P3")
		return self.create_incident(
			tenant_id=tenant_id,
			title=f"Security Alert: {alert_reference}",
			category="security",
			priority=priority,
			description=f"Alert type: {alert_type}. Source: intel_alerts/{alert_id}",
			reported_by="intel_alerts",
			source_alert_id=alert_id,
		)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _get_inc_or_raise(self, tenant_id: str, incident_id: str) -> ItIncident:
		inc = self._incidents.get((tenant_id, incident_id))
		if inc is None:
			raise KeyError(f"incident {incident_id!r} not found for tenant {tenant_id!r}")
		return inc

	def _add_timeline(self, tenant_id: str, incident_id: str, event: str, actor: str, notes: str) -> None:
		key = (tenant_id, incident_id)
		tl = self._timelines.setdefault(key, [])
		tl.append({"ts": _now(), "event": event, "actor": actor, "notes": notes[:200]})

	def _add_update(
		self,
		tenant_id: str,
		incident_id: str,
		update_type: str,
		author_id: str,
		content: str,
		previous_status: str | None = None,
		new_status: str | None = None,
	) -> None:
		upd = ItIncidentUpdate(
			tenant_id=tenant_id,
			incident_id=incident_id,
			update_type=update_type,
			author_id=author_id,
			content=content,
			previous_status=previous_status,
			new_status=new_status,
		)
		self._updates[(tenant_id, upd.id)] = upd

	def _init_sla(self, tenant_id: str, inc: ItIncident) -> None:
		target_min = SLA_MINUTES.get(inc.priority, 1440)
		sla = ItIncidentSLA(
			tenant_id=tenant_id,
			incident_id=inc.id,
			priority=inc.priority,
			sla_type="resolve",
			target_minutes=target_min,
			started_at=inc.created_at,
			due_at=inc.sla_due_at or _sla_due(inc.created_at, inc.priority),
		)
		self._slas[(tenant_id, sla.id)] = sla

	def _evaluate_sla(self, tenant_id: str, incident_id: str, inc: ItIncident) -> None:
		"""Mark SLA as met when incident resolves."""
		for (tid, sid), sla in self._slas.items():
			if tid != tenant_id or sla.incident_id != incident_id or sla.sla_type != "resolve":
				continue
			started_dt = _parse_iso(sla.started_at)
			resolved_dt = _parse_iso(inc.resolved_at)
			if started_dt and resolved_dt:
				elapsed = (resolved_dt - started_dt).total_seconds() / 60.0
				sla.elapsed_minutes = elapsed
				sla.remaining_minutes = max(0.0, sla.target_minutes - elapsed)
				if elapsed > sla.target_minutes:
					sla.breached = True
					sla.breached_at = inc.resolved_at
				else:
					sla.met_at = inc.resolved_at
				sla.evaluated_at = _now()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"ts": _now(),
			"processor": "bytewax",
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", a.get("rule", "inc_policy_denied")) for a in result["actions"])
		raise PermissionError(reasons or "inc_policy_denied")


ItsmIncService = IncidentManagementService
