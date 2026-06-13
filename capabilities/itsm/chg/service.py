"""Executable service layer for APG ITSM Change Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CAB_VOTE_OUTCOMES, SUPPORTED_CHANGE_STATUSES,
		SUPPORTED_CHANGE_TYPES, SUPPORTED_IMPACT_LEVELS,
		SUPPORTED_REVIEW_OUTCOMES, SUPPORTED_RISK_LEVELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import ItCabApproval, ItChange, ItChangeReview, ItChangeSchedule
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_CAB_VOTE_OUTCOMES, SUPPORTED_CHANGE_STATUSES,
		SUPPORTED_CHANGE_TYPES, SUPPORTED_IMPACT_LEVELS,
		SUPPORTED_REVIEW_OUTCOMES, SUPPORTED_RISK_LEVELS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import ItCabApproval, ItChange, ItChangeReview, ItChangeSchedule  # type: ignore

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


class ChangeManagementService:
	"""Tenant-scoped Change Management runtime for APG ITSM."""

	def __init__(self) -> None:
		self._changes: dict[tuple[str, str], ItChange] = {}
		self._cab_approvals: dict[tuple[str, str], ItCabApproval] = {}
		self._schedules: dict[tuple[str, str], ItChangeSchedule] = {}
		self._reviews: dict[tuple[str, str], ItChangeReview] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Change Lifecycle
	# ------------------------------------------------------------------

	def create_change(
		self,
		tenant_id: str,
		title: str,
		change_type: str,
		description: str = "",
		*,
		chg_id: str | None = None,
		risk_level: str = "medium",
		impact_level: str = "medium",
		affected_ci_ids: list[str] | None = None,
		affected_services: list[str] | None = None,
		requester_id: str = "system",
		implementer_id: str | None = None,
		team_id: str | None = None,
		implementation_plan: str = "",
		rollback_plan: str = "",
		test_plan: str = "",
		scheduled_start: str | None = None,
		scheduled_end: str | None = None,
		incident_id: str | None = None,
		problem_id: str | None = None,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "create_change",
			"title_present": _present(title),
			"change_type_supported": change_type in SUPPORTED_CHANGE_TYPES,
		})
		chg = ItChange(
			id=chg_id or _uuid7(),
			tenant_id=tenant_id,
			title=title,
			change_type=change_type,
			description=description,
			risk_level=risk_level if risk_level in SUPPORTED_RISK_LEVELS else "medium",
			impact_level=impact_level if impact_level in SUPPORTED_IMPACT_LEVELS else "medium",
			affected_ci_ids=affected_ci_ids or [],
			affected_services=affected_services or [],
			requester_id=requester_id,
			implementer_id=implementer_id,
			team_id=team_id,
			implementation_plan=implementation_plan,
			rollback_plan=rollback_plan,
			test_plan=test_plan,
			scheduled_start=scheduled_start,
			scheduled_end=scheduled_end,
			incident_id=incident_id,
			problem_id=problem_id,
			tags=tags or [],
		)
		self._changes[(tenant_id, chg.id)] = chg
		self._audit(tenant_id, "change_created", chg.id)
		return chg.model_dump()

	def submit_change(
		self,
		tenant_id: str,
		change_id: str,
		submitted_by: str,
		rollback_plan: str | None = None,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		if rollback_plan:
			chg.rollback_plan = rollback_plan
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_change",
			"rollback_plan_present": _present(chg.rollback_plan),
		})
		chg.status = "submitted"
		chg.submitted_at = _now()
		chg.version += 1
		# Standard changes auto-skip CAB
		if chg.change_type == "standard":
			chg.status = "scheduled"
		else:
			chg.status = "cab_pending"
		self._audit(tenant_id, "change_submitted", change_id)
		return chg.model_dump()

	# ------------------------------------------------------------------
	# CAB Workflow
	# ------------------------------------------------------------------

	def create_cab_meeting(
		self,
		tenant_id: str,
		change_id: str,
		meeting_date: str,
		chair_id: str,
		members: list[str],
		agenda_items: list[str] | None = None,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		cab = ItCabApproval(
			tenant_id=tenant_id,
			change_id=change_id,
			meeting_date=meeting_date,
			chair_id=chair_id,
			members=members,
			quorum_required=max(1, len(members) // 2),
			agenda_items=agenda_items or [chg.title],
		)
		self._cab_approvals[(tenant_id, cab.id)] = cab
		chg.cab_meeting_id = cab.id
		self._audit(tenant_id, "cab_meeting_created", cab.id)
		return cab.model_dump()

	def record_cab_vote(
		self,
		tenant_id: str,
		cab_id: str,
		member_id: str,
		outcome: str,
		notes: str = "",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "record_cab_vote",
			"vote_outcome_supported": outcome in SUPPORTED_CAB_VOTE_OUTCOMES,
		})
		cab = self._cab_approvals.get((tenant_id, cab_id))
		if cab is None:
			raise KeyError(f"CAB meeting {cab_id!r} not found")
		# Idempotent per member
		cab.votes = [v for v in cab.votes if v["member_id"] != member_id]
		cab.votes.append({"member_id": member_id, "outcome": outcome, "notes": notes, "voted_at": _now()})
		self._audit(tenant_id, "cab_vote_recorded", cab_id)
		return cab.model_dump()

	def decide_cab(self, tenant_id: str, cab_id: str) -> dict[str, Any]:
		"""Tally votes, apply quorum+threshold rules, set outcome on CAB and Change."""
		cab = self._cab_approvals.get((tenant_id, cab_id))
		if cab is None:
			raise KeyError(f"CAB meeting {cab_id!r} not found")
		total_votes = len(cab.votes)
		if total_votes < cab.quorum_required:
			return {"cab_id": cab_id, "outcome": "pending", "reason": "quorum_not_met", "votes": total_votes, "required": cab.quorum_required}
		approve_votes = sum(1 for v in cab.votes if v["outcome"] == "approve")
		reject_votes = sum(1 for v in cab.votes if v["outcome"] == "reject")
		approval_pct = approve_votes / total_votes if total_votes else 0.0
		if approval_pct >= 0.6:
			outcome = "approve"
		elif reject_votes > approve_votes:
			outcome = "reject"
		else:
			outcome = "defer"
		cab.outcome = outcome
		cab.decided_at = _now()
		# Propagate to change
		chg = self._changes.get((tenant_id, cab.change_id))
		if chg:
			if outcome == "approve":
				chg.status = "cab_approved"
				chg.cab_approved_at = _now()
				chg.cab_approved_by = cab.chair_id
			elif outcome == "reject":
				chg.status = "cab_rejected"
				chg.cab_rejected_at = _now()
				chg.cab_rejection_reason = cab.outcome_notes
			else:
				chg.status = "cab_pending"  # deferred back
			chg.version += 1
		event = "change_cab_approved" if outcome == "approve" else ("change_cab_rejected" if outcome == "reject" else "change_cab_deferred")
		self._audit(tenant_id, event, cab.change_id)
		return cab.model_dump()

	# ------------------------------------------------------------------
	# Schedule & Conflict Detection
	# ------------------------------------------------------------------

	def create_schedule_window(
		self,
		tenant_id: str,
		name: str,
		schedule_type: str,
		start_datetime: str,
		end_datetime: str,
		*,
		affected_services: list[str] | None = None,
		affected_environments: list[str] | None = None,
		recurrence_rule: str | None = None,
		description: str = "",
		created_by: str = "system",
	) -> dict[str, Any]:
		win = ItChangeSchedule(
			tenant_id=tenant_id,
			name=name,
			schedule_type=schedule_type,
			start_datetime=start_datetime,
			end_datetime=end_datetime,
			affected_services=affected_services or [],
			affected_environments=affected_environments or [],
			recurrence_rule=recurrence_rule,
			description=description,
			created_by=created_by,
		)
		self._schedules[(tenant_id, win.id)] = win
		self._audit(tenant_id, "schedule_window_created", win.id)
		return win.model_dump()

	def detect_conflicts(self, tenant_id: str, change_id: str) -> dict[str, Any]:
		"""Detect schedule conflicts for a change's scheduled window."""
		chg = self._get_chg_or_raise(tenant_id, change_id)
		if not chg.scheduled_start or not chg.scheduled_end:
			return {"change_id": change_id, "conflicts": [], "freeze_windows": [], "as_of": _now()}
		chg_start = _parse_iso(chg.scheduled_start)
		chg_end = _parse_iso(chg.scheduled_end)
		conflicts: list[dict[str, Any]] = []
		freeze_hits: list[dict[str, Any]] = []
		for (tid, _), win in self._schedules.items():
			if tid != tenant_id or not win.is_active:
				continue
			w_start = _parse_iso(win.start_datetime)
			w_end = _parse_iso(win.end_datetime)
			if not w_start or not w_end or not chg_start or not chg_end:
				continue
			overlaps = chg_start < w_end and chg_end > w_start
			if not overlaps:
				continue
			if win.schedule_type == "freeze_window" or win.schedule_type == "blackout":
				freeze_hits.append({"window_id": win.id, "name": win.name, "type": win.schedule_type, "start": win.start_datetime, "end": win.end_datetime})
			# Overlap with other changes
		# Also check other scheduled changes
		other_conflicts: list[dict[str, Any]] = []
		for (tid, cid), other in self._changes.items():
			if tid != tenant_id or cid == change_id:
				continue
			if other.status not in ("scheduled", "in_progress"):
				continue
			if not other.scheduled_start or not other.scheduled_end:
				continue
			o_start = _parse_iso(other.scheduled_start)
			o_end = _parse_iso(other.scheduled_end)
			if chg_start and chg_end and o_start and o_end:
				if chg_start < o_end and chg_end > o_start:
					shared = any(s in other.affected_services for s in chg.affected_services)
					if shared or any(c in other.affected_ci_ids for c in chg.affected_ci_ids):
						other_conflicts.append({"change_id": cid, "title": other.title, "start": other.scheduled_start, "end": other.scheduled_end})
		return {
			"change_id": change_id,
			"freeze_window_conflicts": freeze_hits,
			"change_conflicts": other_conflicts,
			"has_conflicts": bool(freeze_hits or other_conflicts),
			"as_of": _now(),
		}

	def schedule_change(
		self,
		tenant_id: str,
		change_id: str,
		scheduled_start: str,
		scheduled_end: str,
		scheduled_by: str,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		assert chg.status in ("cab_approved", "submitted", "standard"), f"cannot schedule from {chg.status!r}"
		chg.scheduled_start = scheduled_start
		chg.scheduled_end = scheduled_end
		chg.status = "scheduled"
		chg.version += 1
		self._audit(tenant_id, "change_scheduled", change_id)
		return chg.model_dump()

	# ------------------------------------------------------------------
	# Implementation
	# ------------------------------------------------------------------

	def start_change(self, tenant_id: str, change_id: str, started_by: str) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		assert chg.status == "scheduled", f"start requires scheduled, got {chg.status!r}"
		chg.status = "in_progress"
		chg.actual_start = _now()
		chg.version += 1
		self._audit(tenant_id, "change_started", change_id)
		return {"change_id": change_id, "status": "in_progress", "actual_start": chg.actual_start}

	def complete_change(
		self,
		tenant_id: str,
		change_id: str,
		completed_by: str,
		implementation_notes: str,
		success: bool,
		failed_reason: str | None = None,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		chg.actual_end = _now()
		chg.implementation_notes = implementation_notes
		if success:
			chg.status = "implemented"
		else:
			chg.status = "failed"
			chg.failed_reason = failed_reason
		chg.version += 1
		event = "change_implemented" if success else "change_failed"
		self._audit(tenant_id, event, change_id)
		return chg.model_dump()

	def rollback_change(
		self,
		tenant_id: str,
		change_id: str,
		rolled_back_by: str,
		reason: str,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		chg.status = "rolled_back"
		chg.rolled_back = True
		chg.rollback_at = _now()
		chg.version += 1
		self._audit(tenant_id, "change_rolled_back", change_id)
		return {"change_id": change_id, "status": "rolled_back", "rollback_at": chg.rollback_at}

	# ------------------------------------------------------------------
	# Post-Implementation Review
	# ------------------------------------------------------------------

	def create_pir(
		self,
		tenant_id: str,
		change_id: str,
		reviewer_id: str,
		outcome: str,
		*,
		implementation_notes: str = "",
		objectives_met: bool = True,
		incidents_caused: list[str] | None = None,
		rollback_required: bool = False,
		lessons_learned: list[str] | None = None,
		recommendations: list[str] | None = None,
		process_improvements: list[str] | None = None,
	) -> dict[str, Any]:
		assert outcome in SUPPORTED_REVIEW_OUTCOMES, f"unsupported outcome {outcome!r}"
		chg = self._get_chg_or_raise(tenant_id, change_id)
		review = ItChangeReview(
			tenant_id=tenant_id,
			change_id=change_id,
			reviewer_id=reviewer_id,
			outcome=outcome,
			implementation_notes=implementation_notes,
			objectives_met=objectives_met,
			incidents_caused=incidents_caused or [],
			rollback_required=rollback_required,
			lessons_learned=lessons_learned or [],
			recommendations=recommendations or [],
			process_improvements=process_improvements or [],
			completed_at=_now(),
		)
		self._reviews[(tenant_id, review.id)] = review
		chg.pir_id = review.id
		chg.pir_completed = True
		chg.version += 1
		self._audit(tenant_id, "change_pir_completed", review.id)
		return review.model_dump()

	def close_change(
		self,
		tenant_id: str,
		change_id: str,
		closed_by: str,
	) -> dict[str, Any]:
		chg = self._get_chg_or_raise(tenant_id, change_id)
		is_failed = chg.status in ("failed", "rolled_back")
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "close_change",
			"is_failed": is_failed,
			"pir_completed": chg.pir_completed,
		})
		chg.status = "closed"
		chg.closed_at = _now()
		chg.version += 1
		self._audit(tenant_id, "change_closed", change_id)
		return {"change_id": change_id, "status": "closed"}

	# ------------------------------------------------------------------
	# NATS integration hooks
	# ------------------------------------------------------------------

	def handle_ci_change_requested(
		self,
		tenant_id: str,
		ci_id: str,
		ci_name: str,
		change_description: str,
		requester_id: str,
	) -> dict[str, Any]:
		"""Auto-create normal change from itsm_cmdb.ci.change_requested event."""
		return self.create_change(
			tenant_id=tenant_id,
			title=f"CI Change: {ci_name}",
			change_type="normal",
			description=change_description,
			affected_ci_ids=[ci_id],
			requester_id=requester_id,
		)

	# ------------------------------------------------------------------
	# Querying & Analytics
	# ------------------------------------------------------------------

	def get_change(self, tenant_id: str, change_id: str) -> dict[str, Any]:
		return self._get_chg_or_raise(tenant_id, change_id).model_dump()

	def list_changes(
		self,
		tenant_id: str,
		*,
		status: str | None = None,
		change_type: str | None = None,
		risk_level: str | None = None,
	) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		for (tid, _), chg in self._changes.items():
			if tid != tenant_id:
				continue
			if status and chg.status != status:
				continue
			if change_type and chg.change_type != change_type:
				continue
			if risk_level and chg.risk_level != risk_level:
				continue
			results.append(chg.model_dump())
		return sorted(results, key=lambda r: r["created_at"], reverse=True)

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		by_status: dict[str, int] = {s: 0 for s in SUPPORTED_CHANGE_STATUSES}
		by_type: dict[str, int] = {t: 0 for t in SUPPORTED_CHANGE_TYPES}
		failed = 0
		total = 0
		for (tid, _), chg in self._changes.items():
			if tid != tenant_id:
				continue
			total += 1
			by_status[chg.status] = by_status.get(chg.status, 0) + 1
			by_type[chg.change_type] = by_type.get(chg.change_type, 0) + 1
			if chg.status in ("failed", "rolled_back"):
				failed += 1
		return {
			"tenant_id": tenant_id,
			"total": total,
			"by_status": by_status,
			"by_type": by_type,
			"failed_changes": failed,
			"failure_rate": round(failed / total, 4) if total else 0.0,
			"as_of": _now(),
		}

	def change_calendar(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all scheduled/in-progress changes for calendar display."""
		results: list[dict[str, Any]] = []
		for (tid, _), chg in self._changes.items():
			if tid != tenant_id:
				continue
			if chg.status in ("scheduled", "in_progress"):
				results.append({
					"id": chg.id,
					"title": chg.title,
					"type": chg.change_type,
					"status": chg.status,
					"risk": chg.risk_level,
					"start": chg.scheduled_start,
					"end": chg.scheduled_end,
					"services": chg.affected_services,
				})
		return sorted(results, key=lambda r: r.get("start") or "")

	# ------------------------------------------------------------------
	# Private
	# ------------------------------------------------------------------

	def _get_chg_or_raise(self, tenant_id: str, change_id: str) -> ItChange:
		chg = self._changes.get((tenant_id, change_id))
		if chg is None:
			raise KeyError(f"change {change_id!r} not found for tenant {tenant_id!r}")
		return chg

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "ts": _now()})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(a.get("reason", "chg_policy_denied") for a in result["actions"])
		raise PermissionError(reasons or "chg_policy_denied")


ItsmChgService = ChangeManagementService
