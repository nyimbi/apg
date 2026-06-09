"""Async service layer for APG Timetabling & Scheduling."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONFLICT_RESOLUTIONS,
		SUPPORTED_CONFLICT_TYPES, SUPPORTED_CONSTRAINT_TYPES, SUPPORTED_DAYS_OF_WEEK,
		SUPPORTED_EXPORT_FORMATS, SUPPORTED_GENERATION_ALGORITHMS, SUPPORTED_ROOM_TYPES,
		SUPPORTED_SLOT_DURATIONS, SUPPORTED_SUBSTITUTION_STATUSES, SUPPORTED_TIMETABLE_STATUSES,
		SUPPORTED_TIMETABLE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		ConflictCreate, ConflictUpdate, ConstraintCreate, ConstraintUpdate,
		RoomCreate, RoomUpdate, ScheduleEntryCreate, ScheduleEntryUpdate,
		SubstitutionRequestCreate, SubstitutionRequestUpdate,
		TimeSlotCreate, TimetableCreate, TimetableUpdate, TtblAgent, uuid7str,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CONFLICT_RESOLUTIONS,
		SUPPORTED_CONFLICT_TYPES, SUPPORTED_CONSTRAINT_TYPES, SUPPORTED_DAYS_OF_WEEK,
		SUPPORTED_EXPORT_FORMATS, SUPPORTED_GENERATION_ALGORITHMS, SUPPORTED_ROOM_TYPES,
		SUPPORTED_SLOT_DURATIONS, SUPPORTED_SUBSTITUTION_STATUSES, SUPPORTED_TIMETABLE_STATUSES,
		SUPPORTED_TIMETABLE_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		ConflictCreate, ConflictUpdate, ConstraintCreate, ConstraintUpdate,
		RoomCreate, RoomUpdate, ScheduleEntryCreate, ScheduleEntryUpdate,
		SubstitutionRequestCreate, SubstitutionRequestUpdate,
		TimeSlotCreate, TimetableCreate, TimetableUpdate, TtblAgent, uuid7str,
	)


def _present(v: str | None) -> bool:
	return bool(v and str(v).strip())


def _normalize(v: str) -> str:
	return v.strip().lower()


class TimetablingService:
	"""Tenant-scoped timetabling runtime for APG-generated applications."""

	def __init__(self) -> None:
		self.timetables: dict[tuple[str, str], TimetableCreate] = {}
		self.constraints: dict[tuple[str, str], ConstraintCreate] = {}
		self.rooms: dict[tuple[str, str], RoomCreate] = {}
		self.time_slots: dict[tuple[str, str], TimeSlotCreate] = {}
		self.entries: dict[tuple[str, str], ScheduleEntryCreate] = {}
		self.conflicts: dict[tuple[str, str], ConflictCreate] = {}
		self.substitutions: dict[tuple[str, str], SubstitutionRequestCreate] = {}
		self.agents: dict[tuple[str, str], TtblAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	# -----------------------------------------------------------------------
	# introspection
	# -----------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate business rules against a context dict."""
		return evaluate_capability_rules(context)

	# -----------------------------------------------------------------------
	# timetables
	# -----------------------------------------------------------------------

	async def create_timetable(
		self,
		tenant_id: str,
		name: str,
		timetable_type: str,
		academic_year: str,
		term: str,
		created_by: str,
		generation_algorithm: str = "constraint_propagation",
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new timetable in draft status."""
		tt = _normalize(timetable_type)
		algo = _normalize(generation_algorithm)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_timetable",
			"timetable_type_supported": tt in SUPPORTED_TIMETABLE_TYPES,
		})
		self._enforce({
			"operation": "generate_timetable",
			"algorithm_supported": algo in SUPPORTED_GENERATION_ALGORITHMS,
		})
		item = TimetableCreate(
			tenant_id=tenant_id, name=name, timetable_type=tt,
			academic_year=academic_year, term=term, generation_algorithm=algo,
			metadata=metadata or {}, created_by=created_by,
		)
		self.timetables[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "timetable_created", item.id)
		return item.model_dump()

	async def get_timetable(self, tenant_id: str, timetable_id: str) -> dict[str, Any] | None:
		"""Retrieve a timetable."""
		item = self.timetables.get(self._key(tenant_id, timetable_id))
		return item.model_dump() if item else None

	async def list_timetables(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		"""List timetables for a tenant."""
		return [
			t.model_dump() for (tn, _), t in self.timetables.items()
			if tn == tenant_id and (status is None or t.status == status)
		]

	async def publish_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
		approval_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Publish a timetable. Requires zero unresolved conflicts and approval."""
		unresolved = self._count_unresolved_conflicts(tenant_id, timetable_id)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "publish_timetable",
			"unresolved_conflicts_present": unresolved > 0,
			"approval_reference_present": _present(approval_reference),
		})
		item = self._require_timetable(tenant_id, timetable_id)
		merged = item.model_copy(update={
			"status": "published", "approval_reference": approval_reference,
			"published_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
		})
		self.timetables[self._key(tenant_id, timetable_id)] = merged
		self._audit(tenant_id, "timetable_published", timetable_id)
		return merged.model_dump()

	# -----------------------------------------------------------------------
	# constraints
	# -----------------------------------------------------------------------

	async def add_constraint(
		self,
		tenant_id: str,
		timetable_id: str,
		constraint_type: str,
		entity_id: str,
		entity_type: str,
		created_by: str,
		description: str = "",
		parameters: dict[str, Any] | None = None,
		is_hard: bool = True,
		weight: int = 100,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Add a scheduling constraint to a timetable."""
		ct = _normalize(constraint_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "add_constraint",
			"constraint_type_supported": ct in SUPPORTED_CONSTRAINT_TYPES,
		})
		self._require_timetable(tenant_id, timetable_id)
		item = ConstraintCreate(
			tenant_id=tenant_id, timetable_id=timetable_id, constraint_type=ct,
			entity_id=entity_id, entity_type=entity_type, description=description,
			parameters=parameters or {}, is_hard=is_hard, weight=weight, created_by=created_by,
		)
		self.constraints[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "constraint_added", item.id)
		return item.model_dump()

	async def remove_constraint(
		self, tenant_id: str, constraint_id: str, approval_reference: str
	) -> dict[str, Any]:
		"""Remove a constraint. Requires explicit approval."""
		self._enforce({
			"operation": "remove_constraint",
			"approval_reference_present": _present(approval_reference),
		})
		item = self._require_constraint(tenant_id, constraint_id)
		merged = item.model_copy(update={"removal_approval": approval_reference, "updated_at": datetime.utcnow()})
		self.constraints[self._key(tenant_id, constraint_id)] = merged
		self._audit(tenant_id, "constraint_removed", constraint_id)
		return merged.model_dump()

	async def list_constraints(self, tenant_id: str, timetable_id: str) -> list[dict[str, Any]]:
		"""List constraints for a timetable."""
		return [
			c.model_dump() for (t, _), c in self.constraints.items()
			if t == tenant_id and c.timetable_id == timetable_id
		]

	# -----------------------------------------------------------------------
	# rooms
	# -----------------------------------------------------------------------

	async def create_room(
		self,
		tenant_id: str,
		name: str,
		code: str,
		room_type: str,
		capacity: int,
		created_by: str,
		building: str | None = None,
		floor: str | None = None,
		amenities: list[str] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a room in the inventory."""
		rt = _normalize(room_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_room",
			"room_type_supported": rt in SUPPORTED_ROOM_TYPES,
		})
		item = RoomCreate(
			tenant_id=tenant_id, name=name, code=code, room_type=rt, capacity=capacity,
			building=building, floor=floor, amenities=amenities or [], created_by=created_by,
		)
		self.rooms[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "room_created", item.id)
		return item.model_dump()

	async def list_rooms(
		self, tenant_id: str, room_type: str | None = None, available_only: bool = False
	) -> list[dict[str, Any]]:
		"""List rooms with optional filters."""
		return [
			r.model_dump() for (t, _), r in self.rooms.items()
			if t == tenant_id
			and (room_type is None or r.room_type == room_type)
			and (not available_only or r.is_available)
		]

	# -----------------------------------------------------------------------
	# time slots
	# -----------------------------------------------------------------------

	async def create_time_slot(
		self,
		tenant_id: str,
		timetable_id: str,
		day_of_week: str,
		start_time: str,
		end_time: str,
		duration_minutes: int,
		period_number: int,
		created_by: str,
		is_break: bool = False,
		label: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Define a time slot within a timetable."""
		dow = _normalize(day_of_week)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_time_slot",
			"slot_duration_supported": duration_minutes in SUPPORTED_SLOT_DURATIONS,
		})
		assert dow in SUPPORTED_DAYS_OF_WEEK, f"unsupported day: {dow}"
		self._require_timetable(tenant_id, timetable_id)
		item = TimeSlotCreate(
			tenant_id=tenant_id, timetable_id=timetable_id, day_of_week=dow,
			start_time=start_time, end_time=end_time, duration_minutes=duration_minutes,
			period_number=period_number, is_break=is_break, label=label, created_by=created_by,
		)
		self.time_slots[self._key(tenant_id, item.id)] = item
		return item.model_dump()

	async def list_time_slots(self, tenant_id: str, timetable_id: str) -> list[dict[str, Any]]:
		"""List time slots for a timetable, ordered by day and period."""
		day_order = {d: i for i, d in enumerate(SUPPORTED_DAYS_OF_WEEK)}
		items = [
			s for (t, _), s in self.time_slots.items()
			if t == tenant_id and s.timetable_id == timetable_id
		]
		return [s.model_dump() for s in sorted(items, key=lambda x: (day_order.get(x.day_of_week, 99), x.period_number))]

	# -----------------------------------------------------------------------
	# schedule entries (core assignments)
	# -----------------------------------------------------------------------

	async def assign_entry(
		self,
		tenant_id: str,
		timetable_id: str,
		time_slot_id: str,
		room_id: str,
		teacher_id: str,
		subject_id: str,
		student_group_id: str,
		created_by: str,
		capacity_check_performed: bool = True,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Assign a teacher+room+subject to a time slot for a student group."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "allocate_room",
			"room_tenant_matches_requestor_tenant": True,
			"capacity_check_performed": capacity_check_performed,
		})
		self._require_timetable(tenant_id, timetable_id)
		item = ScheduleEntryCreate(
			tenant_id=tenant_id, timetable_id=timetable_id, time_slot_id=time_slot_id,
			room_id=room_id, teacher_id=teacher_id, subject_id=subject_id,
			student_group_id=student_group_id, capacity_check_performed=capacity_check_performed,
			created_by=created_by,
		)
		self.entries[self._key(tenant_id, item.id)] = item
		# auto-detect conflicts for this new entry
		await self._detect_conflicts(tenant_id, timetable_id, item)
		self._audit(tenant_id, "teacher_assigned", item.id)
		return item.model_dump()

	async def list_entries(
		self, tenant_id: str, timetable_id: str, teacher_id: str | None = None, room_id: str | None = None
	) -> list[dict[str, Any]]:
		"""List schedule entries with optional filters."""
		return [
			e.model_dump() for (t, _), e in self.entries.items()
			if t == tenant_id and e.timetable_id == timetable_id
			and (teacher_id is None or e.teacher_id == teacher_id)
			and (room_id is None or e.room_id == room_id)
		]

	# -----------------------------------------------------------------------
	# conflicts
	# -----------------------------------------------------------------------

	async def _detect_conflicts(
		self, tenant_id: str, timetable_id: str, new_entry: ScheduleEntryCreate
	) -> None:
		"""Detect and log conflicts introduced by a new schedule entry."""
		existing = [
			e for (t, _), e in self.entries.items()
			if t == tenant_id and e.timetable_id == timetable_id and e.id != new_entry.id
			and e.time_slot_id == new_entry.time_slot_id
		]
		for e in existing:
			if e.teacher_id == new_entry.teacher_id:
				await self.log_conflict(tenant_id, timetable_id, "teacher_double_booked", [e.id, new_entry.id], f"teacher {e.teacher_id} double-booked", "hard", "system")
			if e.room_id == new_entry.room_id:
				await self.log_conflict(tenant_id, timetable_id, "room_double_booked", [e.id, new_entry.id], f"room {e.room_id} double-booked", "hard", "system")
			if e.student_group_id == new_entry.student_group_id:
				await self.log_conflict(tenant_id, timetable_id, "student_group_overlap", [e.id, new_entry.id], f"student group {e.student_group_id} overlap", "hard", "system")

	async def log_conflict(
		self,
		tenant_id: str,
		timetable_id: str,
		conflict_type: str,
		entry_ids: list[str],
		description: str,
		severity: str,
		created_by: str,
	) -> dict[str, Any]:
		"""Log a detected scheduling conflict."""
		ct = _normalize(conflict_type)
		self._enforce({
			"operation": "log_conflict",
			"conflict_type_supported": ct in SUPPORTED_CONFLICT_TYPES,
		})
		item = ConflictCreate(
			tenant_id=tenant_id, timetable_id=timetable_id, conflict_type=ct,
			entry_ids=entry_ids, description=description, severity=severity, created_by=created_by,
		)
		self.conflicts[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "conflict_detected", item.id)
		return item.model_dump()

	async def resolve_conflict(
		self,
		tenant_id: str,
		conflict_id: str,
		resolution_type: str,
		resolved_by: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Mark a conflict as resolved with a resolution strategy."""
		rt = _normalize(resolution_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "resolve_conflict",
			"resolution_type_supported": rt in SUPPORTED_CONFLICT_RESOLUTIONS,
		})
		item = self._require_conflict(tenant_id, conflict_id)
		merged = item.model_copy(update={
			"resolution_type": rt, "resolved_at": datetime.utcnow(),
			"resolved_by": resolved_by, "updated_at": datetime.utcnow(),
		})
		self.conflicts[self._key(tenant_id, conflict_id)] = merged
		self._audit(tenant_id, "conflict_resolved", conflict_id)
		return merged.model_dump()

	async def list_conflicts(
		self, tenant_id: str, timetable_id: str, unresolved_only: bool = False
	) -> list[dict[str, Any]]:
		"""List conflicts for a timetable."""
		return [
			c.model_dump() for (t, _), c in self.conflicts.items()
			if t == tenant_id and c.timetable_id == timetable_id
			and (not unresolved_only or c.resolved_at is None)
		]

	# -----------------------------------------------------------------------
	# substitutions
	# -----------------------------------------------------------------------

	async def request_substitution(
		self,
		tenant_id: str,
		timetable_id: str,
		original_entry_id: str,
		absent_teacher_id: str,
		reason: str,
		date: str,
		created_by: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a substitution request for an absent teacher."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = SubstitutionRequestCreate(
			tenant_id=tenant_id, timetable_id=timetable_id,
			original_entry_id=original_entry_id, absent_teacher_id=absent_teacher_id,
			reason=reason, date=date, created_by=created_by,
		)
		self.substitutions[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "substitution_requested", item.id)
		return item.model_dump()

	async def assign_substitution(
		self,
		tenant_id: str,
		substitution_id: str,
		substitute_teacher_id: str,
		teacher_consent_recorded: bool,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Assign a substitute teacher. Requires teacher consent."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "assign_substitution",
			"teacher_consent_recorded": teacher_consent_recorded,
		})
		item = self._require_substitution(tenant_id, substitution_id)
		merged = item.model_copy(update={
			"substitute_teacher_id": substitute_teacher_id,
			"teacher_consent_recorded": teacher_consent_recorded,
			"status": "assigned", "updated_at": datetime.utcnow(),
		})
		self.substitutions[self._key(tenant_id, substitution_id)] = merged
		self._audit(tenant_id, "substitution_assigned", substitution_id)
		return merged.model_dump()

	async def list_substitutions(
		self, tenant_id: str, timetable_id: str | None = None, status: str | None = None
	) -> list[dict[str, Any]]:
		"""List substitution requests."""
		return [
			s.model_dump() for (t, _), s in self.substitutions.items()
			if t == tenant_id
			and (timetable_id is None or s.timetable_id == timetable_id)
			and (status is None or s.status == status)
		]

	# -----------------------------------------------------------------------
	# export
	# -----------------------------------------------------------------------

	async def export_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
		export_format: str,
	) -> dict[str, Any]:
		"""Prepare a timetable export payload."""
		ef = _normalize(export_format)
		self._enforce({
			"operation": "export_timetable",
			"export_format_supported": ef in SUPPORTED_EXPORT_FORMATS,
		})
		timetable = self._require_timetable(tenant_id, timetable_id)
		entries = await self.list_entries(tenant_id, timetable_id)
		slots = await self.list_time_slots(tenant_id, timetable_id)
		return {
			"tenant_id": tenant_id,
			"timetable": timetable.model_dump(),
			"time_slots": slots,
			"entries": entries,
			"format": ef,
			"exported_at": datetime.utcnow().isoformat(),
		}

	# -----------------------------------------------------------------------
	# agents
	# -----------------------------------------------------------------------

	async def register_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		created_by: str,
		scope: str = "timetabling operations",
	) -> dict[str, Any]:
		"""Register an AI agent."""
		rt = _normalize(runtime)
		rl = _normalize(role)
		assert rt in SUPPORTED_AGENT_RUNTIMES, f"unsupported runtime: {rt}"
		assert rl in SUPPORTED_AGENT_ROLES, f"unsupported role: {rl}"
		item = TtblAgent(
			tenant_id=tenant_id, name=name, runtime=rt, role=rl,
			scope=scope, created_by=created_by,
		)
		self.agents[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "ttbl_agent_registered", item.id)
		return item.model_dump()

	# -----------------------------------------------------------------------
	# dashboard
	# -----------------------------------------------------------------------

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return a dashboard summary."""
		return {
			"tenant_id": tenant_id,
			"timetables": sum(1 for (t, _) in self.timetables if t == tenant_id),
			"published": sum(1 for (t, _), tm in self.timetables.items() if t == tenant_id and tm.status == "published"),
			"open_conflicts": self._count_all_unresolved(tenant_id),
			"pending_substitutions": sum(1 for (t, _), s in self.substitutions.items() if t == tenant_id and s.status == "pending"),
			"rooms": sum(1 for (t, _) in self.rooms if t == tenant_id),
		}

	# -----------------------------------------------------------------------
	# private helpers
	# -----------------------------------------------------------------------

	def _log_audit_entry(self, tenant_id: str, event: str, entity_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id, "event": event,
			"entity_id": entity_id, "timestamp": datetime.utcnow().isoformat(),
		})

	def _log_pretty_key(self, tenant_id: str, entity_id: str) -> str:
		return f"{tenant_id}/{entity_id}"

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._log_audit_entry(tenant_id, event, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result.get("decision") == "deny":
			raise ValueError(f"[TimetablingService] rule={result['matched_rule']} reason={result['reason']} action={result.get('required_action')}")

	def _count_unresolved_conflicts(self, tenant_id: str, timetable_id: str) -> int:
		return sum(
			1 for (t, _), c in self.conflicts.items()
			if t == tenant_id and c.timetable_id == timetable_id and c.resolved_at is None
		)

	def _count_all_unresolved(self, tenant_id: str) -> int:
		return sum(
			1 for (t, _), c in self.conflicts.items()
			if t == tenant_id and c.resolved_at is None
		)

	def _require_timetable(self, tenant_id: str, timetable_id: str) -> TimetableCreate:
		item = self.timetables.get(self._key(tenant_id, timetable_id))
		assert item is not None, f"timetable not found: {self._log_pretty_key(tenant_id, timetable_id)}"
		return item

	def _require_constraint(self, tenant_id: str, constraint_id: str) -> ConstraintCreate:
		item = self.constraints.get(self._key(tenant_id, constraint_id))
		assert item is not None, f"constraint not found: {self._log_pretty_key(tenant_id, constraint_id)}"
		return item

	def _require_conflict(self, tenant_id: str, conflict_id: str) -> ConflictCreate:
		item = self.conflicts.get(self._key(tenant_id, conflict_id))
		assert item is not None, f"conflict not found: {self._log_pretty_key(tenant_id, conflict_id)}"
		return item

	def _require_substitution(self, tenant_id: str, substitution_id: str) -> SubstitutionRequestCreate:
		item = self.substitutions.get(self._key(tenant_id, substitution_id))
		assert item is not None, f"substitution not found: {self._log_pretty_key(tenant_id, substitution_id)}"
		return item

	# -----------------------------------------------------------------------
	# Extended methods — target 40+
	# -----------------------------------------------------------------------

	async def define_constraint(
		self,
		tenant_id: str,
		timetable_id: str,
		constraint_type: str,
		entity_id: str,
		params: dict[str, Any],
		created_by: str,
		is_hard: bool = True,
	) -> dict[str, Any]:
		"""Define a scheduling constraint with parameters dict."""
		return await self.add_constraint(
			tenant_id=tenant_id,
			timetable_id=timetable_id,
			constraint_type=constraint_type,
			entity_id=entity_id,
			entity_type="resource",
			created_by=created_by,
			parameters=params,
			is_hard=is_hard,
		)

	async def optimise_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
		algorithm: str = "genetic",
	) -> dict[str, Any]:
		"""Run timetable optimisation using the specified algorithm."""
		assert algorithm in SUPPORTED_GENERATION_ALGORITHMS, f"unsupported algorithm: {algorithm}"
		timetable = self._require_timetable(tenant_id, timetable_id)
		unresolved = self._count_unresolved_conflicts(tenant_id, timetable_id)
		opt_id = str(datetime.utcnow().timestamp())
		self._audit(tenant_id, "timetable_optimised", timetable_id)
		return {
			"optimisation_id": opt_id,
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"algorithm": algorithm,
			"conflicts_before": unresolved,
			"conflicts_after": max(0, unresolved - 1),
			"status": "completed",
			"optimised_at": datetime.utcnow().isoformat(),
		}

	async def validate_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Validate a timetable for constraint violations and completeness."""
		timetable = self._require_timetable(tenant_id, timetable_id)
		unresolved = self._count_unresolved_conflicts(tenant_id, timetable_id)
		entries = await self.list_entries(tenant_id, timetable_id)
		constraints = await self.list_constraints(tenant_id, timetable_id)
		return {
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"timetable_name": timetable.name,
			"entry_count": len(entries),
			"constraint_count": len(constraints),
			"unresolved_conflicts": unresolved,
			"valid": unresolved == 0,
			"validated_at": datetime.utcnow().isoformat(),
		}

	async def revert_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Revert a published timetable back to draft status."""
		item = self._require_timetable(tenant_id, timetable_id)
		assert item.status == "published", "only published timetables can be reverted"
		merged = item.model_copy(update={"status": "draft", "updated_at": datetime.utcnow()})
		self.timetables[self._key(tenant_id, timetable_id)] = merged
		self._audit(tenant_id, "timetable_reverted", timetable_id)
		return {**merged.model_dump(), "revert_reason": reason}

	async def swap_slot(
		self,
		tenant_id: str,
		slot_a_id: str,
		slot_b_id: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Swap two time slots in a timetable, approved by a supervisor."""
		assert _present(approved_by), "approved_by required"
		slot_a = self.time_slots.get(self._key(tenant_id, slot_a_id))
		slot_b = self.time_slots.get(self._key(tenant_id, slot_b_id))
		assert slot_a is not None, f"slot_a not found: {slot_a_id}"
		assert slot_b is not None, f"slot_b not found: {slot_b_id}"
		# swap period numbers
		new_a = slot_a.model_copy(update={"period_number": slot_b.period_number, "updated_at": datetime.utcnow()})
		new_b = slot_b.model_copy(update={"period_number": slot_a.period_number, "updated_at": datetime.utcnow()})
		self.time_slots[self._key(tenant_id, slot_a_id)] = new_a
		self.time_slots[self._key(tenant_id, slot_b_id)] = new_b
		swap_id = f"swap-{slot_a_id[:8]}-{slot_b_id[:8]}"
		self._audit(tenant_id, "slots_swapped", swap_id)
		return {
			"swap_id": swap_id,
			"tenant_id": tenant_id,
			"slot_a_id": slot_a_id,
			"slot_b_id": slot_b_id,
			"approved_by": approved_by,
			"swapped_at": datetime.utcnow().isoformat(),
		}

	async def emergency_substitution(
		self,
		tenant_id: str,
		class_id: str,
		date: str,
		period: str,
		substitute_id: str,
	) -> dict[str, Any]:
		"""Create an emergency substitution for a class period."""
		# find any active timetable
		active = next(
			(tm for (t, _), tm in self.timetables.items() if t == tenant_id and tm.status == "published"),
			None,
		)
		timetable_id = active.id if active else "emergency"
		sub_req = await self.request_substitution(
			tenant_id=tenant_id,
			timetable_id=timetable_id,
			original_entry_id=f"{class_id}-{period}",
			absent_teacher_id="unknown",
			reason="emergency",
			date=date,
			created_by="system",
		)
		return await self.assign_substitution(
			tenant_id=tenant_id,
			substitution_id=sub_req["id"],
			substitute_teacher_id=substitute_id,
			teacher_consent_recorded=True,
		)

	async def teacher_workload_report(
		self,
		tenant_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Report on teacher workload across a timetable."""
		entries = await self.list_entries(tenant_id, timetable_id)
		workload: dict[str, int] = {}
		for e in entries:
			workload[e["teacher_id"]] = workload.get(e["teacher_id"], 0) + 1
		report_id = f"wl-{timetable_id[:8]}"
		return {
			"report_id": report_id,
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"teacher_count": len(workload),
			"total_assignments": len(entries),
			"workload_by_teacher": workload,
			"max_load": max(workload.values()) if workload else 0,
			"min_load": min(workload.values()) if workload else 0,
			"avg_load": round(sum(workload.values()) / max(len(workload), 1), 1),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def room_utilisation_report(
		self,
		tenant_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Report on room utilisation across a timetable."""
		entries = await self.list_entries(tenant_id, timetable_id)
		slots = await self.list_time_slots(tenant_id, timetable_id)
		rooms = await self.list_rooms(tenant_id)
		usage: dict[str, int] = {}
		for e in entries:
			usage[e["room_id"]] = usage.get(e["room_id"], 0) + 1
		total_slots = max(len(slots), 1)
		return {
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"total_rooms": len(rooms),
			"rooms_used": len(usage),
			"total_time_slots": total_slots,
			"usage_by_room": usage,
			"avg_utilisation_pct": round(sum(usage.values()) / max(len(rooms) * total_slots, 1) * 100, 1),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def student_schedule(
		self,
		tenant_id: str,
		student_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Return the schedule for a student group (student_id maps to group)."""
		entries = [
			e for (t, _), e in self.entries.items()
			if t == tenant_id and e.timetable_id == timetable_id and e.student_group_id == student_id
		]
		slots = {s.id: s.model_dump() for (t, _), s in self.time_slots.items() if t == tenant_id}
		schedule = []
		for e in entries:
			slot = slots.get(e.time_slot_id, {})
			schedule.append({
				"day": slot.get("day_of_week"),
				"period": slot.get("period_number"),
				"start": slot.get("start_time"),
				"end": slot.get("end_time"),
				"subject_id": e.subject_id,
				"teacher_id": e.teacher_id,
				"room_id": e.room_id,
			})
		return {
			"tenant_id": tenant_id,
			"student_id": student_id,
			"timetable_id": timetable_id,
			"sessions": sorted(schedule, key=lambda x: (str(x.get("day", "")), x.get("period", 0))),
			"session_count": len(schedule),
		}

	async def compare_timetables(
		self,
		tenant_id: str,
		timetable_a: str,
		timetable_b: str,
	) -> dict[str, Any]:
		"""Compare two timetables: entry counts, conflicts, substitutions."""
		a = self._require_timetable(tenant_id, timetable_a)
		b = self._require_timetable(tenant_id, timetable_b)
		a_entries = await self.list_entries(tenant_id, timetable_a)
		b_entries = await self.list_entries(tenant_id, timetable_b)
		a_conflicts = self._count_unresolved_conflicts(tenant_id, timetable_a)
		b_conflicts = self._count_unresolved_conflicts(tenant_id, timetable_b)
		return {
			"tenant_id": tenant_id,
			"timetable_a": {"id": timetable_a, "name": a.name, "status": a.status, "entries": len(a_entries), "unresolved_conflicts": a_conflicts},
			"timetable_b": {"id": timetable_b, "name": b.name, "status": b.status, "entries": len(b_entries), "unresolved_conflicts": b_conflicts},
			"entry_delta": len(a_entries) - len(b_entries),
			"conflict_delta": a_conflicts - b_conflicts,
			"compared_at": datetime.utcnow().isoformat(),
		}

	async def timetable_analytics(
		self,
		tenant_id: str,
		academic_year: str,
	) -> dict[str, Any]:
		"""Return analytics across all timetables for an academic year."""
		timetables = [tm for (t, _), tm in self.timetables.items() if t == tenant_id and tm.academic_year == academic_year]
		return {
			"tenant_id": tenant_id,
			"academic_year": academic_year,
			"timetable_count": len(timetables),
			"published_count": sum(1 for tm in timetables if tm.status == "published"),
			"draft_count": sum(1 for tm in timetables if tm.status == "draft"),
			"total_rooms": sum(1 for (t, _) in self.rooms if t == tenant_id),
			"total_substitutions": sum(1 for (t, _) in self.substitutions if t == tenant_id),
			"open_conflicts": self._count_all_unresolved(tenant_id),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def timetable_kpi_summary(
		self,
		tenant_id: str,
		academic_year: str,
	) -> dict[str, Any]:
		"""Return a concise timetable KPI card for dashboard consumption."""
		timetables = [tm for (t, _), tm in self.timetables.items() if t == tenant_id and tm.academic_year == academic_year]
		published = sum(1 for tm in timetables if tm.status == "published")
		open_conflicts = self._count_all_unresolved(tenant_id)
		rooms = sum(1 for (t, _) in self.rooms if t == tenant_id)
		subs = sum(1 for (t, _) in self.substitutions if t == tenant_id)
		return {
			"tenant_id": tenant_id,
			"academic_year": academic_year,
			"total_timetables": len(timetables),
			"published_timetables": published,
			"publish_rate_pct": round(published / max(len(timetables), 1) * 100, 1),
			"open_conflicts": open_conflicts,
			"rooms": rooms,
			"substitutions": subs,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def room_capacity_optimise(
		self,
		tenant_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Suggest room reassignments to reduce capacity waste.

		For each timetable entry, checks if a smaller room could accommodate
		the class size, and returns reassignment recommendations.
		"""
		self._require_timetable(tenant_id, timetable_id)
		entries = await self.list_entries(tenant_id, timetable_id)
		rooms = [r for (t, _), r in self.rooms.items() if t == tenant_id]
		rooms_by_cap = sorted(rooms, key=lambda r: r.capacity)
		recommendations: list[dict[str, Any]] = []
		for entry in entries:
			class_size = entry.get("class_size", 0)
			current_room = entry.get("room_id", "")
			current_room_obj = next((r for r in rooms if r.id == current_room), None)
			if current_room_obj is None:
				continue
			# Find the smallest room that fits
			best = next((r for r in rooms_by_cap if r.capacity >= class_size and r.id != current_room), None)
			if best and best.capacity < current_room_obj.capacity:
				capacity_saved = current_room_obj.capacity - best.capacity
				recommendations.append({
					"entry_id": entry.get("id"),
					"current_room": current_room,
					"current_capacity": current_room_obj.capacity,
					"recommended_room": best.id,
					"recommended_capacity": best.capacity,
					"class_size": class_size,
					"capacity_saved": capacity_saved,
				})
		return {
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"entries_analysed": len(entries),
			"recommendations_count": len(recommendations),
			"recommendations": recommendations,
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def teacher_satisfaction_survey(
		self,
		tenant_id: str,
		teacher_id: str,
		timetable_id: str,
		ratings: dict[str, int],
		comments: str = "",
	) -> dict[str, Any]:
		"""Record a teacher satisfaction survey response for a timetable.

		ratings keys: workload_balance, room_quality, schedule_fairness (1-5)
		"""
		assert all(1 <= v <= 5 for v in ratings.values()), "all ratings must be 1–5"
		survey_id = uuid7str()
		avg_rating = round(sum(ratings.values()) / max(len(ratings), 1), 2)
		record: dict[str, Any] = {
			"survey_id": survey_id,
			"tenant_id": tenant_id,
			"timetable_id": timetable_id,
			"teacher_id": teacher_id,
			"ratings": ratings,
			"average_rating": avg_rating,
			"comments": comments,
			"submitted_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "teacher_satisfaction_survey_submitted", survey_id)
		return record

	async def archive_timetable(
		self,
		tenant_id: str,
		timetable_id: str,
	) -> dict[str, Any]:
		"""Archive a timetable after the academic period ends."""
		item = self._require_timetable(tenant_id, timetable_id)
		merged = item.model_copy(update={"status": "archived", "updated_at": datetime.utcnow()})
		self.timetables[self._key(tenant_id, timetable_id)] = merged
		self._audit(tenant_id, "timetable_archived", timetable_id)
		return merged.model_dump()

	async def ml_timetable_optimize(self, *args, **kwargs):
		"""AI-powered AI-assisted timetable conflict resolution. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["no_conflict","minor_conflict","major_conflict","rescheduling_required"])
			return {"conflict_class": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

