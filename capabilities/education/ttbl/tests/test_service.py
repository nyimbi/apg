"""Service-layer tests for education_ttbl."""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from service import TimetablingService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


T = "test_school"


# ---------------------------------------------------------------------------
# timetables
# ---------------------------------------------------------------------------

def test_create_and_get_timetable():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "Master 2025", "master", "2025-2026", "term_1", "admin"))
	assert tt["status"] == "draft"
	assert tt["timetable_type"] == "master"
	fetched = run(svc.get_timetable(T, tt["id"]))
	assert fetched["id"] == tt["id"]


def test_list_timetables_by_status():
	svc = TimetablingService()
	run(svc.create_timetable(T, "Draft 1", "master", "2025-2026", "term_1", "admin"))
	run(svc.create_timetable(T, "Draft 2", "class", "2025-2026", "term_1", "admin"))
	drafts = run(svc.list_timetables(T, status="draft"))
	assert len(drafts) == 2


def test_unsupported_timetable_type_denied():
	svc = TimetablingService()
	with pytest.raises(ValueError):
		run(svc.create_timetable(T, "X", "fantasy_type", "2025-2026", "term_1", "admin"))


def test_unsupported_algorithm_denied():
	svc = TimetablingService()
	with pytest.raises(ValueError):
		run(svc.create_timetable(T, "X", "master", "2025-2026", "term_1", "admin", generation_algorithm="magic_wand"))


def test_publish_with_unresolved_conflicts_denied():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	# manually inject an unresolved conflict
	from models import ConflictCreate
	from datetime import datetime
	conflict = ConflictCreate(
		tenant_id=T, timetable_id=tt["id"], conflict_type="teacher_double_booked",
		entry_ids=[], description="test", severity="hard", created_by="system",
	)
	svc.conflicts[(T, conflict.id)] = conflict
	with pytest.raises(ValueError, match="timetable_publish_requires_zero_conflicts"):
		run(svc.publish_timetable(T, tt["id"], "APPR-001"))


def test_publish_timetable_requires_approval():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	with pytest.raises(ValueError, match="timetable_publish_requires_approval"):
		run(svc.publish_timetable(T, tt["id"], ""))


def test_publish_timetable_success():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	published = run(svc.publish_timetable(T, tt["id"], "APPR-PUB-001"))
	assert published["status"] == "published"
	assert published["approval_reference"] == "APPR-PUB-001"


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------

def test_add_constraint():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	c = run(svc.add_constraint(T, tt["id"], "teacher_availability", "teacher_1", "teacher", "admin"))
	assert c["constraint_type"] == "teacher_availability"
	assert c["is_hard"] is True


def test_unsupported_constraint_type_denied():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	with pytest.raises(ValueError):
		run(svc.add_constraint(T, tt["id"], "must_teach_in_space", "teacher_1", "teacher", "admin"))


def test_remove_constraint_requires_approval():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	c = run(svc.add_constraint(T, tt["id"], "teacher_availability", "teacher_1", "teacher", "admin"))
	with pytest.raises(ValueError, match="constraint_removal_requires_approval"):
		run(svc.remove_constraint(T, c["id"], ""))


def test_remove_constraint_with_approval():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	c = run(svc.add_constraint(T, tt["id"], "teacher_availability", "teacher_1", "teacher", "admin"))
	removed = run(svc.remove_constraint(T, c["id"], "APPR-REM-001"))
	assert removed["removal_approval"] == "APPR-REM-001"


# ---------------------------------------------------------------------------
# rooms
# ---------------------------------------------------------------------------

def test_create_room():
	svc = TimetablingService()
	room = run(svc.create_room(T, "Room 101", "R101", "classroom", 40, "admin"))
	assert room["room_type"] == "classroom"
	assert room["capacity"] == 40


def test_unsupported_room_type_denied():
	svc = TimetablingService()
	with pytest.raises(ValueError):
		run(svc.create_room(T, "X", "X1", "cave", 10, "admin"))


def test_list_rooms_available_only():
	svc = TimetablingService()
	run(svc.create_room(T, "R1", "R1", "classroom", 30, "admin"))
	run(svc.create_room(T, "R2", "R2", "lab", 20, "admin"))
	all_rooms = run(svc.list_rooms(T))
	assert len(all_rooms) == 2
	available = run(svc.list_rooms(T, available_only=True))
	assert len(available) == 2


# ---------------------------------------------------------------------------
# time slots
# ---------------------------------------------------------------------------

def test_create_time_slot():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	assert slot["day_of_week"] == "monday"
	assert slot["duration_minutes"] == 45


def test_unsupported_slot_duration_denied():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	with pytest.raises(ValueError):
		run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:13", 13, 1, "admin"))


def test_list_time_slots_ordered():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	run(svc.create_time_slot(T, tt["id"], "tuesday", "09:00", "09:45", 45, 2, "admin"))
	run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	slots = run(svc.list_time_slots(T, tt["id"]))
	assert slots[0]["day_of_week"] == "monday"


# ---------------------------------------------------------------------------
# schedule entries & conflict detection
# ---------------------------------------------------------------------------

def test_assign_entry():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	room = run(svc.create_room(T, "R101", "R101", "classroom", 30, "admin"))
	entry = run(svc.assign_entry(T, tt["id"], slot["id"], room["id"], "teacher_1", "math", "class_1a", "admin"))
	assert entry["teacher_id"] == "teacher_1"


def test_teacher_double_booking_creates_conflict():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	room1 = run(svc.create_room(T, "R1", "R1", "classroom", 30, "admin"))
	room2 = run(svc.create_room(T, "R2", "R2", "classroom", 30, "admin"))
	run(svc.assign_entry(T, tt["id"], slot["id"], room1["id"], "teacher_1", "math", "class_1a", "admin"))
	run(svc.assign_entry(T, tt["id"], slot["id"], room2["id"], "teacher_1", "english", "class_1b", "admin"))
	conflicts = run(svc.list_conflicts(T, tt["id"], unresolved_only=True))
	teacher_conflicts = [c for c in conflicts if c["conflict_type"] == "teacher_double_booked"]
	assert len(teacher_conflicts) >= 1


# ---------------------------------------------------------------------------
# conflict resolution
# ---------------------------------------------------------------------------

def test_resolve_conflict():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	conflict = run(svc.log_conflict(T, tt["id"], "teacher_double_booked", [], "test conflict", "hard", "admin"))
	resolved = run(svc.resolve_conflict(T, conflict["id"], "reassign_teacher", "admin"))
	assert resolved["resolution_type"] == "reassign_teacher"
	assert resolved["resolved_at"] is not None


def test_unsupported_resolution_type_denied():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	conflict = run(svc.log_conflict(T, tt["id"], "teacher_double_booked", [], "test", "hard", "admin"))
	with pytest.raises(ValueError):
		run(svc.resolve_conflict(T, conflict["id"], "magic_fix", "admin"))


# ---------------------------------------------------------------------------
# substitutions
# ---------------------------------------------------------------------------

def test_request_substitution():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	room = run(svc.create_room(T, "R1", "R1", "classroom", 30, "admin"))
	entry = run(svc.assign_entry(T, tt["id"], slot["id"], room["id"], "teacher_1", "math", "class_1a", "admin"))
	sub = run(svc.request_substitution(T, tt["id"], entry["id"], "teacher_1", "sick leave", "2025-09-15", "admin"))
	assert sub["status"] == "pending"


def test_assign_substitution_requires_consent():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	room = run(svc.create_room(T, "R1", "R1", "classroom", 30, "admin"))
	entry = run(svc.assign_entry(T, tt["id"], slot["id"], room["id"], "teacher_1", "math", "class_1a", "admin"))
	sub = run(svc.request_substitution(T, tt["id"], entry["id"], "teacher_1", "sick", "2025-09-15", "admin"))
	with pytest.raises(ValueError, match="substitution_requires_teacher_consent"):
		run(svc.assign_substitution(T, sub["id"], "teacher_2", teacher_consent_recorded=False))


def test_assign_substitution_with_consent():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	slot = run(svc.create_time_slot(T, tt["id"], "monday", "08:00", "08:45", 45, 1, "admin"))
	room = run(svc.create_room(T, "R1", "R1", "classroom", 30, "admin"))
	entry = run(svc.assign_entry(T, tt["id"], slot["id"], room["id"], "teacher_1", "math", "class_1a", "admin"))
	sub = run(svc.request_substitution(T, tt["id"], entry["id"], "teacher_1", "sick", "2025-09-15", "admin"))
	assigned = run(svc.assign_substitution(T, sub["id"], "teacher_2", teacher_consent_recorded=True))
	assert assigned["status"] == "assigned"
	assert assigned["substitute_teacher_id"] == "teacher_2"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_export_timetable():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	export = run(svc.export_timetable(T, tt["id"], "json"))
	assert export["format"] == "json"
	assert "timetable" in export


def test_unsupported_export_format_denied():
	svc = TimetablingService()
	tt = run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	with pytest.raises(ValueError):
		run(svc.export_timetable(T, tt["id"], "docx"))


# ---------------------------------------------------------------------------
# dashboard & agents
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = TimetablingService()
	run(svc.create_timetable(T, "TT", "master", "2025-2026", "term_1", "admin"))
	summary = run(svc.dashboard_summary(T))
	assert summary["timetables"] >= 1
	assert summary["tenant_id"] == T


def test_register_agent():
	svc = TimetablingService()
	agent = run(svc.register_agent(T, "ScheduleBot", "codex", "schedule_optimizer", "admin"))
	assert agent["role"] == "schedule_optimizer"


def test_invalid_agent_runtime_rejected():
	svc = TimetablingService()
	with pytest.raises(AssertionError):
		run(svc.register_agent(T, "Bot", "skynet", "schedule_optimizer", "admin"))
