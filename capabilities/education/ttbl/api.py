"""Flask Blueprint REST API for APG Timetabling & Scheduling."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import TimetablingService
	from .capability_contract import get_capability_contract, evaluate_capability_rules
except ImportError:
	from service import TimetablingService  # type: ignore
	from capability_contract import get_capability_contract, evaluate_capability_rules  # type: ignore


blueprint = Blueprint("education_ttbl", __name__, url_prefix="/api/ttbl")
_service = TimetablingService()


def _loop() -> asyncio.AbstractEventLoop:
	try:
		return asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		return loop


def _run(coro: Any) -> Any:
	return _loop().run_until_complete(coro)


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(message: str, status: int = 400):
	return jsonify({"status": "error", "message": message}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id", request.args.get("tenant_id", "default"))


# ---------------------------------------------------------------------------
# Contract / meta
# ---------------------------------------------------------------------------

@blueprint.get("/contract")
def get_contract():
	"""
	GET /api/ttbl/contract
	Returns the capability contract.
	Permission: education_ttbl:view
	"""
	return _ok(get_capability_contract(_tenant()))


@blueprint.post("/evaluate")
def evaluate_rules():
	"""
	POST /api/ttbl/evaluate
	Evaluate business rules against context.
	Permission: education_ttbl:admin
	"""
	body = request.get_json(force=True) or {}
	return _ok(evaluate_capability_rules(body))


@blueprint.get("/dashboard")
def dashboard():
	"""
	GET /api/ttbl/dashboard
	Dashboard summary.
	Permission: education_ttbl:view
	"""
	return _ok(_run(_service.dashboard_summary(_tenant())))


# ---------------------------------------------------------------------------
# Timetables
# ---------------------------------------------------------------------------

@blueprint.get("/timetables")
def list_timetables():
	"""
	GET /api/ttbl/timetables[?status=...]
	List timetables.
	Permission: education_ttbl:view
	"""
	status = request.args.get("status")
	return _ok(_run(_service.list_timetables(_tenant(), status)))


@blueprint.post("/timetables")
def create_timetable():
	"""
	POST /api/ttbl/timetables
	Create a timetable.
	Permission: education_ttbl:manage_timetables
	Body: {name, timetable_type, academic_year, term, created_by, ...}
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_timetable(
			tenant_id=_tenant(),
			name=body["name"],
			timetable_type=body["timetable_type"],
			academic_year=body["academic_year"],
			term=body["term"],
			created_by=body["created_by"],
			generation_algorithm=body.get("generation_algorithm", "constraint_propagation"),
			metadata=body.get("metadata", {}),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/timetables/<timetable_id>")
def get_timetable(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>
	Retrieve a timetable.
	Permission: education_ttbl:view
	"""
	result = _run(_service.get_timetable(_tenant(), timetable_id))
	if result is None:
		return _err("timetable not found", 404)
	return _ok(result)


@blueprint.post("/timetables/<timetable_id>/publish")
def publish_timetable(timetable_id: str):
	"""
	POST /api/ttbl/timetables/<timetable_id>/publish
	Publish a timetable. Requires zero unresolved conflicts.
	Permission: education_ttbl:manage_timetables
	Body: {approval_reference}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.publish_timetable(
			_tenant(), timetable_id, body["approval_reference"]
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@blueprint.get("/timetables/<timetable_id>/constraints")
def list_constraints(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>/constraints
	List constraints for a timetable.
	Permission: education_ttbl:manage_constraints
	"""
	return _ok(_run(_service.list_constraints(_tenant(), timetable_id)))


@blueprint.post("/timetables/<timetable_id>/constraints")
def add_constraint(timetable_id: str):
	"""
	POST /api/ttbl/timetables/<timetable_id>/constraints
	Add a constraint.
	Permission: education_ttbl:manage_constraints
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.add_constraint(
			tenant_id=_tenant(),
			timetable_id=timetable_id,
			constraint_type=body["constraint_type"],
			entity_id=body["entity_id"],
			entity_type=body["entity_type"],
			created_by=body["created_by"],
			description=body.get("description", ""),
			parameters=body.get("parameters", {}),
			is_hard=body.get("is_hard", True),
			weight=body.get("weight", 100),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.delete("/constraints/<constraint_id>")
def remove_constraint(constraint_id: str):
	"""
	DELETE /api/ttbl/constraints/<constraint_id>
	Remove a constraint. Requires approval_reference.
	Permission: education_ttbl:manage_constraints
	Body: {approval_reference}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.remove_constraint(_tenant(), constraint_id, body["approval_reference"])))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------

@blueprint.get("/rooms")
def list_rooms():
	"""
	GET /api/ttbl/rooms[?room_type=...&available_only=true]
	List rooms.
	Permission: education_ttbl:manage_rooms
	"""
	room_type = request.args.get("room_type")
	available_only = request.args.get("available_only", "false").lower() == "true"
	return _ok(_run(_service.list_rooms(_tenant(), room_type, available_only)))


@blueprint.post("/rooms")
def create_room():
	"""
	POST /api/ttbl/rooms
	Register a room.
	Permission: education_ttbl:manage_rooms
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_room(
			tenant_id=_tenant(),
			name=body["name"],
			code=body["code"],
			room_type=body["room_type"],
			capacity=body["capacity"],
			created_by=body["created_by"],
			building=body.get("building"),
			floor=body.get("floor"),
			amenities=body.get("amenities", []),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Time slots
# ---------------------------------------------------------------------------

@blueprint.get("/timetables/<timetable_id>/slots")
def list_slots(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>/slots
	List time slots for a timetable.
	Permission: education_ttbl:view
	"""
	return _ok(_run(_service.list_time_slots(_tenant(), timetable_id)))


@blueprint.post("/timetables/<timetable_id>/slots")
def create_slot(timetable_id: str):
	"""
	POST /api/ttbl/timetables/<timetable_id>/slots
	Create a time slot.
	Permission: education_ttbl:manage_timetables
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_time_slot(
			tenant_id=_tenant(),
			timetable_id=timetable_id,
			day_of_week=body["day_of_week"],
			start_time=body["start_time"],
			end_time=body["end_time"],
			duration_minutes=body["duration_minutes"],
			period_number=body["period_number"],
			created_by=body["created_by"],
			is_break=body.get("is_break", False),
			label=body.get("label"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Schedule entries
# ---------------------------------------------------------------------------

@blueprint.get("/timetables/<timetable_id>/entries")
def list_entries(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>/entries[?teacher_id=...&room_id=...]
	List schedule entries.
	Permission: education_ttbl:view
	"""
	teacher_id = request.args.get("teacher_id")
	room_id = request.args.get("room_id")
	return _ok(_run(_service.list_entries(_tenant(), timetable_id, teacher_id, room_id)))


@blueprint.post("/timetables/<timetable_id>/entries")
def assign_entry(timetable_id: str):
	"""
	POST /api/ttbl/timetables/<timetable_id>/entries
	Assign a schedule entry (teacher + room + subject to a time slot).
	Permission: education_ttbl:manage_timetables
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.assign_entry(
			tenant_id=_tenant(),
			timetable_id=timetable_id,
			time_slot_id=body["time_slot_id"],
			room_id=body["room_id"],
			teacher_id=body["teacher_id"],
			subject_id=body["subject_id"],
			student_group_id=body["student_group_id"],
			created_by=body["created_by"],
			capacity_check_performed=body.get("capacity_check_performed", True),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------

@blueprint.get("/timetables/<timetable_id>/conflicts")
def list_conflicts(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>/conflicts[?unresolved_only=true]
	List conflicts for a timetable.
	Permission: education_ttbl:resolve_conflicts
	"""
	unresolved_only = request.args.get("unresolved_only", "false").lower() == "true"
	return _ok(_run(_service.list_conflicts(_tenant(), timetable_id, unresolved_only)))


@blueprint.put("/conflicts/<conflict_id>/resolve")
def resolve_conflict(conflict_id: str):
	"""
	PUT /api/ttbl/conflicts/<conflict_id>/resolve
	Resolve a conflict.
	Permission: education_ttbl:resolve_conflicts
	Body: {resolution_type, resolved_by}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.resolve_conflict(
			_tenant(), conflict_id, body["resolution_type"], body["resolved_by"]
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Substitutions
# ---------------------------------------------------------------------------

@blueprint.get("/substitutions")
def list_substitutions():
	"""
	GET /api/ttbl/substitutions[?timetable_id=...&status=...]
	List substitution requests.
	Permission: education_ttbl:manage_substitutions
	"""
	timetable_id = request.args.get("timetable_id")
	status = request.args.get("status")
	return _ok(_run(_service.list_substitutions(_tenant(), timetable_id, status)))


@blueprint.post("/substitutions")
def request_substitution():
	"""
	POST /api/ttbl/substitutions
	Create a substitution request.
	Permission: education_ttbl:manage_substitutions
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.request_substitution(
			tenant_id=_tenant(),
			timetable_id=body["timetable_id"],
			original_entry_id=body["original_entry_id"],
			absent_teacher_id=body["absent_teacher_id"],
			reason=body["reason"],
			date=body["date"],
			created_by=body["created_by"],
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.put("/substitutions/<substitution_id>/assign")
def assign_substitution(substitution_id: str):
	"""
	PUT /api/ttbl/substitutions/<substitution_id>/assign
	Assign a substitute teacher.
	Permission: education_ttbl:manage_substitutions
	Body: {substitute_teacher_id, teacher_consent_recorded}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.assign_substitution(
			_tenant(), substitution_id, body["substitute_teacher_id"],
			body.get("teacher_consent_recorded", False),
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@blueprint.get("/timetables/<timetable_id>/export")
def export_timetable(timetable_id: str):
	"""
	GET /api/ttbl/timetables/<timetable_id>/export?format=ical
	Export a timetable in the specified format.
	Permission: education_ttbl:export
	"""
	export_format = request.args.get("format", "json")
	try:
		return _ok(_run(_service.export_timetable(_tenant(), timetable_id, export_format)))
	except (AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@blueprint.post("/agents")
def register_agent():
	"""
	POST /api/ttbl/agents
	Register a timetabling AI agent.
	Permission: education_ttbl:admin
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.register_agent(
			tenant_id=_tenant(),
			name=body["name"],
			runtime=body["runtime"],
			role=body["role"],
			created_by=body["created_by"],
			scope=body.get("scope", "timetabling operations"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))
