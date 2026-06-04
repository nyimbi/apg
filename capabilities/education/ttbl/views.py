"""View model helpers for APG Timetabling & Scheduling screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import TimetablingService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import TimetablingService  # type: ignore


def dashboard_model(service: TimetablingService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the timetabling dashboard."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.get_event_loop()
	summary = loop.run_until_complete(service.dashboard_summary(tenant_id))
	return {
		"title": "Timetabling & Scheduling",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def timetable_list_model(
	service: TimetablingService, tenant_id: str = "default", status: str | None = None
) -> dict[str, Any]:
	"""Data model for the timetable listing."""
	import asyncio
	loop = asyncio.get_event_loop()
	timetables = loop.run_until_complete(service.list_timetables(tenant_id, status))
	return {
		"tenant_id": tenant_id,
		"timetables": timetables,
		"total": len(timetables),
		"published": [t for t in timetables if t["status"] == "published"],
		"drafts": [t for t in timetables if t["status"] == "draft"],
	}


def timetable_builder_model(
	service: TimetablingService, tenant_id: str, timetable_id: str
) -> dict[str, Any]:
	"""Data model for the interactive timetable builder."""
	import asyncio
	loop = asyncio.get_event_loop()
	timetable = loop.run_until_complete(service.get_timetable(tenant_id, timetable_id))
	entries = loop.run_until_complete(service.list_entries(tenant_id, timetable_id))
	slots = loop.run_until_complete(service.list_time_slots(tenant_id, timetable_id))
	constraints = loop.run_until_complete(service.list_constraints(tenant_id, timetable_id))
	conflicts = loop.run_until_complete(service.list_conflicts(tenant_id, timetable_id, unresolved_only=True))
	rooms = loop.run_until_complete(service.list_rooms(tenant_id, available_only=True))
	return {
		"tenant_id": tenant_id,
		"timetable": timetable,
		"entries": entries,
		"time_slots": slots,
		"constraints": constraints,
		"unresolved_conflicts": conflicts,
		"available_rooms": rooms,
	}


def constraint_editor_model(
	service: TimetablingService, tenant_id: str, timetable_id: str
) -> dict[str, Any]:
	"""Data model for the constraint editor."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.get_event_loop()
	constraints = loop.run_until_complete(service.list_constraints(tenant_id, timetable_id))
	return {
		"tenant_id": tenant_id,
		"timetable_id": timetable_id,
		"constraints": constraints,
		"supported_constraint_types": contract["configuration"]["constraints"]["supported_types"],
		"hard_constraints": [c for c in constraints if c["is_hard"]],
		"soft_constraints": [c for c in constraints if not c["is_hard"]],
	}


def room_inventory_model(
	service: TimetablingService, tenant_id: str = "default", room_type: str | None = None
) -> dict[str, Any]:
	"""Data model for the room inventory."""
	import asyncio
	loop = asyncio.get_event_loop()
	rooms = loop.run_until_complete(service.list_rooms(tenant_id, room_type))
	return {
		"tenant_id": tenant_id,
		"rooms": rooms,
		"total": len(rooms),
		"available": [r for r in rooms if r["is_available"]],
		"room_type_filter": room_type,
	}


def conflict_resolution_model(
	service: TimetablingService, tenant_id: str, timetable_id: str
) -> dict[str, Any]:
	"""Data model for the conflict resolution workbench."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.get_event_loop()
	all_conflicts = loop.run_until_complete(service.list_conflicts(tenant_id, timetable_id))
	unresolved = [c for c in all_conflicts if c["resolved_at"] is None]
	resolved = [c for c in all_conflicts if c["resolved_at"] is not None]
	return {
		"tenant_id": tenant_id,
		"timetable_id": timetable_id,
		"unresolved_conflicts": unresolved,
		"resolved_conflicts": resolved,
		"supported_resolutions": contract["configuration"]["conflicts"]["supported_resolutions"],
	}


def substitution_console_model(
	service: TimetablingService,
	tenant_id: str = "default",
	timetable_id: str | None = None,
	status: str | None = None,
) -> dict[str, Any]:
	"""Data model for the substitution console."""
	import asyncio
	loop = asyncio.get_event_loop()
	substitutions = loop.run_until_complete(service.list_substitutions(tenant_id, timetable_id, status))
	return {
		"tenant_id": tenant_id,
		"substitutions": substitutions,
		"pending": [s for s in substitutions if s["status"] == "pending"],
		"assigned": [s for s in substitutions if s["status"] == "assigned"],
		"confirmed": [s for s in substitutions if s["status"] == "confirmed"],
	}


def teacher_timetable_model(
	service: TimetablingService, tenant_id: str, timetable_id: str, teacher_id: str
) -> dict[str, Any]:
	"""Data model for a teacher's personal timetable view."""
	import asyncio
	loop = asyncio.get_event_loop()
	entries = loop.run_until_complete(service.list_entries(tenant_id, timetable_id, teacher_id=teacher_id))
	slots = loop.run_until_complete(service.list_time_slots(tenant_id, timetable_id))
	slot_map = {s["id"]: s for s in slots}
	return {
		"tenant_id": tenant_id,
		"teacher_id": teacher_id,
		"timetable_id": timetable_id,
		"entries": entries,
		"slot_map": slot_map,
		"period_count": len(entries),
	}


def room_timetable_model(
	service: TimetablingService, tenant_id: str, timetable_id: str, room_id: str
) -> dict[str, Any]:
	"""Data model for a room's occupancy timetable."""
	import asyncio
	loop = asyncio.get_event_loop()
	entries = loop.run_until_complete(service.list_entries(tenant_id, timetable_id, room_id=room_id))
	return {
		"tenant_id": tenant_id,
		"room_id": room_id,
		"timetable_id": timetable_id,
		"entries": entries,
		"utilisation_slots": len(entries),
	}


def agent_workbench_model(
	service: TimetablingService, tenant_id: str = "default"
) -> dict[str, Any]:
	"""Data model for the timetabling agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [a.model_dump() for (t, _), a in service.agents.items() if t == tenant_id],
	}


def _tenant_list(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.model_dump() for (t, _), item in items.items() if t == tenant_id]
