"""View model helpers for APG School Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import SchoolManagementService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import SchoolManagementService  # type: ignore


def dashboard_model(service: SchoolManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the school management dashboard."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.get_event_loop()
	summary = loop.run_until_complete(service.dashboard_summary(tenant_id))
	return {
		"title": "School Management",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def student_registry_model(
	service: SchoolManagementService,
	tenant_id: str = "default",
	grade_level: str | None = None,
	status: str | None = None,
) -> dict[str, Any]:
	"""Data model for the student registry."""
	import asyncio
	loop = asyncio.get_event_loop()
	students = loop.run_until_complete(service.list_students(tenant_id, grade_level, status))
	return {
		"tenant_id": tenant_id,
		"students": students,
		"total": len(students),
		"grade_level_filter": grade_level,
		"status_filter": status,
	}


def student_profile_model(
	service: SchoolManagementService, tenant_id: str, student_id: str
) -> dict[str, Any]:
	"""Data model for a student profile page."""
	import asyncio
	loop = asyncio.get_event_loop()
	student = loop.run_until_complete(service.get_student(tenant_id, student_id))
	invoices = loop.run_until_complete(service.list_fee_invoices(tenant_id, student_id=student_id))
	docs = [
		d.model_dump() for (t, _), d in service.documents.items()
		if t == tenant_id and d.owner_id == student_id
	]
	return {
		"tenant_id": tenant_id,
		"student": student,
		"fee_invoices": invoices,
		"documents": docs,
	}


def admissions_console_model(
	service: SchoolManagementService, tenant_id: str = "default", status: str | None = None
) -> dict[str, Any]:
	"""Data model for the admissions console."""
	import asyncio
	loop = asyncio.get_event_loop()
	applications = loop.run_until_complete(service.list_admissions(tenant_id, status))
	return {
		"tenant_id": tenant_id,
		"applications": applications,
		"total": len(applications),
		"pending_review": [a for a in applications if a["status"] in ("submitted", "under_review")],
		"shortlisted": [a for a in applications if a["status"] == "shortlisted"],
		"offered": [a for a in applications if a["status"] == "offered"],
	}


def fee_management_model(
	service: SchoolManagementService, tenant_id: str = "default"
) -> dict[str, Any]:
	"""Data model for the fee management console."""
	import asyncio
	loop = asyncio.get_event_loop()
	invoices = loop.run_until_complete(service.list_fee_invoices(tenant_id))
	total_outstanding = sum(
		inv["amount"] for inv in invoices if inv["status"] in ("pending", "overdue")
	)
	return {
		"tenant_id": tenant_id,
		"invoices": invoices,
		"total_outstanding": total_outstanding,
		"overdue": [inv for inv in invoices if inv["status"] == "overdue"],
		"paid": [inv for inv in invoices if inv["status"] == "paid"],
	}


def staff_directory_model(
	service: SchoolManagementService, tenant_id: str = "default", role: str | None = None
) -> dict[str, Any]:
	"""Data model for the staff directory."""
	import asyncio
	loop = asyncio.get_event_loop()
	staff = loop.run_until_complete(service.list_staff(tenant_id, role))
	return {
		"tenant_id": tenant_id,
		"staff": staff,
		"total": len(staff),
		"role_filter": role,
	}


def academic_calendar_model(
	service: SchoolManagementService,
	tenant_id: str = "default",
	academic_year: str | None = None,
	term: str | None = None,
) -> dict[str, Any]:
	"""Data model for the academic calendar."""
	import asyncio
	loop = asyncio.get_event_loop()
	events = loop.run_until_complete(service.list_calendar_events(tenant_id, academic_year, term))
	return {
		"tenant_id": tenant_id,
		"events": events,
		"academic_year": academic_year,
		"term": term,
	}


def parent_portal_model(
	service: SchoolManagementService, tenant_id: str, student_id: str
) -> dict[str, Any]:
	"""Data model for the parent portal (read-only student view)."""
	import asyncio
	loop = asyncio.get_event_loop()
	student = loop.run_until_complete(service.get_student(tenant_id, student_id))
	invoices = loop.run_until_complete(service.list_fee_invoices(tenant_id, student_id=student_id))
	events = loop.run_until_complete(service.list_calendar_events(tenant_id))
	return {
		"tenant_id": tenant_id,
		"student": student,
		"fee_invoices": invoices,
		"upcoming_events": [e for e in events if e["is_public"]],
	}


def agent_workbench_model(
	service: SchoolManagementService, tenant_id: str = "default"
) -> dict[str, Any]:
	"""Data model for the school management agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [a.model_dump() for (t, _), a in service.agents.items() if t == tenant_id],
	}


def _tenant_list(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.model_dump() for (t, _), item in items.items() if t == tenant_id]
