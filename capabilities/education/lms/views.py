"""View model helpers for APG Learning Management System screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import LmsService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import LmsService  # type: ignore


def dashboard_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the LMS dashboard view."""
	import asyncio
	contract = get_capability_contract(tenant_id)
	loop = asyncio.get_event_loop()
	summary = loop.run_until_complete(service.dashboard_summary(tenant_id))
	return {
		"title": "Learning Management System",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def course_library_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the course library listing."""
	import asyncio
	loop = asyncio.get_event_loop()
	courses = loop.run_until_complete(service.list_courses(tenant_id))
	return {
		"tenant_id": tenant_id,
		"courses": courses,
		"total": len(courses),
		"published": [c for c in courses if c["status"] == "published"],
		"drafts": [c for c in courses if c["status"] == "draft"],
	}


def course_detail_model(service: LmsService, tenant_id: str, course_id: str) -> dict[str, Any]:
	"""Data model for a single course detail view."""
	import asyncio
	loop = asyncio.get_event_loop()
	course = loop.run_until_complete(service.get_course(tenant_id, course_id))
	content = loop.run_until_complete(service.list_course_content(tenant_id, course_id))
	enrolments = loop.run_until_complete(service.list_enrolments(tenant_id, course_id))
	assessments = [
		a for a in [
			service.assessments.get((tenant_id, k)) for (t, k) in service.assessments
			if t == tenant_id
		]
		if a and a.course_id == course_id
	]
	return {
		"tenant_id": tenant_id,
		"course": course,
		"content_items": content,
		"enrolment_count": len(enrolments),
		"assessment_count": len(assessments),
	}


def enrolment_console_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the enrolment management console."""
	import asyncio
	loop = asyncio.get_event_loop()
	enrolments = loop.run_until_complete(service.list_enrolments(tenant_id))
	return {
		"tenant_id": tenant_id,
		"enrolments": enrolments,
		"active": [e for e in enrolments if e["status"] == "active"],
		"pending": [e for e in enrolments if e["status"] == "pending"],
		"completed": [e for e in enrolments if e["status"] == "completed"],
		"withdrawn": [e for e in enrolments if e["status"] == "withdrawn"],
	}


def gradebook_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the gradebook view."""
	import asyncio
	loop = asyncio.get_event_loop()
	submissions = loop.run_until_complete(service.list_submissions(tenant_id))
	return {
		"tenant_id": tenant_id,
		"submissions": submissions,
		"graded": [s for s in submissions if s["status"] == "graded"],
		"pending_grading": [s for s in submissions if s["status"] == "submitted"],
		"returned": [s for s in submissions if s["status"] == "returned"],
	}


def certificate_console_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the certificate management console."""
	return {
		"tenant_id": tenant_id,
		"certificates": [
			c.model_dump() for (t, _), c in service.certificates.items() if t == tenant_id
		],
	}


def learner_analytics_model(
	service: LmsService, tenant_id: str, learner_id: str, consent_recorded: bool = True
) -> dict[str, Any]:
	"""Data model for the learner analytics view."""
	import asyncio
	loop = asyncio.get_event_loop()
	analytics = loop.run_until_complete(service.learner_analytics(tenant_id, learner_id, consent_recorded))
	return {
		"tenant_id": tenant_id,
		"learner_id": learner_id,
		"analytics": analytics,
	}


def learning_paths_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the learning paths view."""
	import asyncio
	loop = asyncio.get_event_loop()
	paths = loop.run_until_complete(service.list_learning_paths(tenant_id))
	return {
		"tenant_id": tenant_id,
		"learning_paths": paths,
	}


def agent_workbench_model(service: LmsService, tenant_id: str = "default") -> dict[str, Any]:
	"""Data model for the LMS agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [a.model_dump() for (t, _), a in service.agents.items() if t == tenant_id],
	}


def _tenant_list(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.model_dump() for (t, _), item in items.items() if t == tenant_id]
