"""Flask Blueprint REST API for APG Learning Management System."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import LmsService
	from .capability_contract import get_capability_contract, evaluate_capability_rules
except ImportError:
	from service import LmsService  # type: ignore
	from capability_contract import get_capability_contract, evaluate_capability_rules  # type: ignore


blueprint = Blueprint("education_lms", __name__, url_prefix="/api/lms")
_service = LmsService()


def _loop() -> asyncio.AbstractEventLoop:
	try:
		return asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		return loop


def _run(coro):
    """Run a coroutine from Flask sync context. Python 3.12+ compatible."""
    import asyncio
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)
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
	GET /api/lms/contract
	Returns the capability contract for the current tenant.
	Permission: education_lms:view
	"""
	return _ok(get_capability_contract(_tenant()))


@blueprint.post("/evaluate")
def evaluate_rules():
	"""
	POST /api/lms/evaluate
	Evaluate business rules against a context payload.
	Permission: education_lms:admin
	"""
	body = request.get_json(force=True) or {}
	return _ok(evaluate_capability_rules(body))


# ---------------------------------------------------------------------------
# Courses
# ---------------------------------------------------------------------------

@blueprint.get("/courses")
def list_courses():
	"""
	GET /api/lms/courses
	List all courses for the tenant.
	Permission: education_lms:view
	"""
	try:
		return _ok(_run(_service.list_courses(_tenant())))
	except Exception as e:
		return _err(str(e))


@blueprint.post("/courses")
def create_course():
	"""
	POST /api/lms/courses
	Create a new course.
	Permission: education_lms:manage_courses
	Body: {title, code, course_type, owner_id, created_by, ...}
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_course(
			tenant_id=_tenant(),
			title=body["title"],
			code=body["code"],
			course_type=body["course_type"],
			owner_id=body["owner_id"],
			created_by=body["created_by"],
			description=body.get("description", ""),
			enrolment_type=body.get("enrolment_type", "open"),
			max_enrolments=body.get("max_enrolments"),
			duration_weeks=body.get("duration_weeks"),
			grading_scheme=body.get("grading_scheme", "percentage"),
			passing_score=body.get("passing_score", 50.0),
			completion_criteria=body.get("completion_criteria", []),
			tags=body.get("tags", []),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/courses/<course_id>")
def get_course(course_id: str):
	"""
	GET /api/lms/courses/<course_id>
	Retrieve a course by ID.
	Permission: education_lms:view
	"""
	result = _run(_service.get_course(_tenant(), course_id))
	if result is None:
		return _err("course not found", 404)
	return _ok(result)


@blueprint.put("/courses/<course_id>")
def update_course(course_id: str):
	"""
	PUT /api/lms/courses/<course_id>
	Update a course.
	Permission: education_lms:manage_courses
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.update_course(_tenant(), course_id, body)))
	except (AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/courses/<course_id>/publish")
def publish_course(course_id: str):
	"""
	POST /api/lms/courses/<course_id>/publish
	Publish a course. Body: {review_approved: bool}
	Permission: education_lms:manage_courses
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.publish_course(_tenant(), course_id, body.get("review_approved", False))))
	except (AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Content items
# ---------------------------------------------------------------------------

@blueprint.get("/courses/<course_id>/content")
def list_content(course_id: str):
	"""
	GET /api/lms/courses/<course_id>/content
	List content items for a course.
	Permission: education_lms:view
	"""
	try:
		return _ok(_run(_service.list_course_content(_tenant(), course_id)))
	except Exception as e:
		return _err(str(e))


@blueprint.post("/courses/<course_id>/content")
def add_content(course_id: str):
	"""
	POST /api/lms/courses/<course_id>/content
	Add a content item to a course.
	Permission: education_lms:manage_content
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.add_content_item(
			tenant_id=_tenant(),
			course_id=course_id,
			title=body["title"],
			content_type=body["content_type"],
			created_by=body["created_by"],
			url=body.get("url"),
			scorm_version=body.get("scorm_version"),
			duration_minutes=body.get("duration_minutes"),
			order_index=body.get("order_index", 0),
			is_required=body.get("is_required", True),
			metadata=body.get("metadata", {}),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Enrolments
# ---------------------------------------------------------------------------

@blueprint.get("/enrolments")
def list_enrolments():
	"""
	GET /api/lms/enrolments[?course_id=...]
	List enrolments.
	Permission: education_lms:manage_enrolments
	"""
	course_id = request.args.get("course_id")
	return _ok(_run(_service.list_enrolments(_tenant(), course_id)))


@blueprint.post("/enrolments")
def enrol_learner():
	"""
	POST /api/lms/enrolments
	Enrol a learner in a course.
	Permission: education_lms:manage_enrolments
	Body: {course_id, learner_id, enrolment_type, created_by, ...}
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.enrol_learner(
			tenant_id=_tenant(),
			course_id=body["course_id"],
			learner_id=body["learner_id"],
			enrolment_type=body["enrolment_type"],
			created_by=body["created_by"],
			payment_reference=body.get("payment_reference"),
			voucher_code=body.get("voucher_code"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.delete("/enrolments/<enrolment_id>")
def withdraw_enrolment(enrolment_id: str):
	"""
	DELETE /api/lms/enrolments/<enrolment_id>
	Withdraw an enrolment.
	Permission: education_lms:manage_enrolments
	"""
	try:
		return _ok(_run(_service.withdraw_enrolment(_tenant(), enrolment_id)))
	except (AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Assessments & submissions
# ---------------------------------------------------------------------------

@blueprint.post("/assessments")
def create_assessment():
	"""
	POST /api/lms/assessments
	Create an assessment.
	Permission: education_lms:manage_assessments
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_assessment(
			tenant_id=_tenant(),
			course_id=body["course_id"],
			title=body["title"],
			assessment_type=body["assessment_type"],
			created_by=body["created_by"],
			max_score=body.get("max_score", 100.0),
			passing_score=body.get("passing_score", 50.0),
			weight_percent=body.get("weight_percent", 100.0),
			time_limit_minutes=body.get("time_limit_minutes"),
			attempts_allowed=body.get("attempts_allowed", 1),
			instructions=body.get("instructions", ""),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/submissions")
def list_submissions():
	"""
	GET /api/lms/submissions[?assessment_id=...]
	List submissions.
	Permission: education_lms:grade
	"""
	assessment_id = request.args.get("assessment_id")
	return _ok(_run(_service.list_submissions(_tenant(), assessment_id)))


@blueprint.post("/submissions")
def submit_assessment():
	"""
	POST /api/lms/submissions
	Record a learner submission.
	Permission: education_lms:submit
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.submit_assessment(
			tenant_id=_tenant(),
			assessment_id=body["assessment_id"],
			enrolment_id=body["enrolment_id"],
			learner_id=body["learner_id"],
			created_by=body["created_by"],
			attempt_number=body.get("attempt_number", 1),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.put("/submissions/<submission_id>/grade")
def grade_submission(submission_id: str):
	"""
	PUT /api/lms/submissions/<submission_id>/grade
	Grade a submission.
	Permission: education_lms:grade
	Body: {score, graded_by, feedback?, override_approval?}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.grade_submission(
			tenant_id=_tenant(),
			submission_id=submission_id,
			score=body["score"],
			graded_by=body["graded_by"],
			feedback=body.get("feedback", ""),
			override_approval=body.get("override_approval"),
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------

@blueprint.get("/certificates")
def list_certificates():
	"""
	GET /api/lms/certificates[?learner_id=...]
	List certificates.
	Permission: education_lms:manage_certificates
	"""
	learner_id = request.args.get("learner_id")
	return _ok(_run(_service.list_certificates(_tenant(), learner_id)))


@blueprint.post("/certificates")
def issue_certificate():
	"""
	POST /api/lms/certificates
	Issue a certificate.
	Permission: education_lms:manage_certificates
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.issue_certificate(
			tenant_id=_tenant(),
			enrolment_id=body["enrolment_id"],
			learner_id=body["learner_id"],
			course_id=body["course_id"],
			certificate_type=body["certificate_type"],
			issuer_id=body["issuer_id"],
			created_by=body["created_by"],
			completion_criteria_met=body.get("completion_criteria_met", True),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Progress & analytics
# ---------------------------------------------------------------------------

@blueprint.post("/progress")
def record_progress():
	"""
	POST /api/lms/progress
	Record learner progress on a content item.
	Permission: education_lms:submit
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.record_progress(
			tenant_id=_tenant(),
			enrolment_id=body["enrolment_id"],
			learner_id=body["learner_id"],
			course_id=body["course_id"],
			content_item_id=body["content_item_id"],
			created_by=body["created_by"],
			completion_percentage=body.get("completion_percentage", 0.0),
			time_spent_minutes=body.get("time_spent_minutes", 0),
			xapi_statement=body.get("xapi_statement"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/analytics/learner/<learner_id>")
def learner_analytics(learner_id: str):
	"""
	GET /api/lms/analytics/learner/<learner_id>
	Learner analytics. Requires consent header X-Consent: true
	Permission: education_lms:analytics
	"""
	consent = request.headers.get("X-Consent", "false").lower() == "true"
	try:
		return _ok(_run(_service.learner_analytics(_tenant(), learner_id, consent)))
	except ValueError as e:
		return _err(str(e), 403)


@blueprint.get("/dashboard")
def dashboard():
	"""
	GET /api/lms/dashboard
	Dashboard summary for the tenant.
	Permission: education_lms:view
	"""
	return _ok(_run(_service.dashboard_summary(_tenant())))


# ---------------------------------------------------------------------------
# Learning paths
# ---------------------------------------------------------------------------

@blueprint.get("/paths")
def list_paths():
	"""
	GET /api/lms/paths
	List learning paths.
	Permission: education_lms:view
	"""
	return _ok(_run(_service.list_learning_paths(_tenant())))


@blueprint.post("/paths")
def create_path():
	"""
	POST /api/lms/paths
	Create a learning path.
	Permission: education_lms:manage_paths
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_learning_path(
			tenant_id=_tenant(),
			title=body["title"],
			owner_id=body["owner_id"],
			created_by=body["created_by"],
			description=body.get("description", ""),
			course_ids=body.get("course_ids", []),
			required_course_ids=body.get("required_course_ids", []),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@blueprint.post("/agents")
def register_agent():
	"""
	POST /api/lms/agents
	Register an LMS automation agent.
	Permission: education_lms:admin
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.register_agent(
			tenant_id=_tenant(),
			name=body["name"],
			runtime=body["runtime"],
			role=body["role"],
			created_by=body["created_by"],
			scope=body.get("scope", "lms operations"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/agents/validate-action")
def validate_agent_action():
	"""
	POST /api/lms/agents/validate-action
	Validate an agent action against policy rules.
	Permission: education_lms:admin
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.validate_agent_action(
			tenant_id=_tenant(),
			privileged_scope=body.get("privileged_scope", False),
			human_approval_recorded=body.get("human_approval_recorded", False),
		)))
	except Exception as e:
		return _err(str(e))
