"""Service-layer tests for education_lms."""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from service import LmsService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


T = "test_tenant"


# ---------------------------------------------------------------------------
# courses
# ---------------------------------------------------------------------------

def test_create_and_get_course():
	svc = LmsService()
	result = run(svc.create_course(T, "Python 101", "PY101", "self_paced", "instructor_1", "admin"))
	assert result["title"] == "Python 101"
	assert result["status"] == "draft"
	fetched = run(svc.get_course(T, result["id"]))
	assert fetched["id"] == result["id"]


def test_list_courses_by_tenant():
	svc = LmsService()
	run(svc.create_course(T, "Course A", "CA", "self_paced", "o1", "admin"))
	run(svc.create_course(T, "Course B", "CB", "instructor_led", "o1", "admin"))
	run(svc.create_course("other_tenant", "Course X", "CX", "self_paced", "o2", "admin"))
	courses = run(svc.list_courses(T))
	assert len(courses) == 2
	assert all(c["tenant_id"] == T for c in courses)


def test_update_course():
	svc = LmsService()
	course = run(svc.create_course(T, "Old Title", "OT", "self_paced", "o1", "admin"))
	updated = run(svc.update_course(T, course["id"], {"title": "New Title"}))
	assert updated["title"] == "New Title"


def test_publish_course_requires_review():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	with pytest.raises(ValueError, match="course_publish_requires_review"):
		run(svc.publish_course(T, course["id"], review_approved=False))


def test_publish_course_succeeds_with_review():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	published = run(svc.publish_course(T, course["id"], review_approved=True))
	assert published["status"] == "published"


def test_get_course_missing_returns_none():
	svc = LmsService()
	result = run(svc.get_course(T, "nonexistent"))
	assert result is None


def test_unsupported_course_type_denied():
	svc = LmsService()
	with pytest.raises(ValueError):
		run(svc.create_course(T, "X", "X", "invalid_type", "o1", "admin"))


# ---------------------------------------------------------------------------
# content items
# ---------------------------------------------------------------------------

def test_add_content_item():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	item = run(svc.add_content_item(T, course["id"], "Intro Video", "video", "admin", url="http://example.com/vid"))
	assert item["content_type"] == "video"
	assert item["course_id"] == course["id"]


def test_list_course_content_ordered():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	run(svc.add_content_item(T, course["id"], "C", "video", "admin", order_index=2))
	run(svc.add_content_item(T, course["id"], "A", "document", "admin", order_index=0))
	run(svc.add_content_item(T, course["id"], "B", "quiz", "admin", order_index=1))
	items = run(svc.list_course_content(T, course["id"]))
	assert [i["title"] for i in items] == ["A", "B", "C"]


def test_scorm_content_valid_version():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	item = run(svc.add_content_item(T, course["id"], "SCORM Module", "scorm", "admin", scorm_version="scorm_2004_3rd"))
	assert item["scorm_version"] == "scorm_2004_3rd"


# ---------------------------------------------------------------------------
# enrolments
# ---------------------------------------------------------------------------

def test_enrol_learner():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	assert enrolment["status"] == "active"
	assert enrolment["learner_id"] == "learner_1"


def test_paid_enrolment_requires_payment_reference():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	with pytest.raises(ValueError):
		run(svc.enrol_learner(T, course["id"], "learner_1", "paid", "admin"))


def test_paid_enrolment_with_payment_reference():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "paid", "admin", payment_reference="PAY-001"))
	assert enrolment["payment_reference"] == "PAY-001"


def test_withdraw_enrolment():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	withdrawn = run(svc.withdraw_enrolment(T, enrolment["id"]))
	assert withdrawn["status"] == "withdrawn"


# ---------------------------------------------------------------------------
# assessments & grading
# ---------------------------------------------------------------------------

def test_create_assessment():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	assessment = run(svc.create_assessment(T, course["id"], "Quiz 1", "formative_quiz", "admin"))
	assert assessment["assessment_type"] == "formative_quiz"


def test_submit_and_grade():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	assessment = run(svc.create_assessment(T, course["id"], "Final Exam", "summative_exam", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	submission = run(svc.submit_assessment(T, assessment["id"], enrolment["id"], "learner_1", "learner_1"))
	assert submission["status"] == "submitted"
	graded = run(svc.grade_submission(T, submission["id"], 85.0, "teacher_1", "Good work"))
	assert graded["score"] == 85.0
	assert graded["status"] == "graded"


def test_grade_override_requires_approval():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	assessment = run(svc.create_assessment(T, course["id"], "Exam", "summative_exam", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	submission = run(svc.submit_assessment(T, assessment["id"], enrolment["id"], "learner_1", "admin"))
	run(svc.grade_submission(T, submission["id"], 70.0, "teacher_1"))
	with pytest.raises(ValueError, match="grade_override_requires_approval"):
		run(svc.grade_submission(T, submission["id"], 80.0, "teacher_1"))


def test_grade_override_with_approval():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	assessment = run(svc.create_assessment(T, course["id"], "Exam", "summative_exam", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	submission = run(svc.submit_assessment(T, assessment["id"], enrolment["id"], "learner_1", "admin"))
	run(svc.grade_submission(T, submission["id"], 70.0, "teacher_1"))
	updated = run(svc.grade_submission(T, submission["id"], 80.0, "teacher_1", override_approval="APPR-001"))
	assert updated["score"] == 80.0


# ---------------------------------------------------------------------------
# certificates
# ---------------------------------------------------------------------------

def test_issue_certificate():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	cert = run(svc.issue_certificate(T, enrolment["id"], "learner_1", course["id"], "completion", "issuer_1", "admin", completion_criteria_met=True))
	assert cert["certificate_type"] == "completion"
	assert cert["learner_id"] == "learner_1"


def test_certificate_requires_completion():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	with pytest.raises(ValueError, match="certificate_requires_completion"):
		run(svc.issue_certificate(T, enrolment["id"], "learner_1", course["id"], "completion", "issuer_1", "admin", completion_criteria_met=False))


# ---------------------------------------------------------------------------
# progress & analytics
# ---------------------------------------------------------------------------

def test_record_progress():
	svc = LmsService()
	course = run(svc.create_course(T, "Course", "C", "self_paced", "o1", "admin"))
	enrolment = run(svc.enrol_learner(T, course["id"], "learner_1", "open", "admin"))
	item = run(svc.add_content_item(T, course["id"], "Video", "video", "admin"))
	progress = run(svc.record_progress(T, enrolment["id"], "learner_1", course["id"], item["id"], "system", completion_percentage=50.0))
	assert progress["completion_percentage"] == 50.0


def test_learner_analytics_requires_consent():
	svc = LmsService()
	with pytest.raises(ValueError, match="analytics_export_requires_consent"):
		run(svc.learner_analytics(T, "learner_1", consent_recorded=False))


def test_dashboard_summary():
	svc = LmsService()
	run(svc.create_course(T, "Course A", "CA", "self_paced", "o1", "admin"))
	summary = run(svc.dashboard_summary(T))
	assert summary["courses"] >= 1
	assert summary["tenant_id"] == T


# ---------------------------------------------------------------------------
# learning paths
# ---------------------------------------------------------------------------

def test_create_learning_path():
	svc = LmsService()
	path = run(svc.create_learning_path(T, "Data Science Track", "owner_1", "admin"))
	assert path["title"] == "Data Science Track"
	paths = run(svc.list_learning_paths(T))
	assert len(paths) == 1


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

def test_register_agent():
	svc = LmsService()
	agent = run(svc.register_agent(T, "GradeBot", "codex", "assessment_grader", "admin"))
	assert agent["role"] == "assessment_grader"


def test_invalid_agent_runtime_rejected():
	svc = LmsService()
	with pytest.raises(AssertionError):
		run(svc.register_agent(T, "Bot", "invalid_runtime", "assessment_grader", "admin"))


def test_agent_action_privileged_without_approval_denied():
	svc = LmsService()
	result = run(svc.validate_agent_action(T, privileged_scope=True, human_approval_recorded=False))
	assert result["decision"] == "deny"


def test_agent_action_with_approval_allowed():
	svc = LmsService()
	result = run(svc.validate_agent_action(T, privileged_scope=True, human_approval_recorded=True))
	assert result["decision"] == "allow"
