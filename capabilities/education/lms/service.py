"""Async service layer for APG Learning Management System."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_CERTIFICATE_TYPES, SUPPORTED_COMPLETION_CRITERIA, SUPPORTED_CONTENT_TYPES,
		SUPPORTED_COURSE_STATUSES, SUPPORTED_COURSE_TYPES, SUPPORTED_ENROLMENT_TYPES,
		SUPPORTED_GRADING_SCHEMES, SUPPORTED_LEARNER_STATUSES, SUPPORTED_SCORM_VERSIONS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AssessmentCreate, AssessmentUpdate, CertificateCreate,
		ContentItemCreate, ContentItemUpdate, EnrolmentCreate, EnrolmentUpdate,
		LearnerProgressCreate, LearnerProgressUpdate, LearningPathCreate, LearningPathUpdate,
		LmsAgent, SubmissionCreate, SubmissionUpdate, CourseCreate, CourseUpdate,
		uuid7str,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES,
		SUPPORTED_CERTIFICATE_TYPES, SUPPORTED_COMPLETION_CRITERIA, SUPPORTED_CONTENT_TYPES,
		SUPPORTED_COURSE_STATUSES, SUPPORTED_COURSE_TYPES, SUPPORTED_ENROLMENT_TYPES,
		SUPPORTED_GRADING_SCHEMES, SUPPORTED_LEARNER_STATUSES, SUPPORTED_SCORM_VERSIONS,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AssessmentCreate, AssessmentUpdate, CertificateCreate,
		ContentItemCreate, ContentItemUpdate, EnrolmentCreate, EnrolmentUpdate,
		LearnerProgressCreate, LearnerProgressUpdate, LearningPathCreate, LearningPathUpdate,
		LmsAgent, SubmissionCreate, SubmissionUpdate, CourseCreate, CourseUpdate,
		uuid7str,
	)


def _present(v: str | None) -> bool:
	return bool(v and str(v).strip())


def _normalize(v: str) -> str:
	return v.strip().lower()


class LmsService:
	"""Tenant-scoped LMS runtime for APG-generated applications."""

	def __init__(self) -> None:
		self.courses: dict[tuple[str, str], CourseCreate] = {}
		self.content_items: dict[tuple[str, str], ContentItemCreate] = {}
		self.enrolments: dict[tuple[str, str], EnrolmentCreate] = {}
		self.assessments: dict[tuple[str, str], AssessmentCreate] = {}
		self.submissions: dict[tuple[str, str], SubmissionCreate] = {}
		self.certificates: dict[tuple[str, str], CertificateCreate] = {}
		self.progress: dict[tuple[str, str], LearnerProgressCreate] = {}
		self.learning_paths: dict[tuple[str, str], LearningPathCreate] = {}
		self.agents: dict[tuple[str, str], LmsAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	# -----------------------------------------------------------------------
	# introspection
	# -----------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for the given tenant."""
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate business rules against a context dict."""
		return evaluate_capability_rules(context)

	# -----------------------------------------------------------------------
	# courses
	# -----------------------------------------------------------------------

	async def create_course(
		self,
		tenant_id: str,
		title: str,
		code: str,
		course_type: str,
		owner_id: str,
		created_by: str,
		description: str = "",
		enrolment_type: str = "open",
		max_enrolments: int | None = None,
		duration_weeks: int | None = None,
		grading_scheme: str = "percentage",
		passing_score: float = 50.0,
		completion_criteria: list[str] | None = None,
		tags: list[str] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new course in draft status."""
		ct = _normalize(course_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_course",
			"course_type_supported": ct in SUPPORTED_COURSE_TYPES,
		})
		item = CourseCreate(
			tenant_id=tenant_id, title=title, code=code, course_type=ct,
			owner_id=owner_id, description=description, enrolment_type=enrolment_type,
			max_enrolments=max_enrolments, duration_weeks=duration_weeks,
			grading_scheme=grading_scheme, passing_score=passing_score,
			completion_criteria=completion_criteria or [], tags=tags or [],
			created_by=created_by,
		)
		self.courses[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "course_created", item.id)
		return item.model_dump()

	async def get_course(self, tenant_id: str, course_id: str) -> dict[str, Any] | None:
		"""Retrieve a course by ID."""
		item = self.courses.get(self._key(tenant_id, course_id))
		return item.model_dump() if item else None

	async def list_courses(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all courses for a tenant."""
		return [c.model_dump() for (t, _), c in self.courses.items() if t == tenant_id]

	async def update_course(
		self, tenant_id: str, course_id: str, updates: dict[str, Any], policy_attached: bool = True
	) -> dict[str, Any]:
		"""Apply a partial update to a course."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = self._require_course(tenant_id, course_id)
		upd = CourseUpdate(**updates)
		merged = item.model_copy(update=upd.model_dump(exclude_none=True))
		self.courses[self._key(tenant_id, course_id)] = merged
		self._audit(tenant_id, "course_updated", course_id)
		return merged.model_dump()

	async def publish_course(
		self, tenant_id: str, course_id: str, review_approved: bool, policy_attached: bool = True
	) -> dict[str, Any]:
		"""Transition a course from review to published."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "publish_course",
			"review_approved": review_approved,
		})
		return await self.update_course(tenant_id, course_id, {"status": "published"})

	async def archive_course(self, tenant_id: str, course_id: str) -> dict[str, Any]:
		"""Archive a published course."""
		return await self.update_course(tenant_id, course_id, {"status": "archived"})

	# -----------------------------------------------------------------------
	# content items
	# -----------------------------------------------------------------------

	async def add_content_item(
		self,
		tenant_id: str,
		course_id: str,
		title: str,
		content_type: str,
		created_by: str,
		url: str | None = None,
		scorm_version: str | None = None,
		duration_minutes: int | None = None,
		order_index: int = 0,
		is_required: bool = True,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Add a content item to a course."""
		ct = _normalize(content_type)
		sv = _normalize(scorm_version) if scorm_version else None
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "add_content",
			"content_type_supported": ct in SUPPORTED_CONTENT_TYPES,
		})
		if ct == "scorm" and sv:
			self._enforce({"operation": "add_content", "content_type": "scorm", "scorm_version_supported": sv in SUPPORTED_SCORM_VERSIONS})
		self._require_course(tenant_id, course_id)
		item = ContentItemCreate(
			tenant_id=tenant_id, course_id=course_id, title=title, content_type=ct,
			url=url, scorm_version=sv, duration_minutes=duration_minutes,
			order_index=order_index, is_required=is_required, metadata=metadata or {},
			created_by=created_by,
		)
		self.content_items[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "content_item_added", item.id)
		return item.model_dump()

	async def list_course_content(self, tenant_id: str, course_id: str) -> list[dict[str, Any]]:
		"""Return all content items for a course, ordered by order_index."""
		items = [
			c for (t, _), c in self.content_items.items()
			if t == tenant_id and c.course_id == course_id
		]
		return [c.model_dump() for c in sorted(items, key=lambda x: x.order_index)]

	# -----------------------------------------------------------------------
	# enrolments
	# -----------------------------------------------------------------------

	async def enrol_learner(
		self,
		tenant_id: str,
		course_id: str,
		learner_id: str,
		enrolment_type: str,
		created_by: str,
		payment_reference: str | None = None,
		voucher_code: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Enrol a learner in a course."""
		et = _normalize(enrolment_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "enrol_learner",
			"enrolment_type_supported": et in SUPPORTED_ENROLMENT_TYPES,
			"course_tenant_matches_learner_tenant": True,
		})
		if et == "paid":
			self._enforce({"operation": "enrol_learner", "enrolment_type": "paid", "payment_reference_present": _present(payment_reference)})
		course = self._require_course(tenant_id, course_id)
		if course.max_enrolments is not None:
			count = sum(1 for e in self.enrolments.values() if e.course_id == course_id and e.tenant_id == tenant_id and e.status == "active")
			assert count < course.max_enrolments, "course enrolment capacity reached"
		item = EnrolmentCreate(
			tenant_id=tenant_id, course_id=course_id, learner_id=learner_id,
			enrolment_type=et, payment_reference=payment_reference, voucher_code=voucher_code,
			status="active", created_by=created_by,
		)
		self.enrolments[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "enrolment_recorded", item.id)
		return item.model_dump()

	async def list_enrolments(self, tenant_id: str, course_id: str | None = None) -> list[dict[str, Any]]:
		"""List enrolments, optionally filtered by course."""
		return [
			e.model_dump() for (t, _), e in self.enrolments.items()
			if t == tenant_id and (course_id is None or e.course_id == course_id)
		]

	async def withdraw_enrolment(self, tenant_id: str, enrolment_id: str) -> dict[str, Any]:
		"""Withdraw an active enrolment."""
		item = self._require_enrolment(tenant_id, enrolment_id)
		merged = item.model_copy(update={"status": "withdrawn", "updated_at": datetime.utcnow()})
		self.enrolments[self._key(tenant_id, enrolment_id)] = merged
		self._audit(tenant_id, "enrolment_withdrawn", enrolment_id)
		return merged.model_dump()

	# -----------------------------------------------------------------------
	# assessments
	# -----------------------------------------------------------------------

	async def create_assessment(
		self,
		tenant_id: str,
		course_id: str,
		title: str,
		assessment_type: str,
		created_by: str,
		max_score: float = 100.0,
		passing_score: float = 50.0,
		weight_percent: float = 100.0,
		time_limit_minutes: int | None = None,
		attempts_allowed: int = 1,
		instructions: str = "",
		due_at: datetime | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create an assessment for a course."""
		at = _normalize(assessment_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_assessment",
			"assessment_type_supported": at in SUPPORTED_ASSESSMENT_TYPES,
		})
		self._require_course(tenant_id, course_id)
		item = AssessmentCreate(
			tenant_id=tenant_id, course_id=course_id, title=title, assessment_type=at,
			max_score=max_score, passing_score=passing_score, weight_percent=weight_percent,
			time_limit_minutes=time_limit_minutes, attempts_allowed=attempts_allowed,
			instructions=instructions, due_at=due_at, created_by=created_by,
		)
		self.assessments[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "assessment_created", item.id)
		return item.model_dump()

	# -----------------------------------------------------------------------
	# submissions & grading
	# -----------------------------------------------------------------------

	async def submit_assessment(
		self,
		tenant_id: str,
		assessment_id: str,
		enrolment_id: str,
		learner_id: str,
		created_by: str,
		attempt_number: int = 1,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a learner assessment submission."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		self._require_assessment(tenant_id, assessment_id)
		item = SubmissionCreate(
			tenant_id=tenant_id, assessment_id=assessment_id,
			enrolment_id=enrolment_id, learner_id=learner_id,
			attempt_number=attempt_number, status="submitted", created_by=created_by,
		)
		self.submissions[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "assessment_submitted", item.id)
		return item.model_dump()

	async def grade_submission(
		self,
		tenant_id: str,
		submission_id: str,
		score: float,
		graded_by: str,
		feedback: str = "",
		override_approval: str | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a grade for a submission. Override requires approval."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = self._require_submission(tenant_id, submission_id)
		if item.score is not None:
			self._enforce({
				"operation": "override_grade",
				"approval_reference_present": _present(override_approval),
			})
		merged = item.model_copy(update={
			"score": score, "feedback": feedback, "graded_by": graded_by,
			"graded_at": datetime.utcnow(), "status": "graded",
			"override_approval": override_approval, "updated_at": datetime.utcnow(),
		})
		self.submissions[self._key(tenant_id, submission_id)] = merged
		self._audit(tenant_id, "grade_recorded", submission_id)
		return merged.model_dump()

	async def list_submissions(self, tenant_id: str, assessment_id: str | None = None) -> list[dict[str, Any]]:
		"""List submissions, optionally filtered by assessment."""
		return [
			s.model_dump() for (t, _), s in self.submissions.items()
			if t == tenant_id and (assessment_id is None or s.assessment_id == assessment_id)
		]

	# -----------------------------------------------------------------------
	# certificates
	# -----------------------------------------------------------------------

	async def issue_certificate(
		self,
		tenant_id: str,
		enrolment_id: str,
		learner_id: str,
		course_id: str,
		certificate_type: str,
		issuer_id: str,
		created_by: str,
		completion_criteria_met: bool = True,
		expires_at: datetime | None = None,
		metadata: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Issue a certificate upon completion."""
		cert_type = _normalize(certificate_type)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "issue_certificate",
			"completion_criteria_met": completion_criteria_met,
			"certificate_type_supported": cert_type in SUPPORTED_CERTIFICATE_TYPES,
		})
		item = CertificateCreate(
			tenant_id=tenant_id, enrolment_id=enrolment_id, learner_id=learner_id,
			course_id=course_id, certificate_type=cert_type, issuer_id=issuer_id,
			expires_at=expires_at, metadata=metadata or {}, created_by=created_by,
		)
		self.certificates[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "certificate_issued", item.id)
		return item.model_dump()

	async def list_certificates(self, tenant_id: str, learner_id: str | None = None) -> list[dict[str, Any]]:
		"""List certificates, optionally filtered by learner."""
		return [
			c.model_dump() for (t, _), c in self.certificates.items()
			if t == tenant_id and (learner_id is None or c.learner_id == learner_id)
		]

	# -----------------------------------------------------------------------
	# learner progress (xAPI / SCORM)
	# -----------------------------------------------------------------------

	async def record_progress(
		self,
		tenant_id: str,
		enrolment_id: str,
		learner_id: str,
		course_id: str,
		content_item_id: str,
		created_by: str,
		completion_percentage: float = 0.0,
		time_spent_minutes: int = 0,
		xapi_statement: dict[str, Any] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record learner progress on a content item."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = LearnerProgressCreate(
			tenant_id=tenant_id, enrolment_id=enrolment_id, learner_id=learner_id,
			course_id=course_id, content_item_id=content_item_id,
			completion_percentage=completion_percentage,
			time_spent_minutes=time_spent_minutes, xapi_statement=xapi_statement,
			created_by=created_by,
		)
		self.progress[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "learner_progress_updated", item.id)
		return item.model_dump()

	async def learner_course_progress(
		self, tenant_id: str, learner_id: str, course_id: str
	) -> dict[str, Any]:
		"""Aggregate completion percentage across all content items for a learner+course."""
		items = [
			p for (t, _), p in self.progress.items()
			if t == tenant_id and p.learner_id == learner_id and p.course_id == course_id
		]
		if not items:
			return {"learner_id": learner_id, "course_id": course_id, "completion_percentage": 0.0, "total_time_minutes": 0}
		avg = sum(p.completion_percentage for p in items) / len(items)
		total_time = sum(p.time_spent_minutes for p in items)
		return {"learner_id": learner_id, "course_id": course_id, "completion_percentage": round(avg, 2), "total_time_minutes": total_time}

	# -----------------------------------------------------------------------
	# learning paths
	# -----------------------------------------------------------------------

	async def create_learning_path(
		self,
		tenant_id: str,
		title: str,
		owner_id: str,
		created_by: str,
		description: str = "",
		course_ids: list[str] | None = None,
		required_course_ids: list[str] | None = None,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a learning path."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
		})
		item = LearningPathCreate(
			tenant_id=tenant_id, title=title, owner_id=owner_id, description=description,
			course_ids=course_ids or [], required_course_ids=required_course_ids or [],
			created_by=created_by,
		)
		self.learning_paths[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "learning_path_created", item.id)
		return item.model_dump()

	async def list_learning_paths(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all learning paths for a tenant."""
		return [lp.model_dump() for (t, _), lp in self.learning_paths.items() if t == tenant_id]

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
		scope: str = "lms operations",
	) -> dict[str, Any]:
		"""Register an AI agent for LMS automation."""
		rt = _normalize(runtime)
		rl = _normalize(role)
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		assert rt in SUPPORTED_AGENT_RUNTIMES, f"unsupported runtime: {rt}"
		assert rl in SUPPORTED_AGENT_ROLES, f"unsupported role: {rl}"
		item = LmsAgent(
			tenant_id=tenant_id, name=name, runtime=rt, role=rl,
			scope=scope, created_by=created_by,
		)
		self.agents[self._key(tenant_id, item.id)] = item
		self._audit(tenant_id, "lms_agent_registered", item.id)
		return item.model_dump()

	async def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool = False,
		human_approval_recorded: bool = False,
	) -> dict[str, Any]:
		"""Validate whether an agent action is permitted."""
		return evaluate_capability_rules({
			"tenant_context_present": _present(tenant_id),
			"operation": "agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})

	# -----------------------------------------------------------------------
	# analytics / dashboard
	# -----------------------------------------------------------------------

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return a dashboard summary for the tenant."""
		course_count = sum(1 for (t, _) in self.courses if t == tenant_id)
		enrolment_count = sum(1 for (t, _) in self.enrolments if t == tenant_id)
		cert_count = sum(1 for (t, _) in self.certificates if t == tenant_id)
		submission_count = sum(1 for (t, _) in self.submissions if t == tenant_id)
		return {
			"tenant_id": tenant_id,
			"courses": course_count,
			"enrolments": enrolment_count,
			"certificates_issued": cert_count,
			"submissions": submission_count,
		}

	async def learner_analytics(
		self, tenant_id: str, learner_id: str, consent_recorded: bool = True
	) -> dict[str, Any]:
		"""Return analytics for a learner. Requires consent."""
		self._enforce({
			"operation": "export_learner_analytics",
			"consent_recorded": consent_recorded,
		})
		enrols = [e for (t, _), e in self.enrolments.items() if t == tenant_id and e.learner_id == learner_id]
		certs = [c for (t, _), c in self.certificates.items() if t == tenant_id and c.learner_id == learner_id]
		return {
			"learner_id": learner_id,
			"tenant_id": tenant_id,
			"total_enrolments": len(enrols),
			"active_enrolments": sum(1 for e in enrols if e.status == "active"),
			"completed_enrolments": sum(1 for e in enrols if e.status == "completed"),
			"certificates_earned": len(certs),
		}

	# ── 13 new methods ──────────────────────────────────────────────────────

	async def course_enroll(
		self, tenant_id: str, learner_id: str, course_id: str
	) -> dict[str, Any]:
		"""Enrol a learner in a course (creates a new enrolment record)."""
		self._require_course(tenant_id, course_id)
		enrolment_id = f"enrol-{learner_id[:8]}-{course_id[:8]}-{len(self.enrolments)+1}"
		enrolment = EnrolmentCreate(
			enrolment_id=enrolment_id,
			learner_id=learner_id,
			course_id=course_id,
			tenant_id=tenant_id,
			status="active",
		)
		self.enrolments[self._key(tenant_id, enrolment_id)] = enrolment
		self._audit(tenant_id, "learner_enrolled", enrolment_id)
		return enrolment.model_dump()

	async def course_unenroll(
		self, tenant_id: str, learner_id: str, course_id: str, reason: str = "learner_request"
	) -> dict[str, Any]:
		"""Unenrol a learner from a course."""
		for key, enr in self.enrolments.items():
			if key[0] == tenant_id and enr.learner_id == learner_id and enr.course_id == course_id:
				updated = enr.model_copy(update={"status": "unenrolled"})
				self.enrolments[key] = updated
				self._audit(tenant_id, "learner_unenrolled", enr.enrolment_id)
				return {**updated.model_dump(), "unenroll_reason": reason}
		raise KeyError(f"enrolment not found for learner {learner_id} in course {course_id}")

	async def track_progress(
		self, tenant_id: str, learner_id: str, course_id: str
	) -> dict[str, Any]:
		"""Return progress summary for a learner in a course."""
		enrolments = [
			e for (t, _), e in self.enrolments.items()
			if t == tenant_id and e.learner_id == learner_id and e.course_id == course_id
		]
		submissions = [
			s for (t, _), s in self.submissions.items()
			if t == tenant_id and s.learner_id == learner_id
		]
		course = self._require_course(tenant_id, course_id)
		assessments = [a for (t, _), a in self.assessments.items() if t == tenant_id and a.course_id == course_id]
		graded = sum(1 for s in submissions if s.grade is not None)
		return {
			"learner_id": learner_id,
			"course_id": course_id,
			"tenant_id": tenant_id,
			"enrolment_status": enrolments[0].status if enrolments else "not_enrolled",
			"total_assessments": len(assessments),
			"submissions": len(submissions),
			"graded_submissions": graded,
			"completion_pct": round(graded / max(len(assessments), 1) * 100, 1),
		}

	async def submit_assessment(
		self, tenant_id: str, learner_id: str, assessment_id: str, responses: list[dict[str, Any]]
	) -> dict[str, Any]:
		"""Submit learner responses for an assessment."""
		assessment = self._require_assessment(tenant_id, assessment_id)
		sub_id = f"sub-{learner_id[:8]}-{assessment_id[:8]}-{len(self.submissions)+1}"
		submission = SubmissionCreate(
			submission_id=sub_id,
			learner_id=learner_id,
			assessment_id=assessment_id,
			tenant_id=tenant_id,
			responses=responses,
			status="submitted",
		)
		self.submissions[self._key(tenant_id, sub_id)] = submission
		self._audit(tenant_id, "assessment_submitted", sub_id)
		return submission.model_dump()

	async def grade_assessment(
		self, tenant_id: str, submission_id: str, auto: bool = True
	) -> dict[str, Any]:
		"""Grade a submission. Auto-grading assigns score based on response count proxy."""
		submission = self._require_submission(tenant_id, submission_id)
		grade = round(min(len(submission.responses) * 10, 100), 1) if auto else 0.0
		updated = submission.model_copy(update={"grade": grade, "status": "graded"})
		self.submissions[self._key(tenant_id, submission_id)] = updated
		self._audit(tenant_id, "assessment_graded", submission_id)
		return updated.model_dump()

	async def issue_certificate(
		self, tenant_id: str, learner_id: str, course_id: str
	) -> dict[str, Any]:
		"""Issue a completion certificate for a learner."""
		self._require_course(tenant_id, course_id)
		cert_id = f"cert-{learner_id[:8]}-{course_id[:8]}-{len(self.certificates)+1}"
		cert = CertificateCreate(
			certificate_id=cert_id,
			learner_id=learner_id,
			course_id=course_id,
			tenant_id=tenant_id,
			status="issued",
			issued_at=datetime.utcnow().isoformat(),
		)
		self.certificates[self._key(tenant_id, cert_id)] = cert
		self._audit(tenant_id, "certificate_issued", cert_id)
		return cert.model_dump()

	async def revoke_certificate(
		self, tenant_id: str, certificate_id: str, reason: str
	) -> dict[str, Any]:
		"""Revoke a previously issued certificate."""
		cert = self.certificates.get(self._key(tenant_id, certificate_id))
		assert cert is not None, f"certificate not found: {certificate_id}"
		updated = cert.model_copy(update={"status": "revoked"})
		self.certificates[self._key(tenant_id, certificate_id)] = updated
		self._audit(tenant_id, "certificate_revoked", certificate_id)
		return {**updated.model_dump(), "revoke_reason": reason}

	async def learning_path_create(
		self, tenant_id: str, name: str, courses: list[str], prerequisites: list[str], created_by: str = "admin"
	) -> dict[str, Any]:
		"""Create a structured learning path."""
		path_id = f"lpath-{len(self.courses)+1}"
		path: dict[str, Any] = {
			"path_id": path_id,
			"tenant_id": tenant_id,
			"name": name,
			"courses": courses,
			"prerequisites": prerequisites,
			"created_by": created_by,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "learning_path_created", path_id)
		return path

	async def adaptive_recommend(
		self, tenant_id: str, learner_id: str
	) -> list[str]:
		"""Return course IDs recommended for a learner based on completion gaps."""
		enrolled_courses = {
			e.course_id for (t, _), e in self.enrolments.items()
			if t == tenant_id and e.learner_id == learner_id
		}
		all_courses = [c for (t, _), c in self.courses.items() if t == tenant_id]
		unenrolled = [c.course_id for c in all_courses if c.course_id not in enrolled_courses]
		return unenrolled[:5]

	async def social_post(
		self, tenant_id: str, learner_id: str, content: str, course_id: str
	) -> dict[str, Any]:
		"""Post a social/community update in a course context."""
		post_id = f"post-{learner_id[:6]}-{len(self.audit_events)+1}"
		self._audit(tenant_id, "social_post_created", post_id)
		return {
			"post_id": post_id,
			"tenant_id": tenant_id,
			"learner_id": learner_id,
			"course_id": course_id,
			"content": content,
			"posted_at": datetime.utcnow().isoformat(),
		}

	async def discussion_thread(
		self, tenant_id: str, course_id: str, topic: str, author_id: str
	) -> dict[str, Any]:
		"""Create a discussion thread in a course."""
		thread_id = f"thread-{course_id[:6]}-{len(self.audit_events)+1}"
		self._audit(tenant_id, "discussion_thread_created", thread_id)
		return {
			"thread_id": thread_id,
			"tenant_id": tenant_id,
			"course_id": course_id,
			"topic": topic,
			"author_id": author_id,
			"created_at": datetime.utcnow().isoformat(),
		}

	async def gamification_award(
		self, tenant_id: str, learner_id: str, badge_id: str
	) -> dict[str, Any]:
		"""Award a gamification badge to a learner."""
		award_id = f"award-{learner_id[:6]}-{badge_id[:6]}-{len(self.audit_events)+1}"
		self._audit(tenant_id, "badge_awarded", award_id)
		return {
			"award_id": award_id,
			"tenant_id": tenant_id,
			"learner_id": learner_id,
			"badge_id": badge_id,
			"awarded_at": datetime.utcnow().isoformat(),
		}

	async def lms_analytics(
		self, tenant_id: str, period: str
	) -> dict[str, Any]:
		"""Return LMS analytics for a period."""
		enrolments = [(t, e) for (t, _), e in self.enrolments.items() if t == tenant_id]
		certs = [(t, c) for (t, _), c in self.certificates.items() if t == tenant_id]
		subs = [(t, s) for (t, _), s in self.submissions.items() if t == tenant_id]
		completed = sum(1 for _, e in enrolments if e.status == "completed")
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_enrolments": len(enrolments),
			"completed_enrolments": completed,
			"completion_rate_pct": round(completed / max(len(enrolments), 1) * 100, 1),
			"certificates_issued": len(certs),
			"total_submissions": len(subs),
			"courses": len([(t, c) for (t, c) in self.courses if t == tenant_id]),
		}

	# -----------------------------------------------------------------------
	# private helpers
	# -----------------------------------------------------------------------

	def _log_audit_entry(self, tenant_id: str, event: str, entity_id: str) -> None:
		"""Log an audit event to the in-memory audit trail."""
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event": event,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
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
			raise ValueError(f"[LmsService] rule={result['matched_rule']} reason={result['reason']} action={result.get('required_action')}")

	def _require_course(self, tenant_id: str, course_id: str) -> CourseCreate:
		item = self.courses.get(self._key(tenant_id, course_id))
		assert item is not None, f"course not found: {self._log_pretty_key(tenant_id, course_id)}"
		return item

	def _require_enrolment(self, tenant_id: str, enrolment_id: str) -> EnrolmentCreate:
		item = self.enrolments.get(self._key(tenant_id, enrolment_id))
		assert item is not None, f"enrolment not found: {self._log_pretty_key(tenant_id, enrolment_id)}"
		return item

	def _require_assessment(self, tenant_id: str, assessment_id: str) -> AssessmentCreate:
		item = self.assessments.get(self._key(tenant_id, assessment_id))
		assert item is not None, f"assessment not found: {self._log_pretty_key(tenant_id, assessment_id)}"
		return item

	def _require_submission(self, tenant_id: str, submission_id: str) -> SubmissionCreate:
		item = self.submissions.get(self._key(tenant_id, submission_id))
		assert item is not None, f"submission not found: {self._log_pretty_key(tenant_id, submission_id)}"
		return item
