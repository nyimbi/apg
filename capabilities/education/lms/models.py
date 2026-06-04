"""Pydantic v2 models for APG Learning Management System."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, AfterValidator


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "field must be non-empty"
	return v.strip()


def _valid_score(v: float) -> float:
	assert 0.0 <= v <= 100.0, f"score must be 0-100, got {v}"
	return v


def _non_negative(v: float) -> float:
	assert v >= 0.0, f"value must be non-negative, got {v}"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
ValidScore = Annotated[float, AfterValidator(_valid_score)]
NonNegativeFloat = Annotated[float, AfterValidator(_non_negative)]

_BASE_CONFIG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------------

class CourseCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	title: NonEmptyStr
	code: NonEmptyStr
	description: str = ""
	course_type: NonEmptyStr
	owner_id: NonEmptyStr
	status: str = "draft"
	enrolment_type: str = "open"
	max_enrolments: int | None = None
	duration_weeks: int | None = None
	grading_scheme: str = "percentage"
	passing_score: ValidScore = 50.0
	completion_criteria: list[str] = Field(default_factory=list)
	tags: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class CourseUpdate(BaseModel):
	model_config = _BASE_CONFIG

	title: str | None = None
	description: str | None = None
	status: str | None = None
	enrolment_type: str | None = None
	max_enrolments: int | None = None
	duration_weeks: int | None = None
	grading_scheme: str | None = None
	passing_score: ValidScore | None = None
	completion_criteria: list[str] | None = None
	tags: list[str] | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CourseResponse(CourseCreate):
	pass


# ---------------------------------------------------------------------------
# ContentItem
# ---------------------------------------------------------------------------

class ContentItemCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	course_id: NonEmptyStr
	title: NonEmptyStr
	content_type: NonEmptyStr
	order_index: int = 0
	duration_minutes: int | None = None
	url: str | None = None
	scorm_version: str | None = None
	is_required: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)
	compliance_checked: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class ContentItemUpdate(BaseModel):
	model_config = _BASE_CONFIG

	title: str | None = None
	order_index: int | None = None
	duration_minutes: int | None = None
	url: str | None = None
	is_required: bool | None = None
	metadata: dict[str, Any] | None = None
	compliance_checked: bool | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ContentItemResponse(ContentItemCreate):
	pass


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------

class EnrolmentCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	course_id: NonEmptyStr
	learner_id: NonEmptyStr
	enrolment_type: NonEmptyStr
	status: str = "pending"
	payment_reference: str | None = None
	voucher_code: str | None = None
	enrolled_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None
	completion_percentage: ValidScore = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class EnrolmentUpdate(BaseModel):
	model_config = _BASE_CONFIG

	status: str | None = None
	completion_percentage: ValidScore | None = None
	expires_at: datetime | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class EnrolmentResponse(EnrolmentCreate):
	pass


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

class AssessmentCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	course_id: NonEmptyStr
	title: NonEmptyStr
	assessment_type: NonEmptyStr
	max_score: NonNegativeFloat = 100.0
	passing_score: ValidScore = 50.0
	weight_percent: ValidScore = 100.0
	time_limit_minutes: int | None = None
	attempts_allowed: int = 1
	instructions: str = ""
	due_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class AssessmentUpdate(BaseModel):
	model_config = _BASE_CONFIG

	title: str | None = None
	max_score: NonNegativeFloat | None = None
	passing_score: ValidScore | None = None
	weight_percent: ValidScore | None = None
	time_limit_minutes: int | None = None
	attempts_allowed: int | None = None
	instructions: str | None = None
	due_at: datetime | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AssessmentResponse(AssessmentCreate):
	pass


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

class SubmissionCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	assessment_id: NonEmptyStr
	enrolment_id: NonEmptyStr
	learner_id: NonEmptyStr
	status: str = "submitted"
	attempt_number: int = 1
	submitted_at: datetime = Field(default_factory=datetime.utcnow)
	score: ValidScore | None = None
	feedback: str = ""
	graded_by: str | None = None
	graded_at: datetime | None = None
	override_approval: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class SubmissionUpdate(BaseModel):
	model_config = _BASE_CONFIG

	status: str | None = None
	score: ValidScore | None = None
	feedback: str | None = None
	graded_by: str | None = None
	graded_at: datetime | None = None
	override_approval: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class SubmissionResponse(SubmissionCreate):
	pass


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

class CertificateCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	enrolment_id: NonEmptyStr
	learner_id: NonEmptyStr
	course_id: NonEmptyStr
	certificate_type: NonEmptyStr
	issued_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime | None = None
	issuer_id: NonEmptyStr
	verification_code: str = Field(default_factory=uuid7str)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class CertificateResponse(CertificateCreate):
	pass


# ---------------------------------------------------------------------------
# LearnerProgress
# ---------------------------------------------------------------------------

class LearnerProgressCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	enrolment_id: NonEmptyStr
	learner_id: NonEmptyStr
	course_id: NonEmptyStr
	content_item_id: NonEmptyStr
	completion_percentage: ValidScore = 0.0
	time_spent_minutes: int = 0
	last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
	xapi_statement: dict[str, Any] | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class LearnerProgressUpdate(BaseModel):
	model_config = _BASE_CONFIG

	completion_percentage: ValidScore | None = None
	time_spent_minutes: int | None = None
	last_accessed_at: datetime | None = None
	xapi_statement: dict[str, Any] | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class LearnerProgressResponse(LearnerProgressCreate):
	pass


# ---------------------------------------------------------------------------
# LearningPath
# ---------------------------------------------------------------------------

class LearningPathCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	title: NonEmptyStr
	description: str = ""
	owner_id: NonEmptyStr
	course_ids: list[str] = Field(default_factory=list)
	required_course_ids: list[str] = Field(default_factory=list)
	is_published: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class LearningPathUpdate(BaseModel):
	model_config = _BASE_CONFIG

	title: str | None = None
	description: str | None = None
	course_ids: list[str] | None = None
	required_course_ids: list[str] | None = None
	is_published: bool | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class LearningPathResponse(LearningPathCreate):
	pass


# ---------------------------------------------------------------------------
# LmsAgent
# ---------------------------------------------------------------------------

class LmsAgent(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	runtime: NonEmptyStr
	role: NonEmptyStr
	scope: str = "lms operations"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
