"""Pydantic v2 models for APG ECM / Records Management.

All model names use the 'Ec' prefix per convention.
All IDs are UUID7 strings for temporal ordering.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover — graceful fallback during bootstrapping
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _non_empty_str(v: str) -> str:
	assert v and v.strip(), "value must be a non-empty string"
	return v.strip()


def _positive_int(v: int) -> int:
	assert v > 0, "value must be a positive integer"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty_str)]
PositiveInt = Annotated[int, AfterValidator(_positive_int)]


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# EcDocument
# ---------------------------------------------------------------------------

class EcDocument(BaseModel):
	"""A managed document under ECM lifecycle control.

	Tracks content via SHA-256 hash (or equivalent) so the service layer
	remains storage-agnostic — the actual blob lives in an object store.
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str, description="UUID7 document identifier")
	tenant_id: NonEmptyStr
	title: NonEmptyStr
	document_type: NonEmptyStr
	content_hash: NonEmptyStr = Field(description="SHA-256 or equivalent hex digest of the canonical blob")
	version: PositiveInt = Field(default=1, description="Current major version number")
	status: NonEmptyStr = Field(default="draft")
	retention_category: NonEmptyStr
	# optional enrichment — populated after classification
	sensitivity: str | None = None
	regulatory_framework: str | None = None
	# lifecycle timestamps
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)
	# retention-derived dates — computed and stored when a policy is applied
	retention_expires_at: str | None = None
	disposal_due_date: str | None = None
	# links
	retention_policy_id: str | None = None
	current_workflow_id: str | None = None
	# free-form metadata (tags, author, department, etc.)
	metadata: dict[str, Any] = Field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcDocumentVersion
# ---------------------------------------------------------------------------

class EcDocumentVersion(BaseModel):
	"""An immutable point-in-time snapshot of a document.

	Versions are append-only; deletion is denied at the contract layer.
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	document_id: NonEmptyStr
	version_number: PositiveInt
	author: NonEmptyStr
	change_summary: NonEmptyStr
	content_hash: NonEmptyStr
	created_at: str = Field(default_factory=_now_iso)
	metadata: dict[str, Any] = Field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcRetentionPolicy
# ---------------------------------------------------------------------------

class EcRetentionPolicy(BaseModel):
	"""Defines how long documents in a category are kept and how they are disposed.

	trigger values: creation | last_access | last_modified | event
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	category: NonEmptyStr = Field(description="Retention category this policy governs")
	retention_years: PositiveInt
	trigger: NonEmptyStr = Field(description="Clock event that starts the retention countdown")
	disposal_method: NonEmptyStr
	# optional: description and the regulatory framework driving this policy
	description: str = ""
	regulatory_basis: str = ""
	active: bool = True
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcRecordClassification
# ---------------------------------------------------------------------------

class EcRecordClassification(BaseModel):
	"""Sensitivity and regulatory classification of a specific document."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	document_id: NonEmptyStr
	category: NonEmptyStr = Field(description="High-level category label (e.g. 'patient_data', 'financial')")
	sensitivity: NonEmptyStr = Field(description="public | internal | confidential | restricted | secret")
	regulatory_framework: NonEmptyStr = Field(description="Applicable regulatory regime (e.g. 'hipaa', 'gdpr')")
	classified_by: str = ""
	classified_at: str = Field(default_factory=_now_iso)
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcWorkflowStep  (embedded inside EcWorkflowInstance)
# ---------------------------------------------------------------------------

class EcWorkflowStep(BaseModel):
	"""A single step within a content workflow."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	step_number: PositiveInt
	approver_id: NonEmptyStr
	status: str = "pending"  # pending | approved | rejected | returned_for_revision | escalated | skipped
	decision: str | None = None
	comments: str = ""
	decided_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcWorkflowInstance
# ---------------------------------------------------------------------------

class EcWorkflowInstance(BaseModel):
	"""A running content approval/review workflow instance for a document."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	document_id: NonEmptyStr
	workflow_type: NonEmptyStr
	steps: list[EcWorkflowStep] = Field(default_factory=list)
	current_step: int = 1  # 1-based index into steps
	status: str = "in_progress"  # in_progress | completed | rejected | cancelled
	initiated_by: str = ""
	started_at: str = Field(default_factory=_now_iso)
	completed_at: str | None = None
	outcome: str | None = None  # final decision after all steps

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


# ---------------------------------------------------------------------------
# EcDisposalRecord
# ---------------------------------------------------------------------------

class EcDisposalRecord(BaseModel):
	"""Immutable audit record of a document disposal action.

	Created when a document transitions to 'disposed' status.
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	document_id: NonEmptyStr
	document_title: str = ""
	method: NonEmptyStr
	authorized_by: NonEmptyStr
	date: str = Field(default_factory=_now_iso)
	retention_policy_id: str | None = None
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()
