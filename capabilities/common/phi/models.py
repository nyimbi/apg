"""PHI classifier capability — Pydantic v2 models."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	field_name: str
	value: str


class ClassifyBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	fields: list[dict[str, str]] = Field(
		default_factory=list,
		description="List of {field_name, value} dicts",
	)


class RedactRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	record: dict[str, object]


class RedactBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	records: list[dict[str, object]]


class ScanDocumentRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	text: str


class LogPhiAccessRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	accessor_id: str
	record_id: str
	purpose: str


class AddIdentifierRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	pattern: str = Field(..., description="Python regex pattern")


class TestIdentifierRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	pattern: str
	test_value: str


class ClassifyResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	field_name: str
	is_phi: bool
	identifier_type: str | None = None
	confidence: float = 0.0
	regulation: str = "HIPAA"


class RedactResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	redacted_record: dict[str, object]
	phi_fields_found: list[str]
	phi_count: int
	total_fields: int


class ScanResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	phi_fields: list[dict[str, object]]
	phi_count: int
	total_fields: int
	phi_density: float


class ComplianceStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	hipaa_compliant: bool
	identifiers_monitored: int
