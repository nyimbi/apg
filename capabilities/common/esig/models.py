"""Electronic signature capability — Pydantic v2 models."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SignRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	document_id: str
	signer_id: str
	signer_display_name: str
	meaning: str = Field(..., description="Statement of what the signature certifies (21 CFR Part 11 component 1)")
	document_content: str = Field(default="", description="Document text or JSON to hash-bind")


class SignBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	signatures: list[SignRequest]


class VerifyRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	signature_id: str


class RevokeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	signature_id: str
	reason: str


class CreateSignatureRequestModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	document_id: str
	required_signers: list[str] = Field(default_factory=list)
	meaning: str
	deadline: str | None = None


class SignatureRecord(BaseModel):
	model_config = ConfigDict(extra="ignore", validate_by_name=True)

	signature_id: str
	document_id: str
	signer_id: str
	signer_display_name: str
	meaning: str
	timestamp: str
	document_hash: str
	signature_hash: str
	tenant_id: str
	is_valid: bool


class VerifyResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	signature_id: str
	valid: bool
	signer_id: str | None = None
	verified_at: str | None = None
	error: str | None = None


class ComplianceReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	cfr_21_part_11_compliant: bool
	signatures_reviewed: int
	invalid_signatures: int
	tenant_id: str
