"""Vault tokenization capability — Pydantic v2 models."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TokenizeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	pan: str = Field(..., description="Primary Account Number to tokenize")
	tenant_id: str
	actor_id: str = "system"


class TokenizeBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	pans: list[str]
	tenant_id: str
	actor_id: str = "system"


class DetokenizeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	token: str
	tenant_id: str
	actor_id: str
	actor_role: str = Field(default="pci_authorized")


class DetokenizeBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tokens: list[str]
	tenant_id: str
	actor_id: str
	actor_role: str = "pci_authorized"


class StoreSecretRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	key: str
	value: str
	tenant_id: str


class GetSecretRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	key: str
	tenant_id: str


class EncryptRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	plaintext: str
	tenant_id: str


class TokenRecord(BaseModel):
	model_config = ConfigDict(extra="ignore", validate_by_name=True)

	token: str
	last_four: str
	card_type: str
	bin: str
	tenant_id: str
	created_at: str
	masked_pan: str


class DetokenizeResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	token: str
	pan: str
	authorized: bool


class ComplianceStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	pci_dss_compliant: bool
	tokens_issued: int
	pan_never_stored: bool
	luhn_validated: bool
