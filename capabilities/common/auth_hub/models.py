"""Auth Hub — Pydantic v2 request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SignInRequest(BaseModel):
	model_config = ConfigDict(extra="allow", validate_by_name=True)

	username: str | None = None
	email: str | None = None
	password: str | None = None
	token: str | None = None  # pre-existing session token


class SignInResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	user_id: str
	email: str
	access_token: str
	refresh_token: str
	expires_in: int
	token_type: str = "Bearer"
	roles: list[str] = Field(default_factory=list)
	mfa_required: bool = False


class TokenValidateRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	token: str


class TokenPayloadResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	valid: bool
	user_id: str
	email: str
	roles: list[str] = Field(default_factory=list)
	tenant_id: str = "default"
	expires_at: str | None = None


class CreateUserRequest(BaseModel):
	model_config = ConfigDict(extra="allow", validate_by_name=True)

	email: str
	username: str | None = None
	password: str | None = None
	first_name: str = ""
	last_name: str = ""
	roles: list[str] = Field(default_factory=list)


class UserResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	email: str
	username: str = ""
	first_name: str = ""
	last_name: str = ""
	roles: list[str] = Field(default_factory=list)
	is_active: bool = True
	is_email_verified: bool = False
	mfa_enabled: bool = False


class AssignRoleRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	role: str


class CheckPermissionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	user_id: str
	permission: str
	resource_type: str | None = None
	resource_id: str | None = None
	tenant_id: str = "default"


class BulkCheckRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	user_id: str
	checks: list[dict] = Field(default_factory=list)


class RelationshipRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	resource_type: str
	resource_id: str
	relation: str
	subject_type: str
	subject_id: str


class MFASetupRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	mfa_type: str = "totp"


class ProviderInfoResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	auth_provider: str
	authz_provider: str
