"""Pydantic v2 models for APG Mobile App Platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, populate_by_name=True)

# ---------------------------------------------------------------------------
# Mobile App
# ---------------------------------------------------------------------------

class MobileAppCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	name: str
	bundle_id: str
	platform: str
	category: str
	description: str | None = None
	icon_url: str | None = None
	created_by: str

	@field_validator("platform")
	@classmethod
	def platform_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PLATFORMS
		assert v in SUPPORTED_PLATFORMS, f"platform must be one of {SUPPORTED_PLATFORMS}"
		return v

	@field_validator("category")
	@classmethod
	def category_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_APP_CATEGORIES
		assert v in SUPPORTED_APP_CATEGORIES, f"category must be one of {SUPPORTED_APP_CATEGORIES}"
		return v


class MobileAppUpdate(BaseModel):
	model_config = _MODEL_CFG
	name: str | None = None
	description: str | None = None
	icon_url: str | None = None
	state: str | None = None
	suspension_reason: str | None = None
	updated_by: str


class MobileAppResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	bundle_id: str
	platform: str
	category: str
	state: str = "draft"
	description: str | None = None
	icon_url: str | None = None
	suspension_reason: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# App Version
# ---------------------------------------------------------------------------

class AppVersionCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	version_string: str
	channel: str
	update_policy: str
	build_number: int
	release_notes: str | None = None
	environment: str = "staging"
	created_by: str

	@field_validator("channel")
	@classmethod
	def channel_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_VERSION_CHANNELS
		assert v in SUPPORTED_VERSION_CHANNELS, f"channel must be one of {SUPPORTED_VERSION_CHANNELS}"
		return v

	@field_validator("update_policy")
	@classmethod
	def update_policy_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_UPDATE_POLICIES
		assert v in SUPPORTED_UPDATE_POLICIES, f"update_policy must be one of {SUPPORTED_UPDATE_POLICIES}"
		return v


class AppVersionResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	version_string: str
	channel: str
	update_policy: str
	build_number: int
	release_notes: str | None = None
	environment: str
	state: str = "draft"
	approval_reference: str | None = None
	deployed_at: datetime | None = None
	rollback_of: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Sync Session
# ---------------------------------------------------------------------------

class SyncSessionCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	device_id: str
	sync_strategy: str
	offline_mode: str
	conflict_policy: str
	encryption_enabled: bool = True
	compression_algorithm: str = "gzip"
	created_by: str

	@field_validator("sync_strategy")
	@classmethod
	def sync_strategy_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_SYNC_STRATEGIES
		assert v in SUPPORTED_SYNC_STRATEGIES, f"sync_strategy must be one of {SUPPORTED_SYNC_STRATEGIES}"
		return v

	@field_validator("offline_mode")
	@classmethod
	def offline_mode_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_OFFLINE_MODES
		assert v in SUPPORTED_OFFLINE_MODES, f"offline_mode must be one of {SUPPORTED_OFFLINE_MODES}"
		return v

	@field_validator("conflict_policy")
	@classmethod
	def conflict_policy_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_CONFLICT_POLICIES
		assert v in SUPPORTED_CONFLICT_POLICIES, f"conflict_policy must be one of {SUPPORTED_CONFLICT_POLICIES}"
		return v


class SyncSessionResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	device_id: str
	sync_strategy: str
	offline_mode: str
	conflict_policy: str
	encryption_enabled: bool
	compression_algorithm: str
	state: str = "pending"
	records_synced: int = 0
	conflicts_detected: int = 0
	conflicts_resolved: int = 0
	bytes_transferred: int = 0
	started_at: datetime | None = None
	completed_at: datetime | None = None
	error_message: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Push Notification
# ---------------------------------------------------------------------------

class PushNotificationCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	channel: str
	title: str
	body: str
	target_type: str  # "device", "segment", "broadcast"
	target_reference: str
	approval_reference: str | None = None
	deep_link: str | None = None
	payload: dict[str, Any] | None = None
	created_by: str

	@field_validator("channel")
	@classmethod
	def channel_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_NOTIFICATION_CHANNELS
		assert v in SUPPORTED_NOTIFICATION_CHANNELS, f"channel must be one of {SUPPORTED_NOTIFICATION_CHANNELS}"
		return v


class PushNotificationResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	channel: str
	title: str
	body: str
	target_type: str
	target_reference: str
	approval_reference: str | None = None
	deep_link: str | None = None
	payload: dict[str, Any] | None = None
	state: str = "queued"
	delivered_count: int = 0
	failed_count: int = 0
	sent_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Biometric Enrollment
# ---------------------------------------------------------------------------

class BiometricEnrollmentCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	device_id: str
	user_id: str
	auth_method: str
	device_enrolled: bool = True
	created_by: str

	@field_validator("auth_method")
	@classmethod
	def auth_method_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_AUTH_METHODS
		assert v in SUPPORTED_AUTH_METHODS, f"auth_method must be one of {SUPPORTED_AUTH_METHODS}"
		return v


class BiometricEnrollmentResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	device_id: str
	user_id: str
	auth_method: str
	biometric_state: str = "enrolled"
	enrolled_at: datetime = Field(default_factory=datetime.utcnow)
	revoked_at: datetime | None = None
	revocation_reason: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Permission Scope Grant
# ---------------------------------------------------------------------------

class PermissionScopeCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	device_id: str
	scope: str
	granted_by: str
	justification: str
	created_by: str

	@field_validator("scope")
	@classmethod
	def scope_valid(cls, v: str) -> str:
		from .capability_contract import SUPPORTED_PERMISSION_SCOPES
		assert v in SUPPORTED_PERMISSION_SCOPES, f"scope must be one of {SUPPORTED_PERMISSION_SCOPES}"
		return v


class PermissionScopeResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	device_id: str
	scope: str
	granted_by: str
	justification: str
	state: str = "granted"
	revoked_at: datetime | None = None
	revocation_reason: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# App Analytics Event
# ---------------------------------------------------------------------------

class AppAnalyticsEventCreate(BaseModel):
	model_config = _MODEL_CFG
	tenant_id: str
	app_id: str
	device_id: str
	event_type: str
	event_payload: dict[str, Any] | None = None
	session_id: str | None = None
	created_by: str


class AppAnalyticsEventResponse(BaseModel):
	model_config = _MODEL_CFG
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	app_id: str
	device_id: str
	event_type: str
	event_payload: dict[str, Any] | None = None
	session_id: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
