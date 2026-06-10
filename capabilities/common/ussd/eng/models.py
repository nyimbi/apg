"""Pydantic v2 models for ussd_eng capability."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid_extensions import uuid7str
except ImportError:  # pragma: no cover
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())


# ── Session models ──────────────────────────────────────────────────────────

class UssdSessionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	phone_number: str
	service_code: str
	gateway: str = "africastalking"
	language: str = "en"
	tenant_id: str = "default"
	metadata: dict[str, Any] = Field(default_factory=dict)


class UssdSessionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	current_menu: str | None = None
	language: str | None = None
	variables: dict[str, Any] | None = None
	status: str | None = None


class UssdSessionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	phone_number: str
	service_code: str
	gateway: str
	language: str
	current_menu: str
	session_state: str  # active | ended | timeout | error
	variables: dict[str, Any] = Field(default_factory=dict)
	input_history: list[str] = Field(default_factory=list)
	menu_history: list[str] = Field(default_factory=list)
	hop_count: int = 0
	created_at: str
	updated_at: str
	ended_at: str | None = None


class UssdSessionList(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	sessions: list[UssdSessionResponse]
	total: int
	tenant_id: str


class UssdSessionFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str | None = None
	phone_number: str | None = None
	service_code: str | None = None
	gateway: str | None = None
	session_state: str | None = None
	language: str | None = None


# ── Menu models ─────────────────────────────────────────────────────────────

class UssdMenuItemCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	key: str
	label: str
	action: str  # navigate | execute | end | back
	target: str | None = None  # menu id for navigate
	handler: str | None = None  # callable reference for execute
	condition: str | None = None  # expression for conditional display


class UssdMenuCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	menu_id: str
	title: str
	body: str
	items: list[UssdMenuItemCreate] = Field(default_factory=list)
	service_code: str
	tenant_id: str = "default"
	language: str = "en"
	is_end_screen: bool = False
	timeout_seconds: int = 180
	metadata: dict[str, Any] = Field(default_factory=dict)


class UssdMenuUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	title: str | None = None
	body: str | None = None
	items: list[UssdMenuItemCreate] | None = None
	is_end_screen: bool | None = None
	timeout_seconds: int | None = None
	language: str | None = None


class UssdMenuResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	menu_id: str
	title: str
	body: str
	items: list[dict[str, Any]] = Field(default_factory=list)
	service_code: str
	language: str
	is_end_screen: bool
	timeout_seconds: int
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# ── Gateway models ───────────────────────────────────────────────────────────

class UssdGatewayCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	gateway_type: str  # africastalking | safaricom | custom
	service_code: str
	api_key: str | None = None
	api_secret: str | None = None
	username: str | None = None
	webhook_url: str | None = None
	tenant_id: str = "default"
	environment: str = "sandbox"  # sandbox | production
	metadata: dict[str, Any] = Field(default_factory=dict)


class UssdGatewayUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	webhook_url: str | None = None
	environment: str | None = None
	status: str | None = None
	metadata: dict[str, Any] | None = None


class UssdGatewayResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	gateway_type: str
	service_code: str
	username: str | None = None
	webhook_url: str | None = None
	environment: str
	status: str  # active | inactive | error
	session_count: int = 0
	created_at: str
	updated_at: str


# ── USSD callback / request models ──────────────────────────────────────────

class UssdCallbackRequest(BaseModel):
	"""Incoming USSD callback from gateway (Africa's Talking / Safaricom format)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	session_id: str
	service_code: str
	phone_number: str
	text: str = ""
	network_code: str | None = None
	gateway: str = "africastalking"


class UssdCallbackResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	response_type: str  # CON | END
	body: str
	session_id: str


# ── Audit model ──────────────────────────────────────────────────────────────

class UssdAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	resource_id: str
	resource_type: str
	actor_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)
	emitted_at: str
