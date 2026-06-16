"""
USSD Engine — Pydantic models.

Covers: sessions, menus, menu items, gateway requests/responses, session
state enum, and flow definitions.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

SESSION_TTL_SECONDS: int = 180
AT_MAX_CHARS: int = 182  # Africa's Talking hard limit


# ── Enums ────────────────────────────────────────────────────────────────────

class SessionState(str, Enum):
	ACTIVE   = "active"
	ENDED    = "ended"
	EXPIRED  = "expired"
	ERROR    = "error"


class MenuItemAction(str, Enum):
	NAVIGATE = "navigate"
	EXECUTE  = "execute"
	END      = "end"
	BACK     = "back"
	INPUT    = "input"


# ── Menu models ───────────────────────────────────────────────────────────────

class USSDMenuItem(BaseModel):
	"""A single selectable option inside a USSDMenu."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	key: str
	label: str
	labels: dict[str, str] = Field(default_factory=dict)	# lang → label
	action: MenuItemAction = MenuItemAction.NAVIGATE
	target: str | None = None		# menu_id (navigate) or variable name (input)
	handler: str | None = None		# callable reference (execute)
	condition: str | None = None	# simple condition expression


class USSDMenu(BaseModel):
	"""A single USSD screen / menu node."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	menu_id: str = Field(default_factory=uuid7str)
	title: str
	titles: dict[str, str] = Field(default_factory=dict)
	body: str = ""
	bodies: dict[str, str] = Field(default_factory=dict)
	items: list[USSDMenuItem] = Field(default_factory=list)
	is_terminal: bool = False
	show_back: bool = False
	show_exit: bool = False
	metadata: dict[str, Any] = Field(default_factory=dict)


# ── Session model ─────────────────────────────────────────────────────────────

class USSDSession(BaseModel):
	"""Live USSD session keyed by (msisdn, service_code)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	session_id: str = Field(default_factory=uuid7str)
	msisdn: str
	service_code: str
	tenant_id: str = "default"
	gateway: str = "africastalking"		# africastalking | safaricom
	current_menu_id: str = "main"
	navigation_stack: list[str] = Field(default_factory=lambda: ["main"])
	data: dict[str, Any] = Field(default_factory=dict)
	input_history: list[str] = Field(default_factory=list)
	hop_count: int = 0
	language: str = "en"
	state: SessionState = SessionState.ACTIVE
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	expires_at: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)
	)
	ended_at: datetime | None = None


# ── Gateway request / response ────────────────────────────────────────────────

class USSDRequest(BaseModel):
	"""
	Normalised USSD request — gateway-agnostic internal format.

	Callers populate this from the raw AT or Safaricom webhook payload before
	passing it to USSDEngineService.handle_request().
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	session_id: str
	service_code: str
	msisdn: str
	text: str = ""				# full concatenated input chain (AT style)
	network_code: str | None = None
	gateway: str = "africastalking"
	tenant_id: str = "default"
	language: str = "en"
	raw: dict[str, Any] = Field(default_factory=dict)


class USSDResponse(BaseModel):
	"""
	Response returned to the USSD gateway.

	``continue_session=True``  → CON prefix (subscriber sees next screen)
	``continue_session=False`` → END prefix (session terminates)

	The ``text`` field must never exceed AT_MAX_CHARS (182).
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	text: str
	continue_session: bool
	session_id: str
	menu_id: str = ""
	hop_count: int = 0

	@field_validator("text")
	@classmethod
	def _check_length(cls, v: str) -> str:
		if len(v) > AT_MAX_CHARS:
			return v[:AT_MAX_CHARS]
		return v


# ── Flow definition ───────────────────────────────────────────────────────────

class FlowDefinition(BaseModel):
	"""
	Declarative USSD flow definition for a service code.

	A flow is a complete menu tree: menus indexed by menu_id, a pointer to
	the root, and per-language support metadata.
	"""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	flow_id: str = Field(default_factory=uuid7str)
	service_code: str
	name: str
	description: str = ""
	root_menu_id: str
	menus: dict[str, USSDMenu] = Field(default_factory=dict)	# menu_id → USSDMenu
	default_language: str = "en"
	supported_languages: list[str] = Field(default_factory=lambda: ["en"])
	tenant_id: str = "default"
	version: str = "1.0.0"
	active: bool = True
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator("menus")
	@classmethod
	def _root_must_exist(cls, v: dict[str, USSDMenu], info: Any) -> dict[str, USSDMenu]:
		# Only validate when root_menu_id is already set (field order matters in Pydantic v2)
		root = (info.data or {}).get("root_menu_id")
		if root and v and root not in v:
			raise ValueError(f"root_menu_id '{root}' not found in menus")
		return v
