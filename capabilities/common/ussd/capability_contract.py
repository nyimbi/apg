"""
Capability contract for common/ussd.

Defines the stable public interface consumed by the APG platform bus,
capability registry, and cross-capability composition layer.

CAPABILITY_ID : "common_ussd"
PROVIDES      : ussd_session_management, menu_rendering, mpesa_callback,
                gateway_integration, i18n_menus
REQUIRES      : auth, audl, ntfy
PUBLISHES     : ussd.session_started, ussd.session_ended, ussd.payment_confirmed
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


# ── Contract metadata ─────────────────────────────────────────────────────────

CAPABILITY_ID = "common_ussd"
VERSION = "1.1.0"
DOMAIN = "common"
DESCRIPTION = (
	"USSD session state machine, declarative menu DSL with I18N, "
	"Africa's Talking and Safaricom gateway adapters, MPESA C2B and "
	"STK Push callback processing, and NATS event emission."
)

PROVIDES: list[str] = [
	"ussd_session_management",
	"menu_rendering",
	"mpesa_callback",
	"gateway_integration",
	"i18n_menus",
]

REQUIRES: list[str] = [
	"auth",   # token verification, permission checks
	"audl",   # structured audit logging
	"ntfy",   # notification dispatch (session alerts, payment confirmations)
]

PUBLISHES: list[str] = [
	"ussd.session_started",    # new subscriber session opened
	"ussd.session_ended",      # session terminated (user_exit / ttl / max_hops)
	"ussd.payment_confirmed",  # MPESA C2B payment confirmed
	"ussd.stk_result",         # MPESA STK Push result (success or failure)
]

SUBSCRIBES: list[str] = []  # no inbound subscriptions at this time

KEYWORDS: list[str] = [
	"ussd", "mpesa", "c2b", "stk_push", "africa_talking", "safaricom",
	"menu_dsl", "session_manager", "i18n", "swahili", "amharic", "french",
]


# ── Pydantic contract models ───────────────────────────────────────────────────

class UsCapabilityDescriptor(BaseModel):
	"""Machine-readable capability contract descriptor."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	capability_id: str
	version: str
	domain: str
	description: str
	provides: list[str]
	requires: list[str]
	publishes: list[str]
	subscribes: list[str]
	keywords: list[str]
	metadata: dict[str, Any] = Field(default_factory=dict)


class UsHealthReport(BaseModel):
	"""Health check response for common/ussd."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	capability_id: str = CAPABILITY_ID
	status: str  # healthy | degraded | unhealthy
	active_sessions: int = 0
	gateway_adapters: list[str] = Field(default_factory=list)
	nats_connected: bool = False
	checked_at: str = ""
	details: dict[str, Any] = Field(default_factory=dict)


# ── Factory / describe ────────────────────────────────────────────────────────

def describe() -> UsCapabilityDescriptor:
	"""Return the capability contract descriptor."""
	return UsCapabilityDescriptor(
		capability_id=CAPABILITY_ID,
		version=VERSION,
		domain=DOMAIN,
		description=DESCRIPTION,
		provides=PROVIDES,
		requires=REQUIRES,
		publishes=PUBLISHES,
		subscribes=SUBSCRIBES,
		keywords=KEYWORDS,
		metadata={
			"supported_gateways": ["africastalking", "safaricom"],
			"supported_languages": ["en", "sw", "am", "fr"],
			"ussd_max_chars": 180,
			"session_ttl_seconds": 180,
			"mpesa_callback_types": ["c2b_validation", "c2b_confirmation", "stk_push"],
		},
	)


def health(
	*,
	active_sessions: int = 0,
	nats_connected: bool = False,
	gateway_adapters: list[str] | None = None,
	details: dict[str, Any] | None = None,
) -> UsHealthReport:
	"""Build a health report. Call from the /health endpoint."""
	from datetime import datetime, timezone
	status = "healthy"
	if not nats_connected:
		status = "degraded"
	return UsHealthReport(
		status=status,
		active_sessions=active_sessions,
		gateway_adapters=gateway_adapters or [],
		nats_connected=nats_connected,
		checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
		details=details or {},
	)


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the standard APG capability contract dict for this capability."""
	return {
		"capability": CAPABILITY_ID,
		"id": CAPABILITY_ID,
		"version": VERSION,
		"domain": DOMAIN,
		"description": DESCRIPTION,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"publishes": PUBLISHES,
		"subscribes": SUBSCRIBES,
		"configuration": {"tenant_id": tenant_id},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": [
			{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
		]},
		"ui": {"shell": "apg_python", "api_prefix": "/ussd/api/v1", "routes": [
			{"name": "dashboard", "path": "/ussd/dashboard", "component": "UssdDashboard", "permission": "ussd:view", "nav_group": "Overview"},
		]},
		"theme": {
			"name": "common_ussd_theme",
			"tokens": {
				"color.primary": "#1A3A5C",
				"color.accent": "#F59E0B",
				"color.success": "#10B981",
				"color.danger": "#EF4444",
				"surface.canvas": "#F8FAFC",
				"surface.panel": "#FFFFFF",
				"text.primary": "#111827",
				"border.radius": "8px",
			},
			"components": {"button": {}},
		},
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id"],
			"properties": {"tenant_id": {"type": "string"}},
		},
		"streaming": {"processor": "bytewax", "stream": "apg.common.ussd", "key": "tenant_id"},
	}


# ── Composition keywords ───────────────────────────────────────────────────────
#
# These strings trigger capability composition in APG routing rules.
# When a downstream capability emits or subscribes to one of these keywords,
# the platform bus routes the event through common_ussd.
#
# Keyword → capability interaction map:
#   "dial_ussd"          → start_session via gateway adapter
#   "ussd_input"         → navigate()
#   "mpesa_c2b"          → MpesaCallbackHandler.handle_c2b_confirmation
#   "mpesa_validate"     → MpesaCallbackHandler.handle_c2b_validation
#   "mpesa_stk"          → MpesaCallbackHandler.handle_stk_callback
#   "render_menu"        → menu_builder.render()

COMPOSITION_KEYWORDS: list[str] = [
	"dial_ussd",
	"ussd_input",
	"mpesa_c2b",
	"mpesa_validate",
	"mpesa_stk",
	"render_menu",
]
