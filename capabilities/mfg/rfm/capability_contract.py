"""Executable capability contract for APG Repetitive Manufacturing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_rfm"
CAPABILITY_NAME = "Repetitive Manufacturing"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Production line management, rate-based scheduling, daily rate planning, and backflush reporting."

RFM_EVENT_STREAM = "apg.mfg.rfm.lifecycle"

SUPPORTED_SCHEDULE_TYPES = ["daily_rate", "weekly_rate", "takt_based"]
SUPPORTED_LINE_STATUSES = ["active", "idle", "changeover", "maintenance", "decommissioned"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"lines": {"supported_statuses": SUPPORTED_LINE_STATUSES, "takt_time_enabled": True},
	"scheduling": {"supported_types": SUPPORTED_SCHEDULE_TYPES, "backflush_enabled": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mes": "mfg_mes", "event_stream": "bytewax"},
}

PROVIDES = ["production_line_management", "rate_scheduling", "backflush_reporting", "takt_time_analysis"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.rfm.daily_rate_confirmed", "apg.mfg.rfm.backflush_recorded", "apg.mfg.rfm.line_status_changed"]
SUBSCRIBES = ["apg.mfg.ppl.master_schedule_updated"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-rfm/dashboard", "component": "MfgRfmDashboard", "permission": "mfg_rfm:view", "nav_group": "Overview"},
	{"name": "lines", "path": "/mfg-rfm/lines", "component": "MfgRfmLines", "permission": "mfg_rfm:manage", "nav_group": "Lines"},
	{"name": "schedules", "path": "/mfg-rfm/schedules", "component": "MfgRfmSchedules", "permission": "mfg_rfm:manage", "nav_group": "Scheduling"},
	{"name": "backflush", "path": "/mfg-rfm/backflush", "component": "MfgRfmBackflush", "permission": "mfg_rfm:manage", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mfg-rfm/settings", "component": "MfgRfmSettings", "permission": "mfg_rfm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_rfm_theme",
	"tokens": {"color.primary": "#2C4770", "color.accent": "#27AE60", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "schedule_requires_line", "condition": {"operation": "create_schedule", "line_present": False}, "effect": {"decision": "deny", "reason": "production_line_required", "required_action": "specify_production_line"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-rfm/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": RFM_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
