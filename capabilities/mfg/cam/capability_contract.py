"""Executable capability contract for APG Computer-Aided Manufacturing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_cam"
CAPABILITY_NAME = "Computer-Aided Manufacturing"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "CNC program management, tool library, cutting parameter optimisation, and NC post-processing."

CAM_EVENT_STREAM = "apg.mfg.cam.lifecycle"

SUPPORTED_PROGRAM_STATUSES = ["draft", "under_review", "approved", "released", "archived"]
SUPPORTED_MACHINE_TYPES = ["cnc_mill", "cnc_lathe", "cnc_grinder", "edm", "laser", "plasma", "waterjet"]
SUPPORTED_TOOL_TYPES = ["end_mill", "drill", "tap", "insert", "turning_tool", "grinding_wheel", "special"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"supported_statuses": SUPPORTED_PROGRAM_STATUSES, "version_control": True, "simulation_required_before_release": False},
	"tools": {"supported_types": SUPPORTED_TOOL_TYPES, "tool_life_tracking": True},
	"machines": {"supported_types": SUPPORTED_MACHINE_TYPES},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "sfc": "mfg_sfc", "event_stream": "bytewax"},
}

PROVIDES = ["cnc_program_management", "tool_library", "cutting_parameters", "nc_post_processing"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.cam.program_released", "apg.mfg.cam.tool_life_expired"]
SUBSCRIBES = ["apg.mfg.sfc.operation_started"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-cam/dashboard", "component": "MfgCamDashboard", "permission": "mfg_cam:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/mfg-cam/programs", "component": "MfgCamPrograms", "permission": "mfg_cam:manage", "nav_group": "Programs"},
	{"name": "tool_library", "path": "/mfg-cam/tools", "component": "MfgCamTools", "permission": "mfg_cam:manage", "nav_group": "Tools"},
	{"name": "machines", "path": "/mfg-cam/machines", "component": "MfgCamMachines", "permission": "mfg_cam:view", "nav_group": "Machines"},
	{"name": "settings", "path": "/mfg-cam/settings", "component": "MfgCamSettings", "permission": "mfg_cam:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_cam_theme",
	"tokens": {"color.primary": "#1A2E44", "color.accent": "#00B4D8", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "program_release_requires_approval", "condition": {"operation": "release_program", "approval_present": False}, "effect": {"decision": "deny", "reason": "approval_required_for_release", "required_action": "obtain_program_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-cam/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": CAM_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
