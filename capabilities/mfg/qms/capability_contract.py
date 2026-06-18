"""Executable capability contract for APG Quality Management System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_qms"
CAPABILITY_NAME = "Quality Management System"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Inspection plans, non-conformance reports (NCR), corrective and preventive action (CAPA), and SPC chart monitoring."

QMS_EVENT_STREAM = "apg.mfg.qms.lifecycle"

SUPPORTED_INSPECTION_TYPES = ["incoming", "in_process", "final", "customer_return"]
SUPPORTED_NCR_STATUSES = ["open", "under_review", "capa_required", "capa_in_progress", "closed", "rejected"]
SUPPORTED_CAPA_TYPES = ["corrective", "preventive"]
SUPPORTED_DISPOSITION_TYPES = ["accept", "reject", "rework", "return_to_supplier", "use_as_is"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"inspection": {"supported_types": SUPPORTED_INSPECTION_TYPES, "aql_sampling": True},
	"ncr": {"supported_statuses": SUPPORTED_NCR_STATUSES, "auto_capa_threshold": "high"},
	"capa": {"supported_types": SUPPORTED_CAPA_TYPES, "approval_required": True},
	"spc": {"control_chart_types": ["x_bar_r", "p_chart", "np_chart", "c_chart"], "enabled": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mes": "mfg_mes", "event_stream": "bytewax"},
}

PROVIDES = ["inspection_plan", "ncr_workflow", "capa_workflow", "spc_monitoring", "quality_reporting"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.qms.ncr_raised", "apg.mfg.qms.capa_completed", "apg.mfg.qms.inspection_completed", "apg.mfg.qms.spc_alert"]
SUBSCRIBES = ["apg.mfg.mes.scrap_recorded", "apg.mfg.mes.work_order_completed"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-qms/dashboard", "component": "MfgQmsDashboard", "permission": "mfg_qms:view", "nav_group": "Overview"},
	{"name": "inspections", "path": "/mfg-qms/inspections", "component": "MfgQmsInspections", "permission": "mfg_qms:manage", "nav_group": "Inspection"},
	{"name": "ncrs", "path": "/mfg-qms/ncrs", "component": "MfgQmsNcrs", "permission": "mfg_qms:manage", "nav_group": "Non-Conformance"},
	{"name": "capas", "path": "/mfg-qms/capas", "component": "MfgQmsCapas", "permission": "mfg_qms:manage", "nav_group": "CAPA"},
	{"name": "spc", "path": "/mfg-qms/spc", "component": "MfgQmsSpc", "permission": "mfg_qms:view", "nav_group": "SPC"},
	{"name": "settings", "path": "/mfg-qms/settings", "component": "MfgQmsSettings", "permission": "mfg_qms:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_qms_theme",
	"tokens": {"color.primary": "#1B4F72", "color.accent": "#17A589", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ncr_requires_item", "condition": {"operation": "create_ncr", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "capa_approval_required", "condition": {"operation": "close_capa", "approval_present": False}, "effect": {"decision": "deny", "reason": "capa_approval_required", "required_action": "obtain_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-qms/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": QMS_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"inspection": {"type": "object"},
				"ncr": {"type": "object"},
				"capa": {"type": "object"},
				"spc": {"type": "object"},
				"governance": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
