"""Executable capability contract for APG Quantum Computing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"backends": {"backend_approval_required": True, "quota_policy_required": True, "provider_credentials_managed_by_keym": True, "simulator_fallback_enabled": True},
	"circuits": {"circuit_owner_required": True, "versioning_required": True, "sensitive_input_encryption_required": True, "experiment_metadata_required": True},
	"jobs": {"job_quota_required": True, "shot_limit": 10000, "result_retention_days": 90, "retry_policy_required": True},
	"governance": {"require_tenant_context": True, "audit_quantum_jobs": True, "post_quantum_review_required": True, "cost_limit_required": True},
	"ui": {"enable_backend_registry": True, "enable_circuit_library": True, "enable_job_queue": True, "enable_result_viewer": True},
	"theme": {"default_theme": "quan_quantum_lab", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "backends", "circuits", "jobs", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["backends", "circuits", "jobs", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All quantum operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "backend_requires_approval", "description": "Quantum backends require approval.", "condition": {"operation": "register_backend", "backend_approved": False}, "effect": {"decision": "deny", "reason": "backend_approval_required", "required_action": "approve_quantum_backend"}},
	{"name": "circuit_requires_owner", "description": "Quantum circuits require an accountable owner.", "condition": {"operation": "create_circuit", "circuit_owner_assigned": False}, "effect": {"decision": "deny", "reason": "circuit_owner_required", "required_action": "assign_circuit_owner"}},
	{"name": "sensitive_input_requires_encryption", "description": "Sensitive circuit inputs require encryption.", "condition": {"sensitive_input_present": True, "encryption_applied": False}, "effect": {"decision": "deny", "reason": "sensitive_input_encryption_required", "required_action": "encrypt_quantum_inputs"}},
	{"name": "job_requires_quota", "description": "Quantum jobs require quota policy.", "condition": {"operation": "submit_job", "quota_policy_attached": False}, "effect": {"decision": "deny", "reason": "job_quota_required", "required_action": "attach_job_quota_policy"}},
	{"name": "large_job_requires_review", "description": "Large quantum jobs require review.", "condition": {"shot_count_gt": 10000, "job_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_quantum_job_review_required", "required_action": "review_quantum_job"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/quan/dashboard", "component": "QUANDashboard", "permission": "quan:view", "nav_group": "Overview"},
	{"name": "backends", "path": "/quan/backends", "component": "QuantumBackendRegistry", "permission": "quan:manage_backends", "nav_group": "Backends"},
	{"name": "circuits", "path": "/quan/circuits", "component": "CircuitLibrary", "permission": "quan:experiment", "nav_group": "Circuits"},
	{"name": "jobs", "path": "/quan/jobs", "component": "QuantumJobQueue", "permission": "quan:run_jobs", "nav_group": "Jobs"},
	{"name": "experiments", "path": "/quan/experiments", "component": "ExperimentWorkbench", "permission": "quan:experiment", "nav_group": "Experiments"},
	{"name": "results", "path": "/quan/results", "component": "ResultViewer", "permission": "quan:view", "nav_group": "Results"},
	{"name": "governance", "path": "/quan/governance", "component": "QuantumGovernance", "permission": "quan:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/quan/settings", "component": "QUANSettings", "permission": "quan:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "quan_quantum_lab",
	"tokens": {"color.primary": "#2B4C7E", "color.accent": "#6B46C1", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"backend_card": {"icon": "cpu", "status_indicator": "quota-pill", "risk_style": "provider-band"}, "circuit_library": {"visual": "circuit-list", "highlight": "version-chip"}, "job_queue": {"visual": "execution-queue", "status_style": "shot-chip"}, "result_viewer": {"visual": "measurement-grid", "status_style": "confidence-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "quan", "display_name": "Quantum Computing", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/quan/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
