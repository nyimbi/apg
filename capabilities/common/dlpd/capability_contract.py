"""Executable capability contract for APG Data Loss Prevention."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"data_patterns": {
		"enabled_classifiers": ["pii", "phi", "pci", "secrets", "financial_records", "source_code"],
		"nlp_classification_enabled": True,
		"minimum_classifier_confidence": 0.82,
		"custom_pattern_review_required": True
	},
	"channels": {
		"inspected": ["email", "api_export", "file_share", "chat", "clipboard", "object_storage"],
		"egress_policy_required": True,
		"bulk_export_threshold_records": 10000,
		"anomaly_context_required": True
	},
	"response": {
		"block_high_severity": True,
		"quarantine_supported": True,
		"incident_owner_required": True,
		"notification_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_inspection": True,
		"encrypted_quarantine_required": True,
		"legal_hold_supported": True
	},
	"ui": {
		"enable_policy_console": True,
		"enable_incident_queue": True,
		"enable_channel_monitor": True,
		"enable_classifier_workbench": True
	},
	"theme": {
		"default_theme": "dlpd_data_protection_ops",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "data_patterns", "channels", "response", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["data_patterns", "channels", "response", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All DLP operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "inspection_source_requires_policy", "description": "Inspected egress sources require a policy.", "condition": {"operation": "inspect_egress", "egress_policy_attached": False}, "effect": {"decision": "deny", "reason": "egress_policy_required", "required_action": "attach_egress_policy"}},
	{"name": "sensitive_content_requires_classification", "description": "Sensitive content cannot move without classification metadata.", "condition": {"sensitive_content_detected": True, "classification_label_present": False}, "effect": {"decision": "deny", "reason": "classification_label_required", "required_action": "apply_classification_label"}},
	{"name": "high_severity_exfiltration_requires_block", "description": "High-severity exfiltration signals must be blocked or quarantined.", "condition": {"severity": "high", "blocked_or_quarantined": False}, "effect": {"decision": "deny", "reason": "high_severity_block_required", "required_action": "block_or_quarantine_transfer"}},
	{"name": "quarantine_requires_encryption", "description": "Quarantined sensitive data must be encrypted.", "condition": {"quarantine_requested": True, "quarantine_encrypted": False}, "effect": {"decision": "deny", "reason": "encrypted_quarantine_required", "required_action": "encrypt_quarantine"}},
	{"name": "large_export_requires_review", "description": "Large exports require review before release.", "condition": {"export_record_count_gt": 10000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_export_review_required", "required_action": "review_export"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dlpd/dashboard", "component": "DLPDDashboard", "permission": "dlpd:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/dlpd/policies", "component": "DLPPolicyConsole", "permission": "dlpd:manage_policies", "nav_group": "Policies"},
	{"name": "classifiers", "path": "/dlpd/classifiers", "component": "ClassifierWorkbench", "permission": "dlpd:manage_policies", "nav_group": "Policies"},
	{"name": "channels", "path": "/dlpd/channels", "component": "ChannelMonitor", "permission": "dlpd:inspect", "nav_group": "Monitoring"},
	{"name": "incidents", "path": "/dlpd/incidents", "component": "IncidentQueue", "permission": "dlpd:respond", "nav_group": "Response"},
	{"name": "quarantine", "path": "/dlpd/quarantine", "component": "QuarantineVault", "permission": "dlpd:respond", "nav_group": "Response"},
	{"name": "analytics", "path": "/dlpd/analytics", "component": "DLPAnalytics", "permission": "dlpd:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/dlpd/settings", "component": "DLPDSettings", "permission": "dlpd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "dlpd_data_protection_ops",
	"tokens": {
		"color.primary": "#254E58",
		"color.accent": "#B83280",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F9FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"classifier_grid": {"icon": "scan-text", "status_indicator": "classifier-pill", "risk_style": "sensitivity-band"},
		"channel_flow": {"visual": "egress-sankey", "highlight": "blocked-chip"},
		"incident_queue": {"visual": "severity-lanes", "status_style": "response-chip"},
		"quarantine_vault": {"visual": "encrypted-item-list", "status_style": "hold-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable DLPD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "dlpd",
		"display_name": "Data Loss Prevention",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/dlpd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default DLPD governance rules."""
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
