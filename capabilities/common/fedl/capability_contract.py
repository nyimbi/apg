"""Executable capability contract for APG Federated Learning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"federation": {"coordinator_enabled": True, "participant_attestation_required": True, "minimum_participants": 3},
	"privacy": {"secure_aggregation_required": True, "differential_privacy_enabled": True, "max_privacy_epsilon": 8.0},
	"training": {"round_approval_required": True, "model_update_validation": True, "poisoning_detection_enabled": True},
	"governance": {"require_tenant_context": True, "data_residency_required": True, "audit_rounds": True, "participant_contract_required": True},
	"ui": {"enable_federation_console": True, "enable_round_monitor": True, "enable_privacy_budget": True, "enable_participant_map": True},
	"theme": {"default_theme": "fedl_privacy_mesh", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "federation", "privacy", "training", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["federation", "privacy", "training", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All federated learning operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "participant_requires_attestation", "description": "Participants require attestation before joining.", "condition": {"operation": "join_federation", "participant_attested": False}, "effect": {"decision": "deny", "reason": "participant_attestation_required", "required_action": "complete_participant_attestation"}},
	{"name": "round_requires_minimum_participants", "description": "Training rounds require enough participants.", "condition": {"participant_count_lt": 3, "operation": "start_round"}, "effect": {"decision": "deny", "reason": "minimum_participants_required", "required_action": "add_participants"}},
	{"name": "secure_aggregation_required", "description": "Federated updates require secure aggregation.", "condition": {"secure_aggregation_enabled": False, "operation": "aggregate_updates"}, "effect": {"decision": "deny", "reason": "secure_aggregation_required", "required_action": "enable_secure_aggregation"}},
	{"name": "privacy_budget_requires_review", "description": "High privacy budget requires review.", "condition": {"privacy_epsilon_gt": 8.0, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "privacy_budget_review_required", "required_action": "record_privacy_review"}},
	{"name": "poisoning_signal_blocks_round", "description": "Poisoning signals block model aggregation.", "condition": {"poisoning_signal_detected": True, "operation": "aggregate_updates"}, "effect": {"decision": "deny", "reason": "poisoning_signal_detected", "required_action": "quarantine_suspicious_update"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/fedl/dashboard", "component": "FEDLDashboard", "permission": "fedl:view", "nav_group": "Overview"},
	{"name": "federations", "path": "/fedl/federations", "component": "FederationConsole", "permission": "fedl:manage_federations", "nav_group": "Federations"},
	{"name": "participants", "path": "/fedl/participants", "component": "ParticipantMap", "permission": "fedl:view_participants", "nav_group": "Federations"},
	{"name": "rounds", "path": "/fedl/rounds", "component": "TrainingRoundMonitor", "permission": "fedl:run_rounds", "nav_group": "Training"},
	{"name": "privacy", "path": "/fedl/privacy", "component": "PrivacyBudgetConsole", "permission": "fedl:manage_privacy", "nav_group": "Governance"},
	{"name": "security", "path": "/fedl/security", "component": "PoisoningDefense", "permission": "fedl:manage_security", "nav_group": "Governance"},
	{"name": "models", "path": "/fedl/models", "component": "FederatedModelRegistry", "permission": "fedl:view_models", "nav_group": "Models"},
	{"name": "settings", "path": "/fedl/settings", "component": "FEDLSettings", "permission": "fedl:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "fedl_privacy_mesh",
	"tokens": {"color.primary": "#1E5F74", "color.accent": "#9B5DE5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F6F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"participant_node_card": {"icon": "nodes", "status_indicator": "attestation-pill", "risk_style": "privacy-band"},
		"training_round_timeline": {"visual": "round-checkpoints", "highlight": "aggregation-chip"},
		"privacy_budget_meter": {"visual": "segmented-meter", "threshold_style": "epsilon-bands"},
		"federation_topology": {"visual": "privacy-mesh", "edge_style": "secure-channel-line"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "fedl", "display_name": "Federated Learning", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "__init__.py", "api_prefix": "/fedl/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
