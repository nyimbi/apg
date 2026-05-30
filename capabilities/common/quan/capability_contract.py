"""Executable capability contract for APG Quantum Computing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_QUAN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_QUAN_AGENT_ROLES = [
	"backend_reviewer",
	"circuit_reviewer",
	"job_reviewer",
	"result_reviewer",
	"cost_reviewer",
	"post_quantum_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"backends": {
		"backend_approval_required": True,
		"quota_policy_required": True,
		"provider_credentials_managed_by_keym": True,
		"simulator_fallback_enabled": True,
		"qubit_capacity_required": True,
	},
	"circuits": {
		"circuit_owner_required": True,
		"versioning_required": True,
		"sensitive_input_encryption_required": True,
		"experiment_metadata_required": True,
		"gate_validation_required": True,
	},
	"jobs": {
		"job_quota_required": True,
		"shot_limit": 10000,
		"result_retention_days": 90,
		"retry_policy_required": True,
		"event_stream": "bytewax",
	},
	"quan_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_QUAN_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_QUAN_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_quantum_jobs": True,
		"post_quantum_review_required": True,
		"cost_limit_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"backend_metrics_required": True,
		"job_metrics_required": True,
		"result_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.QuanService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"ai_orchestration": "aicr",
		"encryption": "encr",
		"key_management": "keym",
		"audit_sink": "audl",
		"monitoring": "moni",
		"compliance": "comp",
	},
	"ui": {
		"enable_backend_registry": True,
		"enable_circuit_library": True,
		"enable_job_queue": True,
		"enable_result_viewer": True,
		"enable_experiment_workbench": True,
		"enable_agent_panel": True,
		"enable_audit": True,
	},
	"theme": {
		"default_theme": "quan_quantum_lab",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"backends",
		"circuits",
		"jobs",
		"quan_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"backends",
			"circuits",
			"jobs",
			"quan_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All quantum operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "backend_requires_approval", "description": "Quantum backends require approval.", "condition": {"operation": "register_backend", "backend_approved": False}, "effect": {"decision": "deny", "reason": "backend_approval_required", "required_action": "approve_quantum_backend"}},
	{"name": "backend_requires_credentials_reference", "description": "External quantum backends require managed credentials.", "condition": {"operation": "register_backend", "external_provider": True, "credentials_ref_present": False}, "effect": {"decision": "deny", "reason": "provider_credentials_required", "required_action": "attach_keym_credentials_reference"}},
	{"name": "backend_requires_qubit_capacity", "description": "Quantum backends require positive qubit capacity.", "condition": {"operation": "register_backend", "backend_qubit_count_lt": 1}, "effect": {"decision": "deny", "reason": "backend_qubit_capacity_required", "required_action": "set_backend_qubit_capacity"}},
	{"name": "circuit_requires_owner", "description": "Quantum circuits require an accountable owner.", "condition": {"operation": "create_circuit", "circuit_owner_assigned": False}, "effect": {"decision": "deny", "reason": "circuit_owner_required", "required_action": "assign_circuit_owner"}},
	{"name": "circuit_requires_version", "description": "Quantum circuits require explicit version.", "condition": {"operation": "create_circuit", "circuit_version_present": False}, "effect": {"decision": "deny", "reason": "circuit_version_required", "required_action": "version_quantum_circuit"}},
	{"name": "circuit_requires_qubits", "description": "Quantum circuits require positive qubit requirement.", "condition": {"operation": "create_circuit", "circuit_qubits_required_lt": 1}, "effect": {"decision": "deny", "reason": "circuit_qubit_requirement_required", "required_action": "set_circuit_qubit_requirement"}},
	{"name": "circuit_requires_gates", "description": "Quantum circuits require at least one gate.", "condition": {"operation": "create_circuit", "circuit_gates_present": False}, "effect": {"decision": "deny", "reason": "circuit_gates_required", "required_action": "define_quantum_gates"}},
	{"name": "sensitive_input_requires_encryption", "description": "Sensitive circuit inputs require encryption.", "condition": {"sensitive_input_present": True, "encryption_applied": False}, "effect": {"decision": "deny", "reason": "sensitive_input_encryption_required", "required_action": "encrypt_quantum_inputs"}},
	{"name": "circuit_requires_experiment_metadata", "description": "Quantum circuits require experiment metadata.", "condition": {"operation": "create_circuit", "experiment_metadata_present": False}, "effect": {"decision": "deny", "reason": "experiment_metadata_required", "required_action": "attach_experiment_metadata"}},
	{"name": "job_requires_quota", "description": "Quantum jobs require quota policy.", "condition": {"operation": "submit_job", "quota_policy_attached": False}, "effect": {"decision": "deny", "reason": "job_quota_required", "required_action": "attach_job_quota_policy"}},
	{"name": "job_requires_submitter", "description": "Quantum jobs require submitter identity.", "condition": {"operation": "submit_job", "job_submitter_present": False}, "effect": {"decision": "deny", "reason": "job_submitter_required", "required_action": "set_job_submitter"}},
	{"name": "job_requires_retry_policy", "description": "Quantum jobs require retry policy.", "condition": {"operation": "submit_job", "retry_policy_attached": False}, "effect": {"decision": "deny", "reason": "retry_policy_required", "required_action": "attach_retry_policy"}},
	{"name": "job_requires_shot_count", "description": "Quantum jobs require positive shot count.", "condition": {"operation": "submit_job", "shot_count_lt": 1}, "effect": {"decision": "deny", "reason": "job_shot_count_required", "required_action": "set_job_shot_count"}},
	{"name": "large_job_requires_review", "description": "Large quantum jobs require review.", "condition": {"operation": "submit_job", "shot_count_gt": 10000, "job_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_quantum_job_review_required", "required_action": "review_quantum_job"}},
	{"name": "job_requires_bytewax_stream", "description": "Quantum job lifecycle events require Bytewax streams.", "condition": {"operation": "submit_job", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "experiment_requires_hypothesis", "description": "Quantum experiments require a hypothesis.", "condition": {"operation": "create_experiment", "hypothesis_present": False}, "effect": {"decision": "deny", "reason": "experiment_hypothesis_required", "required_action": "attach_experiment_hypothesis"}},
	{"name": "experiment_requires_post_quantum_review", "description": "Post-quantum experiments require review.", "condition": {"operation": "create_experiment", "post_quantum_scope": True, "post_quantum_review_recorded": False}, "effect": {"decision": "deny", "reason": "post_quantum_review_required", "required_action": "review_post_quantum_experiment"}},
	{"name": "quan_agent_requires_registration", "description": "AI quantum agents must be registered.", "condition": {"quan_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "quan_agent_registration_required", "required_action": "register_quan_agent"}},
	{"name": "quan_agent_runtime_supported", "description": "AI quantum agents must use a supported runtime.", "condition": {"quan_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "quan_agent_runtime_not_supported", "required_action": "choose_supported_quan_agent_runtime"}},
	{"name": "quan_agent_role_supported", "description": "AI quantum agents must use a supported role.", "condition": {"quan_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "quan_agent_role_not_supported", "required_action": "choose_supported_quan_agent_role"}},
	{"name": "quan_agent_requires_scope", "description": "AI quantum agents require explicit scope.", "condition": {"quan_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "quan_agent_scope_required", "required_action": "set_quan_agent_scope"}},
	{"name": "quan_agent_requires_disclosure", "description": "AI quantum-agent contributions require disclosure.", "condition": {"quan_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "quan_agent_disclosure_required", "required_action": "disclose_quan_agent"}},
	{"name": "quan_state_change_requires_audit", "description": "Quantum lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "quan_audit_event_required", "required_action": "record_quan_audit_event"}},
	{"name": "batch_quantum_mutation_requires_bytewax", "description": "Batch quantum mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_quantum_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/quan/dashboard", "component": "QUANDashboard", "permission": "quan:view", "nav_group": "Overview"},
	{"name": "backends", "path": "/quan/backends", "component": "QuantumBackendRegistry", "permission": "quan:manage_backends", "nav_group": "Backends"},
	{"name": "circuits", "path": "/quan/circuits", "component": "CircuitLibrary", "permission": "quan:experiment", "nav_group": "Circuits"},
	{"name": "jobs", "path": "/quan/jobs", "component": "QuantumJobQueue", "permission": "quan:run_jobs", "nav_group": "Jobs"},
	{"name": "experiments", "path": "/quan/experiments", "component": "ExperimentWorkbench", "permission": "quan:experiment", "nav_group": "Experiments"},
	{"name": "results", "path": "/quan/results", "component": "ResultViewer", "permission": "quan:view", "nav_group": "Results"},
	{"name": "agents", "path": "/quan/agents", "component": "QUANAgentPanel", "permission": "quan:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/quan/audit", "component": "QuantumAuditTrail", "permission": "quan:admin", "nav_group": "Governance"},
	{"name": "governance", "path": "/quan/governance", "component": "QuantumGovernance", "permission": "quan:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/quan/settings", "component": "QUANSettings", "permission": "quan:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "quan_quantum_lab",
	"tokens": {
		"color.primary": "#2B4C7E",
		"color.accent": "#6B46C1",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"backend_card": {"icon": "cpu", "status_indicator": "quota-pill", "risk_style": "provider-band"},
		"circuit_library": {"visual": "circuit-list", "highlight": "version-chip"},
		"job_queue": {"visual": "execution-queue", "status_style": "shot-chip"},
		"result_viewer": {"visual": "measurement-grid", "status_style": "confidence-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "experiment-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.quan.lifecycle",
		"state": [
			"backends",
			"circuits",
			"quota_policies",
			"jobs",
			"results",
			"experiments",
			"quan_agents",
			"audit_events",
		],
		"events": [
			"quan_backend_registered",
			"quan_quota_policy_attached",
			"quan_circuit_created",
			"quan_job_submitted",
			"quan_result_recorded",
			"quan_experiment_created",
			"quan_agent_registered",
		],
		"batch_mutation_guardrail": "batch_quantum_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "quan",
		"display_name": "Quantum Computing",
		"version": "1.0.0",
		"provides": [
			"quantum_backend_registry",
			"circuit_management",
			"quantum_job_orchestration",
			"result_analysis",
			"post_quantum_governance",
			"quan_agents",
		],
		"requires": ["aicr", "encr", "keym", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/quan/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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


def event_stream_name(value: str) -> str:
	return value.strip().lower().split("://", 1)[0]


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
