"""Composable view models for APG Custom Scripting Engine."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ScptService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Custom Scripting Engine",
		"summary": service.dashboard_summary(tenant_id),
		"streaming": contract["streaming"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def workbench_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"scripts": service.list_scripts(tenant_id),
		"sandboxes": service.list_sandboxes(tenant_id),
		"package_policies": service.list_package_policies(tenant_id),
		"guardrails": ["script_owner_required", "script_source_required", "package_policy_required", "sandbox_required", "blocked_import_denied"],
		"actions": ["create_script", "request_script_review", "approve_script", "publish_script", "bind_workflow", "retire_script"],
	}


def script_registry_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"scripts": service.list_scripts(tenant_id),
		"guardrails": ["script_owner_required", "script_review_required", "dangerous_permission_approval_required", "workflow_binding_policy_required", "script_retirement_reason_required"],
	}


def execution_console_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"executions": service.list_executions(tenant_id),
		"audit_events": service.audit_events(tenant_id),
		"guardrails": ["published_script_required", "sandbox_not_ready", "requested_by_required", "bytewax_event_stream_required", "execution_completion_evidence_required"],
		"actions": ["execute_script", "complete_execution", "cancel_execution"],
	}


def sandbox_monitor_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"sandboxes": service.list_sandboxes(tenant_id),
		"guardrails": ["sandbox_required", "sandbox_owner_required", "network_policy_required", "resource_review_required", "sandbox_block_reason_required"],
		"actions": ["create_sandbox", "change_sandbox_state"],
	}


def package_policy_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"package_policies": service.list_package_policies(tenant_id),
		"guardrails": ["package_policy_owner_required", "package_allowlist_required", "secret_access_policy_required", "filesystem_access_policy_required", "network_policy_required", "dangerous_import_blocking"],
	}


def approvals_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"approvals": service.list_approvals(tenant_id),
		"pending_scripts": [script for script in service.list_scripts(tenant_id) if script["dangerous_permissions"] and not script["approval_recorded"]],
	}


def scripting_agent_panel_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"agents": service.list_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["scripting_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["scripting_agents"]["allowed_roles"],
		"guardrails": ["scripting_agent_registration_required", "scripting_agent_runtime_not_supported", "scripting_agent_scope_required", "scripting_agent_disclosure_required"],
		"actions": ["register_scripting_agent"],
	}


def audit_trail_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"audit_events": service.audit_events(tenant_id),
		"streaming_topic": get_capability_contract(tenant_id)["streaming"]["topic"],
		"guardrails": ["scripting_audit_event_required", "cross_tenant_script_access_denied"],
	}


def analytics_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"summary": summary,
		"execution_health": {
			"succeeded": summary["succeeded_execution_count"],
			"failed": summary["failed_execution_count"],
			"total": summary["execution_count"],
		},
		"theme": get_capability_contract(tenant_id)["theme"],
	}


def settings_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"permissions": ["scpt:view", "scpt:write", "scpt:execute", "scpt:approve", "scpt:audit", "scpt:admin"],
	}
