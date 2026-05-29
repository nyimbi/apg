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
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def workbench_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"scripts": service.list_scripts(tenant_id),
		"sandboxes": service.list_sandboxes(tenant_id),
		"package_policies": service.list_package_policies(tenant_id),
		"actions": ["create_script", "approve_script", "publish_script", "bind_workflow"],
	}


def script_registry_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"scripts": service.list_scripts(tenant_id),
		"guardrails": ["script_owner_required", "dangerous_permission_approval_required", "workflow_binding_policy_required"],
	}


def execution_console_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"executions": service.list_executions(tenant_id),
		"audit_events": service.audit_events(tenant_id),
		"actions": ["execute_script", "complete_execution"],
	}


def sandbox_monitor_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"sandboxes": service.list_sandboxes(tenant_id),
		"guardrails": ["sandbox_required", "network_policy_required", "resource_review_required"],
	}


def package_policy_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"package_policies": service.list_package_policies(tenant_id),
		"guardrails": ["allowlist_required", "secret_access_policy_required", "filesystem_access_policy_required", "dangerous_import_blocking"],
	}


def approvals_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"approvals": service.list_approvals(tenant_id),
		"pending_scripts": [script for script in service.list_scripts(tenant_id) if script["dangerous_permissions"] and not script["approval_recorded"]],
	}


def settings_model(service: ScptService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"permissions": ["scpt:view", "scpt:write", "scpt:execute", "scpt:approve", "scpt:admin"],
	}
