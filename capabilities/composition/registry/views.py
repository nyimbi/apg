"""View models for APG capability registry screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_REGISTRY_AGENT_ROLES,
		SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		get_capability_contract,
	)
	from .service import CompositionRegistryService
except ImportError:
	from capability_contract import (
		SUPPORTED_REGISTRY_AGENT_ROLES,
		SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		get_capability_contract,
	)
	from service import CompositionRegistryService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "dashboard", "title": "Capability Registry", "summary": service.dashboard_summary(tenant_id), "sections": ["catalog_health", "dependency_health", "composition_health", "agent_activity"]}


def catalog_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "catalog", "records": service.list_capabilities(tenant_id), "columns": ["capability_id", "name", "category", "version", "owner", "status"], "actions": ["register_capability", "release_version", "deprecate"]}


def dependency_graph_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "dependencies", "records": service.list_dependencies(tenant_id), "columns": ["source_capability_id", "target_capability_id", "dependency_type", "version_constraint", "status"], "actions": ["add_dependency", "validate_cycle"]}


def composition_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "compositions", "records": service.list_compositions(tenant_id), "columns": ["composition_id", "name", "owner", "capability_ids", "status"], "actions": ["create_composition", "validate", "publish"]}


def version_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "versions", "records": service.list_versions(tenant_id), "columns": ["capability_id", "version", "reviewed_by", "status", "updated_at"], "actions": ["release_version", "compare", "deprecate"]}


def marketplace_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {"screen": "marketplace", "records": service.list_publications(tenant_id), "columns": ["publication_id", "capability_id", "documentation_ref", "reviewed_by", "status"], "actions": ["prepare_publication", "record_review"]}


def rule_center_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"screen": "rules", "rules": contract["rule_engine"]["rules"], "guardrails": contract["streaming"]["guardrails"]}


def agent_workbench_model(service: CompositionRegistryService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "agents",
		"records": service.list_registry_agents(tenant_id),
		"supported_runtimes": SUPPORTED_REGISTRY_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_REGISTRY_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
