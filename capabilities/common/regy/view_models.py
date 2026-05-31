"""Generated-app UI view models for the REGY capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .registry_runtime import RegistryService


def dashboard_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return registry dashboard data for composition UIs."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Service Registry",
		"tenant_id": tenant_id,
		"summary": service.registry_summary(tenant_id),
		"registry_agents": service.list_registry_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def service_catalog_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return service catalog data."""
	return {
		"tenant_id": tenant_id,
		"services": service.list_services(tenant_id),
		"instances": service.list_instances(tenant_id),
		"actions": ["register_service", "register_instance", "discover_services", "publish_to_gateway", "retire_service"],
	}


def registration_console_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return service and instance registration form metadata."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"registration": contract["configuration"]["registration"],
		"instances": contract["configuration"]["instances"],
		"required_fields": ["name", "owner", "api_version", "contract_schema_ref", "health_endpoint"],
		"allowed_protocols": contract["configuration"]["registration"]["allowed_protocols"],
		"allowed_regions": contract["configuration"]["instances"]["allowed_regions"],
	}


def discovery_console_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return discovery console data."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"defaults": contract["configuration"]["discovery"],
		"services": service.list_services(tenant_id),
		"healthy_instance_count": service.registry_summary(tenant_id)["healthy_instance_count"],
	}


def health_dashboard_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return health and reliability dashboard data."""
	instances = service.list_instances(tenant_id)
	return {
		"tenant_id": tenant_id,
		"health": get_capability_contract(tenant_id)["configuration"]["health"],
		"instances": instances,
		"healthy": [instance for instance in instances if instance["health"] == "healthy"],
		"degraded": [instance for instance in instances if instance["health"] != "healthy"],
	}


def version_manager_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return version governance data."""
	return {
		"tenant_id": tenant_id,
		"versions": service.list_versions(tenant_id),
		"reviews": [review for review in service.list_reviews(tenant_id) if review["review_type"] == "compatibility"],
		"contract_rules": get_capability_contract(tenant_id)["configuration"]["contracts"],
	}


def contract_review_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return pending contract and production review data."""
	return {
		"tenant_id": tenant_id,
		"reviews": service.list_reviews(tenant_id),
		"guardrail_rules": get_capability_contract(tenant_id)["rule_engine"]["rules"],
	}


def gateway_sync_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return gateway publication data."""
	return {
		"tenant_id": tenant_id,
		"publications": service.list_gateway_publications(tenant_id),
		"routing": get_capability_contract(tenant_id)["configuration"]["routing"],
	}


def retirement_review_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return retirement review data."""
	return {
		"tenant_id": tenant_id,
		"retired_services": [service_record for service_record in service.list_services(tenant_id) if service_record["status"] == "retired"],
		"required_evidence": ["impact_review_recorded", "gateway_unpublish_recorded"],
	}


def audit_timeline_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return registry audit timeline data."""
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"observability": get_capability_contract(tenant_id)["configuration"]["observability"],
	}


def registry_agent_roster_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return first-class registry-agent composition data."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_registry_agents(tenant_id),
		"agent_contract": contract["agents"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"required_fields": ["name", "runtime", "role", "scope", "owner", "purpose", "contribution_disclosed"],
	}


def lifecycle_batch_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return Bytewax registry lifecycle batch monitor data."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": service.list_lifecycle_batches(tenant_id),
		"streaming": contract["streaming"],
		"required_processor": contract["streaming"]["required_processor"],
		"summary": service.registry_summary(tenant_id),
	}


def settings_model(service: RegistryService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return registry settings data."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"schema": contract["configuration_schema"],
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}
