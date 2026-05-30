"""Generated-application view models for the CACH capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import CacheGovernanceService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return compact dashboard state."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Cache Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "create_namespace", "label": "Create namespace", "permission": "cach:manage_namespaces"},
			{"id": "write_entry", "label": "Write entry", "permission": "cach:write"},
			{"id": "request_warming", "label": "Request warming", "permission": "cach:warm"},
			{"id": "review_eviction", "label": "Review eviction", "permission": "cach:review_eviction"},
		],
	}


def namespace_inventory_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return namespace inventory view state."""
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records("namespaces", tenant_id),
		"columns": [
			"namespace",
			"owner",
			"data_classification",
			"default_ttl_seconds",
			"max_ttl_seconds",
			"default_tier",
			"status",
		],
		"empty_state": "No cache namespaces registered.",
	}


def entry_explorer_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return cache entry explorer state."""
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records("entries", tenant_id),
		"filters": ["namespace", "tier", "status", "data_classification", "encrypted"],
		"columns": [
			"namespace",
			"key",
			"producer",
			"tier",
			"ttl_seconds",
			"status",
			"decision",
			"access_count",
		],
	}


def policy_manager_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return policy manager state derived from namespace policy records."""
	return {
		"tenant_id": tenant_id,
		"policies": service.list_records("namespaces", tenant_id),
		"rule_count": len(get_capability_contract(tenant_id)["rule_engine"]["rules"]),
		"policy_fields": [
			"default_ttl_seconds",
			"max_ttl_seconds",
			"max_entries",
			"encryption_required",
			"critical_reads_require_freshness",
			"stale_while_revalidate_allowed",
		],
	}


def warming_console_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return warming request and review queue state."""
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records("warming_plans", tenant_id),
		"columns": ["namespace", "source_name", "key_count", "requester", "status", "decision"],
		"requires_source_registration": True,
	}


def eviction_review_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return eviction and capacity review queue state."""
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records("eviction_reviews", tenant_id),
		"columns": [
			"namespace",
			"requester",
			"memory_utilization_percent",
			"proposed_action",
			"status",
			"reviewer",
		],
		"review_actions": ["approved", "rejected"],
	}


def hierarchy_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return cache tier topology state."""
	namespaces = service.list_records("namespaces", tenant_id)
	tiers = sorted({tier for namespace in namespaces for tier in namespace.get("allowed_tiers", [])})
	return {
		"tenant_id": tenant_id,
		"tiers": tiers,
		"namespaces": [
			{
				"namespace": namespace["namespace"],
				"default_tier": namespace["default_tier"],
				"allowed_tiers": namespace["allowed_tiers"],
				"status": namespace["status"],
			}
			for namespace in namespaces
		],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Return configured adapter boundary state."""
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"default_backend": adapters["default_backend"],
		"supported_backends": adapters["supported_backends"],
		"backend_binding_required_for_production": adapters["backend_binding_required_for_production"],
		"emit_mqeb_events": adapters["emit_mqeb_events"],
	}


def audit_timeline_model(service: CacheGovernanceService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return cache lifecycle audit timeline state."""
	return {
		"tenant_id": tenant_id,
		"events": service.list_records("audit_events", tenant_id),
		"columns": ["created_at", "event_type", "subject", "actor", "decision", "matched_rules"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Return settings state for generated CACH administration screens."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
