"""Regression coverage for the REGY executable capability contract."""

import pytest

from capabilities.common.regy import register_capability
from capabilities.common.regy.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
)
from capabilities.common.regy.registry_runtime import RegistryService
from capabilities.common.regy.view_models import (
	dashboard_model,
	discovery_console_model,
	lifecycle_batch_model,
	registry_agent_roster_model,
	service_catalog_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-registry", {"discovery": {"cache_ttl_seconds": 15}})

	assert contract["capability"] == "regy"
	assert contract["configuration"]["tenant_id"] == "tenant-registry"
	assert contract["configuration"]["discovery"]["cache_ttl_seconds"] == 15
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"registration",
		"instances",
		"contracts",
		"discovery",
		"health",
		"routing",
		"governance",
		"observability",
		"adapters",
		"agents",
		"streaming",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 33
	assert contract["provides"] == ["service_registry", "service_discovery", "registry_agent_composition"]
	assert contract["requires"] == ["apig", "auth", "conf"]
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "contract_reviewer" in contract["agents"]["supported_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "registry_runtime.RegistryService"
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"services",
		"register",
		"instances",
		"discovery",
		"health",
		"versions",
		"contracts",
		"gateway_sync",
		"retirements",
		"audit",
		"agents",
		"lifecycle",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/api/regy/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "gateway_sync_panel" in contract["theme"]["components"]
	assert "registry_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_registry_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_service",
		"owner_assigned": False,
		"health_endpoint_present": False,
		"api_version_present": False,
		"contract_schema_present": False,
		"duplicate_service_name": True,
		"environment": "production",
		"production_review_recorded": False,
		"trace_propagation_configured": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"service_registration_requires_owner",
		"service_registration_requires_health_endpoint",
		"service_registration_requires_api_version",
		"service_registration_requires_contract_schema",
		"duplicate_service_name_blocked",
		"production_registration_requires_review",
		"production_requires_tracing",
	}


def test_rule_engine_enforces_registry_agent_and_bytewax_guardrails():
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_registry_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_regy_lifecycle_batch",
		"event_stream": "legacy_queue",
	})

	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"registry_agent_runtime_supported",
		"registry_agent_role_supported",
		"registry_agent_requires_scope",
		"registry_agent_requires_owner",
		"registry_agent_requires_purpose",
		"registry_agent_requires_contribution_disclosure",
		"registry_agent_privileged_role_requires_human_approval",
	}
	assert stream_result["decision"] == "deny"
	assert stream_result["matched_rules"] == ["bytewax_regy_stream_required"]


def test_registry_runtime_registers_discovers_and_publishes_services():
	registry = RegistryService()

	service = registry.register_service(
		service_id="orders",
		tenant_id="tenant-a",
		name="orders",
		owner="platform",
		service_type="rest_api",
		environment="production",
		api_version="1.0.0",
		contract_schema_ref="schemas/orders.yaml",
		health_endpoint="/health",
		routing_metadata={"path": "/orders", "strategy": "weighted"},
		production_review_recorded=True,
		trace_propagation_configured=True,
	)
	instance = registry.register_instance(
		instance_id="orders-1",
		tenant_id="tenant-a",
		service_id="orders",
		endpoint="https://orders.internal",
		region="edge-africa",
		health_probe="/health",
		weight=100,
	)
	discovery = registry.discover_services("tenant-a", service_name="orders")
	publication = registry.publish_to_gateway("orders-public", "tenant-a", "orders", "/orders")

	assert service["status"] == "registered"
	assert instance["health"] == "healthy"
	assert discovery["total_count"] == 1
	assert publication["status"] == "published"
	assert registry.registry_summary("tenant-a")["audit_event_count"] == 3


def test_registry_runtime_blocks_missing_evidence():
	registry = RegistryService()

	with pytest.raises(PermissionError, match="service_owner_required"):
		registry.register_service(
			service_id="orders",
			tenant_id="tenant-a",
			name="orders",
			owner="",
			service_type="rest_api",
			environment="development",
			api_version="1.0.0",
			contract_schema_ref="schemas/orders.yaml",
			health_endpoint="/health",
		)

	registry.register_service(
		service_id="orders",
		tenant_id="tenant-a",
		name="orders",
		owner="platform",
		service_type="rest_api",
		environment="development",
		api_version="1.0.0",
		contract_schema_ref="schemas/orders.yaml",
		health_endpoint="/health",
	)

	with pytest.raises(PermissionError, match="instance_endpoint_required"):
		registry.register_instance("orders-1", "tenant-a", "orders", "", "local", "/health")

	with pytest.raises(PermissionError, match="healthy_instance_required"):
		registry.publish_to_gateway("orders-public", "tenant-a", "orders", "/orders")


def test_registry_runtime_records_reviews_and_retirement_evidence():
	registry = RegistryService()
	service = registry.register_service(
		service_id="orders",
		tenant_id="tenant-a",
		name="orders",
		owner="platform",
		service_type="rest_api",
		environment="production",
		api_version="1.0.0",
		contract_schema_ref="schemas/orders.yaml",
		health_endpoint="/health",
		production_review_recorded=False,
		trace_propagation_configured=True,
	)
	version = registry.record_version(
		version_id="orders-2",
		tenant_id="tenant-a",
		service_id="orders",
		version="2.0.0",
		contract_schema_ref="schemas/orders-v2.yaml",
		breaking_change_detected=True,
		compatibility_review_recorded=False,
	)

	assert service["status"] == "pending_review"
	assert version["status"] == "pending_review"
	assert len(registry.list_reviews("tenant-a")) == 2
	registry.register_instance("orders-1", "tenant-a", "orders", "https://orders.internal", "local", "/health")
	with pytest.raises(PermissionError, match="service_review_required"):
		registry.publish_to_gateway("orders-public", "tenant-a", "orders", "/orders")
	discovery = registry.discover_services("tenant-a", requested_result_limit=1500)
	assert discovery["decision"] == "require_review"
	assert any(review["review_type"] == "discovery_limit" for review in registry.list_reviews("tenant-a"))
	owner_review = registry.transfer_owner("tenant-a", "orders", "api-platform", "platform", owner_transfer_review_recorded=False)
	assert owner_review["review_type"] == "owner_transfer"
	transferred = registry.transfer_owner("tenant-a", "orders", "api-platform", "platform", owner_transfer_review_recorded=True)
	assert transferred["owner"] == "api-platform"
	with pytest.raises(PermissionError, match="impact_review_required"):
		registry.retire_service("tenant-a", "orders", "platform", False, True)
	retired = registry.retire_service("tenant-a", "orders", "platform", True, True)
	assert retired["status"] == "retired"


def test_registry_runtime_governs_agents_and_lifecycle_batches():
	registry = RegistryService()

	with pytest.raises(PermissionError, match="unsupported_registry_agent_runtime"):
		registry.register_registry_agent(
			agent_id="unknown-agent",
			tenant_id="tenant-a",
			name="Unknown Agent",
			runtime="unsupported",
			role="contract_reviewer",
			scope="service contracts",
			owner="platform",
			purpose="review registry contracts",
		)

	pending = registry.register_registry_agent(
		agent_id="contract-agent",
		tenant_id="tenant-a",
		name="Contract Agent",
		runtime="Claude Code",
		role="contract reviewer",
		scope="service contracts",
		owner="platform",
		purpose="review registry contracts",
		human_approval_required=False,
	)
	active = registry.register_registry_agent(
		agent_id="catalog-agent",
		tenant_id="tenant-a",
		name="Catalog Agent",
		runtime="codex",
		role="catalog_steward",
		scope="catalog hygiene",
		owner="registry-office",
		purpose="maintain service catalog metadata",
	)

	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		registry.validate_regy_lifecycle_batch("tenant-a", "legacy_queue", 2)
	batch = registry.validate_regy_lifecycle_batch("tenant-a", "bytewax", 4)
	summary = registry.registry_summary("tenant-a")

	assert pending["status"] == "pending_review"
	assert pending["runtime"] == "claude_code"
	assert active["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["registry_agent_count"] == 2
	assert summary["lifecycle_batch_count"] == 2
	assert summary["denied_lifecycle_batch_count"] == 1


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "regy_service_catalog"
	assert registration["ui_components"]["discovery"] == "/regy/discovery"
	assert "apig" in registration["dependencies"]
	assert "auth" in registration["dependencies"]
	assert "cach" in registration["optional_dependencies"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"


def test_generated_ui_models_are_composable():
	registry = RegistryService()
	registry.register_service(
		service_id="orders",
		tenant_id="tenant-a",
		name="orders",
		owner="platform",
		service_type="rest_api",
		environment="development",
		api_version="1.0.0",
		contract_schema_ref="schemas/orders.yaml",
		health_endpoint="/health",
		routing_metadata={"path": "/orders"},
	)
	dashboard = dashboard_model(registry, "tenant-a")
	catalog = service_catalog_model(registry, "tenant-a")
	discovery = discovery_console_model(registry, "tenant-a")
	registry.register_registry_agent(
		agent_id="catalog-agent",
		tenant_id="tenant-a",
		name="Catalog Agent",
		runtime="codex",
		role="catalog_steward",
		scope="catalog hygiene",
		owner="registry-office",
		purpose="maintain service catalog metadata",
	)
	registry.validate_regy_lifecycle_batch("tenant-a", "bytewax", 1)
	agents = registry_agent_roster_model(registry, "tenant-a")
	batches = lifecycle_batch_model(registry, "tenant-a")

	assert dashboard["summary"]["service_count"] == 1
	assert catalog["services"][0]["name"] == "orders"
	assert discovery["defaults"]["service_discovery_enabled"] is True
	assert agents["agents"][0]["runtime"] == "codex"
	assert batches["required_processor"] == "bytewax"
