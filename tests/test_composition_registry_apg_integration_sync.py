"""APG integration synchronization regressions for the composition registry."""

from __future__ import annotations

import pytest

from capabilities.composition.registry.apg_integration import (
	APGCapabilityMetadata,
	APGCompositionConfig,
	APGDiscoveryRegistration,
	APGIntegrationService,
)


@pytest.mark.asyncio
async def test_apg_integration_records_capability_discovery_and_composition_syncs():
	service = APGIntegrationService("tenant-a")
	capability = APGCapabilityMetadata(
		capability_id="cap-ai-agent",
		capability_code="AI_AGENT",
		capability_name="AI Agent",
		provides_interfaces=["/api/agents"],
		runtime_config={"multi_tenant": True},
	)
	discovery = APGDiscoveryRegistration(
		capability_id="cap-ai-agent",
		apg_tenant_id="tenant-a",
		discovery_tags=["composition", "ai"],
		search_keywords=["agent", "codex"],
		category_hierarchy=["composition", "agents"],
		service_endpoints=[{"path": "/api/agents", "protocol": "http"}],
	)
	composition = APGCompositionConfig(
		composition_id="comp-agent-suite",
		name="Agent Suite",
		capability_bindings=[{"capability_id": "cap-ai-agent"}],
		service_mappings={"ai_agent": "apg.tenant-a.ai_agent"},
		execution_order=["cap-ai-agent"],
	)

	await service._register_apg_capability(capability)
	await service._register_apg_discovery(discovery)
	await service._register_apg_composition(composition)
	service.registered_capabilities.add("cap-ai-agent")
	await service._sync_discovery_metadata()
	await service._sync_composition_configs()

	assert service.capability_metadata["cap-ai-agent"] is capability
	assert service.discovery_registrations["cap-ai-agent"] is discovery
	assert service.active_compositions["comp-agent-suite"] is composition
	assert service.apg_config["discovery_sync"]["registration_count"] == 1
	assert service.apg_config["discovery_sync"]["registrations"][0]["service_count"] == 1
	assert service.apg_config["composition_sync"]["composition_count"] == 1
	assert service.apg_config["composition_sync"]["registered_capability_count"] == 1
	assert service.apg_config["composition_sync"]["compositions"][0]["binding_count"] == 1
	assert [entry["type"] for entry in service.sync_history] == [
		"discovery_metadata",
		"composition_configs",
	]


def test_apg_integration_no_longer_contains_sync_placeholders():
	import inspect
	from capabilities.composition.registry import apg_integration

	source = inspect.getsource(apg_integration.APGIntegrationService)
	sync_source = source[source.index("async def _sync_discovery_metadata"):]

	assert "pass" not in sync_source
	assert "In production, would sync with APG" not in sync_source
