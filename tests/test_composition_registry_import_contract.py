"""Composition registry import and model mapping regressions."""

from __future__ import annotations


def test_composition_registry_package_imports_core_runtime_surfaces():
	from capabilities.composition.registry import CRCapability, CRService, MobileOfflineService
	from capabilities.composition.registry.service import CapabilityRegistryService

	assert CRService is CapabilityRegistryService
	assert MobileOfflineService.__name__ == "MobileOfflineService"
	assert CRCapability.__tablename__ == "cr_capabilities"


def test_composition_registry_metadata_column_keeps_legacy_instance_access():
	from capabilities.composition.registry.models import CRCapability, CRRegistry

	capability = CRCapability(
		tenant_id="tenant-a",
		capability_code="AI_AGENT",
		capability_name="AI Agent",
		description="Composable AI agent runtime",
		version="1.0.0",
		category="composition",
		created_by="user-a",
		metadata_json={"protocols": ["apg.agent"], "dependencies": ["nlpc"]},
	)

	assert "metadata" in CRCapability.__table__.c
	assert "metadata_json" not in CRCapability.__table__.c
	assert capability.metadata == {"protocols": ["apg.agent"], "dependencies": ["nlpc"]}

	capability.metadata = {"providers": ["codex", "claude-code", "opencode"]}
	assert capability.metadata_json == {"providers": ["codex", "claude-code", "opencode"]}

	registry = CRRegistry(
		tenant_id="tenant-a",
		name="Tenant registry",
		created_by="user-a",
		metadata_json={"source": "test"},
	)

	assert registry.metadata == {"source": "test"}
