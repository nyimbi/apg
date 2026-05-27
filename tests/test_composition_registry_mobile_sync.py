"""Composition registry mobile/offline sync regressions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from capabilities.composition.registry.mobile_service import MobileOfflineService


class _RegistryFeed:
	def __init__(self) -> None:
		self.capabilities = [
			{
				"capability_id": "cap-ai-agent",
				"capability_code": "AI_AGENT",
				"capability_name": "AI Agent",
				"description": "Composable agent runtime",
				"category": "composition",
				"version": "1.0.0",
				"quality_score": 0.97,
				"status": "active",
				"updated_at": datetime.utcnow().isoformat(),
				"providers": ["codex", "claude-code", "opencode"],
			},
			{
				"capability_id": "cap-nlpc",
				"capability_code": "NLPC",
				"capability_name": "Natural Language Processing Core",
				"description": "NLP capability",
				"category": "common",
				"version": "2.0.0",
				"quality_score": 0.91,
				"status": "active",
				"updated_at": datetime.utcnow().isoformat(),
			},
		]
		self.compositions = [
			{
				"composition_id": "comp-agent-stack",
				"name": "Agent Stack",
				"description": "Agent composition with NLPC",
				"composition_type": "agent_composition",
				"capability_ids": ["cap-ai-agent", "cap-nlpc"],
				"validation_results": {"validation_score": 0.93},
				"validation_status": "valid",
				"updated_at": datetime.utcnow().isoformat(),
			}
		]

	async def search_capabilities(self, **kwargs):
		assert kwargs["limit"] == 1000
		return {"capabilities": self.capabilities}

	async def search_compositions(self, query):
		assert query["limit"] == 1000
		return {"compositions": self.compositions}


@pytest.mark.asyncio
async def test_mobile_full_sync_uses_online_registry_feed_instead_of_mock_rows(tmp_path):
	registry = _RegistryFeed()
	service = MobileOfflineService(
		tenant_id="tenant-a",
		offline_db_path=str(tmp_path / "offline.db"),
	)
	await service.set_online_service(registry)

	result = await service.sync_from_online(force_full_sync=True)

	assert result["success"] is True
	assert result["capabilities_synced"] == 2
	assert result["compositions_synced"] == 1

	capabilities = await service.get_mobile_capabilities(limit=10)
	assert [cap.code for cap in capabilities] == ["AI_AGENT", "NLPC"]
	assert {cap.capability_id for cap in capabilities}.isdisjoint({"cap_001", "cap_002", "cap_003"})

	detail = await service.get_mobile_capability_detail("cap-ai-agent")
	assert detail["providers"] == ["codex", "claude-code", "opencode"]

	compositions = await service.get_mobile_compositions(limit=10)
	assert len(compositions) == 1
	assert compositions[0].composition_id == "comp-agent-stack"
	assert compositions[0].capability_count == 2
	assert compositions[0].validation_score == 0.93


@pytest.mark.asyncio
async def test_mobile_incremental_sync_upserts_only_changed_online_records(tmp_path):
	registry = _RegistryFeed()
	service = MobileOfflineService(
		tenant_id="tenant-a",
		offline_db_path=str(tmp_path / "offline.db"),
	)
	await service.set_online_service(registry)
	await service.sync_from_online(force_full_sync=True)

	registry.capabilities = [
		{
			"capability_id": "cap-ai-agent",
			"capability_code": "AI_AGENT",
			"capability_name": "AI Agent Runtime",
			"description": "Updated agent runtime",
			"category": "composition",
			"version": "1.1.0",
			"quality_score": 0.99,
			"status": "active",
			"updated_at": (datetime.utcnow() + timedelta(seconds=1)).isoformat(),
		},
		{
			"capability_id": "cap-nlpc",
			"capability_code": "NLPC",
			"capability_name": "Natural Language Processing Core",
			"description": "Older unchanged NLP capability",
			"category": "common",
			"version": "2.0.0",
			"quality_score": 0.91,
			"status": "active",
			"updated_at": "2020-01-01T00:00:00",
		},
	]
	registry.compositions = []

	result = await service.sync_from_online(force_full_sync=False)

	assert result["success"] is True
	assert result["capabilities_synced"] == 1
	assert result["compositions_synced"] == 0

	capabilities = await service.get_mobile_capabilities(limit=10)
	by_id = {cap.capability_id: cap for cap in capabilities}
	assert by_id["cap-ai-agent"].name == "AI Agent Runtime"
	assert by_id["cap-ai-agent"].version == "1.1.0"
	assert by_id["cap-nlpc"].name == "Natural Language Processing Core"
