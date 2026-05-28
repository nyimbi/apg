"""Composition registry mobile/offline sync regressions."""

from __future__ import annotations

import json
import sqlite3
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


class _CompositionWriter:
	def __init__(self, *, fail: bool = False) -> None:
		self.fail = fail
		self.created: list[dict] = []

	async def create_composition(self, **kwargs):
		self.created.append(kwargs)
		if self.fail:
			return {"success": False, "message": "composition rejected", "errors": ["invalid capability"]}
		return {
			"success": True,
			"data": {"composition_id": "online-agent-stack"},
		}


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


@pytest.mark.asyncio
async def test_mobile_sync_without_online_service_preserves_offline_cache(tmp_path):
	service = MobileOfflineService(
		tenant_id="tenant-a",
		offline_db_path=str(tmp_path / "offline.db"),
	)
	await service._store_synced_capabilities([
		{
			"capability_id": "cap-offline",
			"capability_code": "OFFLINE",
			"capability_name": "Offline Capability",
			"description": "Cached capability",
			"category": "common",
			"version": "1.0.0",
			"quality_score": 0.8,
			"status": "active",
		}
	], full_sync=True)

	result = await service.sync_from_online(force_full_sync=True)

	assert result["success"] is False
	assert result["offline_preserved"] is True
	assert result["offline_counts"]["capabilities"] == 1
	capabilities = await service.get_mobile_capabilities(limit=10)
	assert [cap.capability_id for cap in capabilities] == ["cap-offline"]


@pytest.mark.asyncio
async def test_mobile_offline_composition_action_calls_online_registry_and_completes(tmp_path):
	registry = _CompositionWriter()
	service = MobileOfflineService(
		tenant_id="tenant-a",
		offline_db_path=str(tmp_path / "offline.db"),
	)
	service.is_online = False
	await service.set_online_service(registry)

	local_id = await service.create_mobile_composition(
		name="Agent Stack",
		description="Offline-created agent stack",
		capability_ids=["cap-ai-agent", "cap-nlpc"],
		composition_type="agent_composition",
	)
	assert len(await service.get_pending_offline_actions()) == 1

	service.is_online = True
	result = await service.sync_offline_actions()

	assert result["synced"] == 1
	assert result["failed"] == 0
	assert await service.get_pending_offline_actions() == []
	assert registry.created == [{
		"name": "Agent Stack",
		"description": "Offline-created agent stack",
		"capability_ids": ["cap-ai-agent", "cap-nlpc"],
		"composition_type": "agent_composition",
		"configuration": None,
	}]

	compositions = await service.get_mobile_compositions(limit=10)
	assert compositions[0].composition_id == local_id
	detail = compositions[0]
	assert detail.is_offline_ready is True

	conn = sqlite3.connect(service.offline_db_path)
	cursor = conn.cursor()
	cursor.execute("SELECT status FROM offline_actions")
	assert cursor.fetchone()[0] == "completed"
	cursor.execute("SELECT data_json FROM offline_compositions WHERE composition_id = ?", (local_id,))
	data = json.loads(cursor.fetchone()[0])
	conn.close()
	assert data["synced"] is True
	assert data["created_offline"] is False
	assert data["online_composition_id"] == "online-agent-stack"


@pytest.mark.asyncio
async def test_mobile_offline_action_retry_state_preserves_failed_online_sync(tmp_path):
	registry = _CompositionWriter(fail=True)
	service = MobileOfflineService(
		tenant_id="tenant-a",
		offline_db_path=str(tmp_path / "offline.db"),
	)
	service.is_online = False
	await service.set_online_service(registry)
	await service.create_mobile_composition(
		name="Bad Stack",
		description="Rejected composition",
		capability_ids=["missing-capability"],
		composition_type="agent_composition",
	)

	service.is_online = True
	result = await service.sync_offline_actions()

	assert result["synced"] == 0
	assert result["failed"] == 1
	pending = await service.get_pending_offline_actions()
	assert len(pending) == 1
	assert pending[0].retry_count == 1
