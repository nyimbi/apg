"""Composition registry marketplace transport regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capabilities.composition.registry.marketplace import (
	LicenseType,
	MarketplaceIntegration,
	MarketplaceMetadata,
	MarketplaceStatus,
	PublicationPackage,
	PublicationType,
	QualityLevel,
)


class _MarketplaceClient:
	def __init__(self) -> None:
		self.calls: list[dict] = []

	async def __call__(self, *, endpoint: str, method: str, data: dict) -> dict:
		self.calls.append({"endpoint": endpoint, "method": method, "data": data})
		if endpoint == "updates":
			return {
				"capabilities": [{"capability_id": "cap-ai-agent"}],
				"templates": [{"template_id": "template-agent"}],
				"last_updated": "2026-05-27T04:45:00",
			}
		return {
			"status": "accepted",
			"external_submission_id": data["submission_id"],
		}


class _MarketplaceUpdateClient:
	async def __call__(self, *, endpoint: str, method: str, data: dict) -> dict:
		assert endpoint == "updates"
		assert method == "GET"
		return {
			"capabilities": [{
				"capability_id": "cap-ai-agent",
				"name": "AI Agent Runtime",
				"description": "Updated marketplace description",
				"latest_version": "1.2.0",
				"quality_score": 0.98,
				"status": "active",
				"composition_keywords": ["agent", "codex", "claude-code"],
				"release_notes": "Adds provider-aware orchestration.",
			}],
			"templates": [{
				"template_id": "template-agent",
				"name": "Agent Suite",
				"description": "Updated agent suite template",
				"latest_version": "2.0.0",
				"configuration": {"agents": ["planner", "executor"]},
			}],
			"last_updated": "2026-05-28T10:57:00",
		}


class _Result:
	def __init__(self, value):
		self.value = value

	def scalar_one_or_none(self):
		return self.value


class _MarketplaceSession:
	def __init__(self):
		self.capability = SimpleNamespace(
			capability_id="cap-ai-agent",
			capability_code="AI_AGENT",
			capability_name="AI Agent",
			description="Old description",
			version="1.0.0",
			quality_score=0.5,
			composition_keywords=[],
			provides_services=[],
			data_models=[],
			api_endpoints=[],
			target_users=[],
			use_cases=[],
			industry_focus=[],
			metadata={},
		)
		self.template = SimpleNamespace(
			composition_id="template-agent",
			name="Old Agent Template",
			description="Old template",
			version="1.0.0",
			is_template=True,
			metadata={},
		)
		self.registry = SimpleNamespace(metadata={}, updated_at=None, updated_by=None)
		self._results = [self.capability, self.template, self.registry]
		self.added = []
		self.commits = 0

	async def execute(self, _query):
		return _Result(self._results.pop(0))

	def add(self, value):
		self.added.append(value)

	async def commit(self):
		self.commits += 1


def _integration(client: _MarketplaceClient) -> MarketplaceIntegration:
	return MarketplaceIntegration(
		db_session=None,
		tenant_id="tenant-a",
		user_id="user-a",
		marketplace_api_client=client,
	)


def _package() -> PublicationPackage:
	metadata = MarketplaceMetadata(
		publication_id="pub-ai-agent",
		publication_type=PublicationType.CAPABILITY,
		title="AI Agent Composition",
		description="Composable AI agent capability",
		long_description="Composable AI agent capability for APG.",
		tags=["agent", "ai"],
		categories=["composition"],
		license_type=LicenseType.MIT,
		license_url=None,
		pricing_model="free",
		price=0.0,
		currency="USD",
		author_name="APG",
		author_email="apg@example.com",
		author_organization="Datacraft",
		support_url=None,
		documentation_url=None,
		repository_url=None,
		demo_url=None,
		screenshots=[],
		videos=[],
		requirements={},
		compatibility={},
		quality_level=QualityLevel.STABLE,
		marketplace_status=MarketplaceStatus.DRAFT,
	)
	return PublicationPackage(
		metadata=metadata,
		capability_data={"capability_id": "cap-ai-agent", "complexity_score": 1.0},
		composition_data=None,
		documentation={"README.md": "# AI Agent Composition"},
		assets=[],
		validation_results={"passed": True},
		quality_score=0.95,
		compliance_check={"passed": True},
	)


@pytest.mark.asyncio
async def test_marketplace_submission_uses_injected_transport_response():
	client = _MarketplaceClient()
	integration = _integration(client)

	result = await integration.submit_to_marketplace(_package())

	assert result["success"] is True
	assert len(client.calls) == 1
	call = client.calls[0]
	assert call["endpoint"] == "submissions"
	assert call["method"] == "POST"
	assert call["data"]["tenant_id"] == "tenant-a"
	assert call["data"]["submitted_by"] == "user-a"
	assert call["data"]["package"]["metadata"]["title"] == "AI Agent Composition"
	assert result["data"]["marketplace_response"] == {
		"status": "accepted",
		"external_submission_id": call["data"]["submission_id"],
	}


@pytest.mark.asyncio
async def test_marketplace_sync_fetches_updates_through_transport():
	client = _MarketplaceClient()
	integration = _integration(client)

	updates = await integration._fetch_marketplace_updates()

	assert client.calls == [{
		"endpoint": "updates",
		"method": "GET",
		"data": {"tenant_id": "tenant-a"},
	}]
	assert updates == {
		"capabilities": [{"capability_id": "cap-ai-agent"}],
		"templates": [{"template_id": "template-agent"}],
		"last_updated": "2026-05-27T04:45:00",
	}


@pytest.mark.asyncio
async def test_marketplace_sync_applies_capability_and_template_updates():
	session = _MarketplaceSession()
	integration = MarketplaceIntegration(
		db_session=session,
		tenant_id="tenant-a",
		user_id="user-a",
		marketplace_api_client=_MarketplaceUpdateClient(),
	)

	result = await integration.sync_with_marketplace()

	assert result["success"] is True
	assert session.capability.capability_name == "AI Agent Runtime"
	assert session.capability.version == "1.2.0"
	assert session.capability.quality_score == 0.98
	assert session.capability.composition_keywords == ["agent", "codex", "claude-code"]
	assert session.capability.metadata["marketplace"]["latest_version"] == "1.2.0"
	assert session.template.name == "Agent Suite"
	assert session.template.version == "2.0.0"
	assert session.template.configuration == {"agents": ["planner", "executor"]}
	assert session.registry.metadata["marketplace_cached_capability_updates"] == 1
	assert session.registry.metadata["marketplace_cached_template_updates"] == 1
	assert session.commits == 1
