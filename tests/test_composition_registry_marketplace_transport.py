"""Composition registry marketplace transport regressions."""

from __future__ import annotations

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
