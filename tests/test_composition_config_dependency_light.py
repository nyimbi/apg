"""Dependency-light executable checks for central configuration composition."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from capabilities.composition.config.service import (
	ConfigFormat,
	ConfigValue,
	RedisConfigStorage,
)
from capabilities.composition.config.integrations import enterprise_connectors as connectors


class InMemoryRedis:
	def __init__(self) -> None:
		self.values: dict[str, str] = {}

	async def get(self, key: str) -> str | None:
		return self.values.get(key)

	async def set(self, key: str, value: Any) -> bool:
		self.values[key] = str(value)
		return True

	async def setex(self, key: str, ttl: int, value: Any) -> bool:
		self.values[key] = str(value)
		return True

	async def delete(self, *keys: str) -> int:
		deleted = 0
		for key in keys:
			if key in self.values:
				deleted += 1
				del self.values[key]
		return deleted

	async def scan_iter(self, match: str):
		prefix = match.removesuffix("*")
		for key in list(self.values):
			if key.startswith(prefix):
				yield key


def test_config_redis_storage_runs_without_external_redis_package():
	async def scenario() -> None:
		storage = RedisConfigStorage(InMemoryRedis())
		value = ConfigValue(
			value={"enabled": True},
			raw_value=json.dumps({"enabled": True}),
			format=ConfigFormat.JSON,
			encrypted=False,
			version=1,
			checksum="checksum",
			expires_at=None,
			metadata={"source": "test"},
		)

		version = await storage.set("tenant:feature", value)
		loaded = await storage.get("tenant:feature")

		assert version == 1
		assert loaded is not None
		assert loaded.raw_value == '{"enabled": true}'
		assert loaded.format == ConfigFormat.JSON
		assert loaded.metadata == {"source": "test"}

	asyncio.run(scenario())


def test_enterprise_event_contract_normalizes_current_dataclass_shape():
	manager = connectors.EnterpriseIntegrationManager(config_engine=object())

	assert manager._event_processor_task is None
	assert {"slack", "teams", "github", "azure_ad"}.issubset(manager.connectors)

	event = connectors.IntegrationEvent(
		event_id="event_test",
		event_type="security_alert",
		timestamp=datetime.now(timezone.utc),
		source="config-service",
		data={"message": "policy drift detected", "policy": "zero-trust"},
		metadata={},
		severity="high",
	)

	assert event.event_type is connectors.EventType.SECURITY_ALERT
	assert event.severity is connectors.EventSeverity.ERROR
	assert event.source_service == "config-service"
	assert event.message == "policy drift detected"
	assert connectors.DiscordConnector()._get_event_title(event) == "Security Alert"
	assert connectors.SlackConnector()._format_slack_message(event)["attachments"][0]["fields"][0]["value"] == "HIGH"


def test_base_connector_sends_generic_webhook_payload(monkeypatch):
	posted: dict[str, Any] = {}

	class Response:
		def raise_for_status(self) -> None:
			return None

	class AsyncClient:
		def __init__(self, *args, **kwargs) -> None:
			pass

		async def __aenter__(self):
			return self

		async def __aexit__(self, *args) -> None:
			return None

		async def post(self, url: str, json: dict[str, Any]) -> Response:
			posted["url"] = url
			posted["json"] = json
			return Response()

	monkeypatch.setattr(connectors.httpx, "AsyncClient", AsyncClient)

	async def scenario() -> None:
		integration = connectors.IntegrationConfig(
			integration_id="integration_test",
			name="Generic Webhook",
			integration_type=connectors.IntegrationType.WORKFLOW,
			platform="custom",
			enabled=True,
			config={"webhook_url": "https://example.test/hook"},
			event_filters=[connectors.EventType.CONFIGURATION_UPDATED],
			rate_limit_per_minute=60,
			retry_config={},
			webhook_url=None,
			authentication={},
			created_at=datetime.now(timezone.utc),
			last_used=None,
		)
		event = connectors.IntegrationEvent(
			event_id="event_test",
			event_type=connectors.EventType.CONFIGURATION_UPDATED,
			timestamp=datetime.now(timezone.utc),
			source="config-service",
			data={"message": "changed"},
			metadata={"tenant_id": "tenant_a"},
			severity=connectors.EventSeverity.MEDIUM,
		)

		await connectors.BaseConnector().send_event(integration, event)

	asyncio.run(scenario())

	assert posted["url"] == "https://example.test/hook"
	assert posted["json"]["event_type"] == "configuration_updated"
	assert posted["json"]["severity"] == "medium"
	assert posted["json"]["source"] == "config-service"
