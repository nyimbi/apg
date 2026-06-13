"""NATS JetStream event adapter for APG.

Implements the AuditAdapter protocol using NATS JetStream for durable
event publishing. Drop-in replacement for NullAuditAdapter — same interface,
crash-resilient message delivery.

Activated via NATS_URL environment variable. Falls back to NullAuditAdapter
when NATS is not available so standalone operation is never broken.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .subject_registry import subject_for
from .stream_setup import setup_apg_stream

_log = logging.getLogger(__name__)

_nats_client: Any = None
_nats_js: Any = None
_setup_lock = asyncio.Lock()


async def _get_js() -> Any:
	"""Return a shared NATS JetStream context, connecting on first use."""
	global _nats_client, _nats_js
	if _nats_js is not None:
		return _nats_js

	async with _setup_lock:
		if _nats_js is not None:
			return _nats_js
		try:
			import nats  # type: ignore[import]
			url = os.environ.get("NATS_URL", "nats://localhost:4222")
			_nats_client = await nats.connect(
				url,
				name="apg-platform",
				connect_timeout=5,
				reconnect_time_wait=2,
				max_reconnect_attempts=10,
				error_cb=_on_error,
				disconnected_cb=_on_disconnected,
				reconnected_cb=_on_reconnected,
			)
			_nats_js = _nats_client.jetstream()
			await setup_apg_stream(_nats_js)
			_log.info("Connected to NATS at %s", url)
		except Exception as exc:
			_log.error("Failed to connect to NATS: %s", exc)
			raise
	return _nats_js


async def _on_error(exc: Exception) -> None:
	_log.error("NATS error: %s", exc)


async def _on_disconnected() -> None:
	_log.warning("NATS disconnected")


async def _on_reconnected() -> None:
	_log.info("NATS reconnected")


class NATSEventAdapter:
	"""AuditAdapter implementation that publishes events to NATS JetStream.

	Subject format: apg.events.{capability_id}.{event_type}

	Events are published with:
	- Msg-Id header for exactly-once deduplication (2-minute window)
	- Retry on transient publish failures (up to 3 attempts)
	"""

	def __init__(self, capability_id: str = "platform") -> None:
		self._capability_id = capability_id

	async def publish_event(self, event: "IntegrationEvent") -> None:
		"""Publish a typed IntegrationEvent envelope to JetStream."""
		from .events import IntegrationEvent  # local import avoids circular deps
		data = json.dumps(event.model_dump(mode="json"), default=str).encode()
		for attempt in range(3):
			try:
				js = await _get_js()
				await js.publish(event.subject(), data, headers={"Msg-Id": event.msg_id()})
				return
			except Exception as exc:
				if attempt == 2:
					_log.error("Failed to publish IntegrationEvent %s after 3 attempts: %s", event.subject(), exc)
				else:
					await asyncio.sleep(0.1 * (attempt + 1))

	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		subject = subject_for(self._capability_id, event_type)
		payload = {
			"event_type": event_type,
			"capability_id": self._capability_id,
			"actor_id": actor_id,
			"tenant_id": tenant_id,
			"resource_id": resource_id,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"details": details,
		}
		msg_id = f"{tenant_id}-{resource_id}-{event_type}-{payload['timestamp']}"
		data = json.dumps(payload, default=str).encode()

		for attempt in range(3):
			try:
				js = await _get_js()
				await js.publish(subject, data, headers={"Msg-Id": msg_id})
				return
			except Exception as exc:
				if attempt == 2:
					_log.error(
						"Failed to publish NATS event %s after 3 attempts: %s",
						subject, exc,
					)
				else:
					await asyncio.sleep(0.1 * (attempt + 1))

	async def subscribe(
		self,
		subject_pattern: str,
		handler: Callable[[dict[str, Any]], Any],
		consumer_name: str | None = None,
	) -> Any:
		"""Subscribe to a NATS subject pattern and call handler on each message."""
		js = await _get_js()
		consumer = consumer_name or f"apg-{subject_pattern.replace('.', '-').replace('*', 'all').replace('>', 'all')}"

		async def _msg_handler(msg: Any) -> None:
			try:
				payload = json.loads(msg.data.decode())
				await handler(payload)
				await msg.ack()
			except Exception as exc:
				_log.error("NATS message handler failed for %s: %s", msg.subject, exc)
				await msg.nak()

		sub = await js.subscribe(subject_pattern, durable=consumer, cb=_msg_handler)
		return sub


class NATSConnector:
	"""BaseConnector-compatible NATS connector for direct publish/subscribe.

	Used by capabilities that need event-driven communication beyond
	the audit adapter pattern (e.g. ckm_rtc for real-time collaboration).
	"""

	def __init__(self, capability_id: str) -> None:
		self._capability_id = capability_id
		self._js: Any = None

	async def connect(self) -> None:
		self._js = await _get_js()

	async def publish_event(self, event: "IntegrationEvent") -> None:
		"""Publish a typed IntegrationEvent envelope."""
		if self._js is None:
			await self.connect()
		data = json.dumps(event.model_dump(mode="json"), default=str).encode()
		await self._js.publish(event.subject(), data, headers={"Msg-Id": event.msg_id()})

	async def publish(self, event_type: str, tenant_id: str, payload: dict[str, Any]) -> None:
		if self._js is None:
			await self.connect()
		subject = subject_for(self._capability_id, event_type)
		data = json.dumps({
			"capability_id": self._capability_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"timestamp": datetime.now(timezone.utc).isoformat(),
			**payload,
		}, default=str).encode()
		await self._js.publish(subject, data)

	async def subscribe(
		self,
		event_type: str,
		handler: Callable[[dict[str, Any]], Any],
		durable_name: str | None = None,
	) -> Any:
		if self._js is None:
			await self.connect()
		subject = subject_for(self._capability_id, event_type)
		durable = durable_name or f"{self._capability_id}-{event_type}"

		async def _handler(msg: Any) -> None:
			try:
				await handler(json.loads(msg.data.decode()))
				await msg.ack()
			except Exception as exc:
				_log.error("NATS handler error: %s", exc)
				await msg.nak()

		return await self._js.subscribe(subject, durable=durable, cb=_handler)


def get_nats_audit_adapter(capability_id: str = "platform") -> NATSEventAdapter | None:
	"""Return NATSEventAdapter if NATS_URL is configured, else None.

	Callers fall back to NullAuditAdapter when None is returned.
	"""
	if os.environ.get("NATS_URL"):
		return NATSEventAdapter(capability_id)
	return None
