"""NATS and adapter protocols for USSD Government Services (gov_usd)."""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


@runtime_checkable
class AuditAdapter(Protocol):
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
		resource_id: str, details: dict[str, Any]) -> None: ...


class NullAuditAdapter:
	"""Standalone fallback — logs to stderr."""
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
		resource_id: str, details: dict[str, Any]) -> None:
		_log.info("AUDIT [%s] actor=%s tenant=%s resource=%s", event_type, actor_id, tenant_id, resource_id)


def get_audit_adapter(capability_id: str = "gov_usd") -> AuditAdapter | None:
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	try:
		from apg_common_audl import AuditService  # type: ignore[import]
		return AuditService.from_env()
	except ImportError as _exc:
		_log.debug("Handled exception: %s", _exc)
	return NullAuditAdapter()


@runtime_checkable
class SMSAdapter(Protocol):
	async def send_sms(self, msisdn: str, message: str) -> dict[str, Any]: ...


class NullSMSAdapter:
	"""Standalone fallback — prints SMS to stdout."""
	async def send_sms(self, msisdn: str, message: str) -> dict[str, Any]:
		_log.info("SMS to %s: %s", msisdn, message[:80])
		return {"status": "sent", "msisdn": msisdn}


def get_sms_adapter() -> SMSAdapter:
	try:
		from apg_common_ntfy import SMSService  # type: ignore[import]
		return SMSService.from_env()
	except ImportError:
		return NullSMSAdapter()
