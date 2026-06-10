"""NATS adapter for Cooperative Management capability."""
from __future__ import annotations
import logging, os
_log = logging.getLogger(__name__)

def get_audit_adapter(capability_id: str = "agr_coo"):
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None
