"""NATS routing adapter for Policy Administration (ins_pol)."""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def get_audit_adapter(capability_id: str = "ins_pol"):
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None
