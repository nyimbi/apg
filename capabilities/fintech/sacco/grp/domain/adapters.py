"""Domain adapters for SACCO Group Lending.

© 2025 Datacraft — Author: Nyimbi Odero
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def get_audit_adapter(capability_id: str = "fintech_sacco_grp"):
	"""Return a NATS audit adapter if NATS_URL is configured, else None."""
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None
