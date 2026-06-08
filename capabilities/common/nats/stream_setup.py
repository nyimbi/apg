"""NATS JetStream stream provisioning for APG."""
from __future__ import annotations

import logging
from typing import Any

from .subject_registry import APG_STREAM_NAME, APG_SUBJECT_PREFIX

_log = logging.getLogger(__name__)


async def setup_apg_stream(js: Any, replicas: int = 1) -> None:
	"""Create or update the APG_EVENTS JetStream stream.

	Safe to call on every startup — uses add_stream which is idempotent
	when the config matches.

	Args:
		js: NATS JetStream context (from nc.jetstream())
		replicas: Number of replicas (1 for dev, 3 for k8s production)
	"""
	try:
		from nats.js.api import StreamConfig, RetentionPolicy, StorageType

		config = StreamConfig(
			name=APG_STREAM_NAME,
			subjects=[f"{APG_SUBJECT_PREFIX}.>"],
			retention=RetentionPolicy.LIMITS,
			storage=StorageType.FILE,
			replicas=replicas,
			max_age=90 * 24 * 60 * 60 * 1_000_000_000,  # 90 days in nanoseconds
			max_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB
			max_msg_size=1 * 1024 * 1024,  # 1 MiB per message
			duplicate_window=2 * 60 * 1_000_000_000,  # 2-minute dedup window
		)
		await js.add_stream(config)
		_log.info("NATS JetStream stream %s ready", APG_STREAM_NAME)
	except Exception as exc:
		# Stream already exists with same config — tolerate
		if "already in use" in str(exc).lower() or "stream name already in use" in str(exc).lower():
			_log.debug("NATS stream %s already exists — skipping setup", APG_STREAM_NAME)
		else:
			_log.warning("NATS stream setup failed: %s", exc)
			raise
