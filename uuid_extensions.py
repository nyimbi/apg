#!/usr/bin/env python3
"""UUIDv7 compatibility helpers used across APG capabilities.

APG code historically imports ``uuid7`` and ``uuid7str`` from a
``uuid_extensions`` module.  Python's stdlib does not provide UUIDv7 on all
supported versions, so this module supplies a compact RFC 9562-compatible
generator without adding another runtime dependency.
"""

from __future__ import annotations

import secrets
import time
import uuid


def uuid7() -> uuid.UUID:
	"""Return a time-ordered UUIDv7 value.

	The high 48 bits contain the Unix epoch timestamp in milliseconds, followed
	by the UUID version, variant bits, and random entropy.
	"""

	timestamp_ms = int(time.time_ns() // 1_000_000) & ((1 << 48) - 1)
	random_a = secrets.randbits(12)
	random_b = secrets.randbits(62)

	value = timestamp_ms << 80
	value |= 0x7 << 76
	value |= random_a << 64
	value |= 0b10 << 62
	value |= random_b

	return uuid.UUID(int=value)


def uuid7str() -> str:
	"""Return a UUIDv7 value as a canonical string."""

	return str(uuid7())


__all__ = ["uuid7", "uuid7str"]
