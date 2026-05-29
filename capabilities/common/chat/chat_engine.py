"""Deterministic message helpers for APG CHAT."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class ChatEngine:
	"""Build stable message fingerprints and lightweight delivery metadata."""

	def digest(self, payload: dict[str, Any]) -> str:
		canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
		return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

	def message_fingerprint(self, payload: dict[str, Any]) -> str:
		return self.digest({"kind": "chat.message", "payload": payload})

	def thread_key(self, room_id: str, sender: str, body: str) -> str:
		return self.digest({"room_id": room_id, "sender": sender, "body": body})[:16]

	def restricted_terms(self, body: str, restricted_terms: tuple[str, ...]) -> tuple[str, ...]:
		lowered = body.lower()
		return tuple(term for term in restricted_terms if term.lower() in lowered)
