"""Deterministic sealing helpers for APG ESGN."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class SigningEngine:
	"""Pure hashing helpers for submissions, envelopes, signatures, and evidence."""

	def validation_hash(self, schema: dict[str, Any], payload: dict[str, Any]) -> str:
		return self._stable_digest({"schema": schema, "payload": payload})

	def tamper_seal(self, submission: dict[str, Any], recipients: list[dict[str, Any]]) -> str:
		return self._stable_digest({"submission": submission, "recipients": recipients})

	def signature_hash(self, envelope_id: str, recipient_id: str, signature_intent: str) -> str:
		return self._stable_digest({
			"envelope_id": envelope_id,
			"recipient_id": recipient_id,
			"signature_intent": signature_intent,
		})

	def evidence_hash(self, envelope: dict[str, Any], ceremonies: list[dict[str, Any]]) -> str:
		return self._stable_digest({"envelope": envelope, "ceremonies": ceremonies})

	def certificate_id(self, envelope_id: str, audit_hash: str) -> str:
		return f"cert:{self._stable_digest({'envelope_id': envelope_id, 'audit_hash': audit_hash})[:16]}"

	def _stable_digest(self, payload: dict[str, Any]) -> str:
		encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
		return hashlib.sha256(encoded).hexdigest()
