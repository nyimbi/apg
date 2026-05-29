"""Deterministic deployment helpers for the APG DEPL capability."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class DeploymentEngine:
	"""Pure helper functions for deployment evidence, rollout, and health state."""

	def stable_hash(self, payload: dict[str, Any]) -> str:
		encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
		return hashlib.sha256(encoded).hexdigest()

	def deployment_fingerprint(self, payload: dict[str, Any]) -> str:
		return f"depl-{self.stable_hash(payload)[:16]}"

	def health_status(self, checks: dict[str, bool], report_reference: str, log_trace_link: str) -> str:
		if not checks:
			return "failed"
		if not report_reference or not log_trace_link:
			return "failed"
		return "passed" if all(bool(value) for value in checks.values()) else "failed"

	def rollout_posture(self, strategy: str, canary_percent: int, max_canary_percent: int = 25) -> str:
		if strategy == "canary" and canary_percent > max_canary_percent:
			return "review_required"
		if strategy == "canary" and canary_percent <= 0:
			return "invalid"
		return "standard"
