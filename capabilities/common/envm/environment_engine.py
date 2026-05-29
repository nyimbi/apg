"""Deterministic helpers for APG Environment Management."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class EnvironmentEngine:
	"""Pure helper functions for environment fingerprints and posture decisions."""

	def environment_fingerprint(self, payload: dict[str, Any]) -> str:
		return self._stable_digest(payload)

	def drift_percent(self, changed_items: int, total_items: int) -> float:
		if total_items <= 0:
			return 0.0
		return round((max(changed_items, 0) / total_items) * 100, 2)

	def drift_status(self, drift_percent: float, threshold_percent: float, reviewed: bool) -> str:
		if drift_percent <= 0:
			return "in_sync"
		if drift_percent > threshold_percent and not reviewed:
			return "review_required"
		if drift_percent > threshold_percent:
			return "approved_drift"
		return "minor_drift"

	def promotion_status(self, approval_recorded: bool, target_stage: str) -> str:
		if target_stage == "production" and not approval_recorded:
			return "blocked"
		return "approved" if approval_recorded else "pending_approval"

	def _stable_digest(self, payload: dict[str, Any]) -> str:
		encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
		return hashlib.sha256(encoded).hexdigest()
