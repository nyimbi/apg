"""Deterministic helpers for APG federated learning workflows."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class FederatedLearningEngine:
	"""Pure helpers for update validation, aggregation, and model versioning."""

	def update_digest(self, payload: dict[str, Any]) -> str:
		return self._stable_digest({"update": payload})

	def poisoning_signal(self, quality_score: float, explicit_signal: bool = False) -> bool:
		return bool(explicit_signal) or quality_score < 0.35

	def aggregate_digest(self, updates: list[dict[str, Any]]) -> str:
		return self._stable_digest({"updates": sorted(updates, key=lambda item: item["id"])})

	def model_version(self, federation_id: str, round_number: int, aggregate_digest: str) -> str:
		return f"{federation_id}.r{round_number}.{aggregate_digest[:12]}"

	def _stable_digest(self, payload: dict[str, Any]) -> str:
		encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
		return hashlib.sha256(encoded).hexdigest()
