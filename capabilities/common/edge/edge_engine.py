"""Deterministic helpers for EDGE runtime behavior."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_digest(payload: dict[str, Any]) -> str:
	"""Return a deterministic digest for artifacts, events, and audit records."""
	encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(encoded).hexdigest()


def artifact_digest(name: str, version: str, artifact_payload: dict[str, Any] | str) -> str:
	"""Build a stable workload artifact digest from metadata and payload."""
	return stable_digest({
		"name": name,
		"version": version,
		"artifact": artifact_payload,
	})


def capacity_fits(capacity: dict[str, float], quota: dict[str, float], current_load: dict[str, float] | None = None) -> bool:
	"""Return whether a workload quota can fit on a node after current load."""
	load = current_load or {}
	for resource, requested in quota.items():
		available = float(capacity.get(resource, 0)) - float(load.get(resource, 0))
		if requested > available:
			return False
	return True


def resource_pressure(capacity: dict[str, float], current_load: dict[str, float]) -> dict[str, float]:
	"""Calculate normalized resource pressure for dashboard and placement decisions."""
	pressure: dict[str, float] = {}
	for resource, total in capacity.items():
		if total <= 0:
			pressure[resource] = 1.0
		else:
			pressure[resource] = round(min(1.0, float(current_load.get(resource, 0)) / float(total)), 4)
	return pressure


def sync_status(offline_hours: int, conflicts: list[str], review_required: bool) -> str:
	"""Summarize sync posture from offline duration, conflicts, and review state."""
	if review_required:
		return "review_required"
	if conflicts:
		return "conflict_pending"
	if offline_hours > 0:
		return "replayed"
	return "synced"
