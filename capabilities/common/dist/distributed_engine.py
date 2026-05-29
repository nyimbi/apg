"""Deterministic distributed-computing helpers for the APG DIST capability."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class DistributedEngine:
	"""Pure helpers for partitioning, result evidence, and scaling decisions."""

	def stable_hash(self, payload: dict[str, Any]) -> str:
		encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
		return hashlib.sha256(encoded).hexdigest()

	def partition_ids(self, job_id: str, partition_count: int) -> list[str]:
		return [f"{job_id}-part-{index:04d}" for index in range(1, partition_count + 1)]

	def result_hash(self, job_id: str, partition_results: list[dict[str, Any]]) -> str:
		return self.stable_hash({"job_id": job_id, "partition_results": partition_results})

	def scaling_posture(self, queued_partitions: int, active_workers: int, capacity_quota: int) -> tuple[str, str, int]:
		current_capacity = max(active_workers, 0)
		if queued_partitions > current_capacity * 4 and current_capacity < capacity_quota:
			return "scale_up", "queue_pressure", min(capacity_quota, max(current_capacity + 1, queued_partitions // 4))
		if queued_partitions == 0 and current_capacity > 1:
			return "scale_down", "idle_capacity", max(1, current_capacity - 1)
		return "hold", "capacity_balanced", current_capacity
