"""Deterministic backup artifact helpers for APG BKUP."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class BackupEngine:
	"""Build stable snapshot hashes and continuity status summaries."""

	def digest(self, payload: dict[str, Any]) -> str:
		canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
		return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

	def snapshot_hash(self, payload: dict[str, Any]) -> str:
		return self.digest({"kind": "bkup.snapshot", "payload": payload})

	def continuity_findings(
		self,
		rpo_minutes: int,
		rpo_target_minutes: int,
		rto_minutes: int,
		rto_target_minutes: int,
		days_since_restore_test: int,
	) -> tuple[str, ...]:
		findings: list[str] = []
		if rpo_minutes > rpo_target_minutes:
			findings.append("rpo target exceeded")
		if rto_minutes > rto_target_minutes:
			findings.append("rto target exceeded")
		if days_since_restore_test > 90:
			findings.append("restore test older than 90 days")
		return tuple(findings)
