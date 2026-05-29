"""Deterministic pipeline artifact helpers for APG CICD."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class CicdEngine:
	"""Build stable trace, artifact, and gate evidence identifiers."""

	def digest(self, payload: dict[str, Any]) -> str:
		canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
		return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

	def build_trace_id(self, payload: dict[str, Any]) -> str:
		return f"trace-{self.digest({'kind': 'cicd.build', 'payload': payload})[:20]}"

	def artifact_digest(self, payload: dict[str, Any]) -> str:
		return self.digest({"kind": "cicd.artifact", "payload": payload})

	def gate_findings(
		self,
		tests_passed: bool,
		security_scan_passed: bool,
		artifact_signed: bool,
		approval_recorded: bool,
	) -> tuple[str, ...]:
		findings: list[str] = []
		if not tests_passed:
			findings.append("tests failed")
		if not security_scan_passed:
			findings.append("security scan failed")
		if not artifact_signed:
			findings.append("artifact signature missing")
		if not approval_recorded:
			findings.append("promotion approval missing")
		return tuple(findings)
