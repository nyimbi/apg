"""Deterministic helpers for APG Logging and Tracing."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class ObservabilityRuntime:
	"""Dependency-light diagnostic helper routines used by LOGT."""

	def stable_id(self, prefix: str, payload: dict[str, Any]) -> str:
		material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
		digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
		return f"{prefix}-{digest}"

	def normalize_severity(self, severity: str) -> str:
		value = severity.lower()
		if value not in {"debug", "info", "warning", "error", "critical"}:
			raise ValueError("unsupported_log_severity")
		return value

	def redact_message(self, message: str, redaction_applied: bool) -> str:
		if redaction_applied:
			return message
		return message.replace("@", "[at]").replace("password", "[redacted]")

	def span_status(self, duration_ms: float, error: bool = False) -> str:
		if error:
			return "error"
		if duration_ms >= 1000:
			return "slow"
		return "ok"

	def query_status(self, query_window_hours: int, review_recorded: bool) -> str:
		if query_window_hours > 168 and not review_recorded:
			return "review_required"
		return "complete"

	def service_map(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
		services = sorted({span["service_name"] for span in spans})
		edges = sorted({
			(span["parent_span_id"], span["span_id"])
			for span in spans
			if span.get("parent_span_id")
		})
		return {
			"service_count": len(services),
			"services": services,
			"edge_count": len(edges),
			"edges": [{"from": source, "to": target} for source, target in edges],
		}

	def match_log(self, log: dict[str, Any], query_text: str) -> bool:
		if not query_text:
			return True
		needle = query_text.lower()
		return (
			needle in log["message"].lower()
			or needle in log["service_name"].lower()
			or needle in log["severity"].lower()
			or any(needle in str(value).lower() for value in log.get("attributes", {}).values())
		)
