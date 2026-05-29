"""Deterministic helpers for APG Multi-Channel Output."""

from __future__ import annotations

import hashlib
import json
from string import Template
from typing import Any


class OutputRuntime:
	"""Dependency-light output routing and rendering helpers."""

	CHANNEL_TYPES = {"email", "sms", "push", "pdf", "web", "api", "print"}
	OUTPUT_FORMATS = {"html", "text", "pdf", "json", "xml", "markdown"}
	DELIVERY_STATES = {"accepted", "sent", "delivered", "failed", "bounced"}

	def stable_id(self, prefix: str, payload: dict[str, Any]) -> str:
		material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
		digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
		return f"{prefix}-{digest}"

	def normalize_channel_type(self, channel_type: str) -> str:
		value = channel_type.lower()
		if value not in self.CHANNEL_TYPES:
			raise ValueError("unsupported_output_channel")
		return value

	def normalize_format(self, output_format: str) -> str:
		value = output_format.lower()
		if value not in self.OUTPUT_FORMATS:
			raise ValueError("unsupported_output_format")
		return value

	def normalize_health(self, health: str) -> str:
		value = health.lower()
		if value not in {"healthy", "degraded", "unhealthy"}:
			raise ValueError("unsupported_channel_health")
		return value

	def render_template(self, template_text: str, variables: dict[str, Any]) -> str:
		mapping = {key: str(value) for key, value in variables.items()}
		return Template(template_text).safe_substitute(mapping)

	def selected_channel_id(self, primary_channel: dict[str, Any], fallback_channels: list[dict[str, Any]]) -> str:
		if primary_channel["health"] != "unhealthy":
			return str(primary_channel["id"])
		for channel in fallback_channels:
			if channel["health"] != "unhealthy":
				return str(channel["id"])
		return str(primary_channel["id"])

	def rendered_status(self, sensitive_output: bool, output_encrypted: bool) -> str:
		if sensitive_output and not output_encrypted:
			return "blocked"
		return "ready"

	def batch_status(self, recipient_count: int, delivery_review_recorded: bool) -> str:
		if recipient_count > 10000 and not delivery_review_recorded:
			return "review_required"
		return "queued"

	def normalize_delivery_state(self, delivery_state: str) -> str:
		value = delivery_state.lower()
		if value not in self.DELIVERY_STATES:
			raise ValueError("unsupported_delivery_state")
		return value
