"""Data models for the Anomaly Detection capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnomRecord:
	"""Tenant-scoped dependency-light capability record."""

	id: str
	tenant_id: str
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"status": self.status,
			"metadata": dict(self.metadata),
		}
