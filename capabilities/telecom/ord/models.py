"""In-memory models for APG Order Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class OrdOrder:
	id: str
	tenant_id: str
	order_type: str
	customer_id: str
	channel: str
	priority: str
	status: str
	submitted_at: str
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OrdTask:
	id: str
	tenant_id: str
	order_id: str
	task_type: str
	status: str
	depends_on: str | None
	assigned_to: str | None
	started_at: str | None
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OrdFallout:
	id: str
	tenant_id: str
	order_id: str
	fallout_category: str
	description: str
	retry_count: int
	resolution: str | None
	resolved_at: str | None
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OrdPortabilityRequest:
	id: str
	tenant_id: str
	order_id: str
	msisdn: str
	donor_operator: str
	recipient_operator: str
	status: str
	submitted_at: str
	porting_date: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OrdBulkOrder:
	id: str
	tenant_id: str
	order_type: str
	item_count: int
	approval_reference: str
	status: str
	submitted_by: str
	submitted_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class OrdAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
