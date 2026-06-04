"""In-memory models for APG Service Provisioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ProWorkflow:
	id: str
	tenant_id: str
	workflow_type: str
	order_reference: str
	status: str
	retry_count: int
	started_at: str
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProResourceReservation:
	id: str
	tenant_id: str
	workflow_id: str
	resource_type: str
	resource_value: str
	conflict_checked: bool
	reserved_at: str
	expires_at: str
	released: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProConfigPush:
	id: str
	tenant_id: str
	workflow_id: str
	ne_reference: str
	push_method: str
	template_reference: str
	dry_run_completed: bool
	status: str
	pushed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProActivation:
	id: str
	tenant_id: str
	workflow_id: str
	service_reference: str
	status: str
	verification_completed: bool
	e2e_test_passed: bool
	activated_at: str | None
	confirmed_by: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProRollback:
	id: str
	tenant_id: str
	workflow_id: str
	trigger: str
	description: str
	status: str
	triggered_at: str
	completed_at: str | None

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProBulkJob:
	id: str
	tenant_id: str
	workflow_type: str
	item_count: int
	approval_reference: str
	status: str
	submitted_by: str
	submitted_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ProAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
