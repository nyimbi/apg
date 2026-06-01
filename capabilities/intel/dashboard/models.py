"""In-memory models for APG Intelligence Dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DashboardAuthority:
	id: str
	tenant_id: str
	authority_type: str
	scope_reference: str
	classification: str
	approver_id: str
	expires_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardWorkspace:
	id: str
	tenant_id: str
	workspace_type: str
	name: str
	classification: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardBoard:
	id: str
	tenant_id: str
	workspace_id: str
	dashboard_type: str
	title: str
	owner_id: str
	classification: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardDataSource:
	id: str
	tenant_id: str
	dashboard_id: str
	source_type: str
	source_reference: str
	custodian_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardMetric:
	id: str
	tenant_id: str
	source_id: str
	metric_type: str
	metric_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardWidget:
	id: str
	tenant_id: str
	dashboard_id: str
	widget_type: str
	widget_reference: str
	metric_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardFilter:
	id: str
	tenant_id: str
	dashboard_id: str
	filter_type: str
	filter_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardView:
	id: str
	tenant_id: str
	dashboard_id: str
	view_type: str
	view_reference: str
	viewer_role: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardShare:
	id: str
	tenant_id: str
	dashboard_id: str
	share_type: str
	recipient_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DashboardAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

