"""In-memory models for APG Project Baseline Management (pbl)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ProjectBaseline:
	id: str
	tenant_id: str
	project_id: str
	baseline_type: str
	status: str
	name: str
	description: str
	owner_id: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ChangeRequest:
	id: str
	tenant_id: str
	baseline_id: str
	change_type: str
	priority: str
	status: str
	title: str
	description: str
	submitter_id: str
	impact_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ChangeImpactAssessment:
	id: str
	tenant_id: str
	change_request_id: str
	impact_areas: str  # comma-separated list
	schedule_impact_days: int
	cost_impact_amount: float
	scope_impact_description: str
	risk_impact_description: str
	assessor_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EarnedValueSnapshot:
	id: str
	tenant_id: str
	baseline_id: str
	snapshot_date: str
	pv: float   # planned value
	ev: float   # earned value
	ac: float   # actual cost
	bac: float  # budget at completion
	forecasting_method: str
	eac: float  # estimate at completion
	etc: float  # estimate to complete

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class VarianceReport:
	id: str
	tenant_id: str
	baseline_id: str
	report_period: str
	schedule_variance: float
	cost_variance: float
	spi: float   # schedule performance index
	cpi: float   # cost performance index
	variance_threshold: str
	threshold_breached: bool
	generated_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BaselineApproval:
	id: str
	tenant_id: str
	reference_id: str
	approval_type: str
	reviewer_id: str
	designated_approver: bool
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BaselineAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
