"""In-memory models for APG Intelligence Analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AnalyticsAuthority:
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
class AnalyticsWorkspace:
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
class AnalyticsDataset:
	id: str
	tenant_id: str
	workspace_id: str
	dataset_type: str
	source_reference: str
	owner_id: str
	lineage_reference: str
	retention_class: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsFeatureSet:
	id: str
	tenant_id: str
	dataset_id: str
	feature_type: str
	feature_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsModel:
	id: str
	tenant_id: str
	feature_set_id: str
	model_type: str
	objective: str
	validation_reference: str
	risk_level: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsRun:
	id: str
	tenant_id: str
	model_id: str
	run_type: str
	result_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsInsight:
	id: str
	tenant_id: str
	run_id: str
	insight_type: str
	claim_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsDashboard:
	id: str
	tenant_id: str
	insight_id: str
	name: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsNarrative:
	id: str
	tenant_id: str
	insight_id: str
	narrative_type: str
	summary_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsRecommendation:
	id: str
	tenant_id: str
	insight_id: str
	recommendation_type: str
	action_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnalyticsAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
