"""In-memory models for APG Geospatial Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GeospatialAuthority:
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
class AreaOfInterest:
	id: str
	tenant_id: str
	name: str
	geometry_reference: str
	classification: str
	owner_id: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ImagerySource:
	id: str
	tenant_id: str
	source_type: str
	sensor_type: str
	resolution_class: str
	owner_id: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CollectionPlan:
	id: str
	tenant_id: str
	authority_id: str
	area_id: str
	source_id: str
	collection_mode: str
	retention_days: int
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GeoObservation:
	id: str
	tenant_id: str
	plan_id: str
	observation_reference: str
	captured_at: str
	geospatial_accuracy_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GeoFeature:
	id: str
	tenant_id: str
	observation_id: str
	feature_type: str
	geometry_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ChangeDetection:
	id: str
	tenant_id: str
	feature_id: str
	change_type: str
	severity: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GeoAssessment:
	id: str
	tenant_id: str
	change_id: str
	assessment_type: str
	classification: str
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GEOINTDissemination:
	id: str
	tenant_id: str
	assessment_id: str
	audience: str
	release_marking: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GEOINTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class GEOINTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
