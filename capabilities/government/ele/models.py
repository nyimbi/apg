"""In-memory models for APG Electoral & Civil Registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class VoterRegistration:
	id: str
	tenant_id: str
	registration_type: str
	national_id: str
	biometric_reference: str
	constituency: str
	polling_station_id: str
	deduplication_status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DeduplicationRecord:
	id: str
	tenant_id: str
	registration_id: str
	method: str
	match_score: float
	duplicate_detected: bool
	resolution: str
	resolved_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PollingStation:
	id: str
	tenant_id: str
	station_type: str
	name: str
	constituency: str
	location_reference: str
	capacity: int
	presiding_officer_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Election:
	id: str
	tenant_id: str
	election_type: str
	name: str
	polling_date: str
	nomination_deadline: str
	constituency: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ElectionResult:
	id: str
	tenant_id: str
	election_id: str
	polling_station_id: str
	candidate_id: str
	votes_cast: int
	rejected_votes: int
	presiding_officer_id: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CivilRegistryEvent:
	id: str
	tenant_id: str
	registration_type: str
	subject_id: str
	registrar_id: str
	witness_id: str
	event_date: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ElectoralVerification:
	id: str
	tenant_id: str
	registration_id: str
	status: str
	biometric_match_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ElectoralReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ElectoralAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
