"""In-memory models for APG Financial Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FinancialAuthority:
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
class FinancialSource:
	id: str
	tenant_id: str
	source_type: str
	jurisdiction: str
	owner_id: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialSubject:
	id: str
	tenant_id: str
	subject_type: str
	subject_reference: str
	risk_tier: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialTransaction:
	id: str
	tenant_id: str
	source_id: str
	subject_id: str
	transaction_reference: str
	amount: float
	currency: str
	transaction_type: str
	occurred_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialPattern:
	id: str
	tenant_id: str
	transaction_id: str
	pattern_type: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialRiskAssessment:
	id: str
	tenant_id: str
	pattern_id: str
	risk_type: str
	risk_level: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FinancialReferral:
	id: str
	tenant_id: str
	assessment_id: str
	referral_type: str
	recipient: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FININTDissemination:
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
class FININTReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FININTAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
