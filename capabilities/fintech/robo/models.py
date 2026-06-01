"""Dependency-light data models for APG Robo Advisory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InvestorProfile:
	id: str
	tenant_id: str
	client_id: str
	kyc_reference: str
	suitability_reference: str
	risk_profile: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class GoalPlan:
	id: str
	tenant_id: str
	profile_id: str
	goal_type: str
	target_amount_minor: int
	currency: str
	horizon_date: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ModelPortfolio:
	id: str
	tenant_id: str
	name: str
	risk_profile: str
	target_allocation: dict[str, float]
	policy_reference: str
	status: str = "published"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__ | {"target_allocation": dict(self.target_allocation)}


@dataclass
class RecommendationPacket:
	id: str
	tenant_id: str
	profile_id: str
	goal_id: str
	model_id: str
	analysis_reference: str
	status: str = "generated"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class AutomationPlan:
	id: str
	tenant_id: str
	recommendation_id: str
	funding_source_reference: str
	cadence: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class DriftRecord:
	id: str
	tenant_id: str
	profile_id: str
	drift_bps: int
	analysis_reference: str
	status: str = "recorded"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class TaxLossCandidate:
	id: str
	tenant_id: str
	profile_id: str
	instrument_id: str
	loss_minor: int
	tax_lot_reference: str
	status: str = "candidate"

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class ReviewRecord:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return self.__dict__.copy()


@dataclass
class RoboEvidence:
	id: str
	tenant_id: str
	kind: str
	reference_id: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {"id": self.id, "tenant_id": self.tenant_id, "kind": self.kind, "reference_id": self.reference_id, "status": self.status, "metadata": dict(self.metadata)}
