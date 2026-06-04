"""Pydantic v2 models for APG Prescriptive Analytics (bia_psa)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

def uuid7str() -> str: return str(uuid7())

class OptimisationType(str, Enum):
	LINEAR_PROGRAMMING="linear_programming"; INTEGER_PROGRAMMING="integer_programming"
	CONSTRAINT_SATISFACTION="constraint_satisfaction"; GENETIC_ALGORITHM="genetic_algorithm"
	SIMULATED_ANNEALING="simulated_annealing"; REINFORCEMENT_LEARNING="reinforcement_learning"
	MULTI_OBJECTIVE="multi_objective"

class AnalysisState(str, Enum):
	DRAFT="draft"; RUNNING="running"; COMPLETED="completed"; FAILED="failed"; ARCHIVED="archived"

class RecommendationType(str, Enum):
	ACTION="action"; ALLOCATION="allocation"; CONFIGURATION="configuration"
	PROCESS_CHANGE="process_change"; INVESTMENT="investment"; RISK_MITIGATION="risk_mitigation"

class ApprovalState(str, Enum):
	PENDING="pending"; APPROVED="approved"; REJECTED="rejected"; AUTO_APPROVED="auto_approved"

class ConstraintType(str, Enum):
	HARD="hard"; SOFT="soft"; PREFERENCE="preference"

class ObjectiveType(str, Enum):
	MINIMISE="minimise"; MAXIMISE="maximise"; SATISFICE="satisfice"; BALANCE="balance"

class DecisionType(str, Enum):
	BINARY="binary"; MULTI_CLASS="multi_class"; RANKING="ranking"
	ALLOCATION="allocation"; SCHEDULING="scheduling"; ROUTING="routing"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class OptimisationCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; optimisation_type: OptimisationType; owner_id: str
	objective_type: ObjectiveType; objective_description: str
	constraints: list[dict[str, Any]] = Field(default_factory=list)
	variables: list[dict[str, Any]] = Field(default_factory=list)
	description: str | None = None

class OptimisationResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	optimisation_type: OptimisationType; state: AnalysisState = AnalysisState.DRAFT
	owner_id: str; objective_type: ObjectiveType; objective_description: str
	constraints: list[dict[str, Any]] = Field(default_factory=list)
	variables: list[dict[str, Any]] = Field(default_factory=list)
	result: dict[str, Any] | None = None; description: str | None = None
	completed_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class RecommendationCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; optimisation_id: str; name: str
	recommendation_type: RecommendationType; description: str
	actions: list[dict[str, Any]] = Field(default_factory=list)
	impact_estimate: dict[str, Any] = Field(default_factory=dict); owner_id: str

class RecommendationResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; optimisation_id: str
	name: str; recommendation_type: RecommendationType; description: str
	actions: list[dict[str, Any]] = Field(default_factory=list)
	impact_estimate: dict[str, Any] = Field(default_factory=dict)
	owner_id: str; approval_state: ApprovalState = ApprovalState.PENDING
	approved_by: str | None = None; approved_at: datetime | None = None
	acted_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class WhatIfCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; baseline_model_id: str
	parameters: list[dict[str, Any]]; owner_id: str; description: str | None = None

class WhatIfResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	baseline_model_id: str; parameters: list[dict[str, Any]]
	owner_id: str; state: AnalysisState = AnalysisState.DRAFT
	results: dict[str, Any] = Field(default_factory=dict)
	description: str | None = None; simulated_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class DecisionRecord(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str
	decision_type: DecisionType; recommendation_id: str | None = None
	rationale: str; decided_by: str; outcome: str | None = None
	decided_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"
