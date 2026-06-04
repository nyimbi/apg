"""In-memory models for APG Project Accounting (pac)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ProjectAccount:
	id: str
	tenant_id: str
	project_id: str
	name: str
	status: str
	currency: str
	budget_amount: float
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CostTransaction:
	id: str
	tenant_id: str
	account_id: str
	cost_type: str
	transaction_type: str
	amount: float
	description: str
	period_reference: str
	evidence_reference: str
	backdated: bool = False
	justification: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RevenueRecognition:
	id: str
	tenant_id: str
	account_id: str
	revenue_type: str
	wip_method: str
	amount: float
	recognition_period: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class WipAdjustment:
	id: str
	tenant_id: str
	account_id: str
	adjustment_amount: float
	description: str
	auditor_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class MilestoneInvoice:
	id: str
	tenant_id: str
	account_id: str
	billing_type: str
	amount: float
	milestone_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BudgetOverride:
	id: str
	tenant_id: str
	account_id: str
	original_budget: float
	revised_budget: float
	reason: str
	controller_approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccountingApproval:
	id: str
	tenant_id: str
	reference_id: str
	approval_type: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccountingAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
