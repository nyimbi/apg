"""In-memory models for APG Budget Management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BudgetProgramme:
	id: str
	tenant_id: str
	budget_type: str
	fund_source: str
	vote_id: str
	total_amount: float
	fiscal_year: str
	approver_id: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class VoteAccount:
	id: str
	tenant_id: str
	vote_code: str
	vote_type: str
	budget_id: str
	allocated_amount: float
	committed_amount: float
	expended_amount: float
	available_balance: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BudgetRevision:
	id: str
	tenant_id: str
	budget_id: str
	revision_type: str
	amount_change: float
	approval_reference: str
	treasury_notification_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CommitmentRecord:
	id: str
	tenant_id: str
	vote_id: str
	commitment_type: str
	amount: float
	approval_reference: str
	supplier_reference: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ExpenditureRecord:
	id: str
	tenant_id: str
	commitment_id: str
	expenditure_type: str
	amount: float
	approval_reference: str
	payee_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FiscalReport:
	id: str
	tenant_id: str
	budget_id: str
	report_type: str
	fiscal_period: str
	author_id: str
	evidence_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BudgetApproval:
	id: str
	tenant_id: str
	reference_id: str
	approver_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BudgetReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BudgetAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
