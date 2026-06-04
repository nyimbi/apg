"""In-memory models for APG Time & Expense Management (tex)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Timesheet:
	id: str
	tenant_id: str
	resource_id: str
	project_id: str
	period_type: str
	period_reference: str
	status: str
	submitted_by: str
	reviewer_id: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TimeEntry:
	id: str
	tenant_id: str
	timesheet_id: str
	project_id: str
	task_id: str
	entry_type: str
	billable_status: str
	hours: float
	entry_date: str
	description: str
	backdated: bool
	justification: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ExpenseClaim:
	id: str
	tenant_id: str
	resource_id: str
	project_id: str
	category: str
	currency: str
	amount: float
	status: str
	receipt_status: str
	expense_date: str
	description: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Reimbursement:
	id: str
	tenant_id: str
	expense_claim_id: str
	resource_id: str
	method: str
	amount: float
	currency: str
	approval_reference: str
	processed_date: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BillingRate:
	id: str
	tenant_id: str
	resource_id: str
	project_id: str
	rate_type: str
	rate_amount: float
	currency: str
	effective_date: str
	approval_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TimesheetApproval:
	id: str
	tenant_id: str
	timesheet_id: str
	reviewer_id: str
	status: str
	comments: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ExpenseApproval:
	id: str
	tenant_id: str
	expense_claim_id: str
	reviewer_id: str
	status: str
	comments: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class TexAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
