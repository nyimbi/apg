"""In-memory models for APG FinTech Compliance Automation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ComplianceObligation:
	id: str
	tenant_id: str
	framework: str
	obligation_type: str
	title: str
	owner_id: str
	evidence_reference: str
	effective_date: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceControl:
	id: str
	tenant_id: str
	obligation_id: str
	control_type: str
	owner_id: str
	evidence_reference: str
	frequency: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceCheck:
	id: str
	tenant_id: str
	obligation_id: str
	control_id: str
	check_type: str
	subject_reference: str
	result: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceEvidence:
	id: str
	tenant_id: str
	reference_id: str
	evidence_type: str
	source_reference: str
	retention_days: int

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceAttestation:
	id: str
	tenant_id: str
	obligation_id: str
	attestor_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceIssue:
	id: str
	tenant_id: str
	obligation_id: str
	severity: str
	owner_id: str
	evidence_reference: str
	due_date: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceRemediation:
	id: str
	tenant_id: str
	issue_id: str
	owner_id: str
	plan_reference: str
	approval_reference: str
	status: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceReport:
	id: str
	tenant_id: str
	report_type: str
	framework: str
	period: str
	evidence_reference: str
	approver_id: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ComplianceAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
