"""In-memory models for APG Licensing & Permits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LicenceApplication:
	id: str
	tenant_id: str
	licence_type: str
	applicant_id: str
	business_registration: str
	status: str
	fee_paid: bool
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Licence:
	id: str
	tenant_id: str
	application_id: str
	licence_type: str
	licence_number: str
	holder_id: str
	issued_date: str
	expiry_date: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LicenceInspection:
	id: str
	tenant_id: str
	licence_id: str
	inspection_type: str
	inspector_id: str
	scheduled_date: str
	outcome: str
	findings: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LicenceRenewal:
	id: str
	tenant_id: str
	licence_id: str
	renewal_type: str
	renewal_fee_paid: bool
	new_expiry_date: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class FeeRecord:
	id: str
	tenant_id: str
	application_id: str
	fee_type: str
	amount: float
	currency: str
	receipt_number: str
	paid: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LicenceRevocation:
	id: str
	tenant_id: str
	licence_id: str
	reason: str
	approval_reference: str
	notice_served: bool
	revoked_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LicensingReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class LicensingAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
