"""In-memory models for APG Citizen Services Portal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ServiceDefinition:
	id: str
	tenant_id: str
	service_type: str
	name: str
	description: str
	fee_amount: float
	fee_currency: str
	sla_days: int
	evidence_required: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ServiceApplication:
	id: str
	tenant_id: str
	service_id: str
	citizen_id: str
	channel: str
	status: str
	submitted_at: str
	reference_number: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PaymentRecord:
	id: str
	tenant_id: str
	application_id: str
	payment_method: str
	amount: float
	currency: str
	receipt_number: str
	status: str
	transaction_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class DocumentVerification:
	id: str
	tenant_id: str
	application_id: str
	verification_type: str
	document_reference: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CitizenNotification:
	id: str
	tenant_id: str
	application_id: str
	citizen_id: str
	notification_type: str
	message: str
	sent: bool

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ServiceDeliveryRecord:
	id: str
	tenant_id: str
	application_id: str
	delivered_at: str
	delivery_method: str
	certificate_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ServiceReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class CitizenServicesAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
