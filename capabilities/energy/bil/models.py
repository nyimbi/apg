"""In-memory models for APG Energy Billing & Tariffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Tariff:
	id: str
	tenant_id: str
	name: str
	tariff_type: str
	customer_class: str
	effective_date: str
	status: str
	created_by: str
	approved_by: str = ""
	approved_at: str = ""
	end_date: str = ""
	currency: str = "KES"
	description: str = ""
	rate_blocks: list[dict[str, Any]] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BillCharge:
	charge_type: str
	description: str
	quantity: float
	unit: str
	unit_rate: float
	amount: float
	currency: str = "KES"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EnergyBill:
	id: str
	tenant_id: str
	customer_id: str
	meter_id: str
	tariff_id: str
	billing_cycle: str
	period_start: str
	period_end: str
	status: str
	generated_at: str
	total_amount: float
	currency: str = "KES"
	issued_at: str = ""
	due_date: str = ""
	charges: list[dict[str, Any]] = field(default_factory=list)
	consumption_kwh: float = 0.0
	peak_demand_kw: float = 0.0

	def balance_due(self, paid_amount: float = 0.0) -> float:
		return round(self.total_amount - paid_amount, 4)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class Payment:
	id: str
	tenant_id: str
	bill_id: str
	customer_id: str
	payment_method: str
	amount: float
	currency: str
	received_at: str
	reconciled: bool = False
	reconciled_at: str = ""
	transaction_reference: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class EnergyCredit:
	id: str
	tenant_id: str
	customer_id: str
	credit_type: str
	amount: float
	currency: str
	issued_at: str
	expires_at: str
	approved_by: str
	status: str = "active"
	applied_to_bill_id: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BillingDispute:
	id: str
	tenant_id: str
	bill_id: str
	customer_id: str
	status: str
	reason: str
	evidence_reference: str
	opened_at: str
	resolved_at: str = ""
	resolution: str = ""
	adjusted_amount: float = 0.0
	assigned_to: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class RevenueAssuranceFlag:
	id: str
	tenant_id: str
	flag_type: str
	entity_id: str
	entity_type: str
	estimated_revenue_impact: float
	currency: str
	flagged_at: str
	status: str = "open"
	investigated_by: str = ""
	resolved_at: str = ""
	notes: str = ""

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class BilAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered_at: str
	active: bool = True

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AuditEvent:
	id: str
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	actor: str
	occurred_at: str
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
