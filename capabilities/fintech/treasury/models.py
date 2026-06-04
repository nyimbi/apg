"""Pydantic v2 models for fintech_treasury capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field

uuid7str = lambda: str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


# ── Enums ─────────────────────────────────────────────────────────────────────

class InstrumentType(str, Enum):
	fx_forward = "fx_forward"
	fx_option = "fx_option"
	interest_rate_swap = "interest_rate_swap"
	cross_currency_swap = "cross_currency_swap"
	fx_swap = "fx_swap"
	ndf = "ndf"


class InstrumentStatus(str, Enum):
	booked = "booked"
	active = "active"
	matured = "matured"
	cancelled = "cancelled"
	settled = "settled"


class LoanStatus(str, Enum):
	active = "active"
	repaid = "repaid"
	defaulted = "defaulted"
	restructured = "restructured"


class ForecastMethod(str, Enum):
	ar_ap_driven = "ar_ap_driven"
	statistical = "statistical"
	scenario_based = "scenario_based"


class PoolingMethod(str, Enum):
	notional = "notional"
	physical = "physical"


class ScenarioType(str, Enum):
	fx_shock = "fx_shock"
	interest_rate_shock = "interest_rate_shock"
	liquidity_stress = "liquidity_stress"
	credit_event = "credit_event"
	combined_stress = "combined_stress"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class CashPosition(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str
	as_of_date: str
	currencies: list[str]
	positions: dict[str, Any] = Field(default_factory=dict)
	generated_at: str = Field(default_factory=_now)


class HedgeInstrument(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str | None = None
	instrument_type: InstrumentType
	notional: float
	currency_pair: str
	strike: float
	maturity: str
	counterparty_id: str | None = None
	status: InstrumentStatus = InstrumentStatus.booked
	fair_value: float = 0.0
	hedge_effectiveness: bool | None = None


class IntercompanyLoan(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	lender_entity: str
	borrower_entity: str
	amount: float
	currency: str
	interest_rate_pct: float
	tenor_months: int
	maturity_date: str
	annual_interest: float
	total_interest: float
	outstanding_balance: float
	status: LoanStatus = LoanStatus.active


class TreasuryKPI(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str
	as_of: str
	cash_positions: dict[str, Any] = Field(default_factory=dict)
	active_fx_deals: int = 0
	active_mm_placements: int = 0
	total_placement_kes: float = 0.0
	wacof_pct: float = 0.0
	total_facility_limit: float = 0.0
	total_facility_utilised: float = 0.0
	overall_facility_utilisation_pct: float = 0.0
	generated_at: str = Field(default_factory=_now)


# ── Request / Response ────────────────────────────────────────────────────────

class CashPositionRequest(_Base):
	entity_id: str
	as_of_date: str
	currencies: list[str]


class HedgeInstrumentRequest(_Base):
	instrument_type: InstrumentType
	notional: float
	currency_pair: str
	strike: float
	maturity: str
	entity_id: str | None = None
	counterparty_id: str | None = None


class IntercompanyLoanRequest(_Base):
	lender_entity: str
	borrower_entity: str
	amount: float
	currency: str
	rate: float
	tenor_months: int


class FXForwardRequest(_Base):
	entity_id: str
	buy_currency: str
	sell_currency: str
	amount: float
	settlement_date: str
	forward_rate: float


class PaymentFactoryRequest(_Base):
	entity_id: str
	payments: list[dict[str, Any]]
	payment_date: str


class ScenarioAnalysisRequest(_Base):
	entity_id: str
	scenario_type: ScenarioType
	parameters: dict[str, Any]


class CovenantMonitoringRequest(_Base):
	facility_id: str
	financial_ratios: dict[str, float]
