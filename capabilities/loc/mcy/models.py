"""Pydantic v2 models for APG Multi-Currency Management."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Annotated

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_APPROVAL_STATUSES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_CURRENCY_STATUSES,
		SUPPORTED_FX_ACCOUNT_TYPES,
		SUPPORTED_RATE_SOURCES,
		SUPPORTED_RATE_TYPES,
		SUPPORTED_REVALUATION_METHODS,
		SUPPORTED_REVALUATION_STATUSES,
		SUPPORTED_ROUNDING_MODES,
		SUPPORTED_TRANSLATION_METHODS,
		SUPPORTED_TRANSLATION_STATUSES,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_APPROVAL_STATUSES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_CURRENCY_STATUSES,
		SUPPORTED_FX_ACCOUNT_TYPES,
		SUPPORTED_RATE_SOURCES,
		SUPPORTED_RATE_TYPES,
		SUPPORTED_REVALUATION_METHODS,
		SUPPORTED_REVALUATION_STATUSES,
		SUPPORTED_ROUNDING_MODES,
		SUPPORTED_TRANSLATION_METHODS,
		SUPPORTED_TRANSLATION_STATUSES,
	)

_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


def _check_currency(v: str) -> str:
	assert v.upper() in SUPPORTED_CURRENCIES, f"currency '{v}' not supported"
	return v.upper()


def _check_rate_type(v: str) -> str:
	assert v in SUPPORTED_RATE_TYPES, f"rate_type '{v}' not supported"
	return v


def _check_rate_source(v: str) -> str:
	assert v in SUPPORTED_RATE_SOURCES, f"rate_source '{v}' not supported"
	return v


def _check_rounding_mode(v: str) -> str:
	assert v in SUPPORTED_ROUNDING_MODES, f"rounding_mode '{v}' not supported"
	return v


def _check_currency_status(v: str) -> str:
	assert v in SUPPORTED_CURRENCY_STATUSES, f"status '{v}' not supported"
	return v


def _check_revaluation_method(v: str) -> str:
	assert v in SUPPORTED_REVALUATION_METHODS, f"revaluation_method '{v}' not supported"
	return v


def _check_revaluation_status(v: str) -> str:
	assert v in SUPPORTED_REVALUATION_STATUSES, f"status '{v}' not supported"
	return v


def _check_translation_method(v: str) -> str:
	assert v in SUPPORTED_TRANSLATION_METHODS, f"translation_method '{v}' not supported"
	return v


def _check_translation_status(v: str) -> str:
	assert v in SUPPORTED_TRANSLATION_STATUSES, f"status '{v}' not supported"
	return v


def _check_fx_account_type(v: str) -> str:
	assert v in SUPPORTED_FX_ACCOUNT_TYPES, f"account_type '{v}' not supported"
	return v


# --- Currency Configuration ---

class CurrencyConfigCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	code: Annotated[str, AfterValidator(_check_currency)]
	name: str
	symbol: str
	decimal_places: int = 2
	rounding_mode: Annotated[str, AfterValidator(_check_rounding_mode)] = "round_half_even"
	is_functional: bool = False
	is_presentation: bool = False
	notes: str | None = None

	@field_validator("decimal_places")
	@classmethod
	def _validate_precision(cls, v: int) -> int:
		assert 0 <= v <= 6, f"decimal_places must be 0-6, got {v}"
		return v


class CurrencyConfigUpdate(BaseModel):
	model_config = _MODEL_CFG

	name: str | None = None
	symbol: str | None = None
	decimal_places: int | None = None
	rounding_mode: Annotated[str | None, AfterValidator(lambda v: _check_rounding_mode(v) if v else v)] = None
	status: Annotated[str | None, AfterValidator(lambda v: _check_currency_status(v) if v else v)] = None
	is_functional: bool | None = None
	is_presentation: bool | None = None
	notes: str | None = None


class CurrencyConfigResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	code: str
	name: str
	symbol: str
	decimal_places: int = 2
	rounding_mode: str = "round_half_even"
	status: str = "active"
	is_functional: bool = False
	is_presentation: bool = False
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Exchange Rate ---

class ExchangeRateCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	from_currency: Annotated[str, AfterValidator(_check_currency)]
	to_currency: Annotated[str, AfterValidator(_check_currency)]
	rate: float
	rate_type: Annotated[str, AfterValidator(_check_rate_type)]
	rate_source: Annotated[str, AfterValidator(_check_rate_source)]
	effective_date: date
	expiry_date: date | None = None
	approval_reference: str | None = None
	backdating_override: str | None = None
	notes: str | None = None

	@field_validator("rate")
	@classmethod
	def _validate_rate(cls, v: float) -> float:
		assert v > 0, f"exchange rate must be positive, got {v}"
		return v

	@field_validator("to_currency")
	@classmethod
	def _no_self_conversion(cls, v: str, info: Any) -> str:
		from_currency = info.data.get("from_currency", "")
		if from_currency:
			assert v.upper() != from_currency.upper(), "from_currency and to_currency must differ"
		return v


class ExchangeRateUpdate(BaseModel):
	model_config = _MODEL_CFG

	rate: float | None = None
	expiry_date: date | None = None
	approval_reference: str | None = None
	notes: str | None = None
	is_active: bool | None = None


class ExchangeRateResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	from_currency: str
	to_currency: str
	rate: float
	rate_type: str
	rate_source: str
	effective_date: date
	expiry_date: date | None = None
	approval_reference: str | None = None
	backdating_override: str | None = None
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- FX Revaluation ---

class RevaluationCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	entity_id: str
	period_start: date
	period_end: date
	revaluation_method: Annotated[str, AfterValidator(_check_revaluation_method)]
	functional_currency: Annotated[str, AfterValidator(_check_currency)]
	fx_gain_account_id: str
	fx_loss_account_id: str
	description: str | None = None


class RevaluationUpdate(BaseModel):
	model_config = _MODEL_CFG

	status: Annotated[str | None, AfterValidator(lambda v: _check_revaluation_status(v) if v else v)] = None
	approval_reference: str | None = None
	posted_date: date | None = None
	reversal_date: date | None = None
	notes: str | None = None


class RevaluationResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str
	period_start: date
	period_end: date
	revaluation_method: str
	functional_currency: str
	fx_gain_account_id: str
	fx_loss_account_id: str
	status: str = "draft"
	fx_gain_amount: float = 0.0
	fx_loss_amount: float = 0.0
	net_fx_impact: float = 0.0
	approval_reference: str | None = None
	posted_date: date | None = None
	reversal_date: date | None = None
	description: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Currency Translation ---

class CurrencyTranslationCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	entity_id: str
	period_start: date
	period_end: date
	source_currency: Annotated[str, AfterValidator(_check_currency)]
	target_currency: Annotated[str, AfterValidator(_check_currency)]
	translation_method: Annotated[str, AfterValidator(_check_translation_method)]
	translation_reserve_account_id: str
	description: str | None = None


class CurrencyTranslationUpdate(BaseModel):
	model_config = _MODEL_CFG

	status: Annotated[str | None, AfterValidator(lambda v: _check_translation_status(v) if v else v)] = None
	approval_reference: str | None = None
	posted_date: date | None = None
	notes: str | None = None


class CurrencyTranslationResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	entity_id: str
	period_start: date
	period_end: date
	source_currency: str
	target_currency: str
	translation_method: str
	translation_reserve_account_id: str
	status: str = "draft"
	translation_difference: float = 0.0
	approval_reference: str | None = None
	posted_date: date | None = None
	description: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- FX Account ---

class FxAccountCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	account_type: Annotated[str, AfterValidator(_check_fx_account_type)]
	account_code: str
	account_name: str
	currency: Annotated[str, AfterValidator(_check_currency)]
	entity_id: str | None = None
	notes: str | None = None


class FxAccountResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	account_type: str
	account_code: str
	account_name: str
	currency: str
	entity_id: str | None = None
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- FX Gain/Loss Report ---

class FxGainLossReport(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	period_start: date
	period_end: date
	entity_id: str | None = None
	total_realised_gain: float = 0.0
	total_realised_loss: float = 0.0
	total_unrealised_gain: float = 0.0
	total_unrealised_loss: float = 0.0
	net_fx_impact: float = 0.0
	currency: str = "USD"
	generated_at: datetime = Field(default_factory=datetime.utcnow)


# --- MCY Agent ---

class McyAgentCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	@field_validator("runtime")
	@classmethod
	def _validate_runtime(cls, v: str) -> str:
		assert v in SUPPORTED_AGENT_RUNTIMES, f"runtime '{v}' not supported"
		return v

	@field_validator("role")
	@classmethod
	def _validate_role(cls, v: str) -> str:
		assert v in SUPPORTED_AGENT_ROLES, f"role '{v}' not supported"
		return v


class McyAgentResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Audit Event ---

class McyAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	reference_id: str
	actor_id: str = "system"
	payload: dict[str, Any] = Field(default_factory=dict)
	processor: str = "bytewax"
	stream: str = "apg.loc.mcy.lifecycle"
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
