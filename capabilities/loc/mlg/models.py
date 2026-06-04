"""Pydantic v2 models for APG Multi-Language & Localisation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
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
		SUPPORTED_CONTENT_TYPES,
		SUPPORTED_CURRENCY_DISPLAY_MODES,
		SUPPORTED_DATE_FORMATS,
		SUPPORTED_LANGUAGES,
		SUPPORTED_LOCALES,
		SUPPORTED_NUMBER_FORMATS,
		SUPPORTED_RTL_LANGUAGES,
		SUPPORTED_SCRIPTS,
		SUPPORTED_TEXT_DIRECTIONS,
		SUPPORTED_TRANSLATION_STATUSES,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CONTENT_TYPES,
		SUPPORTED_CURRENCY_DISPLAY_MODES,
		SUPPORTED_DATE_FORMATS,
		SUPPORTED_LANGUAGES,
		SUPPORTED_LOCALES,
		SUPPORTED_NUMBER_FORMATS,
		SUPPORTED_RTL_LANGUAGES,
		SUPPORTED_SCRIPTS,
		SUPPORTED_TEXT_DIRECTIONS,
		SUPPORTED_TRANSLATION_STATUSES,
	)

_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


def _check_locale(v: str) -> str:
	assert v in SUPPORTED_LOCALES, f"locale '{v}' not in supported locales"
	return v


def _check_language(v: str) -> str:
	assert v in SUPPORTED_LANGUAGES, f"language '{v}' not supported"
	return v


def _check_script(v: str) -> str:
	assert v in SUPPORTED_SCRIPTS, f"script '{v}' not supported"
	return v


def _check_direction(v: str) -> str:
	assert v in SUPPORTED_TEXT_DIRECTIONS, f"direction '{v}' not supported"
	return v


def _check_date_format(v: str) -> str:
	assert v in SUPPORTED_DATE_FORMATS, f"date_format '{v}' not supported"
	return v


def _check_number_format(v: str) -> str:
	assert v in SUPPORTED_NUMBER_FORMATS, f"number_format '{v}' not supported"
	return v


def _check_currency_display(v: str) -> str:
	assert v in SUPPORTED_CURRENCY_DISPLAY_MODES, f"currency_display '{v}' not supported"
	return v


def _check_content_type(v: str) -> str:
	assert v in SUPPORTED_CONTENT_TYPES, f"content_type '{v}' not supported"
	return v


def _check_translation_status(v: str) -> str:
	assert v in SUPPORTED_TRANSLATION_STATUSES, f"status '{v}' not supported"
	return v


# --- Locale Configuration ---

class LocaleConfigCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	locale_code: Annotated[str, AfterValidator(_check_locale)]
	language: Annotated[str, AfterValidator(_check_language)]
	script: Annotated[str, AfterValidator(_check_script)]
	text_direction: Annotated[str, AfterValidator(_check_direction)]
	date_format: Annotated[str, AfterValidator(_check_date_format)]
	number_format: Annotated[str, AfterValidator(_check_number_format)]
	currency_display: Annotated[str, AfterValidator(_check_currency_display)] = "symbol"
	is_default: bool = False
	is_rtl: bool = False
	notes: str | None = None

	@model_validator(mode="after")
	def _rtl_consistency(self) -> "LocaleConfigCreate":
		if self.language in SUPPORTED_RTL_LANGUAGES:
			assert self.is_rtl is True, f"language '{self.language}' is RTL — set is_rtl=True and text_direction='rtl'"
			assert self.text_direction == "rtl", f"RTL language '{self.language}' requires text_direction='rtl'"
		return self


class LocaleConfigUpdate(BaseModel):
	model_config = _MODEL_CFG

	date_format: Annotated[str | None, AfterValidator(lambda v: _check_date_format(v) if v else v)] = None
	number_format: Annotated[str | None, AfterValidator(lambda v: _check_number_format(v) if v else v)] = None
	currency_display: Annotated[str | None, AfterValidator(lambda v: _check_currency_display(v) if v else v)] = None
	is_default: bool | None = None
	is_active: bool | None = None
	notes: str | None = None


class LocaleConfigResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	locale_code: str
	language: str
	script: str
	text_direction: str
	date_format: str
	number_format: str
	currency_display: str = "symbol"
	is_default: bool = False
	is_rtl: bool = False
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Translation Entry ---

class TranslationCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	translation_key: str
	source_language: Annotated[str, AfterValidator(_check_language)]
	target_language: Annotated[str, AfterValidator(_check_language)]
	content_type: Annotated[str, AfterValidator(_check_content_type)]
	source_text: str
	translated_text: str
	translator_id: str
	namespace: str = "default"
	version: int = 1
	notes: str | None = None

	@field_validator("target_language")
	@classmethod
	def _no_self_translation(cls, v: str, info: Any) -> str:
		src = info.data.get("source_language", "")
		if src:
			assert v != src, "target_language must differ from source_language"
		return v


class TranslationUpdate(BaseModel):
	model_config = _MODEL_CFG

	translated_text: str | None = None
	status: Annotated[str | None, AfterValidator(lambda v: _check_translation_status(v) if v else v)] = None
	reviewer_id: str | None = None
	notes: str | None = None


class TranslationResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	translation_key: str
	source_language: str
	target_language: str
	content_type: str
	source_text: str
	translated_text: str
	translator_id: str
	namespace: str = "default"
	version: int = 1
	status: str = "draft"
	reviewer_id: str | None = None
	approved_by: str | None = None
	published_by: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Formatting Rule ---

class FormattingRuleCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	locale_id: str
	date_format: Annotated[str, AfterValidator(_check_date_format)]
	number_format: Annotated[str, AfterValidator(_check_number_format)]
	currency_display: Annotated[str, AfterValidator(_check_currency_display)] = "symbol"
	thousand_separator: str = ","
	decimal_separator: str = "."
	time_format_24h: bool = True
	first_day_of_week: int = 1
	notes: str | None = None

	@field_validator("first_day_of_week")
	@classmethod
	def _validate_day(cls, v: int) -> int:
		assert 0 <= v <= 6, f"first_day_of_week must be 0 (Sunday) to 6 (Saturday), got {v}"
		return v


class FormattingRuleResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	locale_id: str
	date_format: str
	number_format: str
	currency_display: str = "symbol"
	thousand_separator: str = ","
	decimal_separator: str = "."
	time_format_24h: bool = True
	first_day_of_week: int = 1
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- Terminology Entry ---

class TerminologyCreate(BaseModel):
	model_config = _MODEL_CFG

	tenant_id: str
	term: str
	language: Annotated[str, AfterValidator(_check_language)]
	definition: str
	domain: str = "general"
	preferred_translation: str | None = None
	forbidden_terms: list[str] = Field(default_factory=list)
	notes: str | None = None


class TerminologyResponse(BaseModel):
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	term: str
	language: str
	definition: str
	domain: str = "general"
	preferred_translation: str | None = None
	forbidden_terms: list[str] = Field(default_factory=list)
	is_active: bool = True
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = "system"


# --- MLG Agent ---

class MlgAgentCreate(BaseModel):
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


class MlgAgentResponse(BaseModel):
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

class MlgAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	reference_id: str
	actor_id: str = "system"
	payload: dict[str, Any] = Field(default_factory=dict)
	processor: str = "bytewax"
	stream: str = "apg.loc.mlg.lifecycle"
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
