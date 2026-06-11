"""Flask-AppBuilder views and Pydantic v2 request/response models for SASRA Regulatory Reporting."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import ReturnType


# ── Request models ────────────────────────────────────────────────────────────

class QuarterlyReturnRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int
	quarter: int  # 1-4


class AnnualReturnRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int


class AsOfDateRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str | None = None


class PARRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str | None = None
	days: int = 30  # 30 or 90


class LedgerSeedRequest(BaseModel):
	"""Inject ledger data for a given date (dev/test use)."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	as_of_date: str
	data: dict[str, Any] = Field(default_factory=dict)


class FileReturnRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	return_type: ReturnType
	period: str           # e.g. "2025-Q1", "2025-annual"
	filing_officer: str
	submitted_at: str | None = None
	data: dict[str, Any] = Field(default_factory=dict)
	notes: str = ""


class CalendarRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int | None = None


class FilingHistoryRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	from_date: str | None = None
	to_date: str | None = None


class XMLReturnRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	year: int
	quarter: int


class BoardReportRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	period: str | None = None


# ── Response summary models ───────────────────────────────────────────────────

class RatioSummaryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	actual_pct: Decimal
	minimum_pct: Decimal | None = None
	maximum_pct: Decimal | None = None
	compliant: bool
	traffic_light: str
	description: str = ""


class ComplianceSummaryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str
	as_of_date: str
	overall_compliant: bool
	violation_count: int
	warning_count: int
	violations: list[str] = Field(default_factory=list)
	warnings: list[str] = Field(default_factory=list)


class FilingListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0


class CalendarResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	items: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0
	overdue_count: int = 0
	upcoming_count: int = 0
