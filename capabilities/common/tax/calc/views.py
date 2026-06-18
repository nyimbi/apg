"""Pydantic v2 request/response models for Tax Calculation Engine."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonTaxCalcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_calc"
    ok: bool = True
    message: str = ""


class CommonTaxCalcListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_calc"
    items: list[CommonTaxCalcResponse] = Field(default_factory=list)
    total: int = 0
