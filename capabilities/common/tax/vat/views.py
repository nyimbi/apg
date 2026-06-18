"""Pydantic v2 request/response models for common_tax_vat."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonTaxVatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_vat"
    ok: bool = True
    message: str = ""


class CommonTaxVatListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_vat"
    items: list[CommonTaxVatResponse] = Field(default_factory=list)
    total: int = 0
