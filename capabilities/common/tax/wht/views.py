"""Pydantic v2 request/response models for common_tax_wht."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonTaxWhtResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_wht"
    ok: bool = True
    message: str = ""


class CommonTaxWhtListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_tax_wht"
    items: list[CommonTaxWhtResponse] = Field(default_factory=list)
    total: int = 0
