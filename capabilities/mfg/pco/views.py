"""Pydantic v2 request/response models for Product Costing."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgPcoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_pco"
    ok: bool = True
    message: str = ""


class MfgPcoListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_pco"
    items: list[MfgPcoResponse] = Field(default_factory=list)
    total: int = 0
