"""Pydantic v2 request/response models for Product Lifecycle Management."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgPlmResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_plm"
    ok: bool = True
    message: str = ""


class MfgPlmListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_plm"
    items: list[MfgPlmResponse] = Field(default_factory=list)
    total: int = 0
