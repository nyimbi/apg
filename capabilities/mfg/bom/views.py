"""Pydantic v2 request/response models for Bill of Materials."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgBomResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_bom"
    ok: bool = True
    message: str = ""


class MfgBomListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_bom"
    items: list[MfgBomResponse] = Field(default_factory=list)
    total: int = 0
