"""Pydantic v2 request/response models for Lot and Batch Management."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgLmtResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_lmt"
    ok: bool = True
    message: str = ""


class MfgLmtListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_lmt"
    items: list[MfgLmtResponse] = Field(default_factory=list)
    total: int = 0
