"""Pydantic v2 request/response models for Capacity Planning."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgCapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_cap"
    ok: bool = True
    message: str = ""


class MfgCapListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_cap"
    items: list[MfgCapResponse] = Field(default_factory=list)
    total: int = 0
