"""Pydantic v2 request/response models for Repetitive Manufacturing."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgRfmResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_rfm"
    ok: bool = True
    message: str = ""


class MfgRfmListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_rfm"
    items: list[MfgRfmResponse] = Field(default_factory=list)
    total: int = 0
