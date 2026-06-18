"""Pydantic v2 request/response models for Advanced Planning and Scheduling."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgApsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_aps"
    ok: bool = True
    message: str = ""


class MfgApsListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_aps"
    items: list[MfgApsResponse] = Field(default_factory=list)
    total: int = 0
