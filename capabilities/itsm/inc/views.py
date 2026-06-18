"""Pydantic v2 request/response models for Incident Management."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ItsmIncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_inc"
    ok: bool = True
    message: str = ""


class ItsmIncListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_inc"
    items: list[ItsmIncResponse] = Field(default_factory=list)
    total: int = 0
