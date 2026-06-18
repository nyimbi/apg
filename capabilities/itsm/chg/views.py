"""Pydantic v2 request/response models for itsm_chg."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ItsmChgResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_chg"
    ok: bool = True
    message: str = ""


class ItsmChgListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_chg"
    items: list[ItsmChgResponse] = Field(default_factory=list)
    total: int = 0
