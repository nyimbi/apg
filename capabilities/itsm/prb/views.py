"""Pydantic v2 request/response models for itsm_prb."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ItsmPrbResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_prb"
    ok: bool = True
    message: str = ""


class ItsmPrbListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_prb"
    items: list[ItsmPrbResponse] = Field(default_factory=list)
    total: int = 0
