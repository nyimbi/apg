"""Pydantic v2 request/response models for Manufacturing Execution System."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgMesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_mes"
    ok: bool = True
    message: str = ""


class MfgMesListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_mes"
    items: list[MfgMesResponse] = Field(default_factory=list)
    total: int = 0
