"""Pydantic v2 request/response models for Computer-Aided Manufacturing."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgCamResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_cam"
    ok: bool = True
    message: str = ""


class MfgCamListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_cam"
    items: list[MfgCamResponse] = Field(default_factory=list)
    total: int = 0
