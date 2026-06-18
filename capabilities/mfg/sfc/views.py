"""Pydantic v2 request/response models for Shop Floor Control."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgSfcResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_sfc"
    ok: bool = True
    message: str = ""


class MfgSfcListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_sfc"
    items: list[MfgSfcResponse] = Field(default_factory=list)
    total: int = 0
