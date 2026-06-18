"""Pydantic v2 request/response models for Material Requirements Planning."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgMrpResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_mrp"
    ok: bool = True
    message: str = ""


class MfgMrpListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_mrp"
    items: list[MfgMrpResponse] = Field(default_factory=list)
    total: int = 0
