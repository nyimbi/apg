"""Pydantic v2 request/response models for ECM / Records Management."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CkmEcmResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "ckm_ecm"
    ok: bool = True
    message: str = ""


class CkmEcmListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "ckm_ecm"
    items: list[CkmEcmResponse] = Field(default_factory=list)
    total: int = 0
