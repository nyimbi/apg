"""Pydantic v2 request/response models for int_dsy."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IntDsyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "int_dsy"
    ok: bool = True
    message: str = ""


class IntDsyListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "int_dsy"
    items: list[IntDsyResponse] = Field(default_factory=list)
    total: int = 0
