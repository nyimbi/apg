"""Pydantic v2 request/response models for common_mlr."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonMlrResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_mlr"
    ok: bool = True
    message: str = ""


class CommonMlrListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_mlr"
    items: list[CommonMlrResponse] = Field(default_factory=list)
    total: int = 0
