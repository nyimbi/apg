"""Pydantic v2 request/response models for Developer Portal."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonDevpResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_devp"
    ok: bool = True
    message: str = ""


class CommonDevpListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_devp"
    items: list[CommonDevpResponse] = Field(default_factory=list)
    total: int = 0
