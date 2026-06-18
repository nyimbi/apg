"""Pydantic v2 request/response models for int_esb."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IntEsbResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "int_esb"
    ok: bool = True
    message: str = ""


class IntEsbListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "int_esb"
    items: list[IntEsbResponse] = Field(default_factory=list)
    total: int = 0
