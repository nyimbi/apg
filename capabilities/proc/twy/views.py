"""Pydantic v2 request/response models for Three-Way Match Engine."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProcTwyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "proc_twy"
    ok: bool = True
    message: str = ""


class ProcTwyListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "proc_twy"
    items: list[ProcTwyResponse] = Field(default_factory=list)
    total: int = 0
