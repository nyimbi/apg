"""Pydantic v2 request/response models for proc_sup_portal."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProcSupPortalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "proc_sup_portal"
    ok: bool = True
    message: str = ""


class ProcSupPortalListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "proc_sup_portal"
    items: list[ProcSupPortalResponse] = Field(default_factory=list)
    total: int = 0
