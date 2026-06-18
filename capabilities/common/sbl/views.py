"""Pydantic v2 request/response models for SaaS Billing Engine."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommonSblResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_sbl"
    ok: bool = True
    message: str = ""


class CommonSblListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "common_sbl"
    items: list[CommonSblResponse] = Field(default_factory=list)
    total: int = 0
