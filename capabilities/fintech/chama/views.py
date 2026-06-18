"""Pydantic v2 request/response models for Chama & ROSCA Engine."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FintechChamaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "fintech_chama"
    ok: bool = True
    message: str = ""


class FintechChamaListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "fintech_chama"
    items: list[FintechChamaResponse] = Field(default_factory=list)
    total: int = 0
