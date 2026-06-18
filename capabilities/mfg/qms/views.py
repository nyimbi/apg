"""Pydantic v2 request/response models for Quality Management System."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MfgQmsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_qms"
    ok: bool = True
    message: str = ""


class MfgQmsListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "mfg_qms"
    items: list[MfgQmsResponse] = Field(default_factory=list)
    total: int = 0
