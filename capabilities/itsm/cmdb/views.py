"""Pydantic v2 request/response models for Configuration Management Database."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ItsmCmdbResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_cmdb"
    ok: bool = True
    message: str = ""


class ItsmCmdbListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "itsm_cmdb"
    items: list[ItsmCmdbResponse] = Field(default_factory=list)
    total: int = 0
