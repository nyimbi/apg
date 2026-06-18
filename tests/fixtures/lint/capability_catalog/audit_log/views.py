"""Pydantic v2 request/response models for Audit Log."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "audit_log"
    ok: bool = True
    message: str = ""


class AuditLogListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "audit_log"
    items: list[AuditLogResponse] = Field(default_factory=list)
    total: int = 0
