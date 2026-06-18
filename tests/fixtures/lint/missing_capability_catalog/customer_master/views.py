"""Pydantic v2 request/response models for Customer Master."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerMasterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "customer_master"
    ok: bool = True
    message: str = ""


class CustomerMasterListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: str = "customer_master"
    items: list[CustomerMasterResponse] = Field(default_factory=list)
    total: int = 0
