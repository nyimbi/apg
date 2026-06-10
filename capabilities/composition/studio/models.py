"""APG Studio — Pydantic v2 request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    source: str = Field(..., description="APG source code")
    filename: str = Field(default="untitled.apg")


class CompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    success: bool
    files: dict[str, str] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    file_count: int = 0


class CapabilitySummary(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_by_name=True)

    id: str
    display_name: str = ""
    domain: str = ""
    description: str = ""
    provides: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    service_method_count: int = 0


class StudioStats(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    capabilities: int
    domains: int
    tests: int = 1261
    connectors: int = 6


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True)

    files: dict[str, str]
