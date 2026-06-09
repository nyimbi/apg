"""MLX capability — views and Pydantic schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
	MLToolType,
	MLBaseResult,
	MLScoreResult,
	MLClassifyResult,
	MLPredictResult,
	MLSummarizeResult,
	MLExtractResult,
)


class ScoreRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	features: dict[str, Any]
	model: str = "mistral:7b"
	task_description: str = "Score the risk/quality of the provided features."


class ClassifyRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	text: str
	labels: list[str]
	model: str = "mistral:7b"


class PredictRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	series: list[float | dict[str, Any]]
	horizon: int = Field(default=5, ge=1)
	model: str = "mistral:7b"
	context: str = ""


class SummarizeRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	text: str
	max_words: int = Field(default=150, ge=10)
	model: str = "mistral:7b"


class ExtractRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	document: str
	schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for extraction output")
	model: str = "mistral:7b"


class EmbedRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	text: str | list[str]
	model: str = "nomic-embed-text"


class RankRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	query: str
	documents: list[str]
	model: str = "mistral:7b"


class ModelInfo(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	size: int = 0
	loaded: bool = False
	context_window: int = 0


class HealthResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str
	ollama_url: str
	default_model: str
	models_available: list[str] = Field(default_factory=list)


__all__ = [
	"ScoreRequest",
	"ClassifyRequest",
	"PredictRequest",
	"SummarizeRequest",
	"ExtractRequest",
	"EmbedRequest",
	"RankRequest",
	"ModelInfo",
	"HealthResponse",
	"MLToolType",
	"MLScoreResult",
	"MLClassifyResult",
	"MLPredictResult",
	"MLSummarizeResult",
	"MLExtractResult",
]
