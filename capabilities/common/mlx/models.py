"""APG MLX result models."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class MLToolType(str, Enum):
	classify = "classify"
	score = "score"
	predict = "predict"
	summarize = "summarize"
	extract = "extract"


class MLBaseResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str = Field(..., description="Ollama model used")
	tool_type: MLToolType
	input_tokens: int = 0
	output_tokens: int = 0
	latency_ms: float = 0.0
	rationale: str = ""


class MLScoreResult(MLBaseResult):
	"""Result of a scoring/risk-assessment tool call."""
	tool_type: MLToolType = MLToolType.score
	score: float = Field(..., ge=0.0, le=1.0, description="Risk/quality score 0–1")
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	factors: list[str] = Field(default_factory=list, description="Key factors driving the score")


class MLClassifyResult(MLBaseResult):
	"""Result of a classification tool call."""
	tool_type: MLToolType = MLToolType.classify
	label: str = Field(..., description="Predicted class label")
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	probabilities: dict[str, float] = Field(default_factory=dict)


class MLPredictResult(MLBaseResult):
	"""Result of a forecasting/prediction tool call."""
	tool_type: MLToolType = MLToolType.predict
	predictions: list[dict[str, Any]] = Field(default_factory=list)
	horizon: int = 0
	confidence_interval: dict[str, float] = Field(default_factory=dict)


class MLSummarizeResult(MLBaseResult):
	"""Result of a text summarization tool call."""
	tool_type: MLToolType = MLToolType.summarize
	summary: str = ""
	key_points: list[str] = Field(default_factory=list)
	word_count: int = 0


class MLExtractResult(MLBaseResult):
	"""Result of a structured data extraction tool call."""
	tool_type: MLToolType = MLToolType.extract
	extracted: dict[str, Any] = Field(default_factory=dict)
	fields_found: list[str] = Field(default_factory=list)
	fields_missing: list[str] = Field(default_factory=list)
