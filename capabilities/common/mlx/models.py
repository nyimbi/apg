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


# ── Extended result models ─────────────────────────────────────────────────


class MLMultiLabelResult(BaseModel):
	"""Result of a multi-label classification call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	labels: list[str] = Field(default_factory=list, description="Labels above confidence threshold")
	probabilities: dict[str, float] = Field(default_factory=dict)
	threshold: float = 0.5
	rationale: str = ""
	latency_ms: float = 0.0


class MLEntity(BaseModel):
	"""A single named entity span."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	text: str
	entity_type: str
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MLNERResult(BaseModel):
	"""Result of a named entity recognition call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	entities: list[MLEntity] = Field(default_factory=list)
	entity_types_requested: list[str] = Field(default_factory=list)
	latency_ms: float = 0.0


class MLAnomalyResult(BaseModel):
	"""Result of an anomaly scoring call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	anomaly_score: float = Field(..., ge=0.0, le=1.0, description="0=normal, 1=highly anomalous")
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	anomalous_dimensions: list[str] = Field(default_factory=list)
	rationale: str = ""
	latency_ms: float = 0.0


class MLScorecardCriterion(BaseModel):
	"""A single scored criterion in a rubric."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	criterion: str
	score: float
	max_score: float
	reasoning: str = ""


class MLScorecardResult(BaseModel):
	"""Result of a chain-of-thought rubric scoring call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	total_score: float
	max_total_score: float
	normalized_score: float = Field(description="total_score / max_total_score, range 0–1")
	criteria: list[MLScorecardCriterion] = Field(default_factory=list)
	reasoning_chain: str = ""
	latency_ms: float = 0.0


class MLTranslationResult(BaseModel):
	"""Result of a translation call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	source_text: str
	translated_text: str
	source_language: str = ""
	target_language: str
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	latency_ms: float = 0.0


class MLLanguageResult(BaseModel):
	"""Result of a language detection call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	language_code: str = Field(description="ISO-639-1 language code, e.g. 'en', 'sw'")
	language_name: str = ""
	confidence: float = Field(default=0.5, ge=0.0, le=1.0)
	latency_ms: float = 0.0


class MLKeywordResult(BaseModel):
	"""Result of a keyword extraction call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	keywords: list[str] = Field(default_factory=list)
	topics: list[str] = Field(default_factory=list, description="Inferred high-level topics")
	latency_ms: float = 0.0


class MLZeroShotResult(BaseModel):
	"""Result of a zero-shot / hypothesis classification call."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	model: str
	ranked: list[dict[str, Any]] = Field(default_factory=list, description="[{label, score}, ...] sorted desc")
	top_label: str = ""
	top_score: float = 0.0
	latency_ms: float = 0.0
