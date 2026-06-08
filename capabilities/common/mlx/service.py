"""APG MLX — Ollama-backed ML meta-capability.

All inference runs locally via the Ollama REST API. No data is sent to
external services — data sovereignty is preserved for regulated industries.

Ollama must be running at OLLAMA_BASE_URL (default: http://localhost:11434).
Any model available in the local Ollama instance can be used.

The five ML tools are implemented as structured Ollama chat completions that
request JSON-format responses. Each tool has a typed result model that
capabilities can depend on in their service code.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from .models import (
	MLClassifyResult,
	MLExtractResult,
	MLPredictResult,
	MLScoreResult,
	MLSummarizeResult,
	MLToolType,
)

_log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "mistral:7b"
_TIMEOUT = 120.0  # Ollama inference can be slow on first call (model cold start)


class MLCapability:
	"""Ollama-backed ML tools for APG capabilities.

	Provides five general-purpose ML tools:
	  - score:     Rate a feature vector on a 0–1 scale (fraud, credit risk, lead quality)
	  - classify:  Assign a label from a provided set of classes
	  - predict:   Produce a time-series or event forecast
	  - summarize: Produce a brief summary with key points
	  - extract:   Extract structured fields from unstructured text

	All tools use local Ollama models and return typed Pydantic result objects.
	"""

	def __init__(
		self,
		model: str | None = None,
		ollama_url: str | None = None,
	) -> None:
		self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
		self._base_url = (ollama_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip("/")
		self._client = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)

	async def score(
		self,
		features: dict[str, Any],
		task: str,
		labels: dict[str, str] | None = None,
		context: str = "",
	) -> MLScoreResult:
		"""Score a feature vector on a 0–1 risk/quality scale.

		Args:
			features: Dict of feature names to values
			task: Human-readable task description, e.g. "fraud_risk" or "lead_quality"
			labels: Optional {score_range: description} to guide interpretation
			context: Additional domain context for the model
		"""
		label_hint = ""
		if labels:
			label_hint = "\nScore interpretation:\n" + "\n".join(f"  {k}: {v}" for k, v in labels.items())

		prompt = f"""You are an ML scoring engine. Given the following features, assign a risk/quality score.

Task: {task}
{f'Context: {context}' if context else ''}
Features: {json.dumps(features, default=str)}
{label_hint}

Respond ONLY with valid JSON in this exact format:
{{"score": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, "factors": ["<key factor 1>", "<key factor 2>"], "rationale": "<1-2 sentence explanation>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"score": 0.5, "confidence": 0.5, "factors": [], "rationale": ""})
		return MLScoreResult(
			model=self._model,
			score=min(1.0, max(0.0, float(parsed.get("score", 0.5)))),
			confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
			factors=parsed.get("factors", []),
			rationale=parsed.get("rationale", ""),
			latency_ms=latency_ms,
		)

	async def classify(
		self,
		text: str,
		labels: list[str],
		context: str = "",
	) -> MLClassifyResult:
		"""Classify text into one of the provided labels.

		Args:
			text: The text or feature description to classify
			labels: Candidate class labels (2–20 classes)
			context: Additional domain context
		"""
		prompt = f"""You are a classification engine. Classify the given input into exactly one of the provided labels.

{'Context: ' + context if context else ''}
Labels: {json.dumps(labels)}
Input: {text}

Respond ONLY with valid JSON:
{{"label": "<chosen label from the list>", "confidence": <float 0.0-1.0>, "probabilities": {{"<label>": <prob>, ...}}, "rationale": "<brief reason>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"label": labels[0] if labels else "", "confidence": 0.5})
		chosen = parsed.get("label", "")
		if chosen not in labels and labels:
			# Find closest label if model hallucinated
			chosen = min(labels, key=lambda l: abs(len(l) - len(chosen)))

		return MLClassifyResult(
			model=self._model,
			label=chosen,
			confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
			probabilities=parsed.get("probabilities", {}),
			rationale=parsed.get("rationale", ""),
			latency_ms=latency_ms,
		)

	async def predict(
		self,
		series: list[dict[str, Any]],
		horizon: int,
		task: str = "forecast",
	) -> MLPredictResult:
		"""Produce a time-series or event forecast.

		Args:
			series: Historical data points [{date/period: ..., value: ...}, ...]
			horizon: Number of future periods to forecast
			task: Task description, e.g. "monthly_sales_forecast"
		"""
		prompt = f"""You are a forecasting engine. Given historical data, predict the next {horizon} periods.

Task: {task}
Historical data (last {len(series)} points): {json.dumps(series[-20:], default=str)}
Horizon: {horizon} future periods

Respond ONLY with valid JSON:
{{"predictions": [{{"period": "<label>", "value": <number>, "lower": <number>, "upper": <number>}}], "confidence_interval": {{"level": 0.95}}, "rationale": "<brief methodology note>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"predictions": [], "confidence_interval": {}, "rationale": ""})
		return MLPredictResult(
			model=self._model,
			predictions=parsed.get("predictions", []),
			horizon=horizon,
			confidence_interval=parsed.get("confidence_interval", {}),
			rationale=parsed.get("rationale", ""),
			latency_ms=latency_ms,
		)

	async def summarize(
		self,
		text: str,
		max_words: int = 100,
		focus: str = "",
	) -> MLSummarizeResult:
		"""Summarize text with key points.

		Args:
			text: Text to summarize (documents, conversations, records)
			max_words: Maximum summary word count
			focus: Specific aspect to focus on, e.g. "compliance issues"
		"""
		prompt = f"""Summarize the following text in under {max_words} words.
{'Focus on: ' + focus if focus else ''}

Text:
{text[:4000]}

Respond ONLY with valid JSON:
{{"summary": "<concise summary>", "key_points": ["<point 1>", "<point 2>", "<point 3>"]}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"summary": "", "key_points": []})
		summary = parsed.get("summary", "")
		return MLSummarizeResult(
			model=self._model,
			summary=summary,
			key_points=parsed.get("key_points", []),
			word_count=len(summary.split()),
			rationale="",
			latency_ms=latency_ms,
		)

	async def extract(
		self,
		text: str,
		schema: dict[str, str],
		context: str = "",
	) -> MLExtractResult:
		"""Extract structured fields from unstructured text.

		Args:
			text: Unstructured text (invoices, forms, emails, clinical notes)
			schema: {field_name: field_description} describing what to extract
			context: Document type or domain context
		"""
		schema_desc = "\n".join(f"  - {k}: {v}" for k, v in schema.items())
		prompt = f"""Extract the specified fields from the text. Return null for fields not found.

{'Document type: ' + context if context else ''}
Fields to extract:
{schema_desc}

Text:
{text[:4000]}

Respond ONLY with valid JSON where keys match exactly the field names:
{{{', '.join(f'"{k}": <value or null>' for k in schema)}}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {})
		found = [k for k in schema if parsed.get(k) is not None]
		missing = [k for k in schema if parsed.get(k) is None]
		return MLExtractResult(
			model=self._model,
			extracted={k: v for k, v in parsed.items() if k in schema},
			fields_found=found,
			fields_missing=missing,
			rationale="",
			latency_ms=latency_ms,
		)

	async def health_check(self) -> bool:
		"""Return True if Ollama is reachable and the configured model is available."""
		try:
			resp = await self._client.get("/api/tags", timeout=5.0)
			resp.raise_for_status()
			tags = resp.json().get("models", [])
			names = [m.get("name", "").split(":")[0] for m in tags]
			model_base = self._model.split(":")[0]
			return model_base in names
		except Exception:
			return False

	# ── private ──────────────────────────────────────────────────────────

	async def _generate(self, prompt: str) -> str:
		"""Call Ollama /api/generate and return the response text."""
		try:
			resp = await self._client.post(
				"/api/generate",
				json={
					"model": self._model,
					"prompt": prompt,
					"format": "json",
					"stream": False,
					"options": {"temperature": 0.1, "top_p": 0.9},
				},
			)
			resp.raise_for_status()
			return resp.json().get("response", "")
		except Exception as exc:
			_log.warning("Ollama generate failed (%s): %s", self._model, exc)
			return "{}"

	@staticmethod
	def _parse_json(text: str, default: dict[str, Any]) -> dict[str, Any]:
		"""Parse JSON from Ollama response, returning default on failure."""
		text = text.strip()
		# Find first { ... } block in case model prepends prose
		start = text.find("{")
		end = text.rfind("}") + 1
		if start >= 0 and end > start:
			text = text[start:end]
		try:
			return json.loads(text)
		except (json.JSONDecodeError, ValueError):
			_log.debug("Failed to parse MLX JSON response: %r", text[:200])
			return default
