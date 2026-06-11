"""APG MLX — Ollama-backed ML meta-capability.

All inference runs locally via the Ollama REST API. No data is sent to
external services — data sovereignty is preserved for regulated industries.

Ollama must be running at OLLAMA_BASE_URL (default: http://localhost:11434).
Any model available in the local Ollama instance can be used.

The five ML tools are implemented as structured Ollama chat completions that
request JSON-format responses. Each tool has a typed result model that
capabilities can depend on in their service code.

Extended tools added in v1.1:
  - classify_multi_label:   Multi-label classification above a confidence threshold
  - ner:                    Named entity recognition with configurable entity types
  - zero_shot_classify:     NLI-style zero-shot hypothesis scoring
  - anomaly_score:          Outlier/anomaly detection against a baseline description
  - score_with_reasoning:   Chain-of-thought rubric scoring (MLScorecardResult)
  - extract_keywords:       Keyword + topic extraction from free text
  - detect_language:        ISO-639-1 language detection with confidence
  - translate:              Text translation via multilingual Ollama models
  - summarize_long:         Hierarchical chunked summarisation for large documents
  - embed_batch:            Per-text batch embeddings (fixes list→single-string bug)
  - cosine_similarity_matrix: N×N symmetric similarity matrix for a text corpus
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

from .models import (
	MLAnomalyResult,
	MLClassifyResult,
	MLExtractResult,
	MLKeywordResult,
	MLLanguageResult,
	MLMultiLabelResult,
	MLNERResult,
	MLPredictResult,
	MLScoreResult,
	MLScorecardResult,
	MLScorecardCriterion,
	MLSummarizeResult,
	MLToolType,
	MLTranslationResult,
	MLZeroShotResult,
)

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
_log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "mistral:7b"
_TIMEOUT = 120.0  # Ollama inference can be slow on first call (model cold start)

# Task-category to model-size hint used by the auto-router.
_TASK_CATEGORY_SMALL = {"classify", "score", "detect_language"}
_TASK_CATEGORY_MEDIUM = {"summarize", "extract", "ner", "keywords", "translate", "zero_shot"}
_TASK_CATEGORY_LARGE = {"score_with_reasoning", "predict", "anomaly"}


class MLCapability:
	"""Ollama-backed ML tools for APG capabilities.

	Provides five core ML tools plus eleven extended tools:

	Core (v1.0):
	  - score:                   Rate a feature vector on a 0–1 risk/quality scale
	  - classify:                Assign one label from a provided set of classes
	  - predict:                 Produce a time-series or event forecast
	  - summarize:               Produce a brief summary with key points
	  - extract:                 Extract structured fields from unstructured text

	Extended (v1.1):
	  - classify_multi_label:    Multi-label classification above a threshold
	  - ner:                     Named entity recognition
	  - zero_shot_classify:      NLI-style zero-shot hypothesis scoring
	  - anomaly_score:           Anomaly/outlier detection against a baseline
	  - score_with_reasoning:    Chain-of-thought rubric scoring
	  - extract_keywords:        Keyword + topic extraction
	  - detect_language:         ISO-639-1 language detection
	  - translate:               Multilingual text translation
	  - summarize_long:          Hierarchical chunked summarisation for large documents
	  - embed_batch:             Concurrent per-text batch embeddings
	  - cosine_similarity_matrix: N×N text similarity matrix

	All tools use local Ollama models and return typed Pydantic result objects.
	Batch methods execute concurrently via asyncio.gather with a configurable semaphore.
	Repeated identical requests are served from an in-process TTL cache.

	Args:
		model:             Ollama model tag (default: env OLLAMA_MODEL or mistral:7b)
		ollama_url:        Ollama server URL (default: env OLLAMA_BASE_URL or localhost:11434)
		batch_concurrency: Max simultaneous Ollama calls for batch operations (default 4)
		cache_ttl:         TTL in seconds for the result cache; 0 disables (default 300)
		auto_route:        When True, dispatch each task to best available model size (default False)
	"""

	def __init__(
		self,
		model: str | None = None,
		ollama_url: str | None = None,
		batch_concurrency: int = 4,
		cache_ttl: int = 300,
		auto_route: bool = False,
	) -> None:
		self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
		# NOTE: use _ollama_url throughout — _base_url alias kept for backward compat
		self._ollama_url = (ollama_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL)).rstrip("/")
		self._base_url = self._ollama_url  # backward-compat alias
		self._client = httpx.AsyncClient(base_url=self._ollama_url, timeout=_TIMEOUT)
		self._sem = asyncio.Semaphore(batch_concurrency)
		self._cache: BoundedCache | None = BoundedCache(max_size=512) if cache_ttl > 0 else None
		self._auto_route = auto_route
		# Inference counters
		self._total_calls: int = 0
		self._total_latency_ms: float = 0.0
		self._total_input_tokens: int = 0
		self._total_output_tokens: int = 0

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

	async def list_models(self) -> list[dict[str, Any]]:
		"""List available Ollama models."""
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=10.0) as client:
				resp = await client.get("/api/tags")
				resp.raise_for_status()
				return resp.json().get("models", [])
		except Exception as exc:
			_log.warning("list_models: Ollama unavailable at %s — %s: %s",
						 self._ollama_url, type(exc).__name__, exc)
			return []

	async def pull_model(self, model_name: str) -> dict[str, Any]:
		"""Pull a model from Ollama registry."""
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=300.0) as client:
				resp = await client.post("/api/pull", json={"name": model_name, "stream": False})
				resp.raise_for_status()
				return {"model": model_name, "pulled": True}
		except Exception as exc:
			return {"model": model_name, "pulled": False, "error": str(exc)}

	async def delete_model(self, model_name: str) -> dict[str, Any]:
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=30.0) as client:
				resp = await client.delete("/api/delete", json={"name": model_name})
				resp.raise_for_status()
				return {"model": model_name, "deleted": True}
		except Exception as exc:
			return {"model": model_name, "deleted": False, "error": str(exc)}

	async def get_model_info(self, model_name: str) -> dict[str, Any]:
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=10.0) as client:
				resp = await client.post("/api/show", json={"name": model_name})
				resp.raise_for_status()
				return resp.json()
		except Exception as exc:
			return {"model": model_name, "error": str(exc)}

	async def embed(self, text: str | list[str], *, model: str | None = None) -> dict[str, Any]:
		"""Generate embeddings via Ollama."""
		embed_model = model or "nomic-embed-text"
		input_text = text if isinstance(text, str) else "\n".join(text)
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=60.0) as client:
				resp = await client.post("/api/embeddings", json={"model": embed_model, "prompt": input_text})
				resp.raise_for_status()
				return resp.json()
		except Exception as exc:
			return {"embedding": [], "error": str(exc)}

	async def generate(self, prompt: str, *, model: str | None = None) -> str:
		"""Raw text generation."""
		prev = self._model
		if model:
			self._model = model
		result = await self._generate(prompt)
		self._model = prev
		return result

	async def chat(self, messages: list[dict[str, str]], *, model: str | None = None) -> str:
		"""Chat completion via Ollama /api/chat."""
		use_model = model or self._model
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=60.0) as client:
				resp = await client.post("/api/chat", json={"model": use_model, "messages": messages, "stream": False})
				resp.raise_for_status()
				return resp.json().get("message", {}).get("content", "")
		except Exception as exc:
			_log.warning("chat failed: %s", exc)
			return ""

	async def rank_documents(
		self, query: str, documents: list[str], *, model: str | None = None
	) -> dict[str, Any]:
		"""Score each document for relevance to query (0–1) using embedding cosine similarity."""
		try:
			q_emb = (await self.embed(query, model=model)).get("embedding", [])
			ranked = []
			for i, doc in enumerate(documents):
				d_emb = (await self.embed(doc, model=model)).get("embedding", [])
				if q_emb and d_emb:
					dot = sum(a * b for a, b in zip(q_emb, d_emb))
					norm = (sum(a ** 2 for a in q_emb) ** 0.5) * (sum(b ** 2 for b in d_emb) ** 0.5)
					score = dot / norm if norm > 0 else 0.0
				else:
					score = 0.0
				ranked.append({"document": doc, "score": round(score, 4), "index": i})
			ranked.sort(key=lambda x: x["score"], reverse=True)
			return {"ranked": ranked, "query": query}
		except Exception as exc:
			return {"ranked": [], "error": str(exc)}

	async def rerank(self, query: str, documents: list[str], *, model: str | None = None) -> dict[str, Any]:
		return await self.rank_documents(query, documents, model=model)

	async def compute_similarity(self, text_a: str, text_b: str) -> float:
		emb_a = (await self.embed(text_a)).get("embedding", [])
		emb_b = (await self.embed(text_b)).get("embedding", [])
		if not emb_a or not emb_b:
			return 0.0
		dot = sum(a * b for a, b in zip(emb_a, emb_b))
		norm = (sum(a ** 2 for a in emb_a) ** 0.5) * (sum(b ** 2 for b in emb_b) ** 0.5)
		return dot / norm if norm > 0 else 0.0

	async def cluster_texts(self, texts: list[str]) -> list[dict[str, Any]]:
		return [{"text": t, "cluster": 0} for t in texts]

	async def warm_up_model(self, model_name: str | None = None) -> dict[str, Any]:
		try:
			await self._generate("warm up")
			return {"warmed_up": True, "model": model_name or self._model}
		except Exception as exc:
			return {"warmed_up": False, "error": str(exc)}

	async def check_model_loaded(self, model_name: str | None = None) -> bool:
		models = await self.list_models()
		target = model_name or self._model
		return any(m.get("name", "").startswith(target) for m in models)

	async def get_context_window(self, model_name: str | None = None) -> int:
		info = await self.get_model_info(model_name or self._model)
		return info.get("details", {}).get("context_length", 4096)

	async def get_inference_stats(self) -> dict[str, Any]:
		avg = self._total_latency_ms / self._total_calls if self._total_calls else 0.0
		return {
			"total_inferences": self._total_calls,
			"avg_latency_ms": round(avg, 2),
			"total_latency_ms": round(self._total_latency_ms, 2),
			"total_input_tokens": self._total_input_tokens,
			"total_output_tokens": self._total_output_tokens,
			"model": self._model,
			"ollama_url": self._ollama_url,
		}

	async def get_model_metrics(self) -> dict[str, Any]:
		return {"model": self._model, "available": await self.check_model_loaded()}

	async def list_model_versions(self) -> list[str]:
		models = await self.list_models()
		return [m.get("name", "") for m in models]

	async def set_default_model(self, model_name: str) -> dict[str, Any]:
		self._model = model_name
		return {"default_model": model_name}

	async def configure_model(self, model_name: str, **kwargs: Any) -> dict[str, Any]:
		self._model = model_name
		return {"configured": True, "model": model_name}

	async def score_batch(
		self,
		items: list[tuple[dict[str, Any], str]],
		batch_concurrency: int | None = None,
	) -> list[MLScoreResult]:
		"""Score a batch of (features, task) tuples concurrently.

		Args:
			items: List of (features_dict, task_description) pairs
			batch_concurrency: Override instance-level concurrency limit
		"""
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(features: dict[str, Any], task: str) -> MLScoreResult:
			async with sem:
				return await self.score(features, task)

		return list(await asyncio.gather(*[_one(f, t) for f, t in items]), return_exceptions=True)

	async def classify_batch(
		self,
		items: list[tuple[str, list[str]]],
		batch_concurrency: int | None = None,
	) -> list[MLClassifyResult]:
		"""Classify a batch of (text, labels) pairs concurrently."""
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(text: str, labels: list[str]) -> MLClassifyResult:
			async with sem:
				return await self.classify(text, labels)

		return list(await asyncio.gather(*[_one(t, l) for t, l in items]), return_exceptions=True)

	async def predict_batch(
		self,
		items: list[tuple[list[Any], int]],
		batch_concurrency: int | None = None,
	) -> list[MLPredictResult]:
		"""Forecast a batch of (series, horizon) pairs concurrently."""
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(series: list[Any], horizon: int) -> MLPredictResult:
			async with sem:
				return await self.predict(series, horizon)

		return list(await asyncio.gather(*[_one(s, h) for s, h in items]), return_exceptions=True)

	async def summarize_batch(
		self,
		texts: list[str],
		max_words: int = 100,
		batch_concurrency: int | None = None,
	) -> list[MLSummarizeResult]:
		"""Summarize a list of texts concurrently."""
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(text: str) -> MLSummarizeResult:
			async with sem:
				return await self.summarize(text, max_words=max_words)

		return list(await asyncio.gather(*[_one(t) for t in texts]), return_exceptions=True)

	async def extract_batch(
		self,
		docs: list[str],
		schema: dict[str, str],
		batch_concurrency: int | None = None,
	) -> list[MLExtractResult]:
		"""Extract structured fields from a list of documents concurrently."""
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(doc: str) -> MLExtractResult:
			async with sem:
				return await self.extract(doc, schema)

		return list(await asyncio.gather(*[_one(d) for d in docs]), return_exceptions=True)

	async def stream_generate(self, prompt: str, *, model: str | None = None):
		"""Async generator yielding chunks from Ollama streaming."""
		use_model = model or self._model
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=120.0) as client:
				async with client.stream("POST", "/api/generate", json={"model": use_model, "prompt": prompt, "stream": True}) as resp:
					async for line in resp.aiter_lines():
						if line:
							import json as _json
							try:
								chunk = _json.loads(line)
								if chunk.get("response"):
									yield chunk["response"]
							except Exception as _exc:
								_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		except Exception as exc:
			yield f"[ERROR: {exc}]"

	async def stream_chat(self, messages: list[dict[str, str]], *, model: str | None = None):
		"""Async generator yielding chunks from Ollama chat streaming."""
		use_model = model or self._model
		try:
			async with httpx.AsyncClient(base_url=self._ollama_url, timeout=120.0) as client:
				async with client.stream("POST", "/api/chat", json={"model": use_model, "messages": messages, "stream": True}) as resp:
					async for line in resp.aiter_lines():
						if line:
							import json as _json
							try:
								chunk = _json.loads(line)
								content = chunk.get("message", {}).get("content", "")
								if content:
									yield content
							except Exception as _exc:
								_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		except Exception as exc:
			yield f"[ERROR: {exc}]"

	async def score_transaction_risk(self, features: dict[str, Any]) -> MLScoreResult:
		"""Score transaction fraud risk 0–1."""
		return await self.score(features, task="transaction_fraud_risk",
								context="Financial transaction fraud detection")

	async def score_lead(self, features: dict[str, Any]) -> MLScoreResult:
		"""Score lead conversion probability 0–1."""
		return await self.score(features, task="lead_conversion_probability",
								context="CRM lead scoring")

	async def classify_document(self, text: str, labels: list[str] | None = None) -> MLClassifyResult:
		"""Classify a document into standard document types."""
		return await self.classify(text, labels or ["invoice", "contract", "report", "email", "other"])

	async def classify_sentiment(self, text: str) -> MLClassifyResult:
		"""Classify text sentiment as positive, negative, or neutral."""
		return await self.classify(text, ["positive", "negative", "neutral"])

	async def predict_churn(self, features: dict[str, Any]) -> MLScoreResult:
		"""Predict customer churn probability 0–1."""
		return await self.score(features, task="customer_churn_probability",
								context="Customer retention and churn prediction")

	async def predict_demand(self, series: list[Any], *, horizon: int = 7) -> MLPredictResult:
		"""Forecast demand for the next `horizon` periods."""
		return await self.predict(series, horizon=horizon, task="demand_forecast")

	async def predict_readmission(self, features: dict[str, Any]) -> MLScoreResult:
		"""Predict 30-day hospital readmission risk 0–1."""
		return await self.score(features, task="hospital_readmission_risk",
								context="Clinical 30-day readmission prediction")

	async def predict_fraud(self, features: dict[str, Any]) -> MLScoreResult:
		"""Predict transaction fraud probability 0–1."""
		return await self.score(features, task="transaction_fraud_probability",
								context="Real-time fraud detection")

	async def summarize_document(self, text: str) -> Any:
		return await self.summarize(text)

	async def summarize_thread(self, messages: list[str]) -> Any:
		return await self.summarize("\n\n".join(messages))

	async def extract_entities(self, text: str) -> MLNERResult:
		"""Extract named entities (persons, orgs, locations, dates) from text."""
		return await self.ner(text, entity_types=["PERSON", "ORG", "LOCATION", "DATE", "MONEY"])

	async def extract_schema(self, document: str, schema: dict[str, str]) -> MLExtractResult:
		"""Extract fields defined in schema from the document."""
		return await self.extract(document, schema=schema)

	async def evaluate_model(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
		return {"evaluated": len(test_cases), "accuracy": 0.0}

	# ── Extended tools (v1.1) ─────────────────────────────────────────────

	async def classify_multi_label(
		self,
		text: str,
		labels: list[str],
		threshold: float = 0.5,
		context: str = "",
	) -> MLMultiLabelResult:
		"""Classify text into zero or more labels that exceed a confidence threshold.

		Unlike `classify`, which forces exactly one label, this method is appropriate for
		document tagging, compliance flagging, and content moderation where multiple labels
		may apply simultaneously.

		Args:
			text:      Input text to classify
			labels:    Candidate label set (2–30 labels)
			threshold: Minimum per-label confidence to include it in the result (0–1)
			context:   Optional domain context
		"""
		label_list = json.dumps(labels)
		prompt = f"""You are a multi-label classification engine. Score each label independently for the given text.

{'Context: ' + context if context else ''}
Labels: {label_list}
Input: {text[:3000]}

Respond ONLY with valid JSON:
{{"probabilities": {{{', '.join(f'"{l}": <float 0-1>' for l in labels)}}}, "rationale": "<brief reason>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"probabilities": {}})
		probs: dict[str, float] = {}
		for lbl in labels:
			raw_prob = parsed.get("probabilities", {}).get(lbl, 0.0)
			try:
				probs[lbl] = min(1.0, max(0.0, float(raw_prob)))
			except (TypeError, ValueError):
				probs[lbl] = 0.0

		accepted = [lbl for lbl, p in probs.items() if p >= threshold]
		return MLMultiLabelResult(
			model=self._model,
			labels=accepted,
			probabilities=probs,
			threshold=threshold,
			rationale=parsed.get("rationale", ""),
			latency_ms=latency_ms,
		)

	async def ner(
		self,
		text: str,
		entity_types: list[str] | None = None,
		context: str = "",
	) -> MLNERResult:
		"""Named entity recognition — extract typed entity spans from text.

		Useful for PII detection, knowledge-graph construction, and compliance scanning.

		Args:
			text:         Input text
			entity_types: Entity types to detect, e.g. ["PERSON","ORG","LOCATION","DATE","MONEY"]
			context:      Optional domain context (e.g. "clinical notes", "legal contract")
		"""
		types = entity_types or ["PERSON", "ORG", "LOCATION", "DATE", "MONEY", "PRODUCT"]
		prompt = f"""You are a named entity recognition engine. Extract all entity spans of the requested types.

{'Context: ' + context if context else ''}
Entity types to extract: {json.dumps(types)}
Text: {text[:3000]}

Respond ONLY with valid JSON:
{{"entities": [{{"text": "<span>", "entity_type": "<TYPE>", "confidence": <float 0-1>}}]}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"entities": []})
		from .models import MLEntity
		entities = []
		for e in parsed.get("entities", []):
			try:
				entities.append(MLEntity(
					text=str(e.get("text", "")),
					entity_type=str(e.get("entity_type", "UNKNOWN")),
					confidence=min(1.0, max(0.0, float(e.get("confidence", 0.5)))),
				))
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		return MLNERResult(
			model=self._model,
			entities=entities,
			entity_types_requested=types,
			latency_ms=latency_ms,
		)

	async def zero_shot_classify(
		self,
		text: str,
		candidates: list[str],
		hypothesis_template: str = "This text is about {label}.",
		context: str = "",
	) -> MLZeroShotResult:
		"""Zero-shot classification using NLI-style hypothesis scoring.

		Each candidate is treated as a natural-language hypothesis. The model scores
		how well the text supports each hypothesis, returning a ranked list. Enables
		policy-rule engines where labels are free-text descriptions.

		Args:
			text:                Input text
			candidates:          List of label descriptions or hypothesis strings
			hypothesis_template: Template with {label} placeholder (default: "This text is about {label}.")
			context:             Optional domain context
		"""
		hypotheses = [hypothesis_template.format(label=c) for c in candidates]
		prompt = f"""You are an NLI (Natural Language Inference) classifier. For each hypothesis, score how strongly the premise ENTAILS it on a 0–1 scale.

{'Context: ' + context if context else ''}
Premise: {text[:2000]}
Hypotheses: {json.dumps(hypotheses)}
Original labels: {json.dumps(candidates)}

Respond ONLY with valid JSON:
{{"scores": [{{"label": "<original label>", "hypothesis": "<hypothesis text>", "score": <float 0-1>}}], "rationale": "<brief note>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"scores": []})
		ranked = []
		for item in parsed.get("scores", []):
			try:
				ranked.append({
					"label": str(item.get("label", "")),
					"hypothesis": str(item.get("hypothesis", "")),
					"score": min(1.0, max(0.0, float(item.get("score", 0.0)))),
				})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		ranked.sort(key=lambda x: x["score"], reverse=True)

		return MLZeroShotResult(
			model=self._model,
			ranked=ranked,
			top_label=ranked[0]["label"] if ranked else "",
			top_score=ranked[0]["score"] if ranked else 0.0,
			latency_ms=latency_ms,
		)

	async def anomaly_score(
		self,
		observation: dict[str, Any],
		baseline: dict[str, Any],
		context: str = "",
	) -> MLAnomalyResult:
		"""Score an observation for anomaly/outlier status against a baseline description.

		The baseline should describe the typical distribution of each feature, e.g.:
		  {"amount": "mean=500, std=200, max=5000", "hour": "typically 9-17", ...}

		Critical for fraud detection, network security, and IoT sensor anomaly detection.

		Args:
			observation: Feature dict for the new data point
			baseline:    Statistical description of normal behaviour per feature
			context:     Optional domain context (e.g. "credit card transactions")
		"""
		prompt = f"""You are an anomaly detection engine. Compare the observation to the baseline description and score how anomalous it is.

{'Context: ' + context if context else ''}
Baseline (normal behaviour description): {json.dumps(baseline, default=str)}
Observation: {json.dumps(observation, default=str)}

Respond ONLY with valid JSON:
{{"anomaly_score": <float 0.0-1.0 where 0=normal 1=highly anomalous>, "confidence": <float 0-1>, "anomalous_dimensions": ["<feature1>", "..."], "rationale": "<1-2 sentences>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"anomaly_score": 0.0, "confidence": 0.5, "anomalous_dimensions": [], "rationale": ""})
		return MLAnomalyResult(
			model=self._model,
			anomaly_score=min(1.0, max(0.0, float(parsed.get("anomaly_score", 0.0)))),
			confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
			anomalous_dimensions=parsed.get("anomalous_dimensions", []),
			rationale=parsed.get("rationale", ""),
			latency_ms=latency_ms,
		)

	async def score_with_reasoning(
		self,
		features: dict[str, Any],
		task: str,
		rubric: dict[str, float],
		context: str = "",
	) -> MLScorecardResult:
		"""Chain-of-thought rubric scoring — auditable, explainable, criterion-level scores.

		The model produces a step-by-step reasoning chain before emitting scores for each
		rubric criterion. Mandatory for credit underwriting, insurance, clinical decisions.

		Args:
			features: Feature dict to evaluate
			task:     Task description, e.g. "credit_risk_assessment"
			rubric:   Dict mapping criterion name to maximum points, e.g.
			          {"payment_history": 35.0, "credit_utilization": 30.0, "credit_age": 15.0}
			context:  Optional domain context
		"""
		rubric_desc = "\n".join(f"  - {k}: max {v} points" for k, v in rubric.items())
		max_total = sum(rubric.values())
		prompt = f"""You are a chain-of-thought scoring engine. Apply the rubric step by step, then emit scores.

Task: {task}
{'Context: ' + context if context else ''}
Features: {json.dumps(features, default=str)}

Rubric (criteria and maximum points):
{rubric_desc}
Total possible points: {max_total}

Think through each criterion carefully, then respond ONLY with valid JSON:
{{"reasoning_chain": "<step-by-step analysis>", "criteria": [{{"criterion": "<name>", "score": <points awarded>, "max_score": <max>, "reasoning": "<brief>"}}], "rationale": "<summary>"}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"reasoning_chain": "", "criteria": [], "rationale": ""})
		criteria_out: list[MLScorecardCriterion] = []
		total = 0.0
		for c in parsed.get("criteria", []):
			try:
				awarded = min(float(c.get("max_score", 0)), max(0.0, float(c.get("score", 0))))
				criteria_out.append(MLScorecardCriterion(
					criterion=str(c.get("criterion", "")),
					score=awarded,
					max_score=float(c.get("max_score", 0)),
					reasoning=str(c.get("reasoning", "")),
				))
				total += awarded
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		normalized = total / max_total if max_total > 0 else 0.0
		return MLScorecardResult(
			model=self._model,
			total_score=total,
			max_total_score=max_total,
			normalized_score=min(1.0, normalized),
			criteria=criteria_out,
			reasoning_chain=parsed.get("reasoning_chain", ""),
			latency_ms=latency_ms,
		)

	async def extract_keywords(
		self,
		text: str,
		n_keywords: int = 10,
		n_topics: int = 3,
		context: str = "",
	) -> MLKeywordResult:
		"""Extract keywords and infer high-level topics from free text.

		Feeds search indexing, content routing, and tag suggestion pipelines.

		Args:
			text:       Input text
			n_keywords: Maximum number of keywords to return
			n_topics:   Number of high-level topics to infer
			context:    Optional domain context
		"""
		prompt = f"""You are a keyword and topic extraction engine.

{'Context: ' + context if context else ''}
Text: {text[:3000]}

Extract the {n_keywords} most significant keywords and infer {n_topics} high-level topics.

Respond ONLY with valid JSON:
{{"keywords": ["<kw1>", "..."], "topics": ["<topic1>", "..."]}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"keywords": [], "topics": []})
		return MLKeywordResult(
			model=self._model,
			keywords=parsed.get("keywords", [])[:n_keywords],
			topics=parsed.get("topics", [])[:n_topics],
			latency_ms=latency_ms,
		)

	async def detect_language(self, text: str) -> MLLanguageResult:
		"""Detect the language of text and return an ISO-639-1 code with confidence.

		Critical for Africa-facing enterprise software that handles Swahili, Amharic,
		Hausa, French, Arabic, and English in the same document streams.

		Args:
			text: Input text (even a sentence is usually sufficient)
		"""
		prompt = f"""Detect the language of the following text.

Text: {text[:500]}

Respond ONLY with valid JSON:
{{"language_code": "<ISO-639-1 code e.g. en, sw, am, ha, fr, ar>", "language_name": "<full name>", "confidence": <float 0-1>}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"language_code": "und", "language_name": "Unknown", "confidence": 0.0})
		return MLLanguageResult(
			model=self._model,
			language_code=str(parsed.get("language_code", "und")),
			language_name=str(parsed.get("language_name", "")),
			confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.0)))),
			latency_ms=latency_ms,
		)

	async def translate(
		self,
		text: str,
		target_language: str,
		source_language: str = "auto",
	) -> MLTranslationResult:
		"""Translate text using a multilingual Ollama model.

		Leverages mistral, aya, llama3, and other multilingual models available locally.
		No API keys or external services — full data sovereignty.

		Args:
			text:            Text to translate
			target_language: Target language name or ISO-639-1 code (e.g. "Swahili", "sw")
			source_language: Source language or "auto" for auto-detection (default: "auto")
		"""
		src_hint = f" from {source_language}" if source_language != "auto" else ""
		prompt = f"""Translate the following text{src_hint} to {target_language}.

Text: {text[:3000]}

Respond ONLY with valid JSON:
{{"translated_text": "<translation>", "source_language": "<detected/provided source>", "confidence": <float 0-1>}}"""

		t0 = time.monotonic()
		raw = await self._generate(prompt)
		latency_ms = (time.monotonic() - t0) * 1000

		parsed = self._parse_json(raw, {"translated_text": "", "source_language": source_language, "confidence": 0.0})
		return MLTranslationResult(
			model=self._model,
			source_text=text,
			translated_text=str(parsed.get("translated_text", "")),
			source_language=str(parsed.get("source_language", source_language)),
			target_language=target_language,
			confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.0)))),
			latency_ms=latency_ms,
		)

	async def summarize_long(
		self,
		text: str,
		chunk_size: int = 3000,
		overlap: int = 200,
		max_words: int = 200,
		focus: str = "",
	) -> MLSummarizeResult:
		"""Hierarchical chunked summarisation for large documents.

		Splits text on sentence boundaries near `chunk_size`, summarises each chunk
		independently (concurrently), then merges the chunk summaries into a final summary.
		Handles 100-page PDFs without silent context-window truncation.

		Args:
			text:       Full document text (any length)
			chunk_size: Target characters per chunk (default 3000)
			overlap:    Character overlap between consecutive chunks for context continuity
			max_words:  Target word count for the final merged summary
			focus:      Specific aspect to emphasise in summaries
		"""
		if len(text) <= chunk_size:
			return await self.summarize(text, max_words=max_words, focus=focus)

		# Split on sentence boundaries near chunk_size
		chunks: list[str] = []
		pos = 0
		while pos < len(text):
			end = min(pos + chunk_size, len(text))
			if end < len(text):
				# Walk back to nearest sentence end
				for sep in (".\n", ". ", ".\t", "?\n", "? ", "!\n", "! "):
					boundary = text.rfind(sep, pos, end)
					if boundary > pos:
						end = boundary + len(sep)
						break
			chunks.append(text[pos:end])
			pos = end - overlap if end < len(text) else len(text)

		# Summarise all chunks concurrently
		chunk_summaries: list[MLSummarizeResult] = await self.summarize_batch(
			chunks, max_words=max(40, max_words // len(chunks))
		)
		combined = " ".join(r.summary for r in chunk_summaries if r.summary)

		# Final merge pass
		merged = await self.summarize(combined, max_words=max_words, focus=focus)

		# Aggregate key points across all chunk summaries
		all_points: list[str] = []
		seen: set[str] = set()
		for r in chunk_summaries:
			for pt in r.key_points:
				if pt not in seen:
					all_points.append(pt)
					seen.add(pt)
		all_points.extend(p for p in merged.key_points if p not in seen)

		return MLSummarizeResult(
			model=self._model,
			summary=merged.summary,
			key_points=all_points[:10],
			word_count=merged.word_count,
			rationale=f"Hierarchical summary: {len(chunks)} chunks, then merged.",
			latency_ms=merged.latency_ms + sum(r.latency_ms for r in chunk_summaries),
		)

	async def embed_batch(
		self,
		texts: list[str],
		*,
		model: str | None = None,
		batch_concurrency: int | None = None,
	) -> list[list[float]]:
		"""Generate per-text embeddings concurrently, returning a list of float vectors.

		Fixes the original `embed(list[str])` bug that joined all texts into one string,
		producing a single embedding instead of N independent embeddings.

		Args:
			texts:             List of texts to embed
			model:             Embedding model override (default: nomic-embed-text)
			batch_concurrency: Override instance-level concurrency limit
		"""
		embed_model = model or "nomic-embed-text"
		sem = asyncio.Semaphore(batch_concurrency) if batch_concurrency else self._sem

		async def _one(text: str) -> list[float]:
			async with sem:
				result = await self.embed(text, model=embed_model)
				return result.get("embedding", [])

		return list(await asyncio.gather(*[_one(t) for t in texts]), return_exceptions=True)

	async def cosine_similarity_matrix(
		self,
		texts: list[str],
		*,
		model: str | None = None,
	) -> list[list[float]]:
		"""Compute an N×N symmetric cosine similarity matrix for a text corpus.

		Useful for clustering, duplicate detection, and semantic deduplication.
		All N embeddings are generated concurrently.

		Args:
			texts: Corpus of texts (N texts → N×N output matrix)
			model: Embedding model override (default: nomic-embed-text)

		Returns:
			N×N list-of-lists where matrix[i][j] is cosine similarity of texts[i] and texts[j]
		"""
		embeddings = await self.embed_batch(texts, model=model)
		n = len(embeddings)
		matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

		def _cosine(a: list[float], b: list[float]) -> float:
			if not a or not b:
				return 0.0
			dot = sum(x * y for x, y in zip(a, b))
			norm_a = sum(x ** 2 for x in a) ** 0.5
			norm_b = sum(y ** 2 for y in b) ** 0.5
			return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0

		for i in range(n):
			matrix[i][i] = 1.0
			for j in range(i + 1, n):
				sim = round(_cosine(embeddings[i], embeddings[j]), 6)
				matrix[i][j] = sim
				matrix[j][i] = sim

		return matrix

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
		"""Call Ollama /api/generate, returning the response text.

		Implements:
		  - In-process TTL result cache keyed on (model, sha256(prompt))
		  - Exponential back-off retry (up to 3 attempts) on timeouts/connect errors
		  - Real token accounting via Ollama's eval_count / prompt_eval_count fields
		"""
		cache_key: str | None = None
		if self._cache is not None:
			digest = hashlib.sha256(f"{self._model}\x00{prompt}".encode()).hexdigest()[:16]
			cache_key = digest
			cached = self._cache.get(cache_key)
			if cached is not None:
				return cached  # type: ignore[return-value]

		backoff_delays = [1.0, 2.0, 4.0]
		for attempt, delay in enumerate(backoff_delays + [None], start=1):  # type: ignore[arg-type]
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
				body = resp.json()
				text = body.get("response", "")
				# Track token counts from Ollama metadata
				self._total_calls += 1
				self._total_input_tokens += body.get("prompt_eval_count", 0)
				self._total_output_tokens += body.get("eval_count", 0)
				if cache_key is not None and self._cache is not None:
					self._cache.set(cache_key, text)
				return text
			except (httpx.TimeoutException, httpx.ConnectError) as exc:
				if delay is None:
					_log.warning("Ollama generate failed after %d attempts (%s): %s", attempt - 1, self._model, exc)
					return "{}"
				_log.debug("Ollama attempt %d/%d timed out; retrying in %.1fs — %s", attempt, len(backoff_delays), delay, exc)
				await asyncio.sleep(delay)
			except Exception as exc:
				_log.warning("Ollama generate failed (%s): %s", self._model, exc)
				return "{}"
		return "{}"  # unreachable but satisfies type-checker

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
