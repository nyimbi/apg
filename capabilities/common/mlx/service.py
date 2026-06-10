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
		return {"total_inferences": 0, "avg_latency_ms": 0, "model": self._model, "ollama_url": self._ollama_url}

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

	async def score_batch(self, items: list[dict[str, Any]], *, model: str | None = None) -> list[Any]:
		return [await self.score(item, model=model or self._model) for item in items]

	async def classify_batch(self, items: list[tuple[str, list[str]]], *, model: str | None = None) -> list[Any]:
		return [await self.classify(text, labels, model=model or self._model) for text, labels in items]

	async def predict_batch(self, series_list: list[list[Any]], *, model: str | None = None) -> list[Any]:
		return [await self.predict(s, model=model or self._model) for s in series_list]

	async def summarize_batch(self, texts: list[str], *, model: str | None = None) -> list[Any]:
		return [await self.summarize(t, model=model or self._model) for t in texts]

	async def extract_batch(self, docs: list[str], schema: dict[str, Any] | None = None, *, model: str | None = None) -> list[Any]:
		return [await self.extract(d, schema=schema or {}, model=model or self._model) for d in docs]

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
							except Exception:
								pass
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
							except Exception:
								pass
		except Exception as exc:
			yield f"[ERROR: {exc}]"

	async def score_transaction_risk(self, features: dict[str, Any]) -> Any:
		return await self.score(features, task_description="Score transaction fraud risk 0-1")

	async def score_lead(self, features: dict[str, Any]) -> Any:
		return await self.score(features, task_description="Score lead conversion probability 0-1")

	async def classify_document(self, text: str, labels: list[str] | None = None) -> Any:
		return await self.classify(text, labels or ["invoice", "contract", "report", "email", "other"])

	async def classify_sentiment(self, text: str) -> Any:
		return await self.classify(text, ["positive", "negative", "neutral"])

	async def predict_churn(self, features: dict[str, Any]) -> Any:
		return await self.score(features, task_description="Predict customer churn probability 0-1")

	async def predict_demand(self, series: list[Any], *, horizon: int = 7) -> Any:
		return await self.predict(series, horizon=horizon)

	async def predict_readmission(self, features: dict[str, Any]) -> Any:
		return await self.score(features, task_description="Predict 30-day hospital readmission risk 0-1")

	async def predict_fraud(self, features: dict[str, Any]) -> Any:
		return await self.score(features, task_description="Predict transaction fraud probability 0-1")

	async def summarize_document(self, text: str) -> Any:
		return await self.summarize(text)

	async def summarize_thread(self, messages: list[str]) -> Any:
		return await self.summarize("\n\n".join(messages))

	async def extract_entities(self, text: str) -> Any:
		return await self.extract(text, schema={"entities": [{"type": "str", "text": "str"}]})

	async def extract_schema(self, document: str, schema: dict[str, Any]) -> Any:
		return await self.extract(document, schema=schema)

	async def evaluate_model(self, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
		return {"evaluated": len(test_cases), "accuracy": 0.0}

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
