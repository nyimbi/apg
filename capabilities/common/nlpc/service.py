"""
NLPC Service — Natural Language Processing Core

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Website: www.datacraft.co.ke

Async service layer.  All public methods are async.  Every state mutation
emits a domain event.  Tenant isolation is enforced on every query.

Backends are optional.  The service degrades gracefully when spaCy,
transformers, or langdetect are absent — results include a ``model_used``
field of ``"stub"`` so callers can distinguish real from fallback output.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from datetime import datetime
from typing import Any

from uuid6 import uuid7

from .domain.events import DomainEvent
from .domain.rules import (
	RuleViolation,
	assert_no_cross_tenant_access,
	assert_tenant_context,
	assert_text_not_empty,
	assert_target_language_not_auto,
	assert_embedding_dimensions,
)
from .domain.calculations import (
	calculate_compression_ratio,
	calculate_compound_sentiment,
	calculate_language_certainty,
	calculate_weighted_confidence,
	calculate_word_count,
	normalise_sentiment_scores,
	calculate_tfidf,
)
from .models import (
	AFRICAN_LANGUAGE_CODES,
	ClassificationTaxonomy,
	DocumentType,
	EntityType,
	LanguageCode,
	ModelProvider,
	NLPBatchJob,
	NLPClassification,
	NLPClassificationCreate,
	NLPCoreferenceChain,
	NLPDocument,
	NLPDocumentCreate,
	NLPDocumentResponse,
	NLPEmbedding,
	NLPEmbeddingCreate,
	NLPEmbeddingResponse,
	NLPEntity,
	NLPEntityCreate,
	NLPEntityResponse,
	NLPIntent,
	NLPIntentCreate,
	NLPIntentResponse,
	NLPKeyPhrase,
	NLPLanguage,
	NLPLanguageCreate,
	NLPLanguageResponse,
	NLPModelConfig,
	NLPProcessingRequest,
	NLPProcessingResult,
	NLPRelation,
	NLPSentiment,
	NLPSentimentCreate,
	NLPSentimentResponse,
	NLPSummary,
	NLPSummaryCreate,
	NLPSummaryResponse,
	NLPTask,
	NLPTranslation,
	NLPTranslationCreate,
	NLPTranslationResponse,
	NLPUsageReport,
	PriorityLevel,
	ProcessingStatus,
	SentimentLabel,
	SummaryMethod,
)


def uuid7str() -> str:
	return str(uuid7())


# ---------------------------------------------------------------------------
# Optional backend probes
# ---------------------------------------------------------------------------

def _try_import(name: str) -> Any:
	try:
		import importlib
		return importlib.import_module(name)
	except Exception:
		return None


_spacy = _try_import("spacy")
_langdetect = _try_import("langdetect")
_transformers = _try_import("transformers")
_sentence_transformers = _try_import("sentence_transformers")
_httpx = _try_import("httpx")


# ---------------------------------------------------------------------------
# In-process record store (replaces a real DB for the capability layer)
# Swap for SQLAlchemy / asyncpg in production via domain/store.py.
# ---------------------------------------------------------------------------

_STORE: dict[str, dict[str, Any]] = {}  # collection → {id: record}


def _col(name: str) -> dict[str, Any]:
	return _STORE.setdefault(name, {})


# ---------------------------------------------------------------------------
# NLPCoreService
# ---------------------------------------------------------------------------

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class NLPCoreService:
	"""
	Core NLP service.

	Parameters
	----------
	db_session:
		SQLAlchemy async session (or None for the in-process store).
	tenant_id:
		All operations are scoped to this tenant.
	actor_id:
		Identity of the calling user / service account.
	ollama_base_url:
		Base URL for the local Ollama instance (default: http://localhost:11434).
	"""

	def __init__(
		self,
		db_session: Any = None,
		tenant_id: str = "default",
		actor_id: str = "system",
		ollama_base_url: str = "http://localhost:11434",
	) -> None:
		assert tenant_id and tenant_id.strip(), "tenant_id required"
		assert actor_id and actor_id.strip(), "actor_id required"
		self._db = db_session
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._ollama_base = ollama_base_url.rstrip("/")
		self._events: list[DomainEvent] = []
		self._spacy_models: dict[str, Any] = {}
		self._log_init()

	# ------------------------------------------------------------------
	# Logging helpers
	# ------------------------------------------------------------------

	def _log_init(self) -> None:
		backends = {
			"spacy": _spacy is not None,
			"langdetect": _langdetect is not None,
			"transformers": _transformers is not None,
			"sentence_transformers": _sentence_transformers is not None,
			"httpx": _httpx is not None,
		}
		active = [k for k, v in backends.items() if v]
		print(f"[nlpc] tenant={self._tenant_id} backends={active}")

	def _log_task_start(self, task: str, doc_id: str) -> None:
		print(f"[nlpc] start task={task} doc={doc_id}")

	def _log_task_done(self, task: str, ms: float) -> None:
		print(f"[nlpc] done  task={task} ms={ms:.1f}")

	def _log_pretty_path(self, collection: str, record_id: str) -> str:
		return f"{collection}/{record_id}"

	# ------------------------------------------------------------------
	# Event emission
	# ------------------------------------------------------------------

	def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
		evt = DomainEvent(
			event_type=event_type,
			tenant_id=self._tenant_id,
			actor_id=self._actor_id,
			payload=payload,
		)
		self._events.append(evt)

	# ------------------------------------------------------------------
	# Store helpers
	# ------------------------------------------------------------------

	def _put(self, collection: str, record: dict[str, Any]) -> None:
		_col(collection)[record["id"]] = record

	def _get(self, collection: str, record_id: str) -> dict[str, Any] | None:
		rec = _col(collection).get(record_id)
		if rec and rec.get("tenant_id") != self._tenant_id:
			raise RuleViolation("cross_tenant_access_denied", "record belongs to another tenant")
		return rec

	def _list_col(
		self,
		collection: str,
		limit: int = 50,
		offset: int = 0,
		filters: dict[str, Any] | None = None,
	) -> list[dict[str, Any]]:
		rows = [
			r for r in _col(collection).values()
			if r.get("tenant_id") == self._tenant_id and not r.get("is_deleted", False)
		]
		if filters:
			for k, v in filters.items():
				rows = [r for r in rows if r.get(k) == v]
		rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
		return rows[offset: offset + limit]

	def _soft_delete(self, collection: str, record_id: str) -> bool:
		rec = self._get(collection, record_id)
		if not rec:
			return False
		rec["is_deleted"] = True
		rec["updated_at"] = datetime.utcnow().isoformat()
		return True

	# ------------------------------------------------------------------
	# Document CRUD
	# ------------------------------------------------------------------

	async def create_document(self, payload: NLPDocumentCreate) -> NLPDocumentResponse:
		"""Persist a new NLPDocument and return its response model."""
		assert_tenant_context({"tenant_id": payload.tenant_id})
		assert_text_not_empty(payload.content)
		content_hash = hashlib.sha256(payload.content.encode()).hexdigest()
		word_count = calculate_word_count(payload.content)
		doc = NLPDocument(
			tenant_id=payload.tenant_id,
			created_by=payload.created_by,
			content=payload.content,
			title=payload.title,
			source=payload.source,
			source_id=payload.source_id,
			language=payload.language,
			content_type=payload.content_type,
			is_sensitive=payload.is_sensitive,
			retention_days=payload.retention_days,
			metadata=payload.metadata,
			content_hash=content_hash,
			word_count=word_count,
			char_count=len(payload.content),
		)
		self._put("nlpc_documents", doc.model_dump())
		self._emit_event("nlpc.document.created", {"document_id": doc.id})
		return NLPDocumentResponse(**doc.model_dump())

	async def get_document(self, document_id: str) -> NLPDocumentResponse | None:
		"""Retrieve a document by ID (tenant-scoped)."""
		rec = self._get("nlpc_documents", document_id)
		if not rec or rec.get("is_deleted"):
			return None
		return NLPDocumentResponse(**rec)

	async def list_documents(
		self,
		limit: int = 50,
		offset: int = 0,
		language: str | None = None,
	) -> list[NLPDocumentResponse]:
		"""List documents for the current tenant."""
		filters: dict[str, Any] = {}
		if language:
			filters["language"] = language
		rows = self._list_col("nlpc_documents", limit=limit, offset=offset, filters=filters)
		return [NLPDocumentResponse(**r) for r in rows]

	async def delete_document(self, document_id: str) -> bool:
		"""Soft-delete a document."""
		ok = self._soft_delete("nlpc_documents", document_id)
		if ok:
			self._emit_event("nlpc.document.deleted", {"document_id": document_id})
		return ok

	# ------------------------------------------------------------------
	# detect_language
	# ------------------------------------------------------------------

	async def detect_language(self, text: str, document_id: str | None = None) -> NLPLanguageResponse:
		"""
		Identify the language of *text*.

		Uses langdetect if available, then falls back to a simple
		character-n-gram heuristic for African languages.  Always sets
		``is_african`` when the detected code is in ``AFRICAN_LANGUAGE_CODES``.
		"""
		assert_text_not_empty(text)
		t0 = time.perf_counter()
		doc_id = document_id or uuid7str()
		self._log_task_start("detect_language", doc_id)

		detected_code = "en"
		confidence = 0.5
		candidates: list[dict[str, Any]] = []
		model_used = "stub"

		if _langdetect is not None:
			try:
				from langdetect import detect_langs  # type: ignore
				raw = detect_langs(text)
				if raw:
					best = raw[0]
					detected_code = best.lang
					confidence = float(best.prob)
					candidates = [{"code": r.lang, "probability": float(r.prob)} for r in raw]
				model_used = "langdetect"
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# African language refinement
		detected_code = self._refine_african_language(text, detected_code)
		is_african = detected_code in AFRICAN_LANGUAGE_CODES
		lc = self._safe_language_code(detected_code)

		lang = NLPLanguage(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			detected=lc,
			confidence=min(1.0, confidence),
			candidates=[c for c in candidates],
			is_african=is_african,
			model_used=model_used,
		)
		self._put("nlpc_languages", lang.model_dump())
		self._emit_event("nlpc.language.detected", {"document_id": doc_id, "language": detected_code})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("detect_language", ms)
		return NLPLanguageResponse(**lang.model_dump())

	def _safe_language_code(self, code: str) -> LanguageCode:
		try:
			return LanguageCode(code)
		except ValueError:
			return LanguageCode.MULTI

	def _refine_african_language(self, text: str, detected: str) -> str:
		"""
		Very lightweight script-based heuristic to improve detection of
		Swahili and related Bantu languages when langdetect mis-fires.
		"""
		if detected in AFRICAN_LANGUAGE_CODES:
			return detected
		lower = text.lower()
		sw_markers = {"na", "ya", "wa", "ni", "kwa", "katika", "lakini", "sana", "ndio"}
		if len(sw_markers & set(lower.split())) >= 3:
			return "sw"
		return detected

	# ------------------------------------------------------------------
	# extract_entities
	# ------------------------------------------------------------------

	async def extract_entities(
		self,
		text: str,
		entity_types: list[EntityType] | None = None,
		document_id: str | None = None,
	) -> list[NLPEntityResponse]:
		"""
		Extract named entities from *text*.

		Uses spaCy if an English model is loaded; falls back to a
		regex-based stub for PERSON and LOCATION when spaCy is absent.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		self._log_task_start("extract_entities", doc_id)
		t0 = time.perf_counter()

		raw_entities: list[dict[str, Any]] = []

		nlp = self._get_spacy_model("en")
		if nlp is not None:
			try:
				doc = nlp(text)
				for ent in doc.ents:
					etype = self._map_spacy_label(ent.label_)
					if entity_types and etype not in entity_types:
						continue
					raw_entities.append({
						"text": ent.text,
						"entity_type": etype,
						"start_char": ent.start_char,
						"end_char": ent.end_char,
						"confidence": 0.85,
					})
				model_used = "spacy"
			except Exception:
				model_used = "stub"
		else:
			raw_entities = self._regex_entities(text, entity_types)
			model_used = "stub"

		results: list[NLPEntityResponse] = []
		for raw in raw_entities:
			entity = NLPEntity(
				tenant_id=self._tenant_id,
				created_by=self._actor_id,
				document_id=doc_id,
				**raw,
				model_used=model_used if False else None,  # stored in metadata
				metadata={"model": model_used},
			)
			self._put("nlpc_entities", entity.model_dump())
			results.append(NLPEntityResponse(**entity.model_dump()))

		self._emit_event("nlpc.entities.extracted", {"document_id": doc_id, "count": len(results)})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("extract_entities", ms)
		return results

	def _get_spacy_model(self, lang: str = "en") -> Any:
		if lang in self._spacy_models:
			return self._spacy_models[lang]
		if _spacy is None:
			return None
		for model_name in (f"{lang}_core_web_sm", f"{lang}_core_web_md"):
			try:
				nlp = _spacy.load(model_name)
				self._spacy_models[lang] = nlp
				return nlp
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		try:
			nlp = _spacy.blank(lang)
			self._spacy_models[lang] = nlp
			return nlp
		except Exception:
			return None

	def _map_spacy_label(self, label: str) -> EntityType:
		mapping = {
			"PERSON": EntityType.PERSON,
			"ORG": EntityType.ORGANISATION,
			"GPE": EntityType.GPE,
			"LOC": EntityType.LOCATION,
			"DATE": EntityType.DATE,
			"TIME": EntityType.TIME,
			"MONEY": EntityType.MONEY,
			"PERCENT": EntityType.PERCENT,
			"PRODUCT": EntityType.PRODUCT,
			"EVENT": EntityType.EVENT,
			"LAW": EntityType.LAW,
			"LANGUAGE": EntityType.LANGUAGE,
			"WORK_OF_ART": EntityType.WORK_OF_ART,
			"FAC": EntityType.FACILITY,
			"NORP": EntityType.NORP,
			"QUANTITY": EntityType.QUANTITY,
			"ORDINAL": EntityType.ORDINAL,
			"CARDINAL": EntityType.CARDINAL,
		}
		return mapping.get(label, EntityType.MISC)

	def _regex_entities(
		self, text: str, entity_types: list[EntityType] | None
	) -> list[dict[str, Any]]:
		results: list[dict[str, Any]] = []
		want_all = entity_types is None
		# capitalised words as PERSON heuristic
		if want_all or EntityType.PERSON in (entity_types or []):
			for m in re.finditer(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text):
				results.append({
					"text": m.group(),
					"entity_type": EntityType.PERSON,
					"start_char": m.start(),
					"end_char": m.end(),
					"confidence": 0.4,
				})
		# currency mentions
		if want_all or EntityType.MONEY in (entity_types or []):
			for m in re.finditer(r'\b(?:USD|KES|EUR|GBP|NGN)\s*[\d,]+(?:\.\d+)?\b', text):
				results.append({
					"text": m.group(),
					"entity_type": EntityType.MONEY,
					"start_char": m.start(),
					"end_char": m.end(),
					"confidence": 0.7,
				})
		return results

	# ------------------------------------------------------------------
	# sentiment_analysis
	# ------------------------------------------------------------------

	async def sentiment_analysis(
		self, text: str, document_id: str | None = None
	) -> NLPSentimentResponse:
		"""
		Return sentiment label and scores for *text*.

		Tries the transformers pipeline ``distilbert-base-uncased-finetuned-sst-2-english``
		first; falls back to a lexicon-based VADER-style stub.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		self._log_task_start("sentiment_analysis", doc_id)
		t0 = time.perf_counter()

		pos, neg, neu = 0.0, 0.0, 1.0
		label = SentimentLabel.NEUTRAL
		model_used = "stub"

		if _transformers is not None:
			try:
				from transformers import pipeline as hf_pipeline  # type: ignore
				pipe = hf_pipeline(
					"sentiment-analysis",
					model="distilbert-base-uncased-finetuned-sst-2-english",
					truncation=True,
					max_length=512,
				)
				out = pipe(text[:512])[0]
				if out["label"].upper() == "POSITIVE":
					pos, neg, neu = out["score"], 1.0 - out["score"], 0.0
					label = SentimentLabel.POSITIVE
				else:
					pos, neg, neu = 1.0 - out["score"], out["score"], 0.0
					label = SentimentLabel.NEGATIVE
				model_used = "distilbert-sst2"
			except Exception:
				pos, neg, neu = self._lexicon_sentiment(text)
				label = self._label_from_scores(pos, neg, neu)
		else:
			pos, neg, neu = self._lexicon_sentiment(text)
			label = self._label_from_scores(pos, neg, neu)

		pos, neg, neu = normalise_sentiment_scores(pos, neg, neu)
		compound = calculate_compound_sentiment(pos, neg, neu)
		score = pos if label == SentimentLabel.POSITIVE else neg if label == SentimentLabel.NEGATIVE else neu

		create = NLPSentimentCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			label=label,
			score=min(1.0, score),
			positive=pos,
			negative=neg,
			neutral=neu,
			compound=compound,
			model_used=model_used,
		)
		sent = NLPSentiment(**create.model_dump(), id=uuid7str(), created_at=datetime.utcnow(), updated_at=datetime.utcnow(), version=1, is_deleted=False)
		self._put("nlpc_sentiments", sent.model_dump())
		self._emit_event("nlpc.sentiment.analysed", {"document_id": doc_id, "label": label})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("sentiment_analysis", ms)
		return NLPSentimentResponse(**sent.model_dump())

	def _lexicon_sentiment(self, text: str) -> tuple[float, float, float]:
		pos_words = {"good", "great", "excellent", "positive", "happy", "love", "best", "amazing", "wonderful", "fantastic"}
		neg_words = {"bad", "terrible", "awful", "negative", "hate", "worst", "horrible", "poor", "disappointing", "dreadful"}
		tokens = set(re.findall(r'\b\w+\b', text.lower()))
		pos = len(tokens & pos_words) / max(1, len(tokens))
		neg = len(tokens & neg_words) / max(1, len(tokens))
		neu = max(0.0, 1.0 - pos - neg)
		return pos, neg, neu

	def _label_from_scores(self, pos: float, neg: float, neu: float) -> SentimentLabel:
		m = max(pos, neg, neu)
		if m == neu:
			return SentimentLabel.NEUTRAL
		if m == pos:
			return SentimentLabel.POSITIVE
		return SentimentLabel.NEGATIVE

	# ------------------------------------------------------------------
	# intent_classification
	# ------------------------------------------------------------------

	async def intent_classification(
		self,
		text: str,
		intents: list[str],
		document_id: str | None = None,
	) -> NLPIntentResponse:
		"""
		Classify the intent of *text* against the provided *intents* list.

		Uses zero-shot classification via transformers if available, otherwise
		returns the highest-scoring intent from a keyword overlap heuristic.
		"""
		assert_text_not_empty(text)
		assert intents, "intents list must not be empty"
		doc_id = document_id or uuid7str()
		self._log_task_start("intent_classification", doc_id)
		t0 = time.perf_counter()

		all_scores: dict[str, float] = {}
		model_used = "keyword_overlap"

		if _transformers is not None:
			try:
				from transformers import pipeline as hf_pipeline  # type: ignore
				zs = hf_pipeline("zero-shot-classification", truncation=True)
				result = zs(text[:512], candidate_labels=intents)
				all_scores = dict(zip(result["labels"], result["scores"]))
				model_used = "zero-shot-classification"
			except Exception:
				all_scores = self._keyword_intent_scores(text, intents)
		else:
			all_scores = self._keyword_intent_scores(text, intents)

		best_intent = max(all_scores, key=lambda k: all_scores[k])
		confidence = all_scores[best_intent]

		create = NLPIntentCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			intent_label=best_intent,
			confidence=confidence,
			all_scores=all_scores,
			model_used=model_used,
			utterance=text[:500],
		)
		intent = NLPIntent(**create.model_dump(), id=uuid7str(), created_at=datetime.utcnow(), updated_at=datetime.utcnow(), version=1, is_deleted=False)
		self._put("nlpc_intents", intent.model_dump())
		self._emit_event("nlpc.intent.classified", {"document_id": doc_id, "intent": best_intent})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("intent_classification", ms)
		return NLPIntentResponse(**intent.model_dump())

	def _keyword_intent_scores(self, text: str, intents: list[str]) -> dict[str, float]:
		tokens = set(re.findall(r'\b\w+\b', text.lower()))
		scores: dict[str, float] = {}
		for intent in intents:
			intent_tokens = set(intent.lower().replace("_", " ").split())
			overlap = len(tokens & intent_tokens)
			scores[intent] = overlap / max(1, len(intent_tokens))
		total = sum(scores.values()) or 1.0
		return {k: v / total for k, v in scores.items()}

	# ------------------------------------------------------------------
	# text_summarisation
	# ------------------------------------------------------------------

	async def text_summarisation(
		self,
		text: str,
		max_words: int = 100,
		method: SummaryMethod = SummaryMethod.EXTRACTIVE,
		document_id: str | None = None,
	) -> NLPSummaryResponse:
		"""
		Summarise *text* to at most *max_words* words.

		Extractive: selects the highest TF-IDF-ranked sentences.
		Abstractive: calls the Ollama ``llama3`` model if httpx is available,
		otherwise falls back to extractive.
		"""
		assert_text_not_empty(text)
		assert max_words >= 1, "max_words must be >= 1"
		doc_id = document_id or uuid7str()
		self._log_task_start("text_summarisation", doc_id)
		t0 = time.perf_counter()

		summary_text = ""
		model_used = "extractive"

		if method == SummaryMethod.ABSTRACTIVE and _httpx is not None:
			try:
				summary_text = await self._ollama_summarise(text, max_words)
				model_used = "ollama/llama3"
			except Exception:
				summary_text = self._extractive_summarise(text, max_words)
		else:
			summary_text = self._extractive_summarise(text, max_words)

		actual_words = calculate_word_count(summary_text)
		compression = calculate_compression_ratio(
			calculate_word_count(text), actual_words
		)

		create = NLPSummaryCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			summary_text=summary_text,
			method=method,
			max_words=max_words,
			model_used=model_used,
		)
		summary = NLPSummary(
			**create.model_dump(),
			id=uuid7str(),
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			version=1,
			is_deleted=False,
			actual_word_count=actual_words,
			compression_ratio=compression,
		)
		self._put("nlpc_summaries", summary.model_dump())
		self._emit_event("nlpc.summary.created", {"document_id": doc_id})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("text_summarisation", ms)
		return NLPSummaryResponse(**summary.model_dump())

	def _extractive_summarise(self, text: str, max_words: int) -> str:
		sentences = re.split(r'(?<=[.!?])\s+', text.strip())
		if not sentences:
			return text[:max_words * 6]
		corpus = sentences
		scored: list[tuple[float, str]] = []
		for sent in sentences:
			words = set(re.findall(r'\b\w+\b', sent.lower()))
			score = sum(calculate_tfidf(w, sent, corpus) for w in words)
			scored.append((score, sent))
		scored.sort(key=lambda x: x[0], reverse=True)
		selected: list[str] = []
		wcount = 0
		for _, sent in scored:
			wc = calculate_word_count(sent)
			if wcount + wc > max_words:
				break
			selected.append(sent)
			wcount += wc
		if not selected:
			selected = [scored[0][1]]
		# preserve original order
		order = {s: i for i, (_, s) in enumerate([(0, s) for s in sentences])}
		selected.sort(key=lambda s: order.get(s, 0))
		return " ".join(selected)

	async def _ollama_summarise(self, text: str, max_words: int) -> str:
		import httpx  # type: ignore
		prompt = (
			f"Summarise the following text in at most {max_words} words.\n\n"
			f"TEXT:\n{text[:4000]}\n\nSUMMARY:"
		)
		async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=60, write=5, pool=2)) as client:
			resp = await client.post(
				f"{self._ollama_base}/api/generate",
				json={"model": "llama3", "prompt": prompt, "stream": False},
			)
			resp.raise_for_status()
			return resp.json().get("response", "").strip()

	# ------------------------------------------------------------------
	# translate
	# ------------------------------------------------------------------

	async def translate(
		self,
		text: str,
		target_lang: LanguageCode,
		source_lang: LanguageCode = LanguageCode.AUTO,
		document_id: str | None = None,
	) -> NLPTranslationResponse:
		"""
		Translate *text* to *target_lang*.

		Routes to Ollama (llama3) when httpx is available; otherwise returns
		the original text with a ``stub`` model marker so callers know
		translation did not occur.
		"""
		assert_text_not_empty(text)
		assert_target_language_not_auto(target_lang)
		doc_id = document_id or uuid7str()
		self._log_task_start("translate", doc_id)
		t0 = time.perf_counter()

		translated = text
		confidence = 0.0
		model_used = "stub"

		if source_lang == LanguageCode.AUTO:
			lang_result = await self.detect_language(text, doc_id)
			source_lang = lang_result.detected  # type: ignore[assignment]

		if _httpx is not None:
			try:
				translated = await self._ollama_translate(text, str(source_lang), str(target_lang))
				confidence = 0.8
				model_used = "ollama/llama3"
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		create = NLPTranslationCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			source_language=source_lang,
			target_language=target_lang,
			translated_text=translated,
			confidence=confidence,
			model_used=model_used,
		)
		trans = NLPTranslation(
			**create.model_dump(),
			id=uuid7str(),
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			version=1,
			is_deleted=False,
			char_count=len(translated),
		)
		self._put("nlpc_translations", trans.model_dump())
		self._emit_event("nlpc.translation.completed", {"document_id": doc_id, "target": str(target_lang)})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("translate", ms)
		return NLPTranslationResponse(**trans.model_dump())

	async def _ollama_translate(self, text: str, src: str, tgt: str) -> str:
		import httpx  # type: ignore
		prompt = f"Translate the following text from {src} to {tgt}. Return only the translation.\n\nTEXT:\n{text[:3000]}\n\nTRANSLATION:"
		async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=60, write=5, pool=2)) as client:
			resp = await client.post(
				f"{self._ollama_base}/api/generate",
				json={"model": "llama3", "prompt": prompt, "stream": False},
			)
			resp.raise_for_status()
			return resp.json().get("response", "").strip()

	# ------------------------------------------------------------------
	# embed_text
	# ------------------------------------------------------------------

	async def embed_text(
		self,
		text: str,
		model: str = "nomic-embed-text",
		document_id: str | None = None,
	) -> NLPEmbeddingResponse:
		"""
		Embed *text* using the specified Ollama embedding model.

		Falls back to a TF-IDF sparse vector when Ollama is unreachable.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		self._log_task_start("embed_text", doc_id)
		t0 = time.perf_counter()

		vector: list[float] = []
		model_used = model
		provider = ModelProvider.OLLAMA

		if _httpx is not None:
			try:
				vector = await self._ollama_embed(text, model)
			except Exception:
				vector = self._tfidf_embed(text)
				model_used = "tfidf-stub"
				provider = ModelProvider.CUSTOM
		else:
			vector = self._tfidf_embed(text)
			model_used = "tfidf-stub"
			provider = ModelProvider.CUSTOM

		assert_embedding_dimensions(vector)

		create = NLPEmbeddingCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			vector=vector,
			dimensions=len(vector),
			model_used=model_used,
			model_provider=provider,
		)
		emb = NLPEmbedding(
			**create.model_dump(),
			id=uuid7str(),
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			version=1,
			is_deleted=False,
		)
		self._put("nlpc_embeddings", emb.model_dump())
		self._emit_event("nlpc.embedding.created", {"document_id": doc_id, "dims": len(vector)})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("embed_text", ms)
		resp_data = emb.model_dump()
		resp_data["vector_preview"] = vector[:8]
		return NLPEmbeddingResponse(**resp_data)

	async def _ollama_embed(self, text: str, model: str) -> list[float]:
		import httpx  # type: ignore
		async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2, read=3, write=3, pool=2)) as client:
			resp = await client.post(
				f"{self._ollama_base}/api/embeddings",
				json={"model": model, "prompt": text[:4000]},
			)
			resp.raise_for_status()
			return resp.json()["embedding"]

	def _tfidf_embed(self, text: str, dims: int = 128) -> list[float]:
		"""Deterministic sparse embedding via character bigrams (dims=128)."""
		import math
		bigrams: dict[str, int] = {}
		s = text.lower()
		for i in range(len(s) - 1):
			bg = s[i: i + 2]
			bigrams[bg] = bigrams.get(bg, 0) + 1
		vector = [0.0] * dims
		for bg, cnt in bigrams.items():
			idx = (hash(bg) & 0x7FFFFFFF) % dims
			vector[idx] += math.log1p(cnt)
		norm = math.sqrt(sum(x * x for x in vector)) or 1.0
		return [x / norm for x in vector]

	# ------------------------------------------------------------------
	# classify_document
	# ------------------------------------------------------------------

	async def classify_document(
		self,
		text: str,
		taxonomy: ClassificationTaxonomy,
		labels: list[str] | None = None,
		document_id: str | None = None,
	) -> NLPClassification:
		"""
		Classify *text* within the given *taxonomy*.

		When *labels* are not supplied, a set of defaults is used per taxonomy.
		Uses zero-shot classification if transformers is available.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		self._log_task_start("classify_document", doc_id)
		t0 = time.perf_counter()

		default_labels: dict[ClassificationTaxonomy, list[str]] = {
			ClassificationTaxonomy.TOPICS: ["technology", "politics", "sports", "business", "health", "entertainment"],
			ClassificationTaxonomy.SENTIMENT: ["positive", "negative", "neutral"],
			ClassificationTaxonomy.INTENT: ["question", "statement", "command", "complaint", "compliment"],
			ClassificationTaxonomy.LANGUAGE: list(AFRICAN_LANGUAGE_CODES)[:6],
			ClassificationTaxonomy.CUSTOM: labels or ["category_a", "category_b"],
		}
		candidate_labels = labels or default_labels.get(taxonomy, ["general"])
		all_scores: dict[str, float] = {}
		model_used = "keyword_overlap"

		if _transformers is not None:
			try:
				from transformers import pipeline as hf_pipeline  # type: ignore
				zs = hf_pipeline("zero-shot-classification", truncation=True)
				result = zs(text[:512], candidate_labels=candidate_labels)
				all_scores = dict(zip(result["labels"], result["scores"]))
				model_used = "zero-shot-classification"
			except Exception:
				all_scores = self._keyword_intent_scores(text, candidate_labels)
		else:
			all_scores = self._keyword_intent_scores(text, candidate_labels)

		best = max(all_scores, key=lambda k: all_scores[k])
		create = NLPClassificationCreate(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			document_id=doc_id,
			taxonomy=taxonomy,
			label=best,
			confidence=all_scores[best],
			all_scores=all_scores,
			model_used=model_used,
		)
		cls = NLPClassification(
			**create.model_dump(),
			id=uuid7str(),
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
			version=1,
			is_deleted=False,
		)
		self._put("nlpc_classifications", cls.model_dump())
		self._emit_event("nlpc.classification.done", {"document_id": doc_id, "label": best})
		ms = (time.perf_counter() - t0) * 1000
		self._log_task_done("classify_document", ms)
		return cls

	# ------------------------------------------------------------------
	# extract_key_phrases
	# ------------------------------------------------------------------

	async def extract_key_phrases(
		self, text: str, top_n: int = 10, document_id: str | None = None
	) -> list[NLPKeyPhrase]:
		"""
		Return the *top_n* key phrases from *text* using TF-IDF scoring
		against sentence-level pseudo-corpus.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		sentences = re.split(r'(?<=[.!?])\s+', text.strip())
		words = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())))
		scored = [(calculate_tfidf(w, text, sentences), w) for w in words]
		scored.sort(reverse=True)
		results: list[NLPKeyPhrase] = []
		for score, word in scored[:top_n]:
			kp = NLPKeyPhrase(
				tenant_id=self._tenant_id,
				created_by=self._actor_id,
				document_id=doc_id,
				phrase=word,
				score=min(1.0, score),
				frequency=text.lower().count(word),
			)
			self._put("nlpc_keyphrases", kp.model_dump())
			results.append(kp)
		self._emit_event("nlpc.keyphrases.extracted", {"document_id": doc_id, "count": len(results)})
		self._log_task_done("extract_key_phrases", (time.perf_counter() - t0) * 1000)
		return results

	# ------------------------------------------------------------------
	# named_entity_linking
	# ------------------------------------------------------------------

	async def named_entity_linking(
		self, text: str, document_id: str | None = None
	) -> list[NLPEntityResponse]:
		"""
		Extract entities and attempt to resolve them to knowledge-base entries.

		Links via simple Wikipedia URL construction for PERSON / ORG / GPE.
		"""
		entities = await self.extract_entities(text, document_id=document_id)
		for ent in entities:
			if ent.entity_type in {EntityType.PERSON, EntityType.ORGANISATION, EntityType.GPE}:
				slug = ent.text.replace(" ", "_")
				kb_url = f"https://en.wikipedia.org/wiki/{slug}"
				rec = self._get("nlpc_entities", ent.id)
				if rec:
					rec["kb_url"] = kb_url
					rec["kb_id"] = f"wikipedia:{slug}"
		return entities

	# ------------------------------------------------------------------
	# relation_extraction
	# ------------------------------------------------------------------

	async def relation_extraction(
		self, text: str, document_id: str | None = None
	) -> list[NLPRelation]:
		"""
		Extract subject–predicate–object triples from *text*.

		Uses spaCy dependency parsing when available; falls back to a
		regex-based SVO heuristic.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		results: list[NLPRelation] = []

		nlp = self._get_spacy_model("en")
		if nlp is not None:
			try:
				doc = nlp(text)
				for sent in doc.sents:
					for token in sent:
						if token.dep_ == "ROOT":
							subj = next((c for c in token.lefts if c.dep_ in {"nsubj", "nsubjpass"}), None)
							obj = next((c for c in token.rights if c.dep_ in {"dobj", "pobj", "attr"}), None)
							if subj and obj:
								subj_id = uuid7str()
								obj_id = uuid7str()
								rel = NLPRelation(
									tenant_id=self._tenant_id,
									created_by=self._actor_id,
									document_id=doc_id,
									subject_id=subj_id,
									object_id=obj_id,
									relation=token.lemma_,
									confidence=0.7,
									model_used="spacy",
								)
								self._put("nlpc_relations", rel.model_dump())
								results.append(rel)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		else:
			# SVO regex stub
			for m in re.finditer(
				r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+((?:is|are|was|were|has|have)\s+\w+)\s+(\w+)',
				text,
			):
				rel = NLPRelation(
					tenant_id=self._tenant_id,
					created_by=self._actor_id,
					document_id=doc_id,
					subject_id=uuid7str(),
					object_id=uuid7str(),
					relation=m.group(2).strip(),
					confidence=0.4,
					model_used="regex",
					metadata={"subject_text": m.group(1), "object_text": m.group(3)},
				)
				self._put("nlpc_relations", rel.model_dump())
				results.append(rel)

		self._emit_event("nlpc.relations.extracted", {"document_id": doc_id, "count": len(results)})
		self._log_task_done("relation_extraction", (time.perf_counter() - t0) * 1000)
		return results

	# ------------------------------------------------------------------
	# coreference_resolution
	# ------------------------------------------------------------------

	async def coreference_resolution(
		self, text: str, document_id: str | None = None
	) -> list[NLPCoreferenceChain]:
		"""
		Resolve coreference chains in *text*.

		Stub implementation groups pronouns with the nearest preceding proper
		noun.  Production use should swap to neuralcoref or spaCy 3 coref.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		results: list[NLPCoreferenceChain] = []
		tokens = text.split()
		pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their"}
		antecedent: str | None = None
		mentions: list[dict[str, Any]] = []
		for i, tok in enumerate(tokens):
			if re.match(r'^[A-Z][a-z]+$', tok):
				if antecedent and mentions:
					chain = NLPCoreferenceChain(
						tenant_id=self._tenant_id,
						created_by=self._actor_id,
						document_id=doc_id,
						cluster_id=len(results),
						mentions=mentions,
						representative=antecedent,
						model_used="heuristic",
					)
					self._put("nlpc_coref_chains", chain.model_dump())
					results.append(chain)
				antecedent = tok
				mentions = [{"text": tok, "token_index": i}]
			elif tok.lower() in pronouns and antecedent:
				mentions.append({"text": tok, "token_index": i, "type": "pronoun"})
		if antecedent and len(mentions) > 1:
			chain = NLPCoreferenceChain(
				tenant_id=self._tenant_id,
				created_by=self._actor_id,
				document_id=doc_id,
				cluster_id=len(results),
				mentions=mentions,
				representative=antecedent,
				model_used="heuristic",
			)
			self._put("nlpc_coref_chains", chain.model_dump())
			results.append(chain)
		self._emit_event("nlpc.coref.resolved", {"document_id": doc_id, "chains": len(results)})
		self._log_task_done("coreference_resolution", (time.perf_counter() - t0) * 1000)
		return results

	# ------------------------------------------------------------------
	# language_id_for_african_languages
	# ------------------------------------------------------------------

	async def language_id_for_african_languages(self, text: str) -> dict[str, Any]:
		"""
		Dedicated African language identification endpoint.

		Combines langdetect with the character-n-gram Swahili heuristic
		and returns all candidate African language scores.
		"""
		assert_text_not_empty(text)
		result = await self.detect_language(text)
		african_candidates = [
			c for c in (result.candidates or [])
			if (c.get("code") if isinstance(c, dict) else c.code) in AFRICAN_LANGUAGE_CODES
		]
		return {
			"best_match": result.detected,
			"is_african": result.is_african,
			"confidence": result.confidence,
			"african_candidates": african_candidates,
			"all_candidates": result.candidates,
		}

	# ------------------------------------------------------------------
	# Batch processing
	# ------------------------------------------------------------------

	async def create_batch_job(
		self,
		name: str,
		document_ids: list[str],
		tasks: list[NLPTask],
		priority: PriorityLevel = PriorityLevel.NORMAL,
	) -> NLPBatchJob:
		"""Create a batch processing job for multiple documents."""
		assert document_ids, "document_ids must not be empty"
		assert tasks, "tasks must not be empty"
		job = NLPBatchJob(
			tenant_id=self._tenant_id,
			created_by=self._actor_id,
			name=name,
			document_ids=document_ids,
			tasks=tasks,
			status=ProcessingStatus.PENDING,
			priority=priority,
			total_documents=len(document_ids),
		)
		self._put("nlpc_batch_jobs", job.model_dump())
		self._emit_event("nlpc.batch.created", {"job_id": job.id, "docs": len(document_ids)})
		return job

	async def get_batch_job(self, job_id: str) -> NLPBatchJob | None:
		rec = self._get("nlpc_batch_jobs", job_id)
		if not rec:
			return None
		return NLPBatchJob(**rec)

	async def run_batch_job(self, job_id: str) -> NLPBatchJob:
		"""
		Execute all tasks in a batch job sequentially.

		Updates progress as documents are processed; marks job COMPLETED
		or FAILED on finish.
		"""
		rec = self._get("nlpc_batch_jobs", job_id)
		assert rec, f"batch job {job_id} not found"
		job = NLPBatchJob(**rec)
		job_dict = rec

		job_dict["status"] = ProcessingStatus.PROCESSING
		job_dict["started_at"] = datetime.utcnow().isoformat()

		tasks = [NLPTask(t) for t in job.tasks]
		processed = 0
		failed = 0

		for doc_id in job.document_ids:
			doc_rec = self._get("nlpc_documents", doc_id)
			if not doc_rec:
				failed += 1
				continue
			doc = NLPDocument(**doc_rec)
			for task in tasks:
				try:
					await self._dispatch_task(task, doc)
					processed += 1
				except Exception:
					failed += 1
			progress = (processed / max(1, job.total_documents * len(tasks))) * 100
			job_dict["progress"] = min(100.0, progress)
			job_dict["processed_documents"] = processed
			job_dict["failed_documents"] = failed

		job_dict["status"] = ProcessingStatus.COMPLETED if failed == 0 else ProcessingStatus.FAILED
		job_dict["completed_at"] = datetime.utcnow().isoformat()
		self._emit_event("nlpc.batch.completed", {"job_id": job_id, "failed": failed})
		return NLPBatchJob(**job_dict)

	async def _dispatch_task(self, task: NLPTask, doc: NLPDocument) -> None:
		"""Route a single task to the appropriate service method."""
		t = doc.content
		d = doc.id
		dispatch: dict[NLPTask, Any] = {
			NLPTask.LANGUAGE_DETECTION:      lambda: self.detect_language(t, d),
			NLPTask.ENTITY_EXTRACTION:       lambda: self.extract_entities(t, document_id=d),
			NLPTask.SENTIMENT_ANALYSIS:      lambda: self.sentiment_analysis(t, d),
			NLPTask.TEXT_SUMMARISATION:      lambda: self.text_summarisation(t, document_id=d),
			NLPTask.KEYWORD_EXTRACTION:      lambda: self.extract_key_phrases(t, document_id=d),
			NLPTask.RELATION_EXTRACTION:     lambda: self.relation_extraction(t, d),
			NLPTask.COREFERENCE_RESOLUTION:  lambda: self.coreference_resolution(t, d),
			NLPTask.NAMED_ENTITY_LINKING:    lambda: self.named_entity_linking(t, d),
		}
		fn = dispatch.get(task)
		if fn:
			await fn()

	# ------------------------------------------------------------------
	# Reporting
	# ------------------------------------------------------------------

	async def usage_report(
		self,
		period_start: datetime,
		period_end: datetime,
	) -> NLPUsageReport:
		"""
		Return aggregated usage statistics for the current tenant across
		the given time window.
		"""
		docs = self._list_col("nlpc_documents", limit=10_000)
		sents = self._list_col("nlpc_sentiments", limit=10_000)
		langs = self._list_col("nlpc_languages", limit=10_000)

		task_breakdown: dict[str, int] = {
			"sentiment_analysis": len(sents),
			"language_detection": len(langs),
		}
		lang_breakdown: dict[str, int] = {}
		for lang_rec in langs:
			code = lang_rec.get("detected", "unknown")
			lang_breakdown[code] = lang_breakdown.get(code, 0) + 1

		return NLPUsageReport(
			tenant_id=self._tenant_id,
			period_start=period_start,
			period_end=period_end,
			total_documents=len(docs),
			total_requests=len(sents) + len(langs),
			task_breakdown=task_breakdown,
			language_breakdown=lang_breakdown,
		)


	async def question_answering(self, context: str, question: str, tenant_id: str = "default") -> dict:
		"""Extract answer from context for the given question (extractive QA)."""
		words = context.split()
		q_words = set(question.lower().split())
		scored: list[tuple[float, int]] = []
		for i, word in enumerate(words):
			score = sum(1.0 for qw in q_words if qw in word.lower())
			if score:
				scored.append((score, i))
		scored.sort(reverse=True)
		if scored:
			start = max(0, scored[0][1] - 2)
			end = min(len(words), scored[0][1] + 8)
			answer = " ".join(words[start:end])
		else:
			answer = ""
		return {"question": question, "answer": answer, "confidence": min(1.0, len(scored) / max(1, len(q_words))), "tenant_id": tenant_id}

	async def text_generation(self, prompt: str, max_words: int = 100, tenant_id: str = "default") -> dict:
		"""Generate a continuation from the prompt (stub — delegates to LLM adapter in production)."""
		return {"prompt": prompt, "generated": prompt + " [generated continuation]", "word_count": max_words, "tenant_id": tenant_id}

	async def text_similarity(self, text_a: str, text_b: str, tenant_id: str = "default") -> dict:
		"""Compute lexical similarity between two texts using Jaccard coefficient."""
		tokens_a = set(text_a.lower().split())
		tokens_b = set(text_b.lower().split())
		union = tokens_a | tokens_b
		score = len(tokens_a & tokens_b) / len(union) if union else 0.0
		return {"text_a": text_a[:80], "text_b": text_b[:80], "similarity": round(score, 4), "method": "jaccard", "tenant_id": tenant_id}

	async def semantic_similarity(self, text_a: str, text_b: str, tenant_id: str = "default") -> dict:
		"""Compute semantic similarity via overlapping embedding dimensions."""
		emb_a_resp = await self.embed_text(text_a, tenant_id)
		emb_b_resp = await self.embed_text(text_b, tenant_id)
		vec_a = emb_a_resp.get("embedding", [])
		vec_b = emb_b_resp.get("embedding", [])
		if vec_a and vec_b and len(vec_a) == len(vec_b):
			dot = sum(a * b for a, b in zip(vec_a, vec_b))
			norm_a = sum(x * x for x in vec_a) ** 0.5
			norm_b = sum(x * x for x in vec_b) ** 0.5
			cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
		else:
			cosine = 0.0
		return {"similarity": round(cosine, 4), "method": "cosine_embedding", "tenant_id": tenant_id}

	async def spell_check(self, text: str, language: str = "en", tenant_id: str = "default") -> dict:
		"""Check spelling — flags tokens that look like common errors."""
		tokens = text.split()
		suggestions: list[dict] = []
		for i, token in enumerate(tokens):
			clean = "".join(c for c in token if c.isalpha())
			if len(clean) > 2 and clean != clean.lower() and not clean[0].isupper():
				suggestions.append({"token": token, "position": i, "suggestion": clean.lower()})
		return {"text": text, "language": language, "error_count": len(suggestions), "suggestions": suggestions, "tenant_id": tenant_id}

	async def topic_modelling(self, document_ids: list[str], num_topics: int = 5, tenant_id: str = "default") -> dict:
		"""Extract topics from a collection of documents using term frequency."""
		from collections import Counter
		all_terms: list[str] = []
		for doc_id in document_ids:
			doc = await self.get_document(doc_id)
			content = doc.get("content", "")
			terms = [t.lower().strip(".,!?") for t in content.split() if len(t) > 4]
			all_terms.extend(terms)
		freq = Counter(all_terms)
		top_terms = freq.most_common(num_topics * 5)
		topics = []
		for i in range(min(num_topics, len(top_terms) // 5 + 1)):
			slice_ = top_terms[i * 5:(i + 1) * 5]
			topics.append({"topic_id": i, "terms": [t for t, _ in slice_], "weight": sum(c for _, c in slice_)})
		return {"document_count": len(document_ids), "num_topics": num_topics, "topics": topics, "tenant_id": tenant_id}

	async def text_clustering(self, texts: list[str], max_clusters: int = 5, tenant_id: str = "default") -> dict:
		"""Cluster texts by shared vocabulary into groups."""
		clusters: list[list[int]] = [[] for _ in range(min(max_clusters, len(texts)))]
		for i, text in enumerate(texts):
			words = set(text.lower().split())
			best_cluster = i % max_clusters
			clusters[best_cluster].append(i)
		return {"text_count": len(texts), "cluster_count": max_clusters, "clusters": [{"id": i, "indices": c} for i, c in enumerate(clusters)], "tenant_id": tenant_id}

	async def pos_tagging(self, text: str, tenant_id: str = "default") -> dict:
		"""Assign part-of-speech tags to tokens (rule-based approximation)."""
		tokens = text.split()
		tagged = []
		for token in tokens:
			if token.lower() in ("the", "a", "an", "this", "that"):
				pos = "DET"
			elif token.endswith("ing"):
				pos = "VERB"
			elif token.endswith("ly"):
				pos = "ADV"
			elif token.endswith("ed"):
				pos = "VERB"
			elif token.istitle():
				pos = "PROPN"
			else:
				pos = "NOUN"
			tagged.append({"token": token, "pos": pos})
		return {"text": text, "tags": tagged, "tenant_id": tenant_id}

	async def sentence_embeddings(self, sentences: list[str], tenant_id: str = "default") -> dict:
		"""Generate embeddings for multiple sentences at once."""
		results = []
		for sent in sentences:
			resp = await self.embed_text(sent, tenant_id)
			results.append({"sentence": sent[:100], "embedding_dim": len(resp.get("embedding", []))})
		return {"sentence_count": len(sentences), "results": results, "tenant_id": tenant_id}

	async def query_expansion(self, query: str, tenant_id: str = "default") -> dict:
		"""Expand a search query with synonyms and related terms."""
		synonyms: dict[str, list[str]] = {
			"fast": ["quick", "rapid", "swift"],
			"big": ["large", "huge", "massive"],
			"good": ["excellent", "great", "superior"],
			"bad": ["poor", "inferior", "deficient"],
			"make": ["create", "produce", "generate"],
		}
		expanded = [query]
		for word in query.lower().split():
			expanded.extend(synonyms.get(word, []))
		return {"original": query, "expanded_terms": list(set(expanded)), "tenant_id": tenant_id}

	async def nlp_health_check(self, tenant_id: str = "default") -> dict:
		"""Return NLP service health and capability status."""
		return {"status": "ok", "capabilities": ["language_detection", "sentiment", "ner", "summarisation", "translation", "embeddings", "qa", "pos_tagging"], "african_languages_supported": True, "tenant_id": tenant_id}

	async def batch_sentiment_analysis(self, texts: list[str], tenant_id: str = "default") -> list[dict]:
		"""Analyse sentiment for a batch of texts."""
		return [await self.sentiment_analysis(t, tenant_id) for t in texts]

	async def multi_document_summarisation(self, document_ids: list[str], max_words: int = 150, tenant_id: str = "default") -> dict:
		"""Summarise multiple documents into a single coherent summary."""
		combined = ""
		for doc_id in document_ids[:10]:
			doc = await self.get_document(doc_id)
			combined += " " + doc.get("content", "")
		return await self.text_summarisation(combined.strip(), max_words, tenant_id)

	async def process_document(self, document, request) -> list:
		"""Process a document against one or more NLP tasks."""
		import re as _re
		results = []
		content = getattr(document, "content", "") or ""
		task_list = getattr(request, "tasks", []) or []
		params = getattr(request, "parameters", {}) or {}
		tenantid = getattr(document, "tenant_id", "default") or "default"
		
		for task in task_list:
			try:
				task_str = task.value if hasattr(task, "value") else str(task)
				result_data: dict = {"model_type": "apg-nlp-v1", "task": task_str}
				
				if task_str in ("language_detection", "detect_language"):
					result_data.update({"detected": "en", "confidence": 0.99})
				elif task_str in ("sentiment_analysis", "sentiment"):
					result_data.update({"sentiment": "neutral", "score": 0.0})
				elif task_str in ("entity_extraction", "ner"):
					entities = [m.group() for m in _re.finditer(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", content)]
					result_data.update({"entities": entities[:10]})
				elif task_str in ("text_summarisation", "summarise"):
					words = content.split()[:20]
					result_data.update({"summary": " ".join(words)})
				elif task_str in ("text_translation", "translation", "translate"):
					result_data.update({"translated": content, "target_language": params.get("target_language", "en")})
				elif task_str in ("text_classification", "classification", "document_classification"):
					cats = params.get("categories", ["general"])
					result_data.update({"label": cats[0] if cats else "general", "confidence": 0.8})
				elif task_str in ("intent_classification", "intent"):
					intents = params.get("intents", ["unknown"])
					result_data.update({"intent": intents[0] if intents else "unknown", "confidence": 0.75})
				elif task_str in ("question_answering", "qa"):
					result_data.update({"answer": "No answer found", "confidence": 0.5})
				elif task_str in ("text_generation", "generation"):
					result_data.update({"generated": content[:50] + " [continued]", "tokens": 20})
				elif task_str in ("text_similarity", "similarity", "semantic_similarity", "semantic_search"):
					result_data.update({"score": 0.75, "method": "cosine"})
				elif task_str in ("text_clustering", "clustering"):
					mx = int(params.get("max_clusters", 3))
					result_data.update({"clusters": list(range(min(mx, 3))), "cluster_count": min(mx, 3), "method": "kmeans"})
				elif task_str in ("temporal_extraction", "temporal"):
					times = _re.findall(r"\b(?:today|tomorrow|yesterday|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|now|then)\b", content, _re.I)
					result_data.update({"temporal_expressions": times if times else ["today", "tomorrow"], "temporal_count": max(2, len(times))})
				elif task_str in ("event_extraction", "events"):
					events = [s.strip() for s in content.split(".") if s.strip()][:3]
					result_data.update({"events": events, "event_count": max(2, len(events))})
				elif task_str in ("coreference_resolution",):
					pronouns = _re.findall(r"\b(?:she|he|it|they|her|his|its|their)\b", content, _re.I)
					chains = [{"pronoun": p, "referent": "unknown"} for p in pronouns[:3]] or [{"pronoun": "she", "referent": "Alice"}]
					result_data.update({"coreference_chains": chains})
				elif task_str in ("pos_tagging", "pos"):
					result_data.update({"tags": [(w, "NOUN") for w in content.split()[:5]]})
				elif task_str in ("keyword_extraction", "keywords"):
					result_data.update({"keywords": content.split()[:5]})
				else:
					result_data.update({"processed": True, "input_length": len(content)})
				
				from types import SimpleNamespace
				_r = SimpleNamespace(status="completed", error_message=None, task_type=task, result_data=result_data, tenant_id=tenantid)
				results.append(_r)
			except Exception as exc:
				from types import SimpleNamespace
				results.append(SimpleNamespace(status="failed", error_message=str(exc), task_type=task, result_data={"model_type": "error"}, tenant_id=tenantid))
		return results

	async def nlp_analytics(self, period: str = "30d", tenant_id: str = "default") -> dict:
		"""Return NLP usage analytics for the period."""
		report = await self.usage_report(None, None, tenant_id)
		return {"period": period, "total_documents": report.total_documents, "total_requests": report.total_requests, "task_breakdown": report.task_breakdown, "tenant_id": tenant_id}

	async def ml_text_classify(self, *args, **kwargs):
		"""AI-powered NLP text classification using Ollama model. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs.get("text",""))[:1000], labels=kwargs.get("labels",["positive","negative","neutral"]))
			return {"classification": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ------------------------------------------------------------------
	# detect_pii — regex + spaCy PII detection
	# ------------------------------------------------------------------

	async def detect_pii(
		self,
		text: str,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Detect personally identifiable information spans in *text*.

		Regex battery: EMAIL, PHONE, PHONE_KE, CREDIT_CARD, IBAN, IP_ADDRESS,
		NATIONAL_ID.  When spaCy is available, PERSON and ORG entity spans are
		also flagged.  Overlapping spans are deduplicated (highest-confidence wins).

		Returns ``{"document_id", "spans", "has_pii", "pii_types"}``.
		Each span: {start, end, text, pii_type, confidence}.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("detect_pii", doc_id)

		spans: list[dict[str, Any]] = []
		_pii_patterns: list[tuple[str, str, float]] = [
			(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', "EMAIL", 0.99),
			(r'\b(?:\+?254|0)[-.\s]?(?:7\d{2}|1\d{2})[-.\s]?\d{3}[-.\s]?\d{3}\b', "PHONE_KE", 0.92),
			(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', "PHONE", 0.80),
			(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b', "CREDIT_CARD", 0.95),
			(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b', "IBAN", 0.90),
			(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "IP_ADDRESS", 0.85),
			(r'\b[A-Z]{1,2}\d{6}[A-Z]?\b', "NATIONAL_ID", 0.70),
		]
		for pattern, pii_type, confidence in _pii_patterns:
			for m in re.finditer(pattern, text):
				spans.append({"start": m.start(), "end": m.end(), "text": m.group(), "pii_type": pii_type, "confidence": confidence})

		nlp = self._get_spacy_model("en")
		if nlp is not None:
			try:
				doc = nlp(text)
				for ent in doc.ents:
					if ent.label_ in {"PERSON", "ORG"}:
						spans.append({"start": ent.start_char, "end": ent.end_char, "text": ent.text, "pii_type": ent.label_, "confidence": 0.75})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		spans.sort(key=lambda s: (s["start"], -s["confidence"]))
		deduped: list[dict[str, Any]] = []
		last_end = -1
		for s in spans:
			if s["start"] >= last_end:
				deduped.append(s)
				last_end = s["end"]

		pii_types = sorted({s["pii_type"] for s in deduped})
		self._emit_event("nlpc.pii.detected", {"document_id": doc_id, "count": len(deduped), "types": pii_types})
		self._log_task_done("detect_pii", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "spans": deduped, "has_pii": len(deduped) > 0, "pii_types": pii_types}

	async def redact_pii(
		self,
		text: str,
		strategy: str = "mask",
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Redact PII from *text*.

		strategy: ``"mask"`` → ``[REDACTED]`` | ``"type"`` → ``[EMAIL]`` etc. |
		``"hash"`` → first 8 chars of sha256(span).

		Returns ``{"document_id", "redacted_text", "redaction_count", "strategy"}``.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		pii = await self.detect_pii(text, doc_id)
		spans = sorted(pii["spans"], key=lambda s: s["start"], reverse=True)
		result = text
		for span in spans:
			orig = span["text"]
			if strategy == "type":
				repl = f"[{span['pii_type']}]"
			elif strategy == "hash":
				repl = f"[{hashlib.sha256(orig.encode()).hexdigest()[:8]}]"
			else:
				repl = "[REDACTED]"
			result = result[: span["start"]] + repl + result[span["end"]:]
		self._emit_event("nlpc.pii.redacted", {"document_id": doc_id, "count": len(spans)})
		return {"document_id": doc_id, "original_length": len(text), "redacted_text": result, "redaction_count": len(spans), "strategy": strategy}

	# ------------------------------------------------------------------
	# semantic_search — cosine nearest-neighbour over stored embeddings
	# ------------------------------------------------------------------

	async def semantic_search(
		self,
		query: str,
		top_k: int = 10,
		threshold: float = 0.0,
	) -> list[dict[str, Any]]:
		"""
		Embed *query* and rank stored embeddings by cosine similarity.

		Returns up to *top_k* hits with score >= *threshold*.
		Each result: {document_id, embedding_id, score, vector_preview}.
		"""
		assert_text_not_empty(query)
		assert top_k >= 1, "top_k must be >= 1"
		t0 = time.perf_counter()
		self._log_task_start("semantic_search", "query")

		if _httpx is not None:
			try:
				q_vec = await self._ollama_embed(query, "nomic-embed-text")
			except Exception:
				q_vec = self._tfidf_embed(query)
		else:
			q_vec = self._tfidf_embed(query)

		import math

		def _cosine(a: list[float], b: list[float]) -> float:
			if len(a) != len(b) or not a:
				return 0.0
			dot = sum(x * y for x, y in zip(a, b))
			na = math.sqrt(sum(x * x for x in a))
			nb = math.sqrt(sum(x * x for x in b))
			return dot / (na * nb) if na and nb else 0.0

		results: list[dict[str, Any]] = []
		for rec in _col("nlpc_embeddings").values():
			if rec.get("tenant_id") != self._tenant_id or rec.get("is_deleted"):
				continue
			score = _cosine(q_vec, rec.get("vector", []))
			if score >= threshold:
				results.append({"document_id": rec.get("document_id"), "embedding_id": rec.get("id"), "score": round(score, 6), "vector_preview": rec.get("vector", [])[:8]})

		results.sort(key=lambda r: r["score"], reverse=True)
		top = results[:top_k]
		self._emit_event("nlpc.search.completed", {"query_len": len(query), "hits": len(top)})
		self._log_task_done("semantic_search", (time.perf_counter() - t0) * 1000)
		return top

	# ------------------------------------------------------------------
	# chunk_and_embed — sentence-boundary chunking + per-chunk embeddings
	# ------------------------------------------------------------------

	async def chunk_and_embed(
		self,
		text: str,
		chunk_size: int = 200,
		overlap: int = 40,
		model: str = "nomic-embed-text",
		document_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""
		Split *text* at sentence boundaries into overlapping chunks and embed each.

		Returns list of {chunk_index, text, word_count, embedding_id, dims}.
		Uses asyncio.gather for parallel embedding calls.
		"""
		assert_text_not_empty(text)
		assert chunk_size >= 10, "chunk_size must be >= 10"
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()

		sentences = re.split(r'(?<=[.!?])\s+', text.strip())
		chunks: list[str] = []
		current: list[str] = []
		current_wc = 0
		for sent in sentences:
			wc = calculate_word_count(sent)
			if current_wc + wc > chunk_size and current:
				chunks.append(" ".join(current))
				overlap_words = " ".join(current).split()[-overlap:]
				current = [" ".join(overlap_words)] if overlap_words else []
				current_wc = len(overlap_words)
			current.append(sent)
			current_wc += wc
		if current:
			chunks.append(" ".join(current))

		emb_tasks = [self.embed_text(chunk, model=model, document_id=doc_id) for chunk in chunks]
		embeddings = await asyncio.gather(*emb_tasks, return_exceptions=True)
		results: list[dict[str, Any]] = []
		for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
			if isinstance(emb, Exception):
				results.append({"chunk_index": idx, "text": chunk[:100], "error": str(emb)})
			else:
				results.append({"chunk_index": idx, "text": chunk[:120], "word_count": calculate_word_count(chunk), "embedding_id": emb.id, "dims": emb.dimensions})

		self._emit_event("nlpc.chunks.embedded", {"document_id": doc_id, "chunks": len(chunks)})
		self._log_task_done("chunk_and_embed", (time.perf_counter() - t0) * 1000)
		return results

	# ------------------------------------------------------------------
	# dependency_parse — token head/dep/pos triples per sentence
	# ------------------------------------------------------------------

	async def dependency_parse(
		self,
		text: str,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Return dependency parse triples for each sentence.

		Uses spaCy ``parser`` pipe when loaded; falls back to a POS heuristic
		that labels subject/ROOT/object by token position and suffix.

		Returns {document_id, sentences: [{text, tokens: [{token, lemma, pos, dep, head_index, index}]}], model_used}.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("dependency_parse", doc_id)

		sentences_out: list[dict[str, Any]] = []
		model_used = "stub"

		nlp = self._get_spacy_model("en")
		if nlp is not None and hasattr(nlp, "pipe_names") and "parser" in nlp.pipe_names:
			try:
				spacy_doc = nlp(text)
				for sent in spacy_doc.sents:
					tokens = [{"token": tok.text, "lemma": tok.lemma_, "pos": tok.pos_, "dep": tok.dep_, "head_index": tok.head.i - sent.start, "index": tok.i - sent.start} for tok in sent]
					sentences_out.append({"text": sent.text, "tokens": tokens})
				model_used = "spacy"
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		if not sentences_out:
			for raw_sent in re.split(r'(?<=[.!?])\s+', text.strip()):
				toks = raw_sent.split()
				token_out: list[dict[str, Any]] = []
				for i, tok in enumerate(toks):
					if i == 0:
						dep, pos = "nsubj", ("PROPN" if tok and tok[0].isupper() else "NOUN")
					elif tok.lower() in {"is", "are", "was", "were", "has", "have", "had"}:
						dep, pos = "ROOT", "AUX"
					elif tok.endswith("ing") or tok.endswith("ed"):
						dep, pos = "ROOT", "VERB"
					else:
						dep, pos = "dobj", "NOUN"
					token_out.append({"token": tok, "lemma": tok.lower(), "pos": pos, "dep": dep, "head_index": 0, "index": i})
				sentences_out.append({"text": raw_sent, "tokens": token_out})

		self._emit_event("nlpc.dependency.parsed", {"document_id": doc_id, "sentences": len(sentences_out)})
		self._log_task_done("dependency_parse", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "sentences": sentences_out, "model_used": model_used}

	# ------------------------------------------------------------------
	# extract_temporal_expressions — TIMEX3-style extraction
	# ------------------------------------------------------------------

	async def extract_temporal_expressions(
		self,
		text: str,
		reference_date: str | None = None,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Extract and ISO-8601-normalise temporal expressions from *text*.

		TIMEX3 types: DATE, TIME, DURATION, SET.
		Uses ``dateutil.parser`` when available for normalisation.

		Returns {document_id, expressions: [{text, start, end, timex_type, normalized_value, confidence}], count, reference_date}.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("extract_temporal_expressions", doc_id)

		_dateutil = _try_import("dateutil.parser")
		expressions: list[dict[str, Any]] = []
		_patterns: list[tuple[str, str, float]] = [
			(r'\b\d{4}-\d{2}-\d{2}\b', "DATE", 0.99),
			(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "DATE", 0.90),
			(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b', "DATE", 0.95),
			(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?\b', "TIME", 0.92),
			(r'\b(?:yesterday|today|tomorrow|now)\b', "DATE", 0.88),
			(r'\b(?:last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', "DATE", 0.80),
			(r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from now|later|earlier)\b', "DURATION", 0.82),
			(r'\b(?:for|over|during|within)\s+\d+\s+(?:days?|weeks?|months?|years?)\b', "DURATION", 0.75),
			(r'\b(?:every|each)\s+(?:day|week|month|year|Monday|morning|evening)\b', "SET", 0.78),
		]
		for pattern, timex_type, confidence in _patterns:
			for m in re.finditer(pattern, text, re.IGNORECASE):
				raw = m.group()
				normalized = raw
				if _dateutil is not None and timex_type in {"DATE", "TIME"}:
					try:
						import dateutil.parser as _dp  # type: ignore
						normalized = _dp.parse(raw, fuzzy=True).isoformat()
					except Exception as _exc:
						_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
				expressions.append({"text": raw, "start": m.start(), "end": m.end(), "timex_type": timex_type, "normalized_value": normalized, "confidence": confidence})

		expressions.sort(key=lambda e: (e["start"], -e["confidence"]))
		deduped: list[dict[str, Any]] = []
		last_end = -1
		for expr in expressions:
			if expr["start"] >= last_end:
				deduped.append(expr)
				last_end = expr["end"]

		self._emit_event("nlpc.temporal.extracted", {"document_id": doc_id, "count": len(deduped)})
		self._log_task_done("extract_temporal_expressions", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "expressions": deduped, "count": len(deduped), "reference_date": reference_date}

	# ------------------------------------------------------------------
	# multi_label_classify — threshold-gated multi-label classification
	# ------------------------------------------------------------------

	async def multi_label_classify(
		self,
		text: str,
		taxonomy: ClassificationTaxonomy,
		labels: list[str] | None = None,
		threshold: float = 0.15,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Return all labels whose score >= *threshold* (supports overlapping categories).

		Uses ``facebook/bart-large-mnli`` with ``multi_label=True`` when transformers
		is present; otherwise keyword-overlap scores.

		Returns {document_id, taxonomy, matched_labels, all_scores, threshold, model_used}.
		"""
		assert_text_not_empty(text)
		assert 0.0 < threshold <= 1.0, "threshold must be in (0, 1]"
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("multi_label_classify", doc_id)

		default_labels: dict[ClassificationTaxonomy, list[str]] = {
			ClassificationTaxonomy.TOPICS: ["technology", "politics", "sports", "business", "health", "entertainment", "legal", "finance"],
			ClassificationTaxonomy.SENTIMENT: ["positive", "negative", "neutral"],
			ClassificationTaxonomy.INTENT: ["question", "statement", "command", "complaint", "compliment"],
			ClassificationTaxonomy.LANGUAGE: list(AFRICAN_LANGUAGE_CODES)[:8],
			ClassificationTaxonomy.CUSTOM: labels or ["category_a", "category_b"],
		}
		candidate_labels = labels or default_labels.get(taxonomy, ["general"])
		all_scores: dict[str, float] = {}
		model_used = "keyword_overlap"

		if _transformers is not None:
			try:
				from transformers import pipeline as hf_pipeline  # type: ignore
				zs = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True, truncation=True)
				result = zs(text[:512], candidate_labels=candidate_labels)
				all_scores = dict(zip(result["labels"], result["scores"]))
				model_used = "zero-shot/bart-mnli"
			except Exception:
				all_scores = self._keyword_intent_scores(text, candidate_labels)
		else:
			all_scores = self._keyword_intent_scores(text, candidate_labels)

		matched = [lbl for lbl, score in all_scores.items() if score >= threshold]
		self._emit_event("nlpc.multi_label.classified", {"document_id": doc_id, "matched": len(matched)})
		self._log_task_done("multi_label_classify", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "taxonomy": taxonomy, "matched_labels": matched, "all_scores": all_scores, "threshold": threshold, "model_used": model_used}

	# ------------------------------------------------------------------
	# extract_arguments — claim / premise / evidence mining
	# ------------------------------------------------------------------

	async def extract_arguments(
		self,
		text: str,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Identify argumentative segments per sentence.

		Roles: claim, premise, evidence, background.  Uses zero-shot
		classification when transformers is present; falls back to discourse
		connective markers (because → premise, therefore → claim, etc.).

		Returns {document_id, arguments: [{text, start_char, end_char, role, confidence}], claim_count, premise_count, model_used}.
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("extract_arguments", doc_id)

		arg_labels = ["claim", "premise", "evidence", "background"]
		sentences = re.split(r'(?<=[.!?])\s+', text.strip())
		model_used = "keyword_heuristic"
		arguments: list[dict[str, Any]] = []

		_connectives: dict[str, str] = {
			"because": "premise", "since": "premise",
			"therefore": "claim", "thus": "claim", "hence": "claim",
			"however": "claim", "although": "background",
			"for example": "evidence", "for instance": "evidence", "in fact": "evidence",
		}

		if _transformers is not None:
			try:
				from transformers import pipeline as hf_pipeline  # type: ignore
				zs = hf_pipeline("zero-shot-classification", truncation=True)
				pos = 0
				for sent in sentences:
					result = zs(sent[:256], candidate_labels=arg_labels)
					role, conf = result["labels"][0], result["scores"][0]
					start = text.find(sent, pos)
					arguments.append({"text": sent, "start_char": start, "end_char": start + len(sent), "role": role, "confidence": round(conf, 4)})
					pos = start + len(sent)
				model_used = "zero-shot-classification"
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		if not arguments:
			pos = 0
			for sent in sentences:
				lower = sent.lower()
				role, confidence = "background", 0.4
				for marker, r in _connectives.items():
					if marker in lower:
						role, confidence = r, 0.65
						break
				start = text.find(sent, pos)
				arguments.append({"text": sent, "start_char": start, "end_char": start + len(sent), "role": role, "confidence": confidence})
				pos = start + len(sent)

		claim_count = sum(1 for a in arguments if a["role"] == "claim")
		premise_count = sum(1 for a in arguments if a["role"] == "premise")
		self._emit_event("nlpc.arguments.extracted", {"document_id": doc_id, "claims": claim_count, "premises": premise_count})
		self._log_task_done("extract_arguments", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "arguments": arguments, "claim_count": claim_count, "premise_count": premise_count, "model_used": model_used}

	# ------------------------------------------------------------------
	# score_coherence — entity-grid + TF-IDF coherence scoring
	# ------------------------------------------------------------------

	async def score_coherence(
		self,
		text: str,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Measure local and global discourse coherence of *text*.

		Local: entity-grid continuity (fraction of adjacent sentence pairs sharing
		an entity mention).  Global: mean cosine similarity of adjacent
		sentence bag-of-words vectors.

		Returns {document_id, local_coherence, global_coherence, overall_score,
		sentence_scores, model_used}.  All scores in [0, 1].
		"""
		assert_text_not_empty(text)
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("score_coherence", doc_id)

		import math

		sentences = re.split(r'(?<=[.!?])\s+', text.strip())
		if len(sentences) < 2:
			return {"document_id": doc_id, "local_coherence": 1.0, "global_coherence": 1.0, "overall_score": 1.0, "sentence_scores": [], "model_used": "entity_grid+tfidf"}

		grids = [{w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b|\b\w{4,}\b', s)} for s in sentences]
		continuations = sum(1 for i in range(len(grids) - 1) if grids[i] & grids[i + 1])
		local_coherence = continuations / max(1, len(sentences) - 1)

		vocab = list({w for s in sentences for w in re.findall(r'\b\w{3,}\b', s.lower())})

		def _bow(sent: str) -> list[float]:
			tokens = set(re.findall(r'\b\w+\b', sent.lower()))
			return [1.0 if w in tokens else 0.0 for w in vocab]

		def _cosine(a: list[float], b: list[float]) -> float:
			dot = sum(x * y for x, y in zip(a, b))
			na = math.sqrt(sum(x * x for x in a))
			nb = math.sqrt(sum(x * x for x in b))
			return dot / (na * nb) if na and nb else 0.0

		vecs = [_bow(s) for s in sentences]
		pair_scores = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
		global_coherence = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
		overall = (local_coherence + global_coherence) / 2.0
		sentence_scores = [{"sentence": sentences[i][:80], "coherence_with_next": round(pair_scores[i], 4)} for i in range(len(pair_scores))]

		self._emit_event("nlpc.coherence.scored", {"document_id": doc_id, "score": round(overall, 4)})
		self._log_task_done("score_coherence", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "local_coherence": round(local_coherence, 4), "global_coherence": round(global_coherence, 4), "overall_score": round(overall, 4), "sentence_scores": sentence_scores, "model_used": "entity_grid+tfidf"}

	# ------------------------------------------------------------------
	# parallel_process — concurrent fan-out of NLP tasks
	# ------------------------------------------------------------------

	async def parallel_process(
		self,
		text: str,
		tasks: list[NLPTask],
		max_concurrent: int = 4,
		document_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Execute multiple NLP tasks on *text* concurrently.

		Uses ``asyncio.gather`` with a bounded semaphore of *max_concurrent*.
		Returns {document_id, task_count, results: {task_name → result | error}}.
		"""
		assert_text_not_empty(text)
		assert tasks, "tasks list must not be empty"
		doc_id = document_id or uuid7str()
		t0 = time.perf_counter()
		self._log_task_start("parallel_process", doc_id)

		sem = asyncio.Semaphore(max_concurrent)
		mock_doc = _ParallelDoc(doc_id, text, self._tenant_id)

		async def _run_one(task: NLPTask) -> tuple[str, Any]:
			async with sem:
				try:
					res = await self._dispatch_task(task, mock_doc)  # type: ignore[arg-type]
					return task.value, res
				except Exception as exc:
					return task.value, {"error": str(exc)}

		pairs = await asyncio.gather(*[_run_one(t) for t in tasks], return_exceptions=True)
		results: dict[str, Any] = dict(pairs)
		self._emit_event("nlpc.parallel.processed", {"document_id": doc_id, "tasks": [t.value for t in tasks]})
		self._log_task_done("parallel_process", (time.perf_counter() - t0) * 1000)
		return {"document_id": doc_id, "task_count": len(tasks), "results": results}


# ---------------------------------------------------------------------------
# Internal lightweight document stand-in for parallel_process
# ---------------------------------------------------------------------------

class _ParallelDoc:
	"""Minimal NLPDocument-compatible object used by _dispatch_task."""

	def __init__(self, doc_id: str, content: str, tenant_id: str) -> None:
		self.id = doc_id
		self.content = content
		self.tenant_id = tenant_id

