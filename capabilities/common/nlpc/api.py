"""
NLPC REST API — Natural Language Processing Core

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Website: www.datacraft.co.ke

Plain Flask Blueprint.  No flask_restx, no flask_socketio.

All endpoints:
  - Require X-Tenant-Id header (or tenant_id query param).
  - Return structured JSON {ok, data, error, request_id}.
  - Use HTTP 422 for validation errors, 404 for missing records, 500 for
    unexpected failures.

Mount with:
    from capabilities.common.nlpc.api import nlpc_bp
    app.register_blueprint(nlpc_bp)
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request, Response
from uuid6 import uuid7

from .domain.rules import RuleViolation
from .models import (
	ClassificationTaxonomy,
	EntityType,
	LanguageCode,
	NLPDocumentCreate,
	NLPTask,
	PriorityLevel,
	SummaryMethod,
)
from .service import NLPCoreService


def uuid7str() -> str:
	return str(uuid7())


nlpc_bp = Blueprint("nlpc", __name__, url_prefix="/api/nlpc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> NLPCoreService:
	"""Instantiate service from request context headers."""
	tenant_id = (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id")
		or "default"
	)
	actor_id = request.headers.get("X-Actor-Id") or "api"
	ollama_url = request.headers.get("X-Ollama-Url") or "http://localhost:11434"
	return NLPCoreService(tenant_id=tenant_id, actor_id=actor_id, ollama_base_url=ollama_url)


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"ok": True, "data": data, "request_id": uuid7str()}), status


def _err(message: str, status: int, details: Any = None) -> Response:
	body: dict[str, Any] = {"ok": False, "error": message, "request_id": uuid7str()}
	if details is not None:
		body["details"] = details
	return jsonify(body), status


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask handler."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _body() -> dict[str, Any]:
	data = request.get_json(silent=True)
	if data is None:
		data = {}
	return data


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@nlpc_bp.errorhandler(RuleViolation)
def _handle_rule_violation(exc: RuleViolation) -> Response:
	return _err(exc.reason, 422, {"rule": exc.rule_name, "action": exc.required_action})


@nlpc_bp.errorhandler(AssertionError)
def _handle_assertion(exc: AssertionError) -> Response:
	return _err(str(exc), 400)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@nlpc_bp.route("/documents", methods=["GET"])
def list_documents() -> Response:
	"""GET /api/nlpc/documents — list documents for tenant."""
	svc = _svc()
	limit = min(int(request.args.get("limit", 50)), 200)
	offset = int(request.args.get("offset", 0))
	language = request.args.get("language")
	docs = _run(svc.list_documents(limit=limit, offset=offset, language=language))
	return _ok([d.model_dump() for d in docs])


@nlpc_bp.route("/documents", methods=["POST"])
def create_document() -> Response:
	"""POST /api/nlpc/documents — ingest a new document."""
	svc = _svc()
	body = _body()
	tenant_id = (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id")
		or body.get("tenant_id", "default")
	)
	try:
		payload = NLPDocumentCreate(
			tenant_id=tenant_id,
			content=body.get("content", ""),
			title=body.get("title"),
			source=body.get("source"),
			source_id=body.get("source_id"),
			language=body.get("language"),
			is_sensitive=body.get("is_sensitive", False),
			retention_days=body.get("retention_days"),
			metadata=body.get("metadata", {}),
		)
	except Exception as exc:
		return _err(str(exc), 422)
	doc = _run(svc.create_document(payload))
	return _ok(doc.model_dump(), 201)


@nlpc_bp.route("/documents/<document_id>", methods=["GET"])
def get_document(document_id: str) -> Response:
	"""GET /api/nlpc/documents/<id>"""
	svc = _svc()
	doc = _run(svc.get_document(document_id))
	if doc is None:
		return _err("document not found", 404)
	return _ok(doc.model_dump())


@nlpc_bp.route("/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id: str) -> Response:
	"""DELETE /api/nlpc/documents/<id> — soft delete."""
	svc = _svc()
	ok = _run(svc.delete_document(document_id))
	if not ok:
		return _err("document not found", 404)
	return _ok({"deleted": True, "document_id": document_id})


# ---------------------------------------------------------------------------
# NLP Tasks — inline text (no prior document required)
# ---------------------------------------------------------------------------

@nlpc_bp.route("/detect-language", methods=["POST"])
def detect_language() -> Response:
	"""
	POST /api/nlpc/detect-language
	Body: {"text": "...", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	doc_id = body.get("document_id")
	try:
		result = _run(svc.detect_language(text, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/extract-entities", methods=["POST"])
def extract_entities() -> Response:
	"""
	POST /api/nlpc/extract-entities
	Body: {"text": "...", "entity_types": ["PERSON", "ORG"], "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	raw_types = body.get("entity_types")
	entity_types: list[EntityType] | None = None
	if raw_types:
		try:
			entity_types = [EntityType(t) for t in raw_types]
		except ValueError as exc:
			return _err(f"invalid entity_type: {exc}", 422)
	doc_id = body.get("document_id")
	try:
		results = _run(svc.extract_entities(text, entity_types=entity_types, document_id=doc_id))
		return _ok([r.model_dump() for r in results])
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/sentiment", methods=["POST"])
def sentiment_analysis() -> Response:
	"""
	POST /api/nlpc/sentiment
	Body: {"text": "...", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	doc_id = body.get("document_id")
	try:
		result = _run(svc.sentiment_analysis(text, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/intent", methods=["POST"])
def intent_classification() -> Response:
	"""
	POST /api/nlpc/intent
	Body: {"text": "...", "intents": ["book_flight", "cancel_order", ...], "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	intents: list[str] = body.get("intents", [])
	doc_id = body.get("document_id")
	if not intents:
		return _err("intents list is required", 422)
	try:
		result = _run(svc.intent_classification(text, intents=intents, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/summarise", methods=["POST"])
def summarise() -> Response:
	"""
	POST /api/nlpc/summarise
	Body: {"text": "...", "max_words": 100, "method": "extractive|abstractive|hybrid", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	max_words = int(body.get("max_words", 100))
	try:
		method = SummaryMethod(body.get("method", "extractive"))
	except ValueError:
		return _err(f"invalid method; choose from {[m.value for m in SummaryMethod]}", 422)
	doc_id = body.get("document_id")
	try:
		result = _run(svc.text_summarisation(text, max_words=max_words, method=method, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/translate", methods=["POST"])
def translate() -> Response:
	"""
	POST /api/nlpc/translate
	Body: {"text": "...", "target_lang": "sw", "source_lang": "auto", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	try:
		target_lang = LanguageCode(body.get("target_lang", ""))
	except ValueError:
		return _err("invalid target_lang", 422)
	try:
		source_lang = LanguageCode(body.get("source_lang", "auto"))
	except ValueError:
		source_lang = LanguageCode.AUTO
	doc_id = body.get("document_id")
	try:
		result = _run(svc.translate(text, target_lang=target_lang, source_lang=source_lang, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/embed", methods=["POST"])
def embed() -> Response:
	"""
	POST /api/nlpc/embed
	Body: {"text": "...", "model": "nomic-embed-text", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	model = body.get("model", "nomic-embed-text")
	doc_id = body.get("document_id")
	try:
		result = _run(svc.embed_text(text, model=model, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/classify", methods=["POST"])
def classify() -> Response:
	"""
	POST /api/nlpc/classify
	Body: {"text": "...", "taxonomy": "topics", "labels": ["tech", "sports"], "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	try:
		taxonomy = ClassificationTaxonomy(body.get("taxonomy", "topics"))
	except ValueError:
		return _err(f"invalid taxonomy; choose from {[t.value for t in ClassificationTaxonomy]}", 422)
	labels: list[str] | None = body.get("labels")
	doc_id = body.get("document_id")
	try:
		result = _run(svc.classify_document(text, taxonomy=taxonomy, labels=labels, document_id=doc_id))
		return _ok(result.model_dump())
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/key-phrases", methods=["POST"])
def key_phrases() -> Response:
	"""
	POST /api/nlpc/key-phrases
	Body: {"text": "...", "top_n": 10, "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	top_n = int(body.get("top_n", 10))
	doc_id = body.get("document_id")
	try:
		results = _run(svc.extract_key_phrases(text, top_n=top_n, document_id=doc_id))
		return _ok([r.model_dump() for r in results])
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/entity-linking", methods=["POST"])
def entity_linking() -> Response:
	"""
	POST /api/nlpc/entity-linking
	Body: {"text": "...", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	doc_id = body.get("document_id")
	try:
		results = _run(svc.named_entity_linking(text, document_id=doc_id))
		return _ok([r.model_dump() for r in results])
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/relations", methods=["POST"])
def relations() -> Response:
	"""
	POST /api/nlpc/relations
	Body: {"text": "...", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	doc_id = body.get("document_id")
	try:
		results = _run(svc.relation_extraction(text, document_id=doc_id))
		return _ok([r.model_dump() for r in results])
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/coreference", methods=["POST"])
def coreference() -> Response:
	"""
	POST /api/nlpc/coreference
	Body: {"text": "...", "document_id": "optional"}
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	doc_id = body.get("document_id")
	try:
		results = _run(svc.coreference_resolution(text, document_id=doc_id))
		return _ok([r.model_dump() for r in results])
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/african-language-id", methods=["POST"])
def african_language_id() -> Response:
	"""
	POST /api/nlpc/african-language-id
	Body: {"text": "..."}

	Dedicated African language identification endpoint with candidate scores.
	"""
	svc = _svc()
	body = _body()
	text = body.get("text", "")
	try:
		result = _run(svc.language_id_for_african_languages(text))
		return _ok(result)
	except RuleViolation as exc:
		return _err(exc.reason, 422)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


# ---------------------------------------------------------------------------
# Batch jobs
# ---------------------------------------------------------------------------

@nlpc_bp.route("/batch", methods=["POST"])
def create_batch() -> Response:
	"""
	POST /api/nlpc/batch
	Body: {
	  "name": "...",
	  "document_ids": ["id1", "id2"],
	  "tasks": ["sentiment_analysis", "language_detection"],
	  "priority": "normal"
	}
	"""
	svc = _svc()
	body = _body()
	name = body.get("name", f"batch-{uuid7str()[:8]}")
	document_ids: list[str] = body.get("document_ids", [])
	raw_tasks: list[str] = body.get("tasks", [])
	priority_str = body.get("priority", "normal")

	if not document_ids:
		return _err("document_ids required", 422)
	if not raw_tasks:
		return _err("tasks required", 422)
	try:
		tasks = [NLPTask(t) for t in raw_tasks]
	except ValueError as exc:
		return _err(f"invalid task: {exc}", 422)
	try:
		priority = PriorityLevel(priority_str)
	except ValueError:
		priority = PriorityLevel.NORMAL

	try:
		job = _run(svc.create_batch_job(name, document_ids, tasks, priority))
		return _ok(job.model_dump(), 201)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


@nlpc_bp.route("/batch/<job_id>", methods=["GET"])
def get_batch(job_id: str) -> Response:
	"""GET /api/nlpc/batch/<job_id>"""
	svc = _svc()
	job = _run(svc.get_batch_job(job_id))
	if job is None:
		return _err("batch job not found", 404)
	return _ok(job.model_dump())


@nlpc_bp.route("/batch/<job_id>/run", methods=["POST"])
def run_batch(job_id: str) -> Response:
	"""POST /api/nlpc/batch/<job_id>/run — execute all tasks in the job."""
	svc = _svc()
	try:
		job = _run(svc.run_batch_job(job_id))
		return _ok(job.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		return _err(str(exc), 500, traceback.format_exc())


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@nlpc_bp.route("/reports/usage", methods=["GET"])
def usage_report() -> Response:
	"""
	GET /api/nlpc/reports/usage?period_start=2026-01-01&period_end=2026-06-01
	"""
	svc = _svc()
	try:
		period_start = datetime.fromisoformat(
			request.args.get("period_start", "2026-01-01")
		)
		period_end = datetime.fromisoformat(
			request.args.get("period_end", datetime.utcnow().date().isoformat())
		)
	except ValueError as exc:
		return _err(f"invalid date format: {exc}", 422)
	report = _run(svc.usage_report(period_start, period_end))
	return _ok(report.model_dump())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@nlpc_bp.route("/health", methods=["GET"])
def health() -> Response:
	"""GET /api/nlpc/health"""
	from .service import _spacy, _langdetect, _transformers, _httpx
	return _ok({
		"status": "ok",
		"capability": "nlpc",
		"backends": {
			"spacy": _spacy is not None,
			"langdetect": _langdetect is not None,
			"transformers": _transformers is not None,
			"httpx": _httpx is not None,
		},
	})
