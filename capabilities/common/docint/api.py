"""Document Intelligence — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import DocumentIntelligenceService

_log = logging.getLogger(__name__)

bp = Blueprint("docint", __name__, url_prefix="/api/docint")
_svc: DocumentIntelligenceService | None = None


def _get_service() -> DocumentIntelligenceService:
	global _svc
	if _svc is None:
		_svc = DocumentIntelligenceService()
	return _svc


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


# ── Documents ─────────────────────────────────────────────────────

@bp.get("/documents")
def list_documents():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	document_type = request.args.get("document_type")
	status = request.args.get("status")
	result = _run(svc.list_documents(tenant_id, document_type=document_type, status=status))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/documents")
def submit_document():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.submit_document(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("submit_document error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/documents/<document_id>")
def get_document(document_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_document(tenant_id, document_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/documents/<document_id>")
def delete_document(document_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_document(tenant_id, document_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/documents/<document_id>/full")
def get_document_with_extraction(document_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_document_with_extraction(tenant_id, document_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── OCR ───────────────────────────────────────────────────────────

@bp.post("/documents/<document_id>/ocr")
def run_ocr(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.run_ocr(tenant_id, document_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Extraction ────────────────────────────────────────────────────

@bp.post("/documents/<document_id>/extract")
def extract_fields(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.extract_fields(tenant_id, document_id, model))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/extractions")
def list_extractions():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	document_type = request.args.get("document_type")
	result = _run(svc.list_extractions(tenant_id, document_type))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/extractions/<extraction_id>")
def get_extraction(extraction_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_extraction(tenant_id, extraction_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Pipelines ─────────────────────────────────────────────────────

@bp.post("/documents/<document_id>/process/invoice")
def process_invoice(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.process_invoice(tenant_id, document_id, model))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/documents/<document_id>/process/contract")
def process_contract(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.process_contract(tenant_id, document_id, model))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/documents/<document_id>/verify/id")
def verify_id_document(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.verify_id_document(tenant_id, document_id, model))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/documents/<document_id>/digitize")
def digitize_form(document_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	template_id = body.get("template_id")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.digitize_form(tenant_id, document_id, template_id, model))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Form templates ────────────────────────────────────────────────

@bp.post("/templates")
def create_template():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.create_form_template(tenant_id=tenant_id, **body))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/templates")
def list_templates():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.list_form_templates(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/templates/<template_id>")
def get_template(template_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_form_template(tenant_id, template_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/templates/<template_id>")
def delete_template(template_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_form_template(tenant_id, template_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Batch ─────────────────────────────────────────────────────────

@bp.post("/batches")
def submit_batch():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	documents = body.get("documents", [])
	pipeline = body.get("pipeline", "ocr_llm")
	return jsonify(_run(svc.submit_batch(tenant_id, documents, pipeline))), 201


@bp.get("/batches/<batch_id>")
def get_batch(batch_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_batch(tenant_id, batch_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/batches/<batch_id>/process")
def process_batch(batch_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	model = body.get("model", "ollama/llama3")
	try:
		return jsonify(_run(svc.process_batch(tenant_id, batch_id, model))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Statistics ────────────────────────────────────────────────────

@bp.get("/statistics")
def statistics():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.processing_statistics(tenant_id))), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200
