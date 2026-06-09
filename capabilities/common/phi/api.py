"""PHI classifier capability — REST API endpoints."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from .service import PHIService
from .models import (
	ClassifyRequest,
	ClassifyBatchRequest,
	RedactRequest,
	RedactBatchRequest,
	ScanDocumentRequest,
	LogPhiAccessRequest,
	TestIdentifierRequest,
)

_log = logging.getLogger(__name__)

phi_api = Blueprint("phi_api", __name__, url_prefix="/api/phi")


def _svc() -> PHIService:
	return PHIService(tenant_id=request.headers.get("X-Tenant-Id", "default"))


@phi_api.get("/health")
async def health():
	return jsonify(await _svc().health_check())


@phi_api.post("/classify")
async def classify():
	body = ClassifyRequest.model_validate(request.get_json(force=True))
	return jsonify(await _svc().classify(body.field_name, body.value))


@phi_api.post("/classify/batch")
async def classify_batch():
	body = ClassifyBatchRequest.model_validate(request.get_json(force=True))
	return jsonify({"results": await _svc().classify_batch(body.fields)})


@phi_api.post("/redact")
async def redact():
	body = RedactRequest.model_validate(request.get_json(force=True))
	return jsonify(await _svc().redact(body.record))


@phi_api.post("/redact/batch")
async def redact_batch():
	body = RedactBatchRequest.model_validate(request.get_json(force=True))
	return jsonify({"results": await _svc().redact_batch(body.records)})


@phi_api.post("/scan/record")
async def scan_record():
	body = request.get_json(force=True) or {}
	return jsonify(await _svc().scan_record(body))


@phi_api.post("/scan/document")
async def scan_document():
	body = ScanDocumentRequest.model_validate(request.get_json(force=True))
	return jsonify(await _svc().scan_document(body.text))


@phi_api.get("/identifiers")
async def list_identifiers():
	return jsonify({"identifiers": _svc().get_phi_identifiers()})


@phi_api.post("/identifiers/test")
async def test_identifier():
	body = TestIdentifierRequest.model_validate(request.get_json(force=True))
	return jsonify(await _svc().test_identifier_pattern(body.pattern, body.test_value))


@phi_api.post("/access/log")
async def log_access():
	body = LogPhiAccessRequest.model_validate(request.get_json(force=True))
	return jsonify(await _svc().log_phi_access(body.accessor_id, body.record_id, body.purpose))


@phi_api.get("/compliance")
async def compliance_status():
	return jsonify(await _svc().get_compliance_status())


@phi_api.post("/validate/deidentification")
async def validate_deidentification():
	body = request.get_json(force=True) or {}
	return jsonify(await _svc().validate_deidentification(body))


@phi_api.post("/certify/safe-harbor")
async def certify_safe_harbor():
	body = request.get_json(force=True) or {}
	return jsonify(await _svc().certify_safe_harbor(body))


@phi_api.get("/audit")
async def get_audit():
	limit = int(request.args.get("limit", "50"))
	return jsonify({"events": await _svc().get_phi_audit_events(limit=limit)})


@phi_api.get("/report")
async def get_report():
	return jsonify(await _svc().generate_phi_report())
