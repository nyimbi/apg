"""Electronic signature capability — REST API endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from .service import ESignatureService
from .models import SignRequest, SignBatchRequest, VerifyRequest, RevokeRequest

_log = logging.getLogger(__name__)

esig_api = Blueprint("esig_api", __name__, url_prefix="/api/esig")


def _svc() -> ESignatureService:
	return ESignatureService(tenant_id=request.headers.get("X-Tenant-Id", "default"))


import hashlib as _hashlib


@esig_api.get("/health")
def health():
	return jsonify({"status": "ok", "capability": "esig"})


@esig_api.post("/sign")
async def sign():
	body = SignRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		doc_hash = _hashlib.sha256(body.document_content.encode()).hexdigest() if body.document_content else ""
		record = await svc.sign(
			document_id=body.document_id,
			signer_id=body.signer_id,
			signer_display_name=body.signer_display_name,
			meaning=body.meaning,
			document_hash=doc_hash,
		)
	except Exception as exc:
		_log.exception("sign failed")
		return jsonify({"error": str(exc)}), 500
	return jsonify({
		"signature_id": record.signature_id,
		"document_id": record.document_id,
		"signer_id": record.signer_id,
		"meaning": record.meaning,
		"timestamp": record.timestamp,
		"signature_hash": record.signature_hash,
		"is_valid": record.is_valid,
	}), 201


@esig_api.post("/sign/batch")
async def sign_batch():
	body = SignBatchRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	results = []
	for req in body.signatures:
		doc_hash = _hashlib.sha256(req.document_content.encode()).hexdigest() if req.document_content else ""
		record = await svc.sign(
			document_id=req.document_id,
			signer_id=req.signer_id,
			signer_display_name=req.signer_display_name,
			meaning=req.meaning,
			document_hash=doc_hash,
		)
		results.append({
			"signature_id": record.signature_id,
			"document_id": record.document_id,
			"is_valid": record.is_valid,
		})
	return jsonify({"signatures": results, "total": len(results)}), 201


@esig_api.post("/verify")
async def verify():
	body = VerifyRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	result = await svc.verify(body.signature_id)
	return jsonify({
		"signature_id": body.signature_id,
		"valid": result.get("valid", False),
		"verified_at": datetime.now(timezone.utc).isoformat(),
	})


@esig_api.get("/signatures/<document_id>")
async def list_signatures(document_id: str):
	svc = _svc()
	signatures = await svc.list_signatures(document_id)
	return jsonify({
		"document_id": document_id,
		"signatures": [
			{
				"signature_id": s.signature_id,
				"signer_id": s.signer_id,
				"meaning": s.meaning,
				"timestamp": s.timestamp,
				"is_valid": s.is_valid,
			}
			for s in signatures
		],
		"total": len(signatures),
	})


@esig_api.delete("/signatures/<signature_id>")
async def revoke_signature(signature_id: str):
	body = (request.get_json(force=True) or {})
	reason = body.get("reason", "")
	svc = _svc()
	result = await svc.revoke(signature_id, reason=reason)
	return jsonify(result)


@esig_api.get("/compliance")
async def compliance_report():
	svc = _svc()
	report = await svc.get_compliance_report()
	return jsonify(report)


@esig_api.get("/audit")
async def audit_trail():
	svc = _svc()
	events = await svc.get_audit_trail()
	return jsonify({"events": events, "total": len(events)})
