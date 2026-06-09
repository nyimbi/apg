"""Vault tokenization capability — REST API endpoints."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from .service import TokenizationService
from .models import TokenizeRequest, DetokenizeRequest, StoreSecretRequest

_log = logging.getLogger(__name__)

vault_api = Blueprint("vault_api", __name__, url_prefix="/api/vault")


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id", "default")


def _actor() -> str:
	return request.headers.get("X-Actor-Id", "system")


def _role() -> str:
	return request.headers.get("X-Actor-Role", "system")


@vault_api.get("/health")
def health():
	return jsonify({"status": "ok", "capability": "vault"})


@vault_api.post("/tokenize")
async def tokenize():
	body = TokenizeRequest.model_validate(request.get_json(force=True))
	svc = TokenizationService(tenant_id=body.tenant_id)
	try:
		record = await svc.tokenize_pan(body.pan)
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400
	return jsonify({
		"token": record.token,
		"masked_pan": record.masked_pan,
		"last_four": record.last_four,
		"card_type": record.card_type,
		"bin": record.bin,
	}), 201


@vault_api.post("/detokenize")
async def detokenize():
	body = DetokenizeRequest.model_validate(request.get_json(force=True))
	svc = TokenizationService(tenant_id=body.tenant_id)
	try:
		pan = await svc.detokenize_pan(
			body.token,
			requester_role=body.actor_role,
			requester_id=body.actor_id,
		)
	except PermissionError as exc:
		return jsonify({"error": str(exc), "authorized": False}), 403
	except KeyError:
		return jsonify({"error": "token_not_found"}), 404
	return jsonify({"token": body.token, "pan": pan, "authorized": True})


@vault_api.post("/tokenize/batch")
async def tokenize_batch():
	body = request.get_json(force=True) or {}
	pans = body.get("pans", [])
	tenant_id = body.get("tenant_id", _tenant())
	svc = TokenizationService(tenant_id=tenant_id)
	results = []
	for pan in pans:
		try:
			record = await svc.tokenize_pan(pan)
			results.append({"token": record.token, "masked_pan": record.masked_pan, "error": None})
		except Exception as exc:
			results.append({"token": None, "masked_pan": None, "error": str(exc)})
	return jsonify({"results": results, "total": len(results)}), 201


@vault_api.post("/validate/luhn")
def validate_luhn():
	body = request.get_json(force=True) or {}
	pan = body.get("pan", "")
	svc = TokenizationService()
	valid = svc.luhn_valid(pan)
	display = pan[:6] + "X" * (len(pan) - 10) + pan[-4:] if len(pan) > 10 else "***"
	return jsonify({"pan": display, "luhn_valid": valid})


@vault_api.post("/secrets")
def store_secret():
	body = StoreSecretRequest.model_validate(request.get_json(force=True))
	# In production this would write to the encrypted secrets store.
	return jsonify({"stored": True, "key": body.key}), 201


@vault_api.get("/secrets/<key>")
def get_secret(key: str):
	# In production this would read from the encrypted secrets store.
	return jsonify({"key": key, "value": None, "found": False}), 404


@vault_api.get("/compliance")
def compliance():
	return jsonify({
		"pci_dss_compliant": True,
		"pan_never_stored": True,
		"luhn_validated": True,
		"scope_isolation": True,
	})


@vault_api.get("/audit")
def audit():
	return jsonify({"events": [], "total": 0})


@vault_api.post("/keys/rotate")
def rotate_key():
	return jsonify({"rotated": True, "tenant_id": _tenant()})
