"""Flask Blueprint views for fintech_terminal capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("fintech_terminal", __name__, url_prefix="/fintech/terminal")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/terminals", "methods": ["GET"], "description": "Terminal list"},
	{"path": "/terminals/<terminal_id>", "methods": ["GET"], "description": "Terminal detail"},
	{"path": "/terminals/<terminal_id>/health", "methods": ["GET"], "description": "Terminal health"},
	{"path": "/terminals/<terminal_id>/float", "methods": ["GET"], "description": "Float balance"},
	{"path": "/analytics", "methods": ["GET"], "description": "Network analytics"},
	{"path": "/reconciliation/<terminal_id>", "methods": ["GET"], "description": "Reconciliation"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "fintech_terminal",
		"display_name": "Terminal Management System",
		"version": "1.1.0",
		"routes": _ROUTES,
	})


@blueprint.get("/terminals")
def terminal_list():
	"""Render terminal list page (JSON representation for Blueprint)."""
	return jsonify({
		"view": "terminal_list",
		"description": "Agency banking terminal network overview",
		"filters": {
			"status": request.args.get("status"),
			"agent_id": request.args.get("agent_id"),
			"terminal_type": request.args.get("terminal_type"),
		},
	})


@blueprint.get("/terminals/<terminal_id>")
def terminal_detail(terminal_id: str):
	return jsonify({
		"view": "terminal_detail",
		"terminal_id": terminal_id,
	})


@blueprint.get("/terminals/<terminal_id>/health")
def terminal_health(terminal_id: str):
	return jsonify({
		"view": "terminal_health",
		"terminal_id": terminal_id,
	})


@blueprint.get("/terminals/<terminal_id>/float")
def terminal_float(terminal_id: str):
	return jsonify({
		"view": "terminal_float",
		"terminal_id": terminal_id,
	})


@blueprint.get("/analytics")
def analytics():
	return jsonify({
		"view": "network_analytics",
		"period": request.args.get("period", "2026-06"),
		"network_id": request.args.get("network_id", "default"),
	})


@blueprint.get("/reconciliation/<terminal_id>")
def reconciliation(terminal_id: str):
	return jsonify({
		"view": "reconciliation",
		"terminal_id": terminal_id,
		"recon_date": request.args.get("date"),
	})
