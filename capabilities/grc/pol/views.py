"""Flask Blueprint views for grc_pol capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_pol", __name__, url_prefix="/grc/pol")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/policies", "methods": ["GET"], "description": "Policy library"},
	{"path": "/policies/<policy_id>", "methods": ["GET"], "description": "Policy detail"},
	{"path": "/exceptions", "methods": ["GET"], "description": "Exception list"},
	{"path": "/acknowledgements", "methods": ["GET"], "description": "Acknowledgement status"},
	{"path": "/dashboard", "methods": ["GET"], "description": "Policy dashboard"},
	{"path": "/analytics", "methods": ["GET"], "description": "Policy analytics"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "grc_pol",
		"display_name": "Policy Management",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/policies")
def policy_library():
	return jsonify({
		"view": "policy_library",
		"filters": {
			"category": request.args.get("category"),
			"status": request.args.get("status"),
			"policy_type": request.args.get("policy_type"),
		},
	})


@blueprint.get("/policies/<policy_id>")
def policy_detail(policy_id: str):
	return jsonify({"view": "policy_detail", "policy_id": policy_id})


@blueprint.get("/exceptions")
def exceptions():
	return jsonify({
		"view": "exception_list",
		"status": request.args.get("status", "pending"),
	})


@blueprint.get("/acknowledgements")
def acknowledgements():
	return jsonify({
		"view": "acknowledgement_status",
		"policy_id": request.args.get("policy_id"),
		"employee_id": request.args.get("employee_id"),
	})


@blueprint.get("/dashboard")
def dashboard():
	return jsonify({
		"view": "policy_dashboard",
		"entity_id": request.args.get("entity_id", "default"),
	})


@blueprint.get("/analytics")
def analytics():
	return jsonify({
		"view": "policy_analytics",
		"period": request.args.get("period", "2026-06"),
	})


@blueprint.get("/gap-analysis")
def gap_analysis():
	return jsonify({
		"view": "policy_gap_analysis",
		"entity_id": request.args.get("entity_id"),
		"framework": request.args.get("framework"),
	})
