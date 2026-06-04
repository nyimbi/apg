"""Flask Blueprint views for grc_rsa capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_rsa", __name__, url_prefix="/grc/rsa")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/risks", "methods": ["GET"], "description": "Risk register"},
	{"path": "/risks/<risk_id>", "methods": ["GET"], "description": "Risk detail"},
	{"path": "/controls", "methods": ["GET"], "description": "Control library"},
	{"path": "/kri", "methods": ["GET"], "description": "KRI dashboard"},
	{"path": "/heat-map", "methods": ["GET"], "description": "Risk heat map"},
	{"path": "/dashboard", "methods": ["GET"], "description": "Risk dashboard"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "grc_rsa",
		"display_name": "Risk & Security Assessment",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/risks")
def risk_register():
	return jsonify({
		"view": "risk_register",
		"filters": {
			"entity_id": request.args.get("entity_id"),
			"category": request.args.get("category"),
			"rating": request.args.get("rating"),
			"status": request.args.get("status"),
		},
	})


@blueprint.get("/risks/<risk_id>")
def risk_detail(risk_id: str):
	return jsonify({"view": "risk_detail", "risk_id": risk_id})


@blueprint.get("/controls")
def controls():
	return jsonify({"view": "control_library"})


@blueprint.get("/kri")
def kri_dashboard():
	return jsonify({
		"view": "kri_dashboard",
		"entity_id": request.args.get("entity_id"),
		"period": request.args.get("period"),
	})


@blueprint.get("/heat-map")
def heat_map():
	return jsonify({
		"view": "risk_heat_map",
		"entity_id": request.args.get("entity_id"),
		"as_of_date": request.args.get("as_of_date"),
	})


@blueprint.get("/dashboard")
def dashboard():
	return jsonify({
		"view": "risk_dashboard",
		"entity_id": request.args.get("entity_id", "default"),
	})
