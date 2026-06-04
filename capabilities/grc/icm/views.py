"""Flask Blueprint views for grc_icm capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_icm", __name__, url_prefix="/grc/icm")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/incidents", "methods": ["GET"], "description": "Incident list"},
	{"path": "/incidents/<incident_id>", "methods": ["GET"], "description": "Incident detail"},
	{"path": "/corrective-actions", "methods": ["GET"], "description": "Corrective actions"},
	{"path": "/compliance-tests", "methods": ["GET"], "description": "Compliance tests"},
	{"path": "/deficiencies", "methods": ["GET"], "description": "Compliance deficiencies"},
	{"path": "/dashboard", "methods": ["GET"], "description": "Compliance dashboard"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "grc_icm",
		"display_name": "Incident & Compliance Management",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/incidents")
def incident_list():
	return jsonify({
		"view": "incident_list",
		"filters": {
			"entity_id": request.args.get("entity_id"),
			"severity": request.args.get("severity"),
			"status": request.args.get("status"),
			"incident_type": request.args.get("incident_type"),
		},
	})


@blueprint.get("/incidents/<incident_id>")
def incident_detail(incident_id: str):
	return jsonify({"view": "incident_detail", "incident_id": incident_id})


@blueprint.get("/corrective-actions")
def corrective_actions():
	return jsonify({
		"view": "corrective_action_list",
		"status": request.args.get("status", "open"),
	})


@blueprint.get("/compliance-tests")
def compliance_tests():
	return jsonify({
		"view": "compliance_test_list",
		"entity_id": request.args.get("entity_id"),
		"period": request.args.get("period"),
	})


@blueprint.get("/deficiencies")
def deficiencies():
	return jsonify({
		"view": "deficiency_list",
		"severity": request.args.get("severity"),
		"status": request.args.get("status", "open"),
	})


@blueprint.get("/dashboard")
def dashboard():
	return jsonify({
		"view": "compliance_dashboard",
		"entity_id": request.args.get("entity_id", "default"),
	})


@blueprint.get("/analytics")
def analytics():
	return jsonify({
		"view": "incident_analytics",
		"entity_id": request.args.get("entity_id"),
		"period": request.args.get("period", "2026-06"),
	})
