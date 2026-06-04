"""Flask Blueprint views for grc_aud capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_aud", __name__, url_prefix="/grc/aud")

_ROUTES = [
	{"path": "/", "methods": ["GET"], "description": "Capability index"},
	{"path": "/engagements", "methods": ["GET"], "description": "Audit engagement list"},
	{"path": "/engagements/<engagement_id>", "methods": ["GET"], "description": "Engagement detail"},
	{"path": "/findings", "methods": ["GET"], "description": "Audit findings board"},
	{"path": "/findings/<finding_id>", "methods": ["GET"], "description": "Finding detail"},
	{"path": "/reports", "methods": ["GET"], "description": "Audit reports"},
	{"path": "/dashboard", "methods": ["GET"], "description": "Audit committee dashboard"},
	{"path": "/kpi", "methods": ["GET"], "description": "Audit KPI report"},
]


@blueprint.get("/")
def index():
	return jsonify({
		"capability": "grc_aud",
		"display_name": "Audit Management",
		"version": "1.0.0",
		"routes": _ROUTES,
	})


@blueprint.get("/engagements")
def engagement_list():
	return jsonify({
		"view": "audit_engagement_list",
		"filters": {
			"entity_id": request.args.get("entity_id"),
			"status": request.args.get("status"),
			"audit_type": request.args.get("audit_type"),
			"year": request.args.get("year"),
		},
	})


@blueprint.get("/engagements/<engagement_id>")
def engagement_detail(engagement_id: str):
	return jsonify({"view": "audit_engagement_detail", "engagement_id": engagement_id})


@blueprint.get("/findings")
def findings():
	return jsonify({
		"view": "audit_finding_board",
		"filters": {
			"entity_id": request.args.get("entity_id"),
			"risk_rating": request.args.get("risk_rating"),
			"status": request.args.get("status"),
		},
	})


@blueprint.get("/findings/<finding_id>")
def finding_detail(finding_id: str):
	return jsonify({"view": "audit_finding_detail", "finding_id": finding_id})


@blueprint.get("/reports")
def reports():
	return jsonify({
		"view": "audit_report_list",
		"entity_id": request.args.get("entity_id"),
	})


@blueprint.get("/dashboard")
def dashboard():
	return jsonify({
		"view": "audit_committee_report",
		"entity_id": request.args.get("entity_id", "default"),
		"period": request.args.get("period", "2026"),
	})


@blueprint.get("/kpi")
def kpi():
	return jsonify({
		"view": "audit_kpi_report",
		"entity_id": request.args.get("entity_id", "default"),
		"period": request.args.get("period", "2026"),
	})


@blueprint.get("/universe")
def audit_universe():
	return jsonify({
		"view": "audit_universe",
		"entity_id": request.args.get("entity_id", "default"),
	})
