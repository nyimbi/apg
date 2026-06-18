"""REST API endpoints for Quality Management System."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_qms_api", __name__, url_prefix="/mfg-qms")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_qms"})


@blueprint.get("/")
def list_items():
    """List all Quality Management System records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_qms"})
