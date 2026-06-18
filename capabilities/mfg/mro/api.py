"""REST API endpoints for Maintenance, Repair and Overhaul."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_mro_api", __name__, url_prefix="/mfg-mro")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_mro"})


@blueprint.get("/")
def list_items():
    """List all Maintenance, Repair and Overhaul records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_mro"})
