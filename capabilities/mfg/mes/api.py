"""REST API endpoints for Manufacturing Execution System."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_mes_api", __name__, url_prefix="/mfg-mes")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_mes"})


@blueprint.get("/")
def list_items():
    """List all Manufacturing Execution System records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_mes"})
