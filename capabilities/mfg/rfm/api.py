"""REST API endpoints for Repetitive Manufacturing."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_rfm_api", __name__, url_prefix="/mfg-rfm")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_rfm"})


@blueprint.get("/")
def list_items():
    """List all Repetitive Manufacturing records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_rfm"})
