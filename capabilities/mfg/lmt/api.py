"""REST API endpoints for Lot and Batch Management."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_lmt_api", __name__, url_prefix="/mfg-lmt")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_lmt"})


@blueprint.get("/")
def list_items():
    """List all Lot and Batch Management records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_lmt"})
