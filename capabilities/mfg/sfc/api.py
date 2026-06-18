"""REST API endpoints for Shop Floor Control."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_sfc_api", __name__, url_prefix="/mfg-sfc")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_sfc"})


@blueprint.get("/")
def list_items():
    """List all Shop Floor Control records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_sfc"})
