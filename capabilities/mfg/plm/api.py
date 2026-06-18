"""REST API endpoints for Product Lifecycle Management."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_plm_api", __name__, url_prefix="/mfg-plm")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_plm"})


@blueprint.get("/")
def list_items():
    """List all Product Lifecycle Management records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_plm"})
