"""REST API endpoints for Bill of Materials."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_bom_api", __name__, url_prefix="/mfg-bom")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_bom"})


@blueprint.get("/")
def list_items():
    """List all Bill of Materials records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_bom"})
