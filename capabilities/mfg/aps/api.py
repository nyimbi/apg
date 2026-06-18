"""REST API endpoints for Advanced Planning and Scheduling."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_aps_api", __name__, url_prefix="/mfg-aps")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_aps"})


@blueprint.get("/")
def list_items():
    """List all Advanced Planning and Scheduling records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_aps"})
