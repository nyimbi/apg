"""REST API endpoints for Capacity Planning."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_cap_api", __name__, url_prefix="/mfg-cap")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_cap"})


@blueprint.get("/")
def list_items():
    """List all Capacity Planning records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_cap"})
