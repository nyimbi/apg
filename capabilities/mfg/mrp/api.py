"""REST API endpoints for Material Requirements Planning."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_mrp_api", __name__, url_prefix="/mfg-mrp")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_mrp"})


@blueprint.get("/")
def list_items():
    """List all Material Requirements Planning records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_mrp"})
