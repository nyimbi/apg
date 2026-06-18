"""REST API endpoints for int_esb."""
from flask import Blueprint, jsonify

blueprint = Blueprint("int_esb_api", __name__, url_prefix="/int-esb")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "int_esb"})


@blueprint.get("/")
def list_items():
    """List all int_esb records."""
    return jsonify({"items": [], "total": 0, "capability": "int_esb"})
