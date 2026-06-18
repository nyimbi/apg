"""REST API endpoints for int_dsy."""
from flask import Blueprint, jsonify

blueprint = Blueprint("int_dsy_api", __name__, url_prefix="/int-dsy")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "int_dsy"})


@blueprint.get("/")
def list_items():
    """List all int_dsy records."""
    return jsonify({"items": [], "total": 0, "capability": "int_dsy"})
