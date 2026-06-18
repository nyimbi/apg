"""REST API endpoints for Three-Way Match Engine."""
from flask import Blueprint, jsonify

blueprint = Blueprint("proc_twy_api", __name__, url_prefix="/proc-twy")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "proc_twy"})


@blueprint.get("/")
def list_items():
    """List all Three-Way Match Engine records."""
    return jsonify({"items": [], "total": 0, "capability": "proc_twy"})
