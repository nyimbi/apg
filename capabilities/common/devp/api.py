"""REST API endpoints for Developer Portal."""
from flask import Blueprint, jsonify

blueprint = Blueprint("common_devp_api", __name__, url_prefix="/common-devp")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "common_devp"})


@blueprint.get("/")
def list_items():
    """List all Developer Portal records."""
    return jsonify({"items": [], "total": 0, "capability": "common_devp"})
