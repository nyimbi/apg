"""REST API endpoints for common_mlr."""
from flask import Blueprint, jsonify

blueprint = Blueprint("common_mlr_api", __name__, url_prefix="/common-mlr")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "common_mlr"})


@blueprint.get("/")
def list_items():
    """List all common_mlr records."""
    return jsonify({"items": [], "total": 0, "capability": "common_mlr"})
