"""REST API endpoints for Incident Management."""
from flask import Blueprint, jsonify

blueprint = Blueprint("itsm_inc_api", __name__, url_prefix="/itsm-inc")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "itsm_inc"})


@blueprint.get("/")
def list_items():
    """List all Incident Management records."""
    return jsonify({"items": [], "total": 0, "capability": "itsm_inc"})
