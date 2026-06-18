"""REST API endpoints for itsm_prb."""
from flask import Blueprint, jsonify

blueprint = Blueprint("itsm_prb_api", __name__, url_prefix="/itsm-prb")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "itsm_prb"})


@blueprint.get("/")
def list_items():
    """List all itsm_prb records."""
    return jsonify({"items": [], "total": 0, "capability": "itsm_prb"})
