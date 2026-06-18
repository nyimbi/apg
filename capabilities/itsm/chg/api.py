"""REST API endpoints for itsm_chg."""
from flask import Blueprint, jsonify

blueprint = Blueprint("itsm_chg_api", __name__, url_prefix="/itsm-chg")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "itsm_chg"})


@blueprint.get("/")
def list_items():
    """List all itsm_chg records."""
    return jsonify({"items": [], "total": 0, "capability": "itsm_chg"})
