"""REST API endpoints for ECM / Records Management."""
from flask import Blueprint, jsonify

blueprint = Blueprint("ckm_ecm_api", __name__, url_prefix="/ckm-ecm")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "ckm_ecm"})


@blueprint.get("/")
def list_items():
    """List all ECM / Records Management records."""
    return jsonify({"items": [], "total": 0, "capability": "ckm_ecm"})
