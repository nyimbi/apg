"""REST API endpoints for SaaS Billing Engine."""
from flask import Blueprint, jsonify

blueprint = Blueprint("common_sbl_api", __name__, url_prefix="/common-sbl")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "common_sbl"})


@blueprint.get("/")
def list_items():
    """List all SaaS Billing Engine records."""
    return jsonify({"items": [], "total": 0, "capability": "common_sbl"})
