"""REST API endpoints for Customer Master."""
from flask import Blueprint, jsonify

blueprint = Blueprint("customer_master_api", __name__, url_prefix="/customer-master")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "customer_master"})


@blueprint.get("/")
def list_items():
    """List all Customer Master records."""
    return jsonify({"items": [], "total": 0, "capability": "customer_master"})
