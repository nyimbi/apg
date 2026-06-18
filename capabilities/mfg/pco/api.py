"""REST API endpoints for Product Costing."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_pco_api", __name__, url_prefix="/mfg-pco")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_pco"})


@blueprint.get("/")
def list_items():
    """List all Product Costing records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_pco"})
