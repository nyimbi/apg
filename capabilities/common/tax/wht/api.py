"""REST API endpoints for common_tax_wht."""
from flask import Blueprint, jsonify

blueprint = Blueprint("common_tax_wht_api", __name__, url_prefix="/common-tax-wht")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "common_tax_wht"})


@blueprint.get("/")
def list_items():
    """List all common_tax_wht records."""
    return jsonify({"items": [], "total": 0, "capability": "common_tax_wht"})
