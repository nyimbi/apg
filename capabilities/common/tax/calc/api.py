"""REST API endpoints for Tax Calculation Engine."""
from flask import Blueprint, jsonify

blueprint = Blueprint("common_tax_calc_api", __name__, url_prefix="/common-tax-calc")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "common_tax_calc"})


@blueprint.get("/")
def list_items():
    """List all Tax Calculation Engine records."""
    return jsonify({"items": [], "total": 0, "capability": "common_tax_calc"})
