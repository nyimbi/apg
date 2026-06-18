"""REST API endpoints for Chama & ROSCA Engine."""
from flask import Blueprint, jsonify

blueprint = Blueprint("fintech_chama_api", __name__, url_prefix="/fintech-chama")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "fintech_chama"})


@blueprint.get("/")
def list_items():
    """List all Chama & ROSCA Engine records."""
    return jsonify({"items": [], "total": 0, "capability": "fintech_chama"})
