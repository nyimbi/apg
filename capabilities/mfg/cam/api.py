"""REST API endpoints for Computer-Aided Manufacturing."""
from flask import Blueprint, jsonify

blueprint = Blueprint("mfg_cam_api", __name__, url_prefix="/mfg-cam")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "mfg_cam"})


@blueprint.get("/")
def list_items():
    """List all Computer-Aided Manufacturing records."""
    return jsonify({"items": [], "total": 0, "capability": "mfg_cam"})
