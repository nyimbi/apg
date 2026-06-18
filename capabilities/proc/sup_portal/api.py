"""REST API endpoints for proc_sup_portal."""
from flask import Blueprint, jsonify

blueprint = Blueprint("proc_sup_portal_api", __name__, url_prefix="/proc-sup-portal")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "proc_sup_portal"})


@blueprint.get("/")
def list_items():
    """List all proc_sup_portal records."""
    return jsonify({"items": [], "total": 0, "capability": "proc_sup_portal"})
