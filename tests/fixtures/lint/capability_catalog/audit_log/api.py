"""REST API endpoints for Audit Log."""
from flask import Blueprint, jsonify

blueprint = Blueprint("audit_log_api", __name__, url_prefix="/audit-log")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "audit_log"})


@blueprint.get("/")
def list_items():
    """List all Audit Log records."""
    return jsonify({"items": [], "total": 0, "capability": "audit_log"})
