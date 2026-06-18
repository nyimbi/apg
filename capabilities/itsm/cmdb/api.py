"""REST API endpoints for Configuration Management Database."""
from flask import Blueprint, jsonify

blueprint = Blueprint("itsm_cmdb_api", __name__, url_prefix="/itsm-cmdb")


@blueprint.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "capability": "itsm_cmdb"})


@blueprint.get("/")
def list_items():
    """List all Configuration Management Database records."""
    return jsonify({"items": [], "total": 0, "capability": "itsm_cmdb"})
