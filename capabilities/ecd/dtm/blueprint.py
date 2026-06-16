"""
Ecd Dtm — APG Blueprint Registration
Auto-generated APG blueprint wrapper.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

CAPABILITY_METADATA: dict[str, Any] = {
	"capability_id": "ecd_dtm",
	"domain": "ecd",
	"name": "Ecd Dtm",
	"version": "1.0.0",
	"category": "Edge Computing & Devices",
	"api_enabled": True,
	"web_interface": True,
	"multi_tenant": True,
}


def init_subcapability(appbuilder) -> dict[str, Any]:
	"""Register this capability with the APG Flask application."""
	try:
		from .api import bp
		appbuilder.get_app.register_blueprint(bp)
		_log.info("Registered blueprint: %s", "ecd_dtm")
		return {"success": True, "capability_id": "ecd_dtm"}
	except Exception as exc:
		_log.error("Blueprint registration failed for ecd_dtm: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	"""Return capability metadata for APG discovery."""
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	"""Return capability health for APG health checks."""
	return {"status": "healthy", "capability_id": "ecd_dtm"}
