"""
Pha Reg — APG Blueprint Registration
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
	"capability_id": "pha_reg",
	"domain": "pharma",
	"name": "Pha Reg",
	"version": "1.0.0",
	"category": "Pharma",
	"api_enabled": True,
	"web_interface": True,
	"multi_tenant": True,
}


def init_subcapability(appbuilder) -> dict[str, Any]:
	"""Register this capability with the APG Flask application."""
	try:
		from .api import bp
		appbuilder.get_app.register_blueprint(bp)
		_log.info("Registered blueprint: %s", "pha_reg")
		return {"success": True, "capability_id": "pha_reg"}
	except Exception as exc:
		_log.error("Blueprint registration failed for pha_reg: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	"""Return capability metadata for APG discovery."""
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	"""Return capability health for APG health checks."""
	return {"status": "healthy", "capability_id": "pha_reg"}
