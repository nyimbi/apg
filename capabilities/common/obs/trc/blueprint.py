"""
Obs Trc — APG Blueprint Registration
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
	"capability_id": "obs_trc",
	"domain": "common",
	"name": "Obs Trc",
	"version": "1.0.0",
	"category": "Common",
	"api_enabled": True,
	"web_interface": True,
	"multi_tenant": True,
}


def init_subcapability(appbuilder) -> dict[str, Any]:
	try:
		from .api import bp
		appbuilder.get_app.register_blueprint(bp)
		_log.info("Registered blueprint: %s", "obs_trc")
		return {"success": True, "capability_id": "obs_trc"}
	except Exception as exc:
		_log.error("Blueprint registration failed for obs_trc: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	return {"status": "healthy", "capability_id": "obs_trc"}
