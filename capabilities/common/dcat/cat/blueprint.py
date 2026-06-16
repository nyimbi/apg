"""
Dcat Cat — APG Blueprint Registration
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
	"capability_id": "dcat_cat",
	"domain": "common",
	"name": "Dcat Cat",
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
		_log.info("Registered blueprint: %s", "dcat_cat")
		return {"success": True, "capability_id": "dcat_cat"}
	except Exception as exc:
		_log.error("Blueprint registration failed for dcat_cat: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	return {"status": "healthy", "capability_id": "dcat_cat"}
