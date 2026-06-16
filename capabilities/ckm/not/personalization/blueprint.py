"""
Ckm Not Personalization — APG Blueprint Registration
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
	"capability_id": "ckm_not_personalization",
	"domain": "ckm",
	"name": "Ckm Not Personalization",
	"version": "1.0.0",
	"category": "Collaboration",
	"api_enabled": True,
	"web_interface": True,
	"multi_tenant": True,
}


def init_subcapability(appbuilder) -> dict[str, Any]:
	try:
		from .api import bp
		appbuilder.get_app.register_blueprint(bp)
		_log.info("Registered blueprint: %s", "ckm_not_personalization")
		return {"success": True, "capability_id": "ckm_not_personalization"}
	except Exception as exc:
		_log.error("Blueprint registration failed for ckm_not_personalization: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	return {"status": "healthy", "capability_id": "ckm_not_personalization"}
