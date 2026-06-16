"""
Fintech Sacco Lnd — APG Blueprint Registration
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
	"capability_id": "fintech_sacco_lnd",
	"domain": "fintech",
	"name": "Fintech Sacco Lnd",
	"version": "1.0.0",
	"category": "Fintech",
	"api_enabled": True,
	"web_interface": True,
	"multi_tenant": True,
}


def init_subcapability(appbuilder) -> dict[str, Any]:
	try:
		from .api import bp
		appbuilder.get_app.register_blueprint(bp)
		_log.info("Registered blueprint: %s", "fintech_sacco_lnd")
		return {"success": True, "capability_id": "fintech_sacco_lnd"}
	except Exception as exc:
		_log.error("Blueprint registration failed for fintech_sacco_lnd: %s", exc)
		return {"success": False, "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	return {"status": "healthy", "capability_id": "fintech_sacco_lnd"}
