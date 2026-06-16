"""
USSD Engine — APG capability registration entry point.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from __future__ import annotations

CAPABILITY_METADATA: dict = {
	"capability_id": "ussd",
	"domain":        "common",
	"name":          "USSD Engine",
	"version":       "1.0.0",
	"category":      "Common",
	"description":   "Session state machine for Africa's Talking + Safaricom USSD gateways. Menu DSL, conditional routing, i18n support.",
	"api_enabled":   True,
	"multi_tenant":  True,
	"keywords": [
		"ussd", "session", "menu", "gateway",
		"africastalking", "safaricom", "i18n", "sms",
	],
}


def init_subcapability(appbuilder: object) -> dict:
	"""
	Register the USSD Engine capability with Flask-AppBuilder.

	Called by the APG capability loader at startup.  Registers the API
	blueprint on the underlying Flask app.

	Args:
		appbuilder: The Flask-AppBuilder AppBuilder instance.

	Returns:
		Dict with success flag and capability_id.
	"""
	try:
		from .api import bp
		app = appbuilder.get_app  # type: ignore[attr-defined]
		app.register_blueprint(bp)
		return {"success": True, "capability_id": "ussd"}
	except Exception as exc:
		import logging
		logging.getLogger(__name__).error("USSD capability init failed: %s", exc, exc_info=True)
		return {"success": False, "capability_id": "ussd", "error": str(exc)}


def get_capability_info() -> dict:
	"""Return static capability metadata."""
	return CAPABILITY_METADATA


def get_health_status() -> dict:
	"""Return runtime health status."""
	return {"status": "healthy", "capability_id": "ussd"}
