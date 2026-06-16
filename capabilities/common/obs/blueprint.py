# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""Observability (obs) capability registration entry point.

Called by the APG capability loader to register this capability with
Flask-AppBuilder.  Follows the standard APG blueprint contract.
"""
from __future__ import annotations

from typing import Any

CAPABILITY_METADATA: dict[str, Any] = {
	"capability_id": "obs",
	"domain": "common",
	"name": "Observability",
	"version": "1.0.0",
	"category": "Common",
	"description": (
		"OpenTelemetry-compatible distributed tracing, RED metrics, "
		"structured logging with correlation IDs, health endpoints, "
		"SLO management and burn-rate alerting."
	),
	"api_enabled": True,
	"multi_tenant": True,
	"subcapabilities": ["obs_trc", "obs_mtx", "obs_log"],
}


def init_subcapability(appbuilder: Any) -> dict[str, Any]:
	"""Register the obs Flask Blueprint with the AppBuilder app.

	Args:
		appbuilder: Flask-AppBuilder AppBuilder instance.

	Returns:
		dict with "success" bool and "capability_id".
	"""
	try:
		from .api import bp
		appbuilder.get_app().register_blueprint(bp)
		return {"success": True, "capability_id": "obs"}
	except Exception as exc:
		return {"success": False, "capability_id": "obs", "error": str(exc)}


def get_capability_info() -> dict[str, Any]:
	"""Return static capability metadata."""
	return CAPABILITY_METADATA


def get_health_status() -> dict[str, Any]:
	"""Return a lightweight health signal (no I/O)."""
	return {"status": "healthy", "capability_id": "obs"}
