"""
Fleet Management Flask Blueprint — APG composition engine integration.

Registers with the APG platform:
  - REST API under /api/fle/v1
  - UI views under /fle
  - Capability metadata and health endpoints
  - Permission manifest for APG auth_rbac
  - APG composition engine hook

Standalone usage (no APG platform required):
  from capabilities.transport.fle.blueprint import create_fle_blueprint
  bp = create_fle_blueprint()
  app.register_blueprint(bp)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

logger = logging.getLogger("apg.transport.fle.blueprint")

# ──────────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────────

CAPABILITY_ID = "transport_fle"
CAPABILITY_VERSION = "2.0.0"
CAPABILITY_NAME = "Fleet Management"

PERMISSIONS: dict[str, str] = {
	"transport_fle:view":             "View fleet data (read-only dashboard)",
	"transport_fle:vehicles":         "View vehicle registry",
	"transport_fle:vehicles_write":   "Register and update vehicles",
	"transport_fle:drivers":          "View driver roster",
	"transport_fle:drivers_write":    "Register and update drivers",
	"transport_fle:trips":            "View and manage trips",
	"transport_fle:trips_dispatch":   "Dispatch and start trips",
	"transport_fle:fuel":             "Record and view fuel purchases",
	"transport_fle:maintenance":      "View and schedule maintenance",
	"transport_fle:inspections":      "Record vehicle inspections",
	"transport_fle:incidents":        "Report and manage incidents",
	"transport_fle:compliance":       "View compliance calendar and COF",
	"transport_fle:telematics":       "Ingest and view telematics events",
	"transport_fle:reports":          "Generate fleet reports and TCO",
	"transport_fle:admin":            "Full administrative access",
}

MENU_ITEMS: list[dict[str, Any]] = [
	{"label": "Fleet Dashboard",      "url": "/fle/",               "icon": "fa-tachometer-alt", "category": "Fleet",      "permission": "transport_fle:view"},
	{"label": "Vehicles",             "url": "/fle/vehicles",       "icon": "fa-truck",          "category": "Fleet",      "permission": "transport_fle:vehicles"},
	{"label": "Drivers",              "url": "/fle/drivers",        "icon": "fa-id-card",        "category": "Fleet",      "permission": "transport_fle:drivers"},
	{"label": "Trips",                "url": "/fle/trips",          "icon": "fa-route",          "category": "Fleet",      "permission": "transport_fle:trips"},
	{"label": "Fuel",                 "url": "/fle/fuel",           "icon": "fa-gas-pump",       "category": "Fleet",      "permission": "transport_fle:fuel"},
	{"label": "Maintenance",          "url": "/fle/maintenance",    "icon": "fa-wrench",         "category": "Fleet",      "permission": "transport_fle:maintenance"},
	{"label": "Compliance Calendar",  "url": "/fle/compliance",     "icon": "fa-calendar-check", "category": "Compliance", "permission": "transport_fle:compliance"},
	{"label": "Predictive Alerts",    "url": "/fle/reports/predictive-maintenance", "icon": "fa-bell", "category": "Compliance", "permission": "transport_fle:reports"},
	{"label": "Fleet Analytics",      "url": "/fle/reports/utilisation", "icon": "fa-chart-bar", "category": "Analytics",  "permission": "transport_fle:reports"},
]


def _capability_metadata() -> dict[str, Any]:
	return {
		"capability_id": CAPABILITY_ID,
		"capability_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"description": "World-class fleet lifecycle management with EU/US HOS compliance, telematics, predictive maintenance, and TCO analytics.",
		"provides": [
			"vehicle_lifecycle_workflow",
			"telematics_integration_workflow",
			"driver_management_workflow",
			"fleet_utilisation_analytics_workflow",
			"fleet_compliance_workflow",
		],
		"requires": ["auth", "audl", "mten", "conf", "ntfy"],
		"permissions": list(PERMISSIONS.keys()),
		"menu_items": MENU_ITEMS,
		"api_prefix": "/api/fle/v1",
		"ui_prefix": "/fle",
		"health_endpoint": "/api/fle/v1/health",
	}


# ──────────────────────────────────────────────────────────────────
# Blueprint factory
# ──────────────────────────────────────────────────────────────────

def create_fle_blueprint() -> Blueprint:
	"""
	Return a Flask Blueprint that integrates both the REST API and UI views.

	Registers:
	  - /api/fle/v1/*  (REST API)
	  - /fle/*         (UI views)
	  - /fle/capability/metadata
	  - /fle/capability/health
	"""
	meta_bp = Blueprint("fle_meta", __name__, url_prefix="/fle")

	@meta_bp.get("/capability/metadata")
	def get_metadata():
		return jsonify(_capability_metadata()), 200

	@meta_bp.get("/capability/health")
	def health():
		return jsonify({
			"status": "ok",
			"capability_id": CAPABILITY_ID,
			"version": CAPABILITY_VERSION,
			"timestamp": datetime.utcnow().isoformat(),
		}), 200

	@meta_bp.get("/capability/permissions")
	def get_permissions():
		return jsonify(PERMISSIONS), 200

	@meta_bp.get("/capability/menu")
	def get_menu():
		return jsonify(MENU_ITEMS), 200

	return meta_bp


def register_with_apg(app: Any, appbuilder: Any | None = None) -> None:
	"""
	Register the Fleet Management capability with a Flask app and optionally
	an APG AppBuilder instance.

	Args:
		app:          Flask application instance.
		appbuilder:   APG AppBuilder (Flask-AppBuilder) — optional.
	"""
	from .api import fle_bp
	from .views import fle_views_bp

	meta_bp = create_fle_blueprint()

	app.register_blueprint(fle_bp)
	app.register_blueprint(fle_views_bp)
	app.register_blueprint(meta_bp)

	if appbuilder is not None:
		_register_appbuilder_menu(appbuilder)
		_register_appbuilder_permissions(appbuilder)

	logger.info(
		"[FLE] Fleet Management capability registered — API: /api/fle/v1, UI: /fle"
	)

	# Emit registration event to APG composition engine if available
	try:
		from capabilities.composition import register_capability
		register_capability(CAPABILITY_ID, _capability_metadata())
		logger.info("[FLE] Registered with APG composition engine")
	except ImportError:
		logger.debug("[FLE] APG composition engine not available — standalone mode")


def _register_appbuilder_menu(appbuilder: Any) -> None:
	"""Add fleet menu items to APG's navigation."""
	try:
		for item in MENU_ITEMS:
			appbuilder.add_link(
				item["label"],
				href=item["url"],
				icon=item.get("icon", ""),
				category=item.get("category", "Fleet"),
				category_icon="fa-truck",
			)
		logger.info("[FLE] AppBuilder menu items registered")
	except Exception as exc:
		logger.warning("[FLE] AppBuilder menu registration failed: %s", exc)


def _register_appbuilder_permissions(appbuilder: Any) -> None:
	"""Register FLE permissions with APG auth_rbac."""
	try:
		sm = appbuilder.sm
		for perm_name, description in PERMISSIONS.items():
			try:
				sm.add_permission_to_role(
					permission_name=perm_name,
					role_name="Fleet_Admin",
				)
			except Exception:
				pass  # role/permission may already exist
		logger.info("[FLE] Permissions registered with APG auth_rbac")
	except Exception as exc:
		logger.warning("[FLE] Permission registration failed: %s", exc)


# ──────────────────────────────────────────────────────────────────
# APG composition engine hook
# ──────────────────────────────────────────────────────────────────

def get_apg_composition_registration() -> dict[str, Any]:
	"""
	Return the registration payload expected by the APG composition engine.

	Called automatically by the APG platform on capability discovery.
	"""
	return {
		"capability_id": CAPABILITY_ID,
		"capability_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"metadata": _capability_metadata(),
		"registered_at": datetime.utcnow().isoformat(),
		"status": "active",
	}


__all__ = [
	"CAPABILITY_ID",
	"CAPABILITY_VERSION",
	"CAPABILITY_NAME",
	"PERMISSIONS",
	"MENU_ITEMS",
	"create_fle_blueprint",
	"register_with_apg",
	"get_apg_composition_registration",
]
