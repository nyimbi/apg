"""
APG Composition Engine Blueprint for Digital Lending.

Registers the lending capability with the APG platform:
  - Flask blueprint mounting
  - Menu integration
  - Permission definitions
  - Composition engine metadata
  - Health checks

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Capability metadata for composition engine
# ---------------------------------------------------------------------------

CAPABILITY_META: dict[str, Any] = {
	"id": "fintech_lending",
	"name": "Digital Lending",
	"category": "fintech",
	"version": "2.0.0",
	"description": (
		"World-class digital lending platform covering the full loan lifecycle: "
		"origination, underwriting, disbursement, repayment, collections, IFRS 9 "
		"provisioning, and portfolio analytics."
	),
	"menu_category": "FinTech",
	"menu_icon": "fa-hand-holding-usd",
	"url_prefix": "/api/v1/lending",
	"ui_prefix": "/lending",
	"dependencies": ["auth_rbac", "audit_compliance", "fintech_kyc", "fintech_aml"],
	"optional_dependencies": ["notification_engine", "document_management"],
	"provides": [
		"loan_product_governance",
		"borrower_lifecycle",
		"credit_application_workflow",
		"underwriting_decisioning",
		"loan_offer_workflow",
		"disbursement_control",
		"repayment_tracking",
		"delinquency_management",
		"ifrs9_provisioning",
		"portfolio_analytics",
	],
	"permissions": [
		"lending.product.read",
		"lending.product.write",
		"lending.application.read",
		"lending.application.write",
		"lending.underwrite",
		"lending.disburse",
		"lending.repayment.post",
		"lending.collections.manage",
		"lending.restructure",
		"lending.writeoff",
		"lending.reports.view",
		"lending.admin",
	],
	"database_prefix": "ld_",
	"streaming_topic": "apg.fintech.lending.lifecycle",
}


# ---------------------------------------------------------------------------
# APG blueprint registration
# ---------------------------------------------------------------------------

def register_blueprint(app: Any, url_prefix: str | None = None) -> None:
	"""
	Register the lending API blueprint with a Flask app.

	Args:
		app: Flask application instance.
		url_prefix: Override URL prefix (defaults to /api/v1/lending).
	"""
	try:
		from .api import lending_bp
	except ImportError:
		from api import lending_bp  # type: ignore

	prefix = url_prefix or CAPABILITY_META["url_prefix"]
	lending_bp.url_prefix = prefix
	app.register_blueprint(lending_bp)
	_log_registered(prefix)


def register_ui_blueprint(app: Any) -> None:
	"""Register the UI views blueprint."""
	try:
		from .views import lending_ui_bp
	except (ImportError, AttributeError):
		return  # UI blueprint optional
	app.register_blueprint(lending_ui_bp)


def init_capability(app: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""
	Full capability initialisation — call from APG app factory.

	Registers API + UI blueprints and returns capability metadata.
	"""
	register_blueprint(app)
	register_ui_blueprint(app)
	_register_with_composition_engine(app)
	return get_capability_info()


def _register_with_composition_engine(app: Any) -> None:
	"""Register capability metadata with APG composition registry if available."""
	try:
		from capabilities.composition.capability_registry import CRCapability, CRCapabilityStatus
		cap = CRCapability(
			id=CAPABILITY_META["id"],
			name=CAPABILITY_META["name"],
			category=CAPABILITY_META["category"],
			status=CRCapabilityStatus.ACTIVE,
		)
		# Registry may expose register() or add() depending on APG version
		registry = getattr(app, "capability_registry", None)
		if registry and hasattr(registry, "register"):
			registry.register(cap)
	except Exception:  # noqa: BLE001
		pass  # Composition engine not present in standalone mode


def _log_registered(prefix: str) -> None:
	import logging
	logging.getLogger("apg.fintech.lending").info(
		"Digital Lending blueprint registered at %s", prefix
	)


# ---------------------------------------------------------------------------
# Composition engine query helpers
# ---------------------------------------------------------------------------

def get_capability_info() -> dict[str, Any]:
	"""Return full capability metadata."""
	return dict(CAPABILITY_META)


def get_permissions() -> list[str]:
	"""Return all permission keys this capability defines."""
	return list(CAPABILITY_META["permissions"])


def get_dependencies() -> list[str]:
	"""Return hard capability dependencies."""
	return list(CAPABILITY_META["dependencies"])


def health_check() -> dict[str, Any]:
	"""Return capability health status."""
	return {
		"capability": CAPABILITY_META["id"],
		"status": "healthy",
		"version": CAPABILITY_META["version"],
	}


# ---------------------------------------------------------------------------
# APG menu registration helper
# ---------------------------------------------------------------------------

MENU_ITEMS: list[dict[str, Any]] = [
	{
		"label": "Lending Dashboard",
		"icon": "fa-tachometer-alt",
		"url": "/lending/",
		"permission": "lending.reports.view",
		"order": 1,
	},
	{
		"label": "Applications",
		"icon": "fa-file-alt",
		"url": "/lending/applications",
		"permission": "lending.application.read",
		"order": 2,
	},
	{
		"label": "Loans",
		"icon": "fa-hand-holding-usd",
		"url": "/lending/loans",
		"permission": "lending.application.read",
		"order": 3,
	},
	{
		"label": "Collections",
		"icon": "fa-exclamation-triangle",
		"url": "/lending/collections",
		"permission": "lending.collections.manage",
		"order": 4,
	},
	{
		"label": "Portfolio",
		"icon": "fa-chart-bar",
		"url": "/lending/portfolio",
		"permission": "lending.reports.view",
		"order": 5,
	},
	{
		"label": "Loan Products",
		"icon": "fa-cogs",
		"url": "/lending/products",
		"permission": "lending.product.read",
		"order": 6,
	},
]


def get_menu_items() -> list[dict[str, Any]]:
	"""Return menu item definitions for APG navigation."""
	return list(MENU_ITEMS)
