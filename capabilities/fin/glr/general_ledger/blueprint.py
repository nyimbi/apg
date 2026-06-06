"""
General Ledger — Flask Blueprint registration.

Registers the REST API blueprint and exposes APG composition metadata.
Uses plain Flask Blueprint (no Flask-AppBuilder dependency).

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Flask

try:
	from .context import get_current_user_id, get_tenant_id_from_request
	from .service import GeneralLedgerService
except ImportError:
	from context import get_current_user_id, get_tenant_id_from_request  # type: ignore
	from service import GeneralLedgerService  # type: ignore


def _run_async_initialization(factory):
	"""Run an async initialisation coroutine/factory synchronously."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				future = pool.submit(asyncio.run, factory())
				return future.result()
		return loop.run_until_complete(factory())
	except RuntimeError:
		return asyncio.run(factory())

# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------

SUBCAPABILITY_META: dict[str, Any] = {
	"name": "General Ledger",
	"code": "GLR",
	"version": "2.0.0",
	"description": (
		"World-class double-entry general ledger with full IFRS/GAAP support, "
		"multi-currency revaluation, period lifecycle management, consolidation, "
		"XBRL tagging, and AI-assisted reconciliation."
	),
	"url_prefix": "/api/glr",
	"menu_category": "Financials",
	"menu_icon": "fa-book",
	"capabilities": [
		"chart_of_accounts",
		"journal_entry_lifecycle",
		"period_management",
		"financial_statements",
		"budget_vs_actual",
		"reconciliation",
		"consolidation",
		"year_end_close",
		"multi_currency",
		"xbrl_reporting",
	],
	"integrations": [
		"fin.apy",  # Accounts Payable feeds AP journals
		"fin.arc",  # Accounts Receivable feeds AR journals
		"fin.cbm",  # Cash & Bank feeds bank reconciliation
		"fin.fam",  # Fixed Assets feeds depreciation journals
		"fin.bfc",  # Budgeting provides budget data for BvA
		"fin.txm",  # Tax module provides tax codes
		"fin.fco",  # Financial Consolidation uses GL data
	],
}


def init_subcapability(appbuilder_or_app: Any, tenant_data: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Register the GLR blueprint with the Flask application.

	Accepts either a Flask app or a Flask-AppBuilder instance.
	Returns the initialization result dict.
	"""
	from .api import bp as glr_bp

	# Extract the Flask app from either Flask or Flask-AppBuilder
	if hasattr(appbuilder_or_app, "get_app"):
		# Flask-AppBuilder wrapper
		app: Flask = appbuilder_or_app.get_app
	elif hasattr(appbuilder_or_app, "app"):
		app = appbuilder_or_app.app
	else:
		# Plain Flask app
		app = appbuilder_or_app

	# Avoid duplicate registration
	if "glr_general_ledger" not in app.blueprints:
		app.register_blueprint(glr_bp)

	# Initialise a tenant-scoped service instance when tenant_data is provided.
	if tenant_data:
		tenant_id = get_tenant_id_from_request(tenant_data)
		user_id = get_current_user_id(tenant_data)
		gl_service = GeneralLedgerService(tenant_id, user_id)
		_run_async_initialization(lambda: gl_service.setup_tenant(tenant_data))

	return {
		"capability": "General Ledger",
		"code": "GLR",
		"blueprint": "glr_general_ledger",
		"url_prefix": "/api/glr",
		"status": "registered",
	}


def get_capability_info() -> dict[str, Any]:
	"""Return capability metadata for the APG composition engine."""
	return SUBCAPABILITY_META


def register_with_composition_engine(registry: Any | None = None) -> dict[str, Any]:
	"""Register this capability with the APG capability registry.

	Accepts the registry service if available; otherwise returns metadata only.
	"""
	meta = get_capability_info()
	if registry is not None and hasattr(registry, "register"):
		registry.register(
			capability_id="fin.glr.general_ledger",
			name=meta["name"],
			version=meta["version"],
			url_prefix=meta["url_prefix"],
			metadata=meta,
		)
	return {
		"registered": True,
		"capability_id": "fin.glr.general_ledger",
		**meta,
	}
