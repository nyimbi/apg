"""Executable capability contract for APG Supplier Self-Service Portal."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = "proc_sup_portal"
CAPABILITY_NAME = "Supplier Self-Service Portal"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "proc"
CAPABILITY_DESCRIPTION = (
    "Supplier-facing portal: quote submission, invoice submission, delivery confirmation, "
    "dispute resolution, PO acknowledgement. Highest cost-reducer in procurement operations."
)

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["procurement_manager", "supplier_manager", "supplier"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"portal": {
		"self_registration_enabled": True,
		"po_auto_acknowledge_enabled": False,
		"invoice_ocr_enabled": True,
		"dispute_sla_days": 5,
	},
	"governance": {"require_tenant_context": True, "audit_events": True},
}

PROVIDES = [
	"supplier_registration", "po_acknowledgement", "quote_submission",
	"invoice_submission", "delivery_confirmation", "dispute_management",
	"supplier_performance_dashboard",
]
REQUIRES = ["auth", "audl", "ntfy", "scm_srm", "fin_arc"]
PUBLISHES = [
	"quote.submitted", "invoice.submitted", "delivery.confirmed",
	"dispute.raised", "po.acknowledged",
]
SUBSCRIBES = []

UI_ROUTES = [
	{"name": "dashboard", "path": "/supplier/dashboard", "component": "SupplierDashboard", "permission": "proc_sup_portal:view", "nav_group": "Overview"},
	{"name": "pos", "path": "/supplier/purchase-orders", "component": "SupplierPOList", "permission": "proc_sup_portal:view", "nav_group": "Orders"},
	{"name": "invoices", "path": "/supplier/invoices", "component": "SupplierInvoices", "permission": "proc_sup_portal:invoice", "nav_group": "Invoices"},
	{"name": "disputes", "path": "/supplier/disputes", "component": "SupplierDisputes", "permission": "proc_sup_portal:dispute", "nav_group": "Disputes"},
	{"name": "profile", "path": "/supplier/profile", "component": "SupplierProfile", "permission": "proc_sup_portal:manage", "nav_group": "Account"},
]

THEME = {
	"name": "proc_sup_portal_theme",
	"tokens": {
		"color.primary": "#1A3A5C", "color.accent": "#F59E0B",
		"surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF",
		"text.primary": "#111827", "border.radius": "8px", "density": "compact",
	},
}


def get_capability_contract() -> dict[str, Any]:
	return {"id": CAPABILITY_ID, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION,
		"provides": PROVIDES, "requires": REQUIRES, "publishes": PUBLISHES,
		"subscribes": SUBSCRIBES, "ui_routes": UI_ROUTES, "theme": THEME,
		"configuration": DEFAULT_CONFIGURATION}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	if not context.get("tenant_context_present"):
		return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": [{"type": "deny", "reason": "missing_tenant_context"}]}
	return {"decision": "allow", "matched_rules": [], "actions": []}
