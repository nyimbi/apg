"""Screen-model helpers for the Fintech Gateway capability."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import FintechGatewayService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import FintechGatewayService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/fintech-gateway/dashboard", "icon": "layout-dashboard"},
	{"name": "Merchants", "route": "/fintech-gateway/merchants", "icon": "store"},
	{"name": "Providers", "route": "/fintech-gateway/providers", "icon": "plug-zap"},
	{"name": "Payment Methods", "route": "/fintech-gateway/payment-methods", "icon": "credit-card"},
	{"name": "Payments", "route": "/fintech-gateway/payments", "icon": "badge-dollar-sign"},
	{"name": "Routing", "route": "/fintech-gateway/routing", "icon": "route"},
	{"name": "Risk", "route": "/fintech-gateway/risk", "icon": "shield-alert"},
	{"name": "Webhooks", "route": "/fintech-gateway/webhooks", "icon": "inbox"},
	{"name": "Settlements", "route": "/fintech-gateway/settlements", "icon": "landmark"},
	{"name": "Disputes", "route": "/fintech-gateway/disputes", "icon": "message-square-warning"},
	{"name": "Agents", "route": "/fintech-gateway/agents", "icon": "bot"},
	{"name": "Settings", "route": "/fintech-gateway/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"risk_reviews": len([record for record in service.risk_reviews.values() if record["tenant_id"] == tenant_id and record["status"] != "reviewed"]),
		"open_disputes": len([record for record in service.disputes.values() if record["tenant_id"] == tenant_id and record["status"] != "resolved"]),
		"settlement_variances": len([record for record in service.settlements.values() if record["tenant_id"] == tenant_id and record["variance"] != 0]),
	}
	return model


def merchant_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("merchants", tenant_id)
	model["records"] = service.list_records("merchants", tenant_id)
	model["columns"] = ["merchant_code", "legal_name", "country", "risk_level", "reviewed_by", "status"]
	return model


def provider_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("providers", tenant_id)
	model["records"] = service.list_records("provider_connections", tenant_id)
	model["columns"] = ["provider", "provider_type", "credential_reference", "priority", "status"]
	return model


def payment_method_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("payment_methods", tenant_id)
	model["records"] = service.list_records("payment_methods", tenant_id)
	model["columns"] = ["merchant_id", "customer_reference", "method_type", "token_reference", "status"]
	return model


def payment_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("payments", tenant_id)
	model["records"] = service.list_records("payment_intents", tenant_id)
	model["columns"] = ["merchant_id", "amount", "currency", "risk_level", "provider_connection_id", "status"]
	return model


def routing_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("routing", tenant_id)
	model["providers"] = service.list_records("provider_connections", tenant_id)
	model["payments"] = service.list_records("payment_intents", tenant_id)
	model["columns"] = ["provider", "provider_type", "priority", "status"]
	return model


def risk_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("risk", tenant_id)
	model["records"] = service.list_records("risk_reviews", tenant_id)
	model["columns"] = ["payment_intent_id", "risk_level", "risk_score", "reviewed_by", "status"]
	return model


def webhook_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("webhooks", tenant_id)
	model["records"] = service.list_records("webhooks", tenant_id)
	model["columns"] = ["provider_connection_id", "event_id", "idempotency_key", "event_type", "status"]
	return model


def settlement_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("settlements", tenant_id)
	model["records"] = service.list_records("settlements", tenant_id)
	model["columns"] = ["provider_connection_id", "settlement_reference", "amount", "expected_amount", "variance", "reviewed_by", "status"]
	return model


def dispute_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("disputes", tenant_id)
	model["records"] = service.list_records("disputes", tenant_id)
	model["columns"] = ["payment_intent_id", "reason", "owner", "resolution", "reviewed_by", "status"]
	return model


def agent_workbench_model(service: FintechGatewayService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_merchant", "review_route", "review_risk", "review_settlement", "review_dispute"]
	return model
