"""View models for Energy Billing & Tariffs screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import EnergyBillingService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import EnergyBillingService  # type: ignore


def dashboard_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Energy Billing & Tariffs",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def tariff_manager_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"tariffs": svc.list_tariffs(tenant_id),
		"active_tariffs": [t for t in svc.list_tariffs(tenant_id) if t["status"] == "active"],
		"supported_tariff_types": contract["configuration"]["tariffs"]["supported_types"],
		"supported_customer_classes": contract["configuration"]["tariffs"]["supported_customer_classes"],
	}


def tariff_detail_model(svc: EnergyBillingService, tenant_id: str, tariff_id: str) -> dict[str, Any]:
	tariff = svc.tariffs.get((tenant_id, tariff_id))
	if not tariff:
		return {"error": "tariff_not_found", "tariff_id": tariff_id}
	bills_using_tariff = [b for b in svc.list_bills(tenant_id) if b["tariff_id"] == tariff_id]
	return {
		"tenant_id": tenant_id,
		"tariff": tariff.to_dict(),
		"bills_count": len(bills_using_tariff),
		"total_billed": sum(b["total_amount"] for b in bills_using_tariff),
	}


def billing_console_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"bills": svc.list_bills(tenant_id),
		"overdue_bills": svc.list_bills(tenant_id, status="overdue"),
		"supported_billing_cycles": contract["configuration"]["billing"]["supported_cycles"],
		"supported_statuses": contract["configuration"]["billing"]["supported_statuses"],
	}


def bill_detail_model(svc: EnergyBillingService, tenant_id: str, bill_id: str) -> dict[str, Any]:
	bill = svc.bills.get((tenant_id, bill_id))
	if not bill:
		return {"error": "bill_not_found", "bill_id": bill_id}
	payments = svc.list_payments(tenant_id, bill_id=bill_id)
	disputes = [d for d in svc.list_disputes(tenant_id) if d["bill_id"] == bill_id]
	total_paid = sum(p["amount"] for p in payments)
	return {
		"tenant_id": tenant_id,
		"bill": bill.to_dict(),
		"payments": payments,
		"disputes": disputes,
		"total_paid": total_paid,
		"balance_due": round(bill.total_amount - total_paid, 4),
	}


def payment_console_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	payments = svc.list_payments(tenant_id)
	unreconciled = [p for p in payments if not p["reconciled"]]
	return {
		"tenant_id": tenant_id,
		"payments": payments,
		"unreconciled_payments": unreconciled,
		"supported_payment_methods": contract["configuration"]["payments"]["supported_methods"],
	}


def credit_manager_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"credits": svc.list_credits(tenant_id),
		"active_credits": [c for c in svc.list_credits(tenant_id) if c["status"] == "active"],
		"supported_credit_types": contract["configuration"]["credits"]["supported_types"],
	}


def dispute_console_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"disputes": svc.list_disputes(tenant_id),
		"open_disputes": svc.list_disputes(tenant_id, status="open"),
	}


def revenue_assurance_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	flags = svc.list_revenue_flags(tenant_id)
	open_flags = svc.list_revenue_flags(tenant_id, status="open")
	total_at_risk = sum(f["estimated_revenue_impact"] for f in open_flags)
	return {
		"tenant_id": tenant_id,
		"flags": flags,
		"open_flags": open_flags,
		"total_revenue_at_risk": total_at_risk,
		"supported_types": contract["configuration"]["revenue_assurance"]["supported_types"],
	}


def agent_workbench_model(svc: EnergyBillingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
