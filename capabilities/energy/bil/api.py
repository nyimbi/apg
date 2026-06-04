"""REST API for APG Energy Billing & Tariffs."""

from __future__ import annotations

from typing import Any

try:
	from .service import EnergyBillingService
	from .views import (
		agent_workbench_model, bill_detail_model, billing_console_model,
		credit_manager_model, dashboard_model, dispute_console_model,
		payment_console_model, revenue_assurance_model, tariff_detail_model,
		tariff_manager_model,
	)
except ImportError:
	from service import EnergyBillingService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, bill_detail_model, billing_console_model,
		credit_manager_model, dashboard_model, dispute_console_model,
		payment_console_model, revenue_assurance_model, tariff_detail_model,
		tariff_manager_model,
	)

_SERVICE = EnergyBillingService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/contract"""
	return _ok(_SERVICE.describe(_tenant(payload)))


def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/dashboard"""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── tariffs ───────────────────────────────────────────────────────────────────

def list_tariffs(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/tariffs"""
	return _ok(tariff_manager_model(_SERVICE, _tenant(payload)))


def create_tariff(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/tariffs"""
	try:
		result = _SERVICE.create_tariff(
			tariff_id=payload["tariff_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			tariff_type=payload["tariff_type"],
			customer_class=payload["customer_class"],
			effective_date=payload["effective_date"],
			created_by=payload["created_by"],
			rate_blocks=payload.get("rate_blocks", []),
			description=payload.get("description", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_tariff(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/tariffs/<id>"""
	return _ok(tariff_detail_model(_SERVICE, _tenant(payload), payload["tariff_id"]))


def approve_tariff(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/tariffs/<id>/approve"""
	try:
		return _ok(_SERVICE.approve_tariff(
			payload["tariff_id"], _tenant(payload),
			approved_by=payload.get("approved_by", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def activate_tariff(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/tariffs/<id>/activate"""
	try:
		return _ok(_SERVICE.activate_tariff(payload["tariff_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── bills ─────────────────────────────────────────────────────────────────────

def list_bills(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/bills"""
	return _ok(billing_console_model(_SERVICE, _tenant(payload)))


def generate_bill(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/bills"""
	try:
		result = _SERVICE.generate_bill(
			bill_id=payload["bill_id"],
			tenant_id=_tenant(payload),
			customer_id=payload["customer_id"],
			meter_id=payload["meter_id"],
			tariff_id=payload["tariff_id"],
			billing_cycle=payload["billing_cycle"],
			period_start=payload["period_start"],
			period_end=payload["period_end"],
			consumption_kwh=float(payload["consumption_kwh"]),
			peak_demand_kw=float(payload.get("peak_demand_kw", 0)),
			charges=payload.get("charges", []),
			total_amount=float(payload["total_amount"]),
			currency=payload.get("currency", "KES"),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_bill(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/bills/<id>"""
	return _ok(bill_detail_model(_SERVICE, _tenant(payload), payload["bill_id"]))


def issue_bill(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/bills/<id>/issue"""
	try:
		return _ok(_SERVICE.issue_bill(
			payload["bill_id"], _tenant(payload),
			due_date=payload.get("due_date", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def write_off_bill(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/bills/<id>/write-off"""
	try:
		return _ok(_SERVICE.write_off_bill(
			payload["bill_id"], _tenant(payload),
			approved_by=payload.get("approved_by", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── payments ──────────────────────────────────────────────────────────────────

def list_payments(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/payments"""
	return _ok(payment_console_model(_SERVICE, _tenant(payload)))


def record_payment(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/payments"""
	try:
		result = _SERVICE.record_payment(
			payment_id=payload["payment_id"],
			tenant_id=_tenant(payload),
			bill_id=payload["bill_id"],
			customer_id=payload["customer_id"],
			payment_method=payload["payment_method"],
			amount=float(payload["amount"]),
			currency=payload.get("currency", "KES"),
			transaction_reference=payload.get("transaction_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def reconcile_payment(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/payments/<id>/reconcile"""
	try:
		return _ok(_SERVICE.reconcile_payment(payload["payment_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── credits ───────────────────────────────────────────────────────────────────

def list_credits(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/credits"""
	return _ok(credit_manager_model(_SERVICE, _tenant(payload)))


def issue_credit(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/credits"""
	try:
		result = _SERVICE.issue_credit(
			credit_id=payload["credit_id"],
			tenant_id=_tenant(payload),
			customer_id=payload["customer_id"],
			credit_type=payload["credit_type"],
			amount=float(payload["amount"]),
			currency=payload.get("currency", "KES"),
			expires_at=payload["expires_at"],
			approved_by=payload["approved_by"],
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── disputes ──────────────────────────────────────────────────────────────────

def list_disputes(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/disputes"""
	return _ok(dispute_console_model(_SERVICE, _tenant(payload)))


def open_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/disputes"""
	try:
		result = _SERVICE.open_dispute(
			dispute_id=payload["dispute_id"],
			tenant_id=_tenant(payload),
			bill_id=payload["bill_id"],
			customer_id=payload["customer_id"],
			reason=payload["reason"],
			evidence_reference=payload.get("evidence_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def resolve_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-bil/api/v1/disputes/<id>/resolve"""
	try:
		return _ok(_SERVICE.resolve_dispute(
			payload["dispute_id"], _tenant(payload),
			resolution=payload.get("resolution", ""),
			adjusted_amount=float(payload.get("adjusted_amount", 0)),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── revenue assurance ─────────────────────────────────────────────────────────

def list_revenue_flags(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/revenue-assurance"""
	return _ok(revenue_assurance_model(_SERVICE, _tenant(payload)))


def flag_revenue_issue(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/revenue-assurance"""
	try:
		result = _SERVICE.flag_revenue_issue(
			flag_id=payload["flag_id"],
			tenant_id=_tenant(payload),
			flag_type=payload["flag_type"],
			entity_id=payload["entity_id"],
			entity_type=payload.get("entity_type", "unknown"),
			estimated_revenue_impact=float(payload.get("estimated_revenue_impact", 0)),
			currency=payload.get("currency", "KES"),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-bil/api/v1/agents"""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-bil/api/v1/agents"""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "energy billing operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> EnergyBillingService:
	return _SERVICE
