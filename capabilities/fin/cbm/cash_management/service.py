"""Dependency-light Cash Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		CBM_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ACCOUNT_TYPES,
		SUPPORTED_CBM_AGENT_ROLES,
		SUPPORTED_CBM_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FLOW_TYPES,
		SUPPORTED_FORECAST_SCENARIOS,
		SUPPORTED_INVESTMENT_TYPES,
		evaluate_capability_rules,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		CBM_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ACCOUNT_TYPES,
		SUPPORTED_CBM_AGENT_ROLES,
		SUPPORTED_CBM_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FLOW_TYPES,
		SUPPORTED_FORECAST_SCENARIOS,
		SUPPORTED_INVESTMENT_TYPES,
		evaluate_capability_rules,
	)


class CashManagementService:
	"""In-memory executable service for the CBM lifecycle packet."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.banks: dict[str, dict[str, Any]] = {}
		self.cash_accounts: dict[str, dict[str, Any]] = {}
		self.cash_positions: dict[str, dict[str, Any]] = {}
		self.cash_flows: dict[str, dict[str, Any]] = {}
		self.cash_forecasts: dict[str, dict[str, Any]] = {}
		self.liquidity_reviews: dict[str, dict[str, Any]] = {}
		self.reconciliations: dict[str, dict[str, Any]] = {}
		self.investments: dict[str, dict[str, Any]] = {}
		self.payment_runs: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": CBM_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def create_bank(self, bank_id: str, tenant_id: str, code: str, name: str, connectivity_status: str = "manual") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_bank",
			"operation_type": "write",
			"policy_attached": True,
			"bank_code_present": bool(code),
			"bank_name_present": bool(name),
		})
		record = {
			"id": self._record_id("bank", bank_id),
			"type": "bank_relationship",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"connectivity_status": connectivity_status,
			"status": "active",
			"created_at": self._now(),
		}
		self.banks[record["id"]] = record
		self._emit(tenant, "bank_created", record)
		return deepcopy(record)

	def create_cash_account(
		self,
		account_id: str,
		tenant_id: str,
		bank_id: str,
		account_number: str,
		name: str,
		account_type: str,
		currency: str = "USD",
		minimum_buffer: float = 0,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		bank = self.banks.get(bank_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_cash_account",
			"operation_type": "write",
			"policy_attached": True,
			"bank_present": bool(bank and bank["tenant_id"] == tenant),
			"account_number_present": bool(account_number),
			"account_name_present": bool(name),
			"account_type_supported": account_type in SUPPORTED_ACCOUNT_TYPES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		record = {
			"id": self._record_id("cashacct", account_id),
			"type": "cash_account",
			"tenant_id": tenant,
			"bank_id": bank_id,
			"account_number": account_number,
			"name": name,
			"account_type": account_type,
			"currency": currency,
			"minimum_buffer": Decimal(str(minimum_buffer)),
			"status": "active",
			"created_at": self._now(),
		}
		self.cash_accounts[record["id"]] = record
		self._emit(tenant, "cash_account_created", record)
		return deepcopy(record)

	def record_cash_position(
		self,
		position_id: str,
		tenant_id: str,
		account_id: str,
		as_of_date: str,
		available_balance: float,
		ledger_balance: float | None = None,
		liquidity_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(account_id)
		below_buffer = bool(account and Decimal(str(available_balance)) < account["minimum_buffer"])
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "record_cash_position",
			"operation_type": "write",
			"policy_attached": True,
			"account_present": bool(account and account["tenant_id"] == tenant),
			"as_of_date_present": bool(as_of_date),
			"available_balance_present": available_balance is not None,
			"below_minimum_buffer": below_buffer,
			"liquidity_review_recorded": bool(liquidity_reviewed_by),
		})
		record = {
			"id": self._record_id("position", position_id),
			"type": "cash_position",
			"tenant_id": tenant,
			"account_id": account_id,
			"as_of_date": as_of_date,
			"available_balance": Decimal(str(available_balance)),
			"ledger_balance": Decimal(str(ledger_balance if ledger_balance is not None else available_balance)),
			"liquidity_reviewed_by": liquidity_reviewed_by,
			"status": "reviewed" if liquidity_reviewed_by else "recorded",
			"created_at": self._now(),
		}
		self.cash_positions[record["id"]] = record
		self._emit(tenant, "cash_position_recorded", record)
		return deepcopy(record)

	def record_cash_flow(self, flow_id: str, tenant_id: str, account_id: str, flow_type: str, amount: float, category: str, expected_date: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(account_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "record_cash_flow",
			"operation_type": "write",
			"policy_attached": True,
			"account_present": bool(account and account["tenant_id"] == tenant),
			"flow_type_supported": flow_type in SUPPORTED_FLOW_TYPES,
			"amount": amount,
			"category_present": bool(category),
			"expected_date_present": bool(expected_date),
		})
		record = {
			"id": self._record_id("flow", flow_id),
			"type": "cash_flow",
			"tenant_id": tenant,
			"account_id": account_id,
			"flow_type": flow_type,
			"amount": Decimal(str(amount)),
			"category": category,
			"expected_date": expected_date,
			"status": "recorded",
			"created_at": self._now(),
		}
		self.cash_flows[record["id"]] = record
		self._emit(tenant, "cash_flow_recorded", record)
		return deepcopy(record)

	def create_cash_forecast(self, forecast_id: str, tenant_id: str, horizon_days: int, scenario: str, confidence_score: float, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_cash_forecast",
			"operation_type": "write",
			"policy_attached": True,
			"horizon_days": horizon_days,
			"scenario_supported": scenario in SUPPORTED_FORECAST_SCENARIOS,
			"confidence_score": confidence_score,
			"forecast_review_recorded": bool(reviewed_by),
		})
		flows = [flow for flow in self.cash_flows.values() if flow["tenant_id"] == tenant]
		net_amount = sum((flow["amount"] if flow["flow_type"] == "inflow" else -flow["amount"]) for flow in flows)
		record = {
			"id": self._record_id("forecast", forecast_id),
			"type": "cash_forecast",
			"tenant_id": tenant,
			"horizon_days": horizon_days,
			"scenario": scenario,
			"confidence_score": confidence_score,
			"reviewed_by": reviewed_by,
			"projected_net_cash": net_amount,
			"source_flow_count": len(flows),
			"status": "reviewed" if reviewed_by else "forecasted",
			"created_at": self._now(),
		}
		self.cash_forecasts[record["id"]] = record
		self._emit(tenant, "cash_forecast_created", record)
		return deepcopy(record)

	def record_liquidity_review(self, review_id: str, tenant_id: str, position_id: str, reviewer: str, decision: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if position_id not in self.cash_positions or not reviewer:
			raise PermissionError("liquidity_review_required")
		record = {
			"id": self._record_id("liquidity", review_id),
			"type": "liquidity_review",
			"tenant_id": tenant,
			"position_id": position_id,
			"reviewer": reviewer,
			"decision": decision,
			"status": "reviewed",
			"created_at": self._now(),
		}
		self.liquidity_reviews[record["id"]] = record
		self._emit(tenant, "liquidity_review_recorded", record)
		return deepcopy(record)

	def record_bank_reconciliation(
		self,
		reconciliation_id: str,
		tenant_id: str,
		account_id: str,
		bank_statement_balance: float,
		ledger_balance: float,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		variance = Decimal(str(bank_statement_balance)) - Decimal(str(ledger_balance))
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "record_bank_reconciliation",
			"operation_type": "write",
			"policy_attached": True,
			"bank_statement_present": bank_statement_balance is not None,
			"ledger_balance_present": ledger_balance is not None,
			"variance": float(variance),
			"reconciliation_review_recorded": bool(reviewed_by),
		})
		record = {
			"id": self._record_id("recon", reconciliation_id),
			"type": "bank_reconciliation",
			"tenant_id": tenant,
			"account_id": account_id,
			"bank_statement_balance": Decimal(str(bank_statement_balance)),
			"ledger_balance": Decimal(str(ledger_balance)),
			"variance": variance,
			"reviewed_by": reviewed_by,
			"status": "matched" if variance == 0 else "reviewed",
			"created_at": self._now(),
		}
		self.reconciliations[record["id"]] = record
		self._emit(tenant, "bank_reconciliation_recorded", record)
		return deepcopy(record)

	def create_treasury_investment(
		self,
		investment_id: str,
		tenant_id: str,
		investment_type: str,
		counterparty: str,
		principal: float,
		maturity_date: str,
		yield_rate: float,
		approved_by: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "create_treasury_investment",
			"operation_type": "write",
			"policy_attached": True,
			"investment_type_supported": investment_type in SUPPORTED_INVESTMENT_TYPES,
			"counterparty_present": bool(counterparty),
			"maturity_date_present": bool(maturity_date),
			"approval_recorded": bool(approved_by),
		})
		record = {
			"id": self._record_id("investment", investment_id),
			"type": "treasury_investment",
			"tenant_id": tenant,
			"investment_type": investment_type,
			"counterparty": counterparty,
			"principal": Decimal(str(principal)),
			"maturity_date": maturity_date,
			"yield_rate": yield_rate,
			"approved_by": approved_by,
			"status": "approved",
			"created_at": self._now(),
		}
		self.investments[record["id"]] = record
		self._emit(tenant, "treasury_investment_created", record)
		return deepcopy(record)

	def validate_payment_run(self, payment_run_id: str, tenant_id: str, funding_account_id: str, payment_total: float, approved_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(funding_account_id)
		positions = [position for position in self.cash_positions.values() if position["tenant_id"] == tenant and position["account_id"] == funding_account_id]
		current_position = positions[-1] if positions else None
		projected_deficit = bool(current_position and current_position["available_balance"] - Decimal(str(payment_total)) < 0)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "validate_payment_run",
			"operation_type": "write",
			"policy_attached": True,
			"funding_account_present": bool(account and account["tenant_id"] == tenant),
			"cash_position_present": bool(current_position),
			"projected_deficit": projected_deficit,
			"approval_recorded": bool(approved_by),
		})
		record = {
			"id": self._record_id("payrun", payment_run_id),
			"type": "payment_run",
			"tenant_id": tenant,
			"funding_account_id": funding_account_id,
			"payment_total": Decimal(str(payment_total)),
			"approved_by": approved_by,
			"status": "funded",
			"created_at": self._now(),
		}
		self.payment_runs[record["id"]] = record
		self._emit(tenant, "payment_run_validated", record)
		return deepcopy(record)

	def register_cbm_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "register_cbm_agent",
			"operation_type": "write",
			"policy_attached": True,
			"agent_runtime_supported": runtime in SUPPORTED_CBM_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_CBM_AGENT_ROLES,
		})
		record = {
			"id": self._record_id("agent"),
			"type": "cbm_agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "cbm_agent_registered", record)
		return deepcopy(record)

	def validate_agent_cbm_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if agent_id not in self.agents:
			raise PermissionError("cbm_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "agent_cbm_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "cbm_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": CBM_EVENT_STREAM}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"bank_count": len([record for record in self.banks.values() if record["tenant_id"] == tenant]),
			"cash_account_count": len([record for record in self.cash_accounts.values() if record["tenant_id"] == tenant]),
			"cash_position_count": len([record for record in self.cash_positions.values() if record["tenant_id"] == tenant]),
			"cash_flow_count": len([record for record in self.cash_flows.values() if record["tenant_id"] == tenant]),
			"forecast_count": len([record for record in self.cash_forecasts.values() if record["tenant_id"] == tenant]),
			"reconciliation_count": len([record for record in self.reconciliations.values() if record["tenant_id"] == tenant]),
			"investment_count": len([record for record in self.investments.values() if record["tenant_id"] == tenant]),
			"payment_run_count": len([record for record in self.payment_runs.values() if record["tenant_id"] == tenant]),
			"cbm_agent_count": len([record for record in self.agents.values() if record["tenant_id"] == tenant]),
			"audit_event_count": len([event for event in self._audit_events if event["tenant_id"] == tenant]),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		store = getattr(self, collection)
		return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]


CBMService = CashManagementService
