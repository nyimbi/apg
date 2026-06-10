"""Dependency-light Cash Management lifecycle service — expanded implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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
except ImportError:  # pragma: no cover
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


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _today() -> str:
	return date.today().isoformat()


class CashManagementService:
	"""
	In-memory executable service for the CBM lifecycle packet.

	Expanded with: bank_account_balance, import_bank_statement,
	auto_reconcile_statement, manual_match, reconciliation_report,
	cash_position_report, liquidity_forecast, fx_position,
	cash_pooling_sweep, intercompany_settlement, bank_covenant_compliance,
	mobile_money_reconciliation.
	"""

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
		# New stores
		self._bank_statements: dict[str, dict[str, Any]] = {}
		self._statement_matches: dict[str, dict[str, Any]] = {}
		self._fx_positions: dict[str, dict[str, Any]] = {}
		self._cash_pools: dict[str, dict[str, Any]] = {}
		self._pool_sweeps: list[dict[str, Any]] = []
		self._intercompany_settlements: list[dict[str, Any]] = []
		self._covenant_checks: list[dict[str, Any]] = []
		self._mobile_money_reconciliations: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		decision = result.get("decision")
		effects = result.get("effects") or result.get("actions") or []
		reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
		# Hard-block on deny
		if decision == "deny":
			raise PermissionError(",".join(reasons) or "operation_denied")
		# require_review blocks when the review flag was not satisfied in context
		# (the rule fires because the flag is False — the caller must supply a reviewer)
		if decision == "require_review":
			raise PermissionError(",".join(reasons) or "review_required")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": CBM_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": _now(),
		})

	def _get_account(self, account_id: str, tenant_id: str) -> dict[str, Any]:
		account = self.cash_accounts.get(account_id)
		if not account or account["tenant_id"] != tenant_id:
			raise KeyError(f"cash_account_not_found:{account_id}")
		return account

	# ------------------------------------------------------------------
	# bank_account_balance
	# ------------------------------------------------------------------

	def bank_account_balance(
		self,
		account_id: str,
		as_of_date: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Return the latest recorded balance for a cash account as of a date.

		Scans cash_positions for the most recent entry on or before as_of_date.
		Returns available_balance, ledger_balance, currency, and as_of_date.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account(account_id, tenant)
		# Find the latest position on or before as_of_date
		positions = [
			p for p in self.cash_positions.values()
			if p["tenant_id"] == tenant and p["account_id"] == account_id
			and p["as_of_date"] <= as_of_date
		]
		if not positions:
			return {
				"account_id": account_id,
				"account_name": account["name"],
				"currency": account["currency"],
				"as_of_date": as_of_date,
				"available_balance": Decimal("0"),
				"ledger_balance": Decimal("0"),
				"status": "no_position_found",
			}
		latest = max(positions, key=lambda p: p["as_of_date"])
		return {
			"account_id": account_id,
			"account_name": account["name"],
			"currency": account["currency"],
			"as_of_date": latest["as_of_date"],
			"available_balance": latest["available_balance"],
			"ledger_balance": latest["ledger_balance"],
			"position_id": latest["id"],
			"status": latest["status"],
		}

	def import_bank_statement(
		self,
		account_id: str,
		statement_date: str,
		transactions: list[dict[str, Any]],
		tenant_id: str | None = None,
		statement_id: str | None = None,
		imported_by: str = "system",
	) -> dict[str, Any]:
		"""
		Import a bank statement with a list of transaction records.

		Each transaction dict should have: date, description, amount, type (credit/debit), reference.
		Returns the statement record with transaction count and totals.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account(account_id, tenant)
		if not statement_date:
			raise ValueError("statement_date_required")
		if not transactions:
			raise ValueError("transactions_required")
		credits = sum(Decimal(str(t.get("amount", 0))) for t in transactions if t.get("type") == "credit")
		debits = sum(Decimal(str(t.get("amount", 0))) for t in transactions if t.get("type") == "debit")
		stmt_id = self._record_id("stmt", statement_id)
		record = {
			"id": stmt_id,
			"type": "bank_statement",
			"tenant_id": tenant,
			"account_id": account_id,
			"account_name": account["name"],
			"statement_date": statement_date,
			"transaction_count": len(transactions),
			"total_credits": credits,
			"total_debits": debits,
			"net_movement": credits - debits,
			"transactions": list(transactions),
			"imported_by": imported_by,
			"status": "imported",
			"created_at": _now(),
		}
		self._bank_statements[stmt_id] = record
		self._emit(tenant, "bank_statement_imported", record)
		return deepcopy(record)

	def auto_reconcile_statement(
		self,
		statement_id: str,
		tenant_id: str | None = None,
		tolerance: float = 0.01,
	) -> dict[str, Any]:
		"""
		Automatically reconcile a bank statement against recorded cash flows.

		Matches statement transactions to cash_flows by amount and reference.
		Returns reconciliation summary with matched, unmatched, and variance counts.
		"""
		tenant = self._tenant(tenant_id)
		statement = self._bank_statements.get(statement_id)
		if not statement or statement["tenant_id"] != tenant:
			raise KeyError(f"bank_statement_not_found:{statement_id}")
		account_flows = [
			f for f in self.cash_flows.values()
			if f["tenant_id"] == tenant and f["account_id"] == statement["account_id"]
		]
		matched: list[dict[str, Any]] = []
		unmatched_stmt: list[dict[str, Any]] = []
		flow_lookup = {str(f["amount"]): f for f in account_flows}
		tol = Decimal(str(tolerance))
		for txn in statement.get("transactions", []):
			txn_amount = Decimal(str(txn.get("amount", 0)))
			flow = flow_lookup.get(str(txn_amount))
			if flow and abs(txn_amount - Decimal(str(flow["amount"]))) <= tol:
				matched.append({"statement_txn": txn, "cash_flow_id": flow["id"]})
			else:
				unmatched_stmt.append(txn)
		recon_id = self._record_id("recon")
		total_variance = sum(Decimal(str(t.get("amount", 0))) for t in unmatched_stmt)
		record = {
			"id": recon_id,
			"type": "auto_reconciliation",
			"tenant_id": tenant,
			"statement_id": statement_id,
			"account_id": statement["account_id"],
			"matched_count": len(matched),
			"unmatched_count": len(unmatched_stmt),
			"matched_items": matched,
			"unmatched_items": unmatched_stmt,
			"total_variance": total_variance,
			"reconciled": len(unmatched_stmt) == 0,
			"status": "matched" if len(unmatched_stmt) == 0 else "partial",
			"created_at": _now(),
		}
		self.reconciliations[recon_id] = record
		self._emit(tenant, "statement_auto_reconciled", record)
		return deepcopy(record)

	def manual_match(
		self,
		gl_entry_id: str,
		bank_transaction_id: str,
		tenant_id: str | None = None,
		matched_by: str = "system",
		notes: str = "",
	) -> dict[str, Any]:
		"""
		Manually match a GL entry to a bank transaction.

		Stores a match record linking gl_entry_id and bank_transaction_id.
		"""
		tenant = self._tenant(tenant_id)
		if not gl_entry_id:
			raise ValueError("gl_entry_id_required")
		if not bank_transaction_id:
			raise ValueError("bank_transaction_id_required")
		match_id = self._record_id("match")
		record = {
			"id": match_id,
			"type": "manual_match",
			"tenant_id": tenant,
			"gl_entry_id": gl_entry_id,
			"bank_transaction_id": bank_transaction_id,
			"matched_by": matched_by,
			"notes": notes,
			"status": "matched",
			"created_at": _now(),
		}
		self._statement_matches[match_id] = record
		self._emit(tenant, "manual_match_recorded", record)
		return deepcopy(record)

	def reconciliation_report(
		self,
		account_id: str,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Generate a reconciliation report for an account over a period.

		Period format: 'YYYY-MM'.  Returns counts, variances, and match rates.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account(account_id, tenant)
		period_recons = [
			r for r in self.reconciliations.values()
			if r["tenant_id"] == tenant and r.get("account_id") == account_id
			and r["created_at"][:7] == period
		]
		total_matched = sum(r.get("matched_count", 0) for r in period_recons)
		total_unmatched = sum(r.get("unmatched_count", 0) for r in period_recons)
		total_transactions = total_matched + total_unmatched
		match_rate = round(total_matched / total_transactions, 4) if total_transactions > 0 else 0.0
		total_variance = sum(r.get("total_variance", Decimal("0")) for r in period_recons)
		return {
			"account_id": account_id,
			"account_name": account["name"],
			"tenant_id": tenant,
			"period": period,
			"reconciliation_count": len(period_recons),
			"total_matched": total_matched,
			"total_unmatched": total_unmatched,
			"match_rate": match_rate,
			"total_variance": total_variance,
			"fully_reconciled_count": sum(1 for r in period_recons if r.get("reconciled")),
			"generated_at": _now(),
		}

	def cash_position_report(
		self,
		as_of_date: str,
		currencies: list[str],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Generate a cash position report across all accounts for specified currencies.

		Returns per-account and per-currency position summaries.
		"""
		tenant = self._tenant(tenant_id)
		accounts = [a for a in self.cash_accounts.values() if a["tenant_id"] == tenant]
		if currencies:
			accounts = [a for a in accounts if a["currency"] in currencies]
		positions_by_currency: dict[str, Decimal] = {}
		account_positions: list[dict[str, Any]] = []
		for account in accounts:
			bal = self.bank_account_balance(account["id"], as_of_date, tenant)
			ccy = account["currency"]
			positions_by_currency[ccy] = positions_by_currency.get(ccy, Decimal("0")) + Decimal(str(bal["available_balance"]))
			account_positions.append({
				"account_id": account["id"],
				"account_name": account["name"],
				"currency": ccy,
				"available_balance": bal["available_balance"],
				"as_of_date": bal.get("as_of_date", as_of_date),
			})
		return {
			"tenant_id": tenant,
			"as_of_date": as_of_date,
			"currencies": currencies,
			"account_count": len(account_positions),
			"positions_by_currency": {k: str(v) for k, v in positions_by_currency.items()},
			"account_positions": account_positions,
			"generated_at": _now(),
		}

	def liquidity_forecast(
		self,
		days: int = 90,
		tenant_id: str | None = None,
		scenario: str = "base",
	) -> dict[str, Any]:
		"""
		Generate a liquidity forecast for the next N days.

		Projects net cash flows from recorded cash_flows with expected_date
		within the forecast horizon.  Returns daily and cumulative projections.
		"""
		tenant = self._tenant(tenant_id)
		if days < 1:
			raise ValueError("forecast_days_must_be_positive")
		today = _today()
		horizon_end = (date.today() + timedelta(days=days)).isoformat()
		# Aggregate flows within horizon
		inflows = sum(
			Decimal(str(f["amount"])) for f in self.cash_flows.values()
			if f["tenant_id"] == tenant and f["flow_type"] == "inflow"
			and today <= f.get("expected_date", today) <= horizon_end
		)
		outflows = sum(
			Decimal(str(f["amount"])) for f in self.cash_flows.values()
			if f["tenant_id"] == tenant and f["flow_type"] == "outflow"
			and today <= f.get("expected_date", today) <= horizon_end
		)
		# Starting balance: sum of latest positions
		starting_balance = Decimal("0")
		for account in self.cash_accounts.values():
			if account["tenant_id"] != tenant:
				continue
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == account["id"]]
			if positions:
				latest = max(positions, key=lambda p: p["as_of_date"])
				starting_balance += Decimal(str(latest["available_balance"]))
		net_change = inflows - outflows
		ending_balance = starting_balance + net_change
		# Simple weekly projection buckets
		weekly_buckets: list[dict[str, Any]] = []
		for week in range(0, min(days, 90), 7):
			bucket_start = (date.today() + timedelta(days=week)).isoformat()
			bucket_end = (date.today() + timedelta(days=min(week + 6, days))).isoformat()
			weekly_buckets.append({
				"week": week // 7 + 1,
				"period_start": bucket_start,
				"period_end": bucket_end,
				"projected_net": str(round(net_change / (days // 7 + 1), 2)),
			})
		return {
			"tenant_id": tenant,
			"scenario": scenario,
			"forecast_days": days,
			"forecast_from": today,
			"forecast_to": horizon_end,
			"starting_balance": str(starting_balance),
			"projected_inflows": str(inflows),
			"projected_outflows": str(outflows),
			"projected_net_change": str(net_change),
			"projected_ending_balance": str(ending_balance),
			"weekly_buckets": weekly_buckets,
			"generated_at": _now(),
		}

	def fx_position(
		self,
		as_of_date: str,
		tenant_id: str | None = None,
		base_currency: str = "USD",
	) -> dict[str, Any]:
		"""
		Calculate the FX position (net exposure by currency) as of a date.

		Returns long/short exposures per currency relative to base_currency.
		"""
		tenant = self._tenant(tenant_id)
		accounts = [a for a in self.cash_accounts.values() if a["tenant_id"] == tenant]
		exposure_by_currency: dict[str, Decimal] = {}
		for account in accounts:
			ccy = account["currency"]
			if ccy == base_currency:
				continue
			bal = self.bank_account_balance(account["id"], as_of_date, tenant)
			exposure_by_currency[ccy] = exposure_by_currency.get(ccy, Decimal("0")) + Decimal(str(bal["available_balance"]))
		positions = []
		for ccy, exposure in exposure_by_currency.items():
			positions.append({
				"currency": ccy,
				"exposure": str(exposure),
				"direction": "long" if exposure > 0 else "short",
				"base_currency": base_currency,
			})
		fx_id = self._record_id("fx")
		record = {
			"id": fx_id,
			"type": "fx_position",
			"tenant_id": tenant,
			"as_of_date": as_of_date,
			"base_currency": base_currency,
			"currency_count": len(positions),
			"positions": positions,
			"status": "calculated",
			"created_at": _now(),
		}
		self._fx_positions[fx_id] = record
		self._emit(tenant, "fx_position_calculated", record)
		return deepcopy(record)

	def cash_pooling_sweep(
		self,
		pool_id: str,
		value_date: str,
		tenant_id: str | None = None,
		sweep_type: str = "zero_balance",
		approved_by: str = "treasury",
	) -> dict[str, Any]:
		"""
		Execute a cash pooling sweep for a pool on a given value date.

		sweep_type: 'zero_balance' (sweep all to header account) or 'target_balance'.
		Returns sweep record with participating accounts and amounts swept.
		"""
		tenant = self._tenant(tenant_id)
		if not pool_id:
			raise ValueError("pool_id_required")
		if not value_date:
			raise ValueError("value_date_required")
		# Get pool definition or create synthetic
		pool = self._cash_pools.get(f"{tenant}:{pool_id}")
		if pool is None:
			pool = {
				"pool_id": pool_id,
				"tenant_id": tenant,
				"header_account_id": None,
				"participant_accounts": [],
			}
		accounts = [a for a in self.cash_accounts.values() if a["tenant_id"] == tenant]
		swept_amounts: list[dict[str, Any]] = []
		total_swept = Decimal("0")
		for account in accounts:
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == account["id"] and p["as_of_date"] <= value_date]
			if not positions:
				continue
			latest = max(positions, key=lambda p: p["as_of_date"])
			available = Decimal(str(latest["available_balance"]))
			min_buffer = Decimal(str(account.get("minimum_buffer", 0)))
			sweepable = max(Decimal("0"), available - min_buffer)
			if sweep_type == "zero_balance" and sweepable > 0:
				swept_amounts.append({"account_id": account["id"], "amount_swept": str(sweepable), "currency": account["currency"]})
				total_swept += sweepable
		sweep_id = self._record_id("sweep")
		record = {
			"id": sweep_id,
			"type": "cash_pooling_sweep",
			"tenant_id": tenant,
			"pool_id": pool_id,
			"value_date": value_date,
			"sweep_type": sweep_type,
			"accounts_swept": len(swept_amounts),
			"swept_amounts": swept_amounts,
			"total_swept": str(total_swept),
			"approved_by": approved_by,
			"status": "completed",
			"created_at": _now(),
		}
		self._pool_sweeps.append(record)
		self._emit(tenant, "cash_pool_swept", record)
		return deepcopy(record)

	def intercompany_settlement(
		self,
		from_entity: str,
		to_entity: str,
		amount: float,
		currency: str,
		value_date: str,
		tenant_id: str | None = None,
		settlement_id: str | None = None,
		approved_by: str = "treasury",
		reference: str = "",
	) -> dict[str, Any]:
		"""
		Record an intercompany cash settlement between two entities.

		Creates a settlement record with debit/credit entries for each entity.
		"""
		tenant = self._tenant(tenant_id)
		if not from_entity or not to_entity:
			raise ValueError("from_entity_and_to_entity_required")
		if from_entity == to_entity:
			raise ValueError("settlement_entities_must_differ")
		if float(amount) <= 0:
			raise ValueError("settlement_amount_must_be_positive")
		if currency not in SUPPORTED_CURRENCIES:
			raise ValueError(f"unsupported_currency:{currency}")
		if not approved_by:
			raise PermissionError("settlement_approval_required")
		sett_id = self._record_id("sett", settlement_id)
		record = {
			"id": sett_id,
			"type": "intercompany_settlement",
			"tenant_id": tenant,
			"from_entity": from_entity,
			"to_entity": to_entity,
			"amount": Decimal(str(amount)),
			"currency": currency,
			"value_date": value_date,
			"reference": reference,
			"approved_by": approved_by,
			"debit_entry": {"entity": from_entity, "amount": str(amount), "currency": currency, "type": "debit"},
			"credit_entry": {"entity": to_entity, "amount": str(amount), "currency": currency, "type": "credit"},
			"status": "settled",
			"created_at": _now(),
		}
		self._intercompany_settlements.append(record)
		self._emit(tenant, "intercompany_settlement_recorded", record)
		return deepcopy(record)

	def bank_covenant_compliance(
		self,
		facility_id: str,
		period: str,
		tenant_id: str | None = None,
		covenants: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Check bank covenant compliance for a credit facility over a period.

		covenants: list of dicts with keys: name, metric, threshold, direction ('min'|'max'), actual.
		Returns compliance status per covenant and overall compliance flag.
		"""
		tenant = self._tenant(tenant_id)
		if not facility_id:
			raise ValueError("facility_id_required")
		if not period:
			raise ValueError("period_required")
		covenant_list = covenants or []
		results: list[dict[str, Any]] = []
		for covenant in covenant_list:
			name = covenant.get("name", "unnamed")
			threshold = float(covenant.get("threshold", 0))
			actual = float(covenant.get("actual", 0))
			direction = covenant.get("direction", "min")
			compliant = actual >= threshold if direction == "min" else actual <= threshold
			results.append({
				"covenant": name,
				"metric": covenant.get("metric", ""),
				"threshold": threshold,
				"actual": actual,
				"direction": direction,
				"compliant": compliant,
				"variance": round(actual - threshold, 4),
			})
		overall_compliant = all(r["compliant"] for r in results)
		breach_count = sum(1 for r in results if not r["compliant"])
		check_id = self._record_id("cov")
		record = {
			"id": check_id,
			"type": "covenant_compliance_check",
			"tenant_id": tenant,
			"facility_id": facility_id,
			"period": period,
			"covenant_count": len(results),
			"compliant_count": len(results) - breach_count,
			"breach_count": breach_count,
			"overall_compliant": overall_compliant,
			"results": results,
			"status": "compliant" if overall_compliant else "breach",
			"created_at": _now(),
		}
		self._covenant_checks.append(record)
		self._emit(tenant, "covenant_compliance_checked", record)
		return deepcopy(record)

	def mobile_money_reconciliation(
		self,
		wallet_id: str,
		period: str,
		tenant_id: str | None = None,
		transactions: list[dict[str, Any]] | None = None,
		provider: str = "mpesa",
	) -> dict[str, Any]:
		"""
		Reconcile mobile money wallet transactions for a period.

		transactions: List of wallet transaction dicts with: date, amount, type, reference.
		Returns reconciliation record with totals, match rate, and unmatched items.
		"""
		tenant = self._tenant(tenant_id)
		if not wallet_id:
			raise ValueError("wallet_id_required")
		if not period:
			raise ValueError("period_required")
		txns = transactions or []
		credits = sum(Decimal(str(t.get("amount", 0))) for t in txns if t.get("type") == "credit")
		debits = sum(Decimal(str(t.get("amount", 0))) for t in txns if t.get("type") == "debit")
		# Match against cash_flows
		account_flows = [
			f for f in self.cash_flows.values()
			if f["tenant_id"] == tenant
			and f.get("expected_date", "")[:7] == period
		]
		matched: list[dict[str, Any]] = []
		unmatched: list[dict[str, Any]] = []
		flow_amounts = {str(f["amount"]): f for f in account_flows}
		for txn in txns:
			flow = flow_amounts.get(str(txn.get("amount", 0)))
			if flow:
				matched.append({"txn": txn, "flow_id": flow["id"]})
			else:
				unmatched.append(txn)
		match_rate = round(len(matched) / len(txns), 4) if txns else 0.0
		recon_id = self._record_id("mmrecon")
		record = {
			"id": recon_id,
			"type": "mobile_money_reconciliation",
			"tenant_id": tenant,
			"wallet_id": wallet_id,
			"provider": provider,
			"period": period,
			"transaction_count": len(txns),
			"total_credits": str(credits),
			"total_debits": str(debits),
			"net_movement": str(credits - debits),
			"matched_count": len(matched),
			"unmatched_count": len(unmatched),
			"match_rate": match_rate,
			"unmatched_items": unmatched,
			"status": "reconciled" if not unmatched else "partial",
			"created_at": _now(),
		}
		self._mobile_money_reconciliations.append(record)
		self._emit(tenant, "mobile_money_reconciled", record)
		return deepcopy(record)

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def create_bank(self, bank_id: str, tenant_id: str, code: str, name: str, connectivity_status: str = "manual") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "create_bank", "operation_type": "write", "policy_attached": True, "bank_code_present": bool(code), "bank_name_present": bool(name)})
		record = {"id": self._record_id("bank", bank_id), "type": "bank_relationship", "tenant_id": tenant, "code": code, "name": name, "connectivity_status": connectivity_status, "status": "active", "created_at": _now()}
		self.banks[record["id"]] = record
		self._emit(tenant, "bank_created", record)
		return deepcopy(record)

	def create_cash_account(self, account_id: str, tenant_id: str, bank_id: str, account_number: str, name: str, account_type: str, currency: str = "USD", minimum_buffer: float = 0) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		bank = self.banks.get(bank_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "create_cash_account", "operation_type": "write", "policy_attached": True, "bank_present": bool(bank and bank["tenant_id"] == tenant), "account_number_present": bool(account_number), "account_name_present": bool(name), "account_type_supported": account_type in SUPPORTED_ACCOUNT_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES})
		record = {"id": self._record_id("cashacct", account_id), "type": "cash_account", "tenant_id": tenant, "bank_id": bank_id, "account_number": account_number, "name": name, "account_type": account_type, "currency": currency, "minimum_buffer": Decimal(str(minimum_buffer)), "status": "active", "created_at": _now()}
		self.cash_accounts[record["id"]] = record
		self._emit(tenant, "cash_account_created", record)
		return deepcopy(record)

	def record_cash_position(self, position_id: str, tenant_id: str, account_id: str, as_of_date: str, available_balance: float, ledger_balance: float | None = None, liquidity_reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(account_id)
		below_buffer = bool(account and Decimal(str(available_balance)) < account["minimum_buffer"])
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "record_cash_position", "operation_type": "write", "policy_attached": True, "account_present": bool(account and account["tenant_id"] == tenant), "as_of_date_present": bool(as_of_date), "available_balance_present": available_balance is not None, "below_minimum_buffer": below_buffer, "liquidity_review_recorded": bool(liquidity_reviewed_by)})
		record = {"id": self._record_id("position", position_id), "type": "cash_position", "tenant_id": tenant, "account_id": account_id, "as_of_date": as_of_date, "available_balance": Decimal(str(available_balance)), "ledger_balance": Decimal(str(ledger_balance if ledger_balance is not None else available_balance)), "liquidity_reviewed_by": liquidity_reviewed_by, "status": "reviewed" if liquidity_reviewed_by else "recorded", "created_at": _now()}
		self.cash_positions[record["id"]] = record
		self._emit(tenant, "cash_position_recorded", record)
		return deepcopy(record)

	def record_cash_flow(self, flow_id: str, tenant_id: str, account_id: str, flow_type: str, amount: float, category: str, expected_date: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(account_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "record_cash_flow", "operation_type": "write", "policy_attached": True, "account_present": bool(account and account["tenant_id"] == tenant), "flow_type_supported": flow_type in SUPPORTED_FLOW_TYPES, "amount": amount, "category_present": bool(category), "expected_date_present": bool(expected_date)})
		record = {"id": self._record_id("flow", flow_id), "type": "cash_flow", "tenant_id": tenant, "account_id": account_id, "flow_type": flow_type, "amount": Decimal(str(amount)), "category": category, "expected_date": expected_date, "status": "recorded", "created_at": _now()}
		self.cash_flows[record["id"]] = record
		self._emit(tenant, "cash_flow_recorded", record)
		return deepcopy(record)

	def create_cash_forecast(self, forecast_id: str, tenant_id: str, horizon_days: int, scenario: str, confidence_score: float, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "create_cash_forecast", "operation_type": "write", "policy_attached": True, "horizon_days": horizon_days, "scenario_supported": scenario in SUPPORTED_FORECAST_SCENARIOS, "confidence_score": confidence_score, "forecast_review_recorded": bool(reviewed_by)})
		flows = [f for f in self.cash_flows.values() if f["tenant_id"] == tenant]
		net_amount = sum((f["amount"] if f["flow_type"] == "inflow" else -f["amount"]) for f in flows)
		record = {"id": self._record_id("forecast", forecast_id), "type": "cash_forecast", "tenant_id": tenant, "horizon_days": horizon_days, "scenario": scenario, "confidence_score": confidence_score, "reviewed_by": reviewed_by, "projected_net_cash": net_amount, "source_flow_count": len(flows), "status": "reviewed" if reviewed_by else "forecasted", "created_at": _now()}
		self.cash_forecasts[record["id"]] = record
		self._emit(tenant, "cash_forecast_created", record)
		return deepcopy(record)

	def record_liquidity_review(self, review_id: str, tenant_id: str, position_id: str, reviewer: str, decision: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if position_id not in self.cash_positions or not reviewer:
			raise PermissionError("liquidity_review_required")
		record = {"id": self._record_id("liquidity", review_id), "type": "liquidity_review", "tenant_id": tenant, "position_id": position_id, "reviewer": reviewer, "decision": decision, "status": "reviewed", "created_at": _now()}
		self.liquidity_reviews[record["id"]] = record
		self._emit(tenant, "liquidity_review_recorded", record)
		return deepcopy(record)

	def record_bank_reconciliation(self, reconciliation_id: str, tenant_id: str, account_id: str, bank_statement_balance: float, ledger_balance: float, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		variance = Decimal(str(bank_statement_balance)) - Decimal(str(ledger_balance))
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "record_bank_reconciliation", "operation_type": "write", "policy_attached": True, "bank_statement_present": bank_statement_balance is not None, "ledger_balance_present": ledger_balance is not None, "variance": float(variance), "reconciliation_review_recorded": bool(reviewed_by)})
		record = {"id": self._record_id("recon", reconciliation_id), "type": "bank_reconciliation", "tenant_id": tenant, "account_id": account_id, "bank_statement_balance": Decimal(str(bank_statement_balance)), "ledger_balance": Decimal(str(ledger_balance)), "variance": variance, "reviewed_by": reviewed_by, "status": "matched" if variance == 0 else "reviewed", "created_at": _now()}
		self.reconciliations[record["id"]] = record
		self._emit(tenant, "bank_reconciliation_recorded", record)
		return deepcopy(record)

	def create_treasury_investment(self, investment_id: str, tenant_id: str, investment_type: str, counterparty: str, principal: float, maturity_date: str, yield_rate: float, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "create_treasury_investment", "operation_type": "write", "policy_attached": True, "investment_type_supported": investment_type in SUPPORTED_INVESTMENT_TYPES, "counterparty_present": bool(counterparty), "maturity_date_present": bool(maturity_date), "approval_recorded": bool(approved_by)})
		record = {"id": self._record_id("investment", investment_id), "type": "treasury_investment", "tenant_id": tenant, "investment_type": investment_type, "counterparty": counterparty, "principal": Decimal(str(principal)), "maturity_date": maturity_date, "yield_rate": yield_rate, "approved_by": approved_by, "status": "approved", "created_at": _now()}
		self.investments[record["id"]] = record
		self._emit(tenant, "treasury_investment_created", record)
		return deepcopy(record)

	def validate_payment_run(self, payment_run_id: str, tenant_id: str, funding_account_id: str, payment_total: float, approved_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(funding_account_id)
		positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == funding_account_id]
		current_position = positions[-1] if positions else None
		projected_deficit = bool(current_position and current_position["available_balance"] - Decimal(str(payment_total)) < 0)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "validate_payment_run", "operation_type": "write", "policy_attached": True, "funding_account_present": bool(account and account["tenant_id"] == tenant), "cash_position_present": bool(current_position), "projected_deficit": projected_deficit, "approval_recorded": bool(approved_by)})
		record = {"id": self._record_id("payrun", payment_run_id), "type": "payment_run", "tenant_id": tenant, "funding_account_id": funding_account_id, "payment_total": Decimal(str(payment_total)), "approved_by": approved_by, "status": "funded", "created_at": _now()}
		self.payment_runs[record["id"]] = record
		self._emit(tenant, "payment_run_validated", record)
		return deepcopy(record)

	def register_cbm_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "register_cbm_agent", "operation_type": "write", "policy_attached": True, "agent_runtime_supported": runtime in SUPPORTED_CBM_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_CBM_AGENT_ROLES})
		record = {"id": self._record_id("agent"), "type": "cbm_agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "scope": scope, "status": "active", "created_at": _now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "cbm_agent_registered", record)
		return deepcopy(record)

	def validate_agent_cbm_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		if agent_id not in self.agents:
			raise PermissionError("cbm_agent_required")
		result = evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "agent_cbm_action", "action": action, "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		if result["decision"] == "deny":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "cbm_batch", "event_stream": event_stream})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": CBM_EVENT_STREAM}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		# Total cash: sum latest position per account
		acct_ids = [a["id"] for a in self.cash_accounts.values() if a["tenant_id"] == tenant]
		total_cash = Decimal("0")
		for aid in acct_ids:
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == aid]
			if positions:
				latest = max(positions, key=lambda p: p["as_of_date"])
				total_cash += latest["available_balance"]
		return {
			"tenant_id": tenant,
			"total_cash_balance": total_cash,
			"bank_count": sum(1 for r in self.banks.values() if r["tenant_id"] == tenant),
			"cash_account_count": sum(1 for r in self.cash_accounts.values() if r["tenant_id"] == tenant),
			"cash_position_count": sum(1 for r in self.cash_positions.values() if r["tenant_id"] == tenant),
			"cash_flow_count": sum(1 for r in self.cash_flows.values() if r["tenant_id"] == tenant),
			"forecast_count": sum(1 for r in self.cash_forecasts.values() if r["tenant_id"] == tenant),
			"reconciliation_count": sum(1 for r in self.reconciliations.values() if r["tenant_id"] == tenant),
			"investment_count": sum(1 for r in self.investments.values() if r["tenant_id"] == tenant),
			"payment_run_count": sum(1 for r in self.payment_runs.values() if r["tenant_id"] == tenant),
			"bank_statement_count": sum(1 for r in self._bank_statements.values() if r["tenant_id"] == tenant),
			"intercompany_settlement_count": sum(1 for r in self._intercompany_settlements if r["tenant_id"] == tenant),
			"mobile_money_recon_count": sum(1 for r in self._mobile_money_reconciliations if r["tenant_id"] == tenant),
			"cbm_agent_count": sum(1 for r in self.agents.values() if r["tenant_id"] == tenant),
			"audit_event_count": sum(1 for e in self._audit_events if e["tenant_id"] == tenant),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		store = getattr(self, collection)
		return [deepcopy(r) for r in store.values() if r["tenant_id"] == tenant]

	# ------------------------------------------------------------------ convenience list methods

	def list_banks(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all bank relationships for tenant."""
		return self.list_records("banks", tenant_id)

	def list_cash_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all cash accounts for tenant."""
		return self.list_records("cash_accounts", tenant_id)

	def list_cash_positions(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all cash positions for tenant."""
		return self.list_records("cash_positions", tenant_id)

	def list_cash_flows(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all cash flows for tenant."""
		return self.list_records("cash_flows", tenant_id)

	def list_forecasts(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all forecasts for tenant."""
		return self.list_records("cash_forecasts", tenant_id)

	def list_reconciliations(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all reconciliations for tenant."""
		return self.list_records("reconciliations", tenant_id)

	def list_investments(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all investments for tenant."""
		return self.list_records("investments", tenant_id)

	# ------------------------------------------------------------------ import_bank_statement

	def import_bank_statement(
		self,
		statement_id: str,
		tenant_id: str,
		account_id: str,
		raw_content: str,
		fmt: str = "mt940",
	) -> dict[str, Any]:
		"""Import a bank statement in MT940, camt.053, or M-Pesa CSV format.

		Parses transactions from the raw content, stores the statement record,
		and returns a summary.  Actual line parsing is heuristic for demo purposes —
		production implementations wire to a dedicated parser library.

		Args:
			statement_id: Caller-supplied deduplication key.
			tenant_id: Tenant scope.
			account_id: Target cash account.
			raw_content: Raw statement text (MT940 / ISO 20022 XML / M-Pesa CSV).
			fmt: One of "mt940", "camt053", "mpesa", "manual".
		"""
		tenant = self._tenant(tenant_id)
		account = self.cash_accounts.get(account_id)
		if not account or account["tenant_id"] != tenant:
			raise KeyError(f"cash_account_not_found:{account_id}")
		assert fmt in ("mt940", "camt053", "mpesa", "manual"), f"unsupported_format:{fmt}"

		txn_count = 0
		closing_balance: Decimal | None = None

		if fmt == "mt940":
			# Count :61: tags as transactions
			import re as _re
			txn_count = len(_re.findall(r"^:61:", raw_content, _re.MULTILINE))
			# :62F: closing balance tag  e.g.  :62F:C260601KES1500000,00
			m = _re.search(r":62[FM]:[CD](\d+)[A-Z]{3}([\d,]+)", raw_content)
			if m:
				closing_balance = Decimal(m.group(2).replace(",", "."))

		elif fmt == "mpesa":
			lines = [l for l in raw_content.strip().splitlines() if l.strip()]
			# Skip header
			txn_count = max(0, len(lines) - 1)

		elif fmt == "camt053":
			import re as _re
			txn_count = len(_re.findall(r"<Ntry>", raw_content))

		record = {
			"id": self._record_id("stmt", statement_id),
			"type": "bank_statement_import",
			"tenant_id": tenant,
			"account_id": account_id,
			"format": fmt,
			"transaction_count": txn_count,
			"closing_balance": str(closing_balance) if closing_balance is not None else None,
			"status": "imported",
			"created_at": _now(),
		}
		self._bank_statements[record["id"]] = record
		self._emit(tenant, "bank_statement_imported", record)
		return deepcopy(record)

	# ------------------------------------------------------------------ sweep_accounts

	def sweep_accounts(
		self,
		sweep_id: str,
		tenant_id: str,
		source_account_ids: list[str],
		target_account_id: str,
		sweep_date: str,
	) -> dict[str, Any]:
		"""Sweep excess cash from source accounts into a concentration account.

		For each source account, if the available balance on sweep_date exceeds
		the account's minimum_buffer, the excess is notionally transferred to the
		target account.  A sweep record is returned with per-account detail.

		Args:
			sweep_id: Deduplication key.
			tenant_id: Tenant scope.
			source_account_ids: Accounts to sweep from.
			target_account_id: Concentration / notional pool account.
			sweep_date: ISO date string (YYYY-MM-DD).
		"""
		tenant = self._tenant(tenant_id)
		target = self.cash_accounts.get(target_account_id)
		if not target or target["tenant_id"] != tenant:
			raise KeyError(f"target_account_not_found:{target_account_id}")

		sweep_lines: list[dict[str, Any]] = []
		total_swept = Decimal("0")

		for acct_id in source_account_ids:
			acct = self.cash_accounts.get(acct_id)
			if not acct or acct["tenant_id"] != tenant:
				continue
			# Find latest position on or before sweep_date
			positions = [
				p for p in self.cash_positions.values()
				if p["tenant_id"] == tenant
				and p["account_id"] == acct_id
				and p["as_of_date"] <= sweep_date
			]
			if not positions:
				continue
			latest = max(positions, key=lambda p: p["as_of_date"])
			balance = latest["available_balance"]
			buffer = acct.get("minimum_buffer", Decimal("0"))
			excess = max(Decimal("0"), balance - buffer)
			if excess > Decimal("0"):
				sweep_lines.append({
					"account_id": acct_id,
					"balance": str(balance),
					"buffer": str(buffer),
					"swept_amount": str(excess),
				})
				total_swept += excess

		record = {
			"id": self._record_id("sweep", sweep_id),
			"type": "cash_sweep",
			"tenant_id": tenant,
			"target_account_id": target_account_id,
			"sweep_date": sweep_date,
			"sweep_id": sweep_id,
			"total_swept": str(total_swept),
			"line_count": len(sweep_lines),
			"lines": sweep_lines,
			"status": "completed",
			"created_at": _now(),
		}
		# Store in bank_statements as generic store — or use a dedicated dict if present
		if hasattr(self, "_sweeps"):
			self._sweeps[record["id"]] = record  # type: ignore[attr-defined]
		self._emit(tenant, "cash_sweep_completed", record)
		return deepcopy(record)

	# ------------------------------------------------------------------ dashboard total_cash_balance

	# ------------------------------------------------------------------
	# IFRS/GAAP compliance and regulatory reporting
	# ------------------------------------------------------------------

	def ifrs_cash_flow_statement(
		self,
		period: str,
		tenant_id: str | None = None,
		method: str = "indirect",
	) -> dict[str, Any]:
		"""
		Generate an IAS 7-compliant cash flow statement for a period.

		Returns operating, investing, and financing activity sections.
		method: 'direct' or 'indirect'.
		"""
		tenant = self._tenant(tenant_id)
		if not period:
			raise ValueError("period_required")
		flows = [f for f in self.cash_flows.values() if f["tenant_id"] == tenant and f.get("expected_date", "")[:7] == period]
		operating = sum(Decimal(str(f["amount"])) * (1 if f["flow_type"] == "inflow" else -1) for f in flows if f.get("category") in ("operating", "collections", "payments"))
		investing = sum(Decimal(str(f["amount"])) * (1 if f["flow_type"] == "inflow" else -1) for f in flows if f.get("category") in ("investing", "capex", "asset_sale"))
		financing = sum(Decimal(str(f["amount"])) * (1 if f["flow_type"] == "inflow" else -1) for f in flows if f.get("category") in ("financing", "debt", "equity", "dividend"))
		net_change = operating + investing + financing
		stmt_id = self._record_id("ifrs7")
		record = {
			"id": stmt_id,
			"type": "ifrs_cash_flow_statement",
			"tenant_id": tenant,
			"period": period,
			"method": method,
			"standard": "IAS 7",
			"operating_activities": str(operating),
			"investing_activities": str(investing),
			"financing_activities": str(financing),
			"net_increase_decrease": str(net_change),
			"source_flow_count": len(flows),
			"status": "generated",
			"created_at": _now(),
		}
		self._emit(tenant, "ifrs_cash_flow_statement_generated", record)
		return deepcopy(record)

	def gaap_disclosure_note(
		self,
		period: str,
		tenant_id: str | None = None,
		framework: str = "IFRS",
	) -> dict[str, Any]:
		"""
		Generate a cash and cash equivalents disclosure note per IFRS/US GAAP.

		Summarises balances, restrictions, and concentrations by bank.
		"""
		tenant = self._tenant(tenant_id)
		accounts = [a for a in self.cash_accounts.values() if a["tenant_id"] == tenant and a["status"] == "active"]
		total_by_currency: dict[str, Decimal] = {}
		restricted_accounts: list[str] = []
		for acct in accounts:
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == acct["id"]]
			if positions:
				latest = max(positions, key=lambda p: p["as_of_date"])
				ccy = acct["currency"]
				total_by_currency[ccy] = total_by_currency.get(ccy, Decimal("0")) + Decimal(str(latest["available_balance"]))
			if acct.get("account_type") == "restricted":
				restricted_accounts.append(acct["id"])
		note_id = self._record_id("gaap_note")
		record = {
			"id": note_id,
			"type": "gaap_disclosure_note",
			"tenant_id": tenant,
			"period": period,
			"framework": framework,
			"account_count": len(accounts),
			"balances_by_currency": {k: str(v) for k, v in total_by_currency.items()},
			"restricted_account_count": len(restricted_accounts),
			"restricted_account_ids": restricted_accounts,
			"concentration_risk_note": len(accounts) > 0 and any(
				sum(1 for a2 in accounts if a2["bank_id"] == acct["bank_id"]) / len(accounts) > 0.5
				for acct in accounts
			),
			"status": "disclosed",
			"created_at": _now(),
		}
		self._emit(tenant, "gaap_disclosure_note_generated", record)
		return deepcopy(record)

	def regulatory_reporting_package(
		self,
		report_type: str,
		period: str,
		tenant_id: str | None = None,
		submitted_to: str = "",
	) -> dict[str, Any]:
		"""
		Compile a regulatory reporting package (e.g. CBK liquidity return, IMF data).

		report_type: 'cbk_liquidity' | 'imf_data' | 'bis_statistics' | 'internal'
		"""
		tenant = self._tenant(tenant_id)
		if not report_type or not period:
			raise ValueError("report_type_and_period_required")
		accounts = [a for a in self.cash_accounts.values() if a["tenant_id"] == tenant]
		flows_period = [f for f in self.cash_flows.values() if f["tenant_id"] == tenant and f.get("expected_date", "")[:7] == period]
		total_assets = sum(Decimal(str(p["available_balance"])) for p in self.cash_positions.values() if p["tenant_id"] == tenant)
		total_inflows = sum(Decimal(str(f["amount"])) for f in flows_period if f["flow_type"] == "inflow")
		total_outflows = sum(Decimal(str(f["amount"])) for f in flows_period if f["flow_type"] == "outflow")
		pkg_id = self._record_id("regpkg")
		record = {
			"id": pkg_id,
			"type": "regulatory_reporting_package",
			"tenant_id": tenant,
			"report_type": report_type,
			"period": period,
			"submitted_to": submitted_to,
			"account_count": len(accounts),
			"total_liquid_assets": str(total_assets),
			"period_inflows": str(total_inflows),
			"period_outflows": str(total_outflows),
			"lcr_proxy": str(round(total_assets / max(total_outflows, Decimal("1")), 4)),
			"status": "submitted" if submitted_to else "draft",
			"created_at": _now(),
		}
		self._emit(tenant, "regulatory_report_generated", record)
		return deepcopy(record)

	def bank_fee_analysis(
		self,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Analyse bank service charges and fees paid in a period.

		Scans cash_flows of category 'bank_fees' and returns breakdown by bank.
		"""
		tenant = self._tenant(tenant_id)
		fee_flows = [
			f for f in self.cash_flows.values()
			if f["tenant_id"] == tenant
			and f.get("category") in ("bank_fees", "charges", "service_charges")
			and f.get("expected_date", "")[:7] == period
		]
		by_bank: dict[str, Decimal] = {}
		for flow in fee_flows:
			acct = self.cash_accounts.get(flow.get("account_id", ""))
			bank_id = acct["bank_id"] if acct else "unknown"
			by_bank[bank_id] = by_bank.get(bank_id, Decimal("0")) + Decimal(str(flow["amount"]))
		total_fees = sum(by_bank.values())
		analysis_id = self._record_id("fee_analysis")
		record = {
			"id": analysis_id,
			"type": "bank_fee_analysis",
			"tenant_id": tenant,
			"period": period,
			"total_fees": str(total_fees),
			"fee_count": len(fee_flows),
			"by_bank": {k: str(v) for k, v in by_bank.items()},
			"largest_fee_bank": max(by_bank, key=by_bank.get) if by_bank else None,
			"status": "analysed",
			"created_at": _now(),
		}
		self._emit(tenant, "bank_fee_analysis_generated", record)
		return deepcopy(record)

	def cash_flow_variance_analysis(
		self,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Compare actual cash flows against forecasted flows for a period.

		Returns variance amounts and percentage deviations per category.
		"""
		tenant = self._tenant(tenant_id)
		actual_flows = [f for f in self.cash_flows.values() if f["tenant_id"] == tenant and f.get("expected_date", "")[:7] == period]
		forecasts = [f for f in self.cash_forecasts.values() if f["tenant_id"] == tenant and f.get("created_at", "")[:7] == period]
		actual_in = sum(Decimal(str(f["amount"])) for f in actual_flows if f["flow_type"] == "inflow")
		actual_out = sum(Decimal(str(f["amount"])) for f in actual_flows if f["flow_type"] == "outflow")
		forecast_net = sum(Decimal(str(f.get("projected_net_cash", 0))) for f in forecasts)
		actual_net = actual_in - actual_out
		variance = actual_net - forecast_net
		variance_pct = float(variance / forecast_net * 100) if forecast_net != 0 else 0.0
		va_id = self._record_id("var_analysis")
		record = {
			"id": va_id,
			"type": "cash_flow_variance_analysis",
			"tenant_id": tenant,
			"period": period,
			"actual_inflows": str(actual_in),
			"actual_outflows": str(actual_out),
			"actual_net": str(actual_net),
			"forecast_net": str(forecast_net),
			"variance": str(variance),
			"variance_pct": round(variance_pct, 2),
			"favourable": variance >= 0,
			"forecast_count": len(forecasts),
			"actual_flow_count": len(actual_flows),
			"status": "analysed",
			"created_at": _now(),
		}
		self._emit(tenant, "cash_flow_variance_analysed", record)
		return deepcopy(record)

	def interest_income_accrual(
		self,
		period: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Calculate accrued interest income on treasury investments for a period.

		Uses yield_rate × principal × (days_in_period / 365) per investment.
		"""
		tenant = self._tenant(tenant_id)
		investments = [i for i in self.investments.values() if i["tenant_id"] == tenant and i["status"] == "approved"]
		accruals: list[dict[str, Any]] = []
		total_accrued = Decimal("0")
		days_in_period = 30  # month approximation
		for inv in investments:
			principal = Decimal(str(inv["principal"]))
			yield_rate = Decimal(str(inv.get("yield_rate", 0)))
			accrued = round(principal * yield_rate * Decimal(str(days_in_period)) / Decimal("365"), 4)
			total_accrued += accrued
			accruals.append({
				"investment_id": inv["id"],
				"investment_type": inv["investment_type"],
				"principal": str(principal),
				"yield_rate": str(yield_rate),
				"accrued_interest": str(accrued),
			})
		accrual_id = self._record_id("int_accrual")
		record = {
			"id": accrual_id,
			"type": "interest_income_accrual",
			"tenant_id": tenant,
			"period": period,
			"investment_count": len(accruals),
			"total_accrued_interest": str(total_accrued),
			"accruals": accruals,
			"days_in_period": days_in_period,
			"accounting_entry": {"debit": "interest_receivable", "credit": "interest_income", "amount": str(total_accrued)},
			"status": "accrued",
			"created_at": _now(),
		}
		self._emit(tenant, "interest_income_accrued", record)
		return deepcopy(record)

	def bulk_create_cash_flows(
		self,
		flows: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""
		Bulk create cash flow records from a list of flow dicts.

		Each dict must have: account_id, flow_type, amount, category, expected_date.
		Returns list of created flow records.
		"""
		tenant = self._tenant(tenant_id)
		if not flows:
			raise ValueError("flows_list_required")
		results: list[dict[str, Any]] = []
		for i, flow in enumerate(flows):
			rec = self.record_cash_flow(
				flow.get("flow_id") or self._record_id("flow"),
				tenant,
				str(flow["account_id"]),
				str(flow["flow_type"]),
				float(flow["amount"]),
				str(flow["category"]),
				str(flow["expected_date"]),
			)
			results.append(rec)
		return results

	def export_cash_flows(
		self,
		period: str,
		tenant_id: str | None = None,
		format: str = "json",
	) -> dict[str, Any]:
		"""
		Export cash flow records for a period in the requested format metadata.

		format: 'json' | 'csv' | 'excel'
		Returns export manifest with record count and download reference.
		"""
		tenant = self._tenant(tenant_id)
		flows = [f for f in self.cash_flows.values() if f["tenant_id"] == tenant and f.get("expected_date", "")[:7] == period]
		export_id = self._record_id("export")
		record = {
			"id": export_id,
			"type": "cash_flow_export",
			"tenant_id": tenant,
			"period": period,
			"format": format,
			"record_count": len(flows),
			"download_ref": f"/exports/{tenant}/{export_id}.{format}",
			"status": "ready",
			"created_at": _now(),
		}
		self._emit(tenant, "cash_flow_exported", record)
		return deepcopy(record)

	def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "CashManagementService",
			"status": "healthy",
			"banks": len(self.banks),
			"cash_accounts": len(self.cash_accounts),
			"cash_positions": len(self.cash_positions),
			"cash_flows": len(self.cash_flows),
			"reconciliations": len(self.reconciliations),
			"investments": len(self.investments),
			"bank_statements": len(self._bank_statements),
			"audit_events": len(self._audit_events),
			"checked_at": _now(),
		}

	def working_capital_analysis(
		self,
		as_of_date: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Compute working capital metrics: current ratio, quick ratio, cash ratio.

		Uses available cash positions as current assets proxy.
		"""
		tenant = self._tenant(tenant_id)
		total_cash = Decimal("0")
		for acct in self.cash_accounts.values():
			if acct["tenant_id"] != tenant:
				continue
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == acct["id"] and p["as_of_date"] <= as_of_date]
			if positions:
				latest = max(positions, key=lambda p: p["as_of_date"])
				total_cash += Decimal(str(latest["available_balance"]))
		# Outflows due within 90 days as proxy for current liabilities
		current_liabilities = sum(
			Decimal(str(f["amount"])) for f in self.cash_flows.values()
			if f["tenant_id"] == tenant and f["flow_type"] == "outflow"
			and f.get("expected_date", "9999") <= (date.today() + timedelta(days=90)).isoformat()
		)
		cash_ratio = float(total_cash / current_liabilities) if current_liabilities > 0 else None
		wc_id = self._record_id("wc_analysis")
		record = {
			"id": wc_id,
			"type": "working_capital_analysis",
			"tenant_id": tenant,
			"as_of_date": as_of_date,
			"total_cash": str(total_cash),
			"current_liabilities_proxy": str(current_liabilities),
			"cash_ratio": round(cash_ratio, 4) if cash_ratio is not None else None,
			"cash_coverage_adequate": cash_ratio is not None and cash_ratio >= 1.0,
			"status": "analysed",
			"created_at": _now(),
		}
		self._emit(tenant, "working_capital_analysed", record)
		return deepcopy(record)

	def investment_maturity_schedule(
		self,
		tenant_id: str | None = None,
		horizon_days: int = 365,
	) -> dict[str, Any]:
		"""
		Return a schedule of treasury investment maturities within the horizon.

		Returns investments sorted by maturity date with projected principal receipts.
		"""
		tenant = self._tenant(tenant_id)
		today = _today()
		horizon_end = (date.today() + timedelta(days=horizon_days)).isoformat()
		maturing = [
			i for i in self.investments.values()
			if i["tenant_id"] == tenant
			and today <= i.get("maturity_date", "9999-12-31") <= horizon_end
		]
		maturing_sorted = sorted(maturing, key=lambda i: i["maturity_date"])
		total_principal = sum(Decimal(str(i["principal"])) for i in maturing_sorted)
		schedule_id = self._record_id("mat_schedule")
		record = {
			"id": schedule_id,
			"type": "investment_maturity_schedule",
			"tenant_id": tenant,
			"horizon_days": horizon_days,
			"investment_count": len(maturing_sorted),
			"total_maturing_principal": str(total_principal),
			"schedule": [
				{
					"investment_id": i["id"],
					"investment_type": i["investment_type"],
					"counterparty": i["counterparty"],
					"maturity_date": i["maturity_date"],
					"principal": str(i["principal"]),
					"yield_rate": i["yield_rate"],
				}
				for i in maturing_sorted
			],
			"status": "generated",
			"created_at": _now(),
		}
		self._emit(tenant, "investment_maturity_schedule_generated", record)
		return deepcopy(record)

	def payment_run_funding_check(
		self,
		payment_run_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Re-validate funding adequacy for a payment run at execution time.

		Checks current available balance against payment_total.
		"""
		tenant = self._tenant(tenant_id)
		pay_run = self.payment_runs.get(payment_run_id)
		if not pay_run or pay_run["tenant_id"] != tenant:
			raise KeyError(f"payment_run_not_found:{payment_run_id}")
		account_id = pay_run["funding_account_id"]
		acct = self._get_account(account_id, tenant)
		positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == account_id]
		current_balance = Decimal("0")
		if positions:
			latest = max(positions, key=lambda p: p["as_of_date"])
			current_balance = Decimal(str(latest["available_balance"]))
		payment_total = Decimal(str(pay_run["payment_total"]))
		shortfall = max(Decimal("0"), payment_total - current_balance)
		check_id = self._record_id("fund_check")
		record = {
			"id": check_id,
			"type": "payment_run_funding_check",
			"tenant_id": tenant,
			"payment_run_id": payment_run_id,
			"account_id": account_id,
			"account_name": acct["name"],
			"current_balance": str(current_balance),
			"payment_total": str(payment_total),
			"shortfall": str(shortfall),
			"funded": shortfall == 0,
			"status": "passed" if shortfall == 0 else "failed",
			"created_at": _now(),
		}
		self._emit(tenant, "payment_run_funding_checked", record)
		return deepcopy(record)

	def stress_test_liquidity(
		self,
		scenario: str,
		outflow_shock_pct: float,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Apply a liquidity stress scenario by shocking outflows by a percentage.

		Returns stressed ending balance and days_to_zero projection.
		"""
		tenant = self._tenant(tenant_id)
		if outflow_shock_pct < 0:
			raise ValueError("outflow_shock_pct_must_be_non_negative")
		total_cash = Decimal("0")
		for acct in self.cash_accounts.values():
			if acct["tenant_id"] != tenant:
				continue
			positions = [p for p in self.cash_positions.values() if p["tenant_id"] == tenant and p["account_id"] == acct["id"]]
			if positions:
				latest = max(positions, key=lambda p: p["as_of_date"])
				total_cash += Decimal(str(latest["available_balance"]))
		normal_outflows = sum(Decimal(str(f["amount"])) for f in self.cash_flows.values() if f["tenant_id"] == tenant and f["flow_type"] == "outflow")
		shocked_outflows = normal_outflows * (1 + Decimal(str(outflow_shock_pct)) / 100)
		normal_inflows = sum(Decimal(str(f["amount"])) for f in self.cash_flows.values() if f["tenant_id"] == tenant and f["flow_type"] == "inflow")
		stressed_net = normal_inflows - shocked_outflows
		daily_stressed_outflow = shocked_outflows / 30 if shocked_outflows > 0 else Decimal("1")
		days_to_zero = int(total_cash / daily_stressed_outflow) if daily_stressed_outflow > 0 and total_cash > 0 else 999
		stress_id = self._record_id("stress")
		record = {
			"id": stress_id,
			"type": "liquidity_stress_test",
			"tenant_id": tenant,
			"scenario": scenario,
			"outflow_shock_pct": outflow_shock_pct,
			"current_cash": str(total_cash),
			"normal_outflows": str(normal_outflows),
			"shocked_outflows": str(shocked_outflows),
			"normal_inflows": str(normal_inflows),
			"stressed_net_30d": str(stressed_net),
			"days_to_zero": days_to_zero,
			"adequate": days_to_zero >= 30,
			"status": "tested",
			"created_at": _now(),
		}
		self._emit(tenant, "liquidity_stress_tested", record)
		return deepcopy(record)


CBMService = CashManagementService
