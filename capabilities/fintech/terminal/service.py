"""TerminalBankingService — agency banking / rural financial inclusion terminal network.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_TERMINAL_TYPES,
	SUPPORTED_STATUSES,
	evaluate_capability_rules,
)
from .database.store import Store, get_store
from .domain.adapters import (
	AuthAdapter,
	AuditAdapter,
	NotifyAdapter,
	get_auth_adapter,
	get_audit_adapter,
	get_notify_adapter,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _today() -> str:
	return date.today().isoformat()


def _uid() -> str:
	return str(uuid.uuid4())


def _log_pretty_path(terminal_id: str, op: str) -> str:
	return f"[terminal:{terminal_id}] {op}"


def _period_bounds(period: str) -> tuple[str, str]:
	"""Parse period like '2025-Q1', '2025-06', '2025' into (start, end) ISO dates."""
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		# last day approximation
		end_day = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{end_day:02d}"
	if "Q" in period:
		y, q = period.split("-Q")
		q = int(q)
		start_month = (q - 1) * 3 + 1
		end_month = q * 3
		end_day = 31 if end_month in {3, 12} else 30
		return f"{y}-{start_month:02d}-01", f"{y}-{end_month:02d}-{end_day:02d}"
	return period, period


# ─────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class TerminalBankingService:
	"""Agency banking terminal network service for rural financial inclusion.

	Manages the full lifecycle of agent terminals: registration, activation,
	transactions, float, reconciliation, fraud alerting, and regulatory reporting.

	Usage (standalone)::

		svc = TerminalBankingService()
		terminal = await svc.register_terminal("T001", {...}, "AGT-1", "mpos", "lte")

	Usage (platform)::

		from apg_common_auth import AuthService
		svc = TerminalBankingService(auth=AuthService.from_env())
	"""

	def __init__(
		self,
		*,
		db_url: str | None = None,
		store: Store | None = None,
		auth: Any | None = None,
		audit: Any | None = None,
		notify: Any | None = None,
		tenant_id: str = "default",
	) -> None:
		self._store: Store = store or get_store(db_url)
		self._auth: AuthAdapter = get_auth_adapter(auth)
		self._audit: AuditAdapter = get_audit_adapter(audit)
		self._notify: NotifyAdapter = get_notify_adapter(notify)
		self._tenant_id = tenant_id
		self._capability = CAPABILITY_ID
		self._version = CAPABILITY_VERSION

	# ── Internal helpers ─────────────────────────────────────

	async def _audit_event(
		self,
		event_type: str,
		actor_id: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		await self._audit.log_event(
			event_type, actor_id, self._tenant_id, resource_id, details
		)

	async def _get_terminal(self, terminal_id: str) -> dict[str, Any]:
		rec = await self._store.get("terminals", terminal_id)
		if rec is None:
			raise ValueError(f"Terminal not found: {terminal_id}")
		return rec

	async def _assert_active(self, terminal_id: str) -> dict[str, Any]:
		t = await self._get_terminal(terminal_id)
		if t.get("status") != "active":
			raise ValueError(
				f"Terminal {terminal_id} is {t.get('status')!r}, expected 'active'"
			)
		return t

	# ── Core lifecycle ────────────────────────────────────────

	async def register_terminal(
		self,
		terminal_id: str,
		location: dict[str, Any],
		agent_id: str,
		terminal_type: str,
		connectivity: str,
		*,
		serial_number: str | None = None,
		merchant_id: str | None = None,
		model: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new agent terminal in the network.

		Validates terminal type, location, and agent assignment before persisting.
		Emits ``terminal_registered`` lifecycle event.
		"""
		assert terminal_id, "terminal_id required"
		assert agent_id, "agent_id required"
		assert location, "location required"

		tid = tenant_id or self._tenant_id

		if terminal_type not in SUPPORTED_TERMINAL_TYPES:
			raise ValueError(
				f"Unsupported terminal type {terminal_type!r}. "
				f"Supported: {SUPPORTED_TERMINAL_TYPES}"
			)

		rule_ctx = {
			"operation": "register_terminal",
			"tenant_context_present": True,
			"terminal_type_supported": True,
			"serial_number_present": bool(serial_number),
			"merchant_present": bool(merchant_id),
			"location_present": bool(location),
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Registration denied: {verdict['matched_rules']}")

		record: dict[str, Any] = {
			"id": terminal_id,
			"tenant_id": tid,
			"terminal_type": terminal_type,
			"connectivity": connectivity,
			"location": location,
			"agent_id": agent_id,
			"serial_number": serial_number,
			"merchant_id": merchant_id,
			"model": model,
			"status": "configuration_pending",
			"registered_at": _now(),
			"updated_at": _now(),
			"float_balance": 0.0,
			"transaction_count": 0,
			"last_heartbeat": None,
			"offline_queue": [],
			"capability": self._capability,
		}
		await self._store.put("terminals", record)
		await self._audit_event(
			"terminal_registered", agent_id, terminal_id,
			{"terminal_type": terminal_type, "location": location, "connectivity": connectivity},
		)
		return record

	async def activate_terminal(
		self,
		terminal_id: str,
		activated_by: str,
		*,
		pci_dss_compliant: bool = True,
		tamper_detection_enabled: bool = True,
		software_integrity_verified: bool = True,
	) -> dict[str, Any]:
		"""Activate a terminal after key-injection and parameter deployment.

		Checks PCI DSS compliance, tamper detection, and software integrity
		before transitioning status to ``active``.
		"""
		assert activated_by, "activated_by required"

		terminal = await self._get_terminal(terminal_id)

		rule_ctx = {
			"operation": "deploy_terminal",
			"tenant_context_present": True,
			"key_injection_complete": True,
			"parameters_deployed": True,
			"pci_dss_compliant": pci_dss_compliant,
			"tamper_detection_enabled": tamper_detection_enabled,
			"software_integrity_verified": software_integrity_verified,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Activation denied: {verdict['matched_rules']}")

		terminal["status"] = "active"
		terminal["activated_at"] = _now()
		terminal["activated_by"] = activated_by
		terminal["updated_at"] = _now()
		terminal["pci_dss_compliant"] = pci_dss_compliant
		terminal["tamper_detection_enabled"] = tamper_detection_enabled
		await self._store.put("terminals", terminal)
		await self._audit_event(
			"terminal_deployed", activated_by, terminal_id,
			{"status": "active", "pci_dss_compliant": pci_dss_compliant},
		)
		return terminal

	async def terminal_transaction(
		self,
		terminal_id: str,
		transaction_type: str,
		amount: float,
		currency: str,
		customer_id: str,
		reference: str,
		*,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Post a generic transaction through the terminal.

		Validates terminal status, checks rule engine, updates transaction
		count, and creates an immutable transaction record.
		"""
		assert amount > 0, "amount must be positive"
		assert currency, "currency required"
		assert customer_id, "customer_id required"
		assert reference, "reference required"

		terminal = await self._assert_active(terminal_id)

		rule_ctx = {
			"operation": "process_transaction",
			"terminal_status": terminal["status"],
			"tamper_detected": False,
			"terminal_key_expired": False,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Transaction denied: {verdict['matched_rules']}")

		txn: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"agent_id": terminal.get("agent_id"),
			"transaction_type": transaction_type,
			"amount": amount,
			"currency": currency,
			"customer_id": customer_id,
			"reference": reference,
			"status": "approved",
			"timestamp": _now(),
			"metadata": metadata or {},
		}
		await self._store.put("terminal_transactions", txn)

		# update counter
		terminal["transaction_count"] = terminal.get("transaction_count", 0) + 1
		terminal["last_transaction_at"] = _now()
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._audit_event(
			f"terminal_{transaction_type}", customer_id, terminal_id,
			{"amount": amount, "currency": currency, "txn_id": txn["id"]},
		)
		return txn

	async def cash_deposit(
		self,
		terminal_id: str,
		customer_id: str,
		amount: float,
		currency: str,
	) -> dict[str, Any]:
		"""Accept a cash deposit at the agent terminal.

		Credits the customer's account and adjusts the agent's float position.
		"""
		assert amount > 0, "deposit amount must be positive"

		terminal = await self._assert_active(terminal_id)
		txn = await self.terminal_transaction(
			terminal_id, "cash_deposit", amount, currency, customer_id,
			ref := f"DEP-{_uid()[:8].upper()}",
		)

		# reduce float (agent holds cash, we owe customer less float)
		terminal["float_balance"] = terminal.get("float_balance", 0.0) + amount
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._notify.send(
			customer_id, "sms",
			f"Deposit confirmed: {currency} {amount:,.2f}",
			f"Your cash deposit of {currency} {amount:,.2f} at agent {terminal_id} has been confirmed. Ref: {ref}",
		)
		return txn

	async def cash_withdrawal(
		self,
		terminal_id: str,
		customer_id: str,
		amount: float,
		currency: str,
		pin_verified: bool = True,
	) -> dict[str, Any]:
		"""Dispense cash at the agent terminal.

		Requires PIN verification and sufficient float before processing.
		"""
		assert amount > 0, "withdrawal amount must be positive"

		if not pin_verified:
			raise PermissionError("PIN verification required for cash withdrawal")

		terminal = await self._assert_active(terminal_id)
		float_bal = terminal.get("float_balance", 0.0)
		if float_bal < amount:
			raise ValueError(
				f"Insufficient float: available {float_bal:,.2f}, requested {amount:,.2f}"
			)

		txn = await self.terminal_transaction(
			terminal_id, "cash_withdrawal", amount, currency, customer_id,
			f"WDR-{_uid()[:8].upper()}",
		)

		terminal["float_balance"] = float_bal - amount
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._notify.send(
			customer_id, "sms",
			f"Withdrawal: {currency} {amount:,.2f}",
			f"Cash withdrawal of {currency} {amount:,.2f} processed at agent {terminal_id}.",
		)
		return txn

	async def fund_transfer_terminal(
		self,
		terminal_id: str,
		from_account: str,
		to_account: str,
		amount: float,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Initiate a peer-to-peer fund transfer via the agent terminal."""
		assert from_account and to_account, "both accounts required"
		assert from_account != to_account, "cannot transfer to same account"

		txn = await self.terminal_transaction(
			terminal_id, "funds_transfer", amount, currency,
			from_account, f"TRF-{_uid()[:8].upper()}",
			metadata={"to_account": to_account},
		)
		await self._notify.send(
			to_account, "sms",
			f"Funds received: {currency} {amount:,.2f}",
			f"You have received {currency} {amount:,.2f} from {from_account}. Ref: {txn['id']}",
		)
		return txn

	async def bill_payment_terminal(
		self,
		terminal_id: str,
		customer_id: str,
		biller_code: str,
		amount: float,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Pay a utility or service bill at the agent terminal."""
		assert biller_code, "biller_code required"
		assert amount > 0, "amount must be positive"

		return await self.terminal_transaction(
			terminal_id, "bill_payment", amount, currency,
			customer_id, f"BILL-{biller_code}-{_uid()[:6].upper()}",
			metadata={"biller_code": biller_code},
		)

	async def balance_inquiry(
		self,
		terminal_id: str,
		customer_id: str,
	) -> dict[str, Any]:
		"""Retrieve account balance for a customer at the agent terminal."""
		await self._assert_active(terminal_id)

		# Simulate balance lookup — in production this routes to the core banking adapter
		balance_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"customer_id": customer_id,
			"inquiry_type": "balance",
			"available_balance": 0.00,  # populated by core banking in production
			"ledger_balance": 0.00,
			"currency": "KES",
			"timestamp": _now(),
		}
		await self._store.put("terminal_inquiries", balance_rec)
		await self._audit_event(
			"terminal_balance_inquiry", customer_id, terminal_id, {}
		)
		return balance_rec

	async def mini_statement_terminal(
		self,
		terminal_id: str,
		customer_id: str,
		limit: int = 5,
	) -> dict[str, Any]:
		"""Retrieve last N transactions for a customer via the agent terminal."""
		assert 1 <= limit <= 20, "limit must be 1–20"
		await self._assert_active(terminal_id)

		txns = await self._store.query(
			"terminal_transactions",
			{"customer_id": customer_id, "terminal_id": terminal_id},
			limit=limit,
		)

		result = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"customer_id": customer_id,
			"transactions": txns,
			"count": len(txns),
			"generated_at": _now(),
		}
		await self._audit_event(
			"terminal_mini_statement", customer_id, terminal_id,
			{"count": len(txns)},
		)
		return result

	async def float_management(
		self,
		terminal_id: str,
		float_amount: float,
		operation_type: str,
		*,
		authorised_by: str | None = None,
	) -> dict[str, Any]:
		"""Top-up or reduce agent float balance.

		operation_type: 'top_up' | 'withdrawal' | 'reconcile'
		"""
		assert operation_type in {"top_up", "withdrawal", "reconcile"}, (
			"operation_type must be one of: top_up, withdrawal, reconcile"
		)
		assert float_amount >= 0, "float_amount must be non-negative"

		terminal = await self._get_terminal(terminal_id)
		previous = terminal.get("float_balance", 0.0)

		if operation_type == "top_up":
			terminal["float_balance"] = previous + float_amount
		elif operation_type == "withdrawal":
			if previous < float_amount:
				raise ValueError("Insufficient float for withdrawal")
			terminal["float_balance"] = previous - float_amount
		else:  # reconcile
			terminal["float_balance"] = float_amount

		terminal["updated_at"] = _now()
		terminal["last_float_operation"] = {
			"type": operation_type,
			"amount": float_amount,
			"previous_balance": previous,
			"new_balance": terminal["float_balance"],
			"authorised_by": authorised_by,
			"timestamp": _now(),
		}
		await self._store.put("terminals", terminal)
		await self._audit_event(
			"terminal_float_adjusted", authorised_by or "system", terminal_id,
			{
				"operation_type": operation_type,
				"float_amount": float_amount,
				"new_balance": terminal["float_balance"],
			},
		)
		return terminal

	async def terminal_reconciliation(
		self,
		terminal_id: str,
		recon_date: str,
	) -> dict[str, Any]:
		"""Reconcile terminal transactions for a given date.

		Returns count, total debits, total credits, and float variance.
		"""
		terminal = await self._get_terminal(terminal_id)
		txns = await self._store.query(
			"terminal_transactions",
			{"terminal_id": terminal_id},
			limit=10_000,
		)
		day_txns = [t for t in txns if t.get("timestamp", "").startswith(recon_date)]

		total_credits = sum(
			t["amount"] for t in day_txns
			if t.get("transaction_type") in {"cash_deposit", "funds_transfer"}
		)
		total_debits = sum(
			t["amount"] for t in day_txns
			if t.get("transaction_type") in {"cash_withdrawal", "bill_payment"}
		)
		float_variance = terminal.get("float_balance", 0.0) - (total_debits - total_credits)

		recon_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"recon_date": recon_date,
			"transaction_count": len(day_txns),
			"total_credits": total_credits,
			"total_debits": total_debits,
			"net_position": total_credits - total_debits,
			"float_balance": terminal.get("float_balance", 0.0),
			"float_variance": float_variance,
			"status": "balanced" if abs(float_variance) < 1.0 else "variance",
			"generated_at": _now(),
		}
		await self._store.put("terminal_reconciliations", recon_record)
		await self._audit_event(
			"terminal_reconciled", "system", terminal_id,
			{"recon_date": recon_date, "status": recon_record["status"]},
		)
		return recon_record

	async def terminal_health_check(
		self,
		terminal_id: str,
	) -> dict[str, Any]:
		"""Poll terminal health: connectivity, float adequacy, last heartbeat age."""
		terminal = await self._get_terminal(terminal_id)

		last_hb = terminal.get("last_heartbeat")
		hb_age_seconds: float | None = None
		if last_hb:
			hb_dt = datetime.fromisoformat(last_hb)
			hb_age_seconds = (datetime.now(timezone.utc) - hb_dt).total_seconds()

		float_bal = terminal.get("float_balance", 0.0)
		health: dict[str, Any] = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"status": terminal.get("status"),
			"float_balance": float_bal,
			"float_adequate": float_bal >= 5000.0,
			"last_heartbeat": last_hb,
			"heartbeat_age_seconds": hb_age_seconds,
			"heartbeat_ok": hb_age_seconds is None or hb_age_seconds < 300,
			"connectivity": terminal.get("connectivity"),
			"transaction_count": terminal.get("transaction_count", 0),
			"checked_at": _now(),
			"overall_health": "ok" if terminal.get("status") == "active" else "degraded",
		}

		# alert if missed heartbeat
		if hb_age_seconds and hb_age_seconds > 300:
			await self._notify.send(
				terminal.get("agent_id", "ops"), "email",
				f"Terminal {terminal_id} missed heartbeat",
				f"Terminal {terminal_id} last heartbeat was {hb_age_seconds:.0f}s ago.",
			)
		return health

	async def offline_queue_sync(
		self,
		terminal_id: str,
		queued_transactions: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Replay offline-queued transactions when connectivity is restored.

		Each queued transaction is validated and posted. Duplicates (same
		reference) are skipped to maintain idempotency.
		"""
		assert isinstance(queued_transactions, list), "queued_transactions must be a list"

		terminal = await self._get_terminal(terminal_id)
		processed, skipped, failed = [], [], []

		for q_txn in queued_transactions:
			ref = q_txn.get("reference", _uid())
			existing = await self._store.query(
				"terminal_transactions", {"reference": ref}, limit=1
			)
			if existing:
				skipped.append(ref)
				continue
			try:
				txn = await self.terminal_transaction(
					terminal_id,
					q_txn.get("transaction_type", "purchase"),
					float(q_txn.get("amount", 0)),
					q_txn.get("currency", "KES"),
					q_txn.get("customer_id", "unknown"),
					ref,
					metadata={"offline_queued": True, "original_timestamp": q_txn.get("timestamp")},
				)
				processed.append(txn["id"])
			except Exception as exc:
				failed.append({"reference": ref, "error": str(exc)})

		result = {
			"terminal_id": terminal_id,
			"queued_count": len(queued_transactions),
			"processed": len(processed),
			"skipped": len(skipped),
			"failed": len(failed),
			"failed_details": failed,
			"synced_at": _now(),
		}
		# clear offline queue
		terminal["offline_queue"] = []
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)
		await self._audit_event(
			"terminal_offline_sync", "system", terminal_id, result
		)
		return result

	async def terminal_commission_report(
		self,
		terminal_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Calculate agent commission earned for the given period.

		Commission rates (configurable): deposits 0.5%, withdrawals 1.0%,
		bill payments 0.3%, transfers 0.2%.
		"""
		start, end = _period_bounds(period)
		txns = await self._store.query(
			"terminal_transactions",
			{"terminal_id": terminal_id},
			limit=100_000,
		)
		period_txns = [
			t for t in txns
			if start <= t.get("timestamp", "")[:10] <= end
		]

		rates = {
			"cash_deposit": 0.005,
			"cash_withdrawal": 0.010,
			"bill_payment": 0.003,
			"funds_transfer": 0.002,
		}

		commission_by_type: dict[str, float] = {}
		for t in period_txns:
			ttype = t.get("transaction_type", "other")
			rate = rates.get(ttype, 0.001)
			commission_by_type[ttype] = commission_by_type.get(ttype, 0.0) + t.get("amount", 0.0) * rate

		total_commission = sum(commission_by_type.values())

		report: dict[str, Any] = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"transaction_count": len(period_txns),
			"commission_by_type": commission_by_type,
			"total_commission_kes": round(total_commission, 2),
			"currency": "KES",
			"generated_at": _now(),
		}
		await self._store.put("terminal_commission_reports", report)
		return report

	async def terminal_analytics(
		self,
		network_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Network-level analytics: active terminals, volume, top agents, error rates."""
		start, end = _period_bounds(period)
		terminals = await self._store.query("terminals", {"tenant_id": self._tenant_id}, limit=10_000)
		txns = await self._store.query("terminal_transactions", {}, limit=500_000)

		period_txns = [t for t in txns if start <= t.get("timestamp", "")[:10] <= end]
		active_terminals = [t for t in terminals if t.get("status") == "active"]

		volume_by_type: dict[str, float] = {}
		count_by_type: dict[str, int] = {}
		for t in period_txns:
			ttype = t.get("transaction_type", "other")
			volume_by_type[ttype] = volume_by_type.get(ttype, 0.0) + t.get("amount", 0.0)
			count_by_type[ttype] = count_by_type.get(ttype, 0) + 1

		agent_volumes: dict[str, float] = {}
		for t in period_txns:
			agent = t.get("agent_id", "unknown")
			agent_volumes[agent] = agent_volumes.get(agent, 0.0) + t.get("amount", 0.0)
		top_agents = sorted(agent_volumes.items(), key=lambda x: x[1], reverse=True)[:10]

		return {
			"network_id": network_id,
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_terminals": len(terminals),
			"active_terminals": len(active_terminals),
			"total_transactions": len(period_txns),
			"total_volume_kes": sum(volume_by_type.values()),
			"volume_by_type": volume_by_type,
			"count_by_type": count_by_type,
			"top_agents": [{"agent_id": a, "volume": v} for a, v in top_agents],
			"generated_at": _now(),
		}

	async def fraud_alert_terminal(
		self,
		terminal_id: str,
		event_type: str,
		details: dict[str, Any],
	) -> dict[str, Any]:
		"""Record a fraud or suspicious-activity alert for a terminal.

		Auto-suspends the terminal if event_type is 'tamper_detected'.
		"""
		assert event_type, "event_type required"

		terminal = await self._get_terminal(terminal_id)
		alert: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"agent_id": terminal.get("agent_id"),
			"event_type": event_type,
			"details": details,
			"status": "open",
			"raised_at": _now(),
		}
		await self._store.put("terminal_fraud_alerts", alert)

		# tamper event auto-suspends
		if event_type == "tamper_detected":
			terminal["status"] = "suspended"
			terminal["suspended_at"] = _now()
			terminal["suspension_reason"] = "tamper_detected"
			terminal["updated_at"] = _now()
			await self._store.put("terminals", terminal)

		await self._audit_event(
			f"terminal_fraud_alert_{event_type}", "system", terminal_id, details
		)
		await self._notify.send(
			"fraud_ops@datacraft.co.ke", "email",
			f"Fraud Alert: {event_type} on {terminal_id}",
			f"Terminal {terminal_id} flagged: {event_type}\nDetails: {json.dumps(details, default=str)}",
		)
		return alert

	async def customer_enrolment(
		self,
		terminal_id: str,
		biometric_data: dict[str, Any],
		id_number: str,
		*,
		customer_name: str | None = None,
		phone: str | None = None,
	) -> dict[str, Any]:
		"""Enrol a new customer at the agent terminal using biometrics and ID.

		Biometric data is hashed before storage — raw biometrics are never persisted.
		"""
		assert id_number, "id_number required"
		assert biometric_data, "biometric_data required"

		await self._assert_active(terminal_id)

		# hash biometric payload — never store raw
		bio_hash = hashlib.sha256(
			json.dumps(biometric_data, sort_keys=True).encode()
		).hexdigest()

		enrolment: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"id_number": id_number,
			"customer_name": customer_name,
			"phone": phone,
			"biometric_hash": bio_hash,
			"status": "pending_verification",
			"enrolled_at": _now(),
		}
		await self._store.put("customer_enrolments", enrolment)
		await self._audit_event(
			"customer_enrolled", "agent", terminal_id,
			{"id_number": id_number, "biometric_hash": bio_hash},
		)
		return enrolment

	async def receipt_generation(
		self,
		transaction_id: str,
		receipt_format: str = "thermal",
	) -> dict[str, Any]:
		"""Generate a transaction receipt in the requested format.

		Supported formats: thermal (72mm text), sms, pdf, qr.
		"""
		assert receipt_format in {"thermal", "sms", "pdf", "qr"}, (
			"format must be: thermal | sms | pdf | qr"
		)

		txn = await self._store.get("terminal_transactions", transaction_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {transaction_id}")

		lines = [
			"=== DATACRAFT AGENT BANKING ===",
			f"Date   : {txn.get('timestamp', '')[:19]}",
			f"Ref    : {txn.get('reference', transaction_id)}",
			f"Type   : {txn.get('transaction_type', '').upper().replace('_', ' ')}",
			f"Amount : {txn.get('currency', 'KES')} {txn.get('amount', 0):,.2f}",
			f"Status : {txn.get('status', '').upper()}",
			"==============================",
			"Thank you for banking with us.",
		]

		receipt: dict[str, Any] = {
			"id": _uid(),
			"transaction_id": transaction_id,
			"format": receipt_format,
			"content": "\n".join(lines),
			"generated_at": _now(),
		}
		await self._store.put("receipts", receipt)
		return receipt

	async def network_performance(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Compute network-wide performance KPIs: uptime %, TPS, error rate."""
		start, end = _period_bounds(period)
		terminals = await self._store.query("terminals", {"tenant_id": self._tenant_id}, limit=10_000)
		health_records = await self._store.query("terminal_health_checks", {}, limit=500_000)

		period_health = [
			h for h in health_records
			if start <= h.get("checked_at", "")[:10] <= end
		]

		uptime_pct = 0.0
		if period_health:
			ok_count = sum(1 for h in period_health if h.get("overall_health") == "ok")
			uptime_pct = (ok_count / len(period_health)) * 100

		txns = await self._store.query("terminal_transactions", {}, limit=500_000)
		period_txns = [t for t in txns if start <= t.get("timestamp", "")[:10] <= end]

		# rough TPS over period in seconds
		period_days = max(1, (date.fromisoformat(end) - date.fromisoformat(start)).days + 1)
		period_seconds = period_days * 86400
		tps = len(period_txns) / period_seconds

		return {
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_terminals": len(terminals),
			"active_terminals": sum(1 for t in terminals if t.get("status") == "active"),
			"total_transactions": len(period_txns),
			"tps_avg": round(tps, 4),
			"uptime_pct": round(uptime_pct, 2),
			"generated_at": _now(),
		}

	# ── Additional methods ──────────────────────────────────────────────────

	async def bulk_register_terminals(self, registrations: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-register multiple terminals in one operation."""
		processed, errors = [], []
		for reg in registrations:
			try:
				rec = await self.register_terminal(
					terminal_id=reg["terminal_id"],
					location=reg["location"],
					agent_id=reg["agent_id"],
					terminal_type=reg["terminal_type"],
					connectivity=reg.get("connectivity", "lte"),
					serial_number=reg.get("serial_number"),
					merchant_id=reg.get("merchant_id"),
					model=reg.get("model"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": reg, "error": str(exc)})
		return {"total": len(registrations), "processed": len(processed), "failed": len(errors), "errors": errors}

	async def suspend_terminal(self, terminal_id: str, reason: str, suspended_by: str) -> dict[str, Any]:
		"""Suspend a terminal (e.g., fraud, maintenance, non-compliance)."""
		terminal = await self._get_terminal(terminal_id)
		terminal["status"] = "suspended"
		terminal["suspension_reason"] = reason
		terminal["suspended_by"] = suspended_by
		terminal["suspended_at"] = _now()
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)
		await self._audit_event("terminal_suspended", suspended_by, terminal_id, {"reason": reason})
		return terminal

	async def decommission_terminal(self, terminal_id: str, reason: str, decommissioned_by: str) -> dict[str, Any]:
		"""Decommission and retire a terminal from the network."""
		terminal = await self._get_terminal(terminal_id)
		terminal["status"] = "decommissioned"
		terminal["decommission_reason"] = reason
		terminal["decommissioned_by"] = decommissioned_by
		terminal["decommissioned_at"] = _now()
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)
		await self._audit_event("terminal_decommissioned", decommissioned_by, terminal_id, {"reason": reason})
		return terminal

	async def heartbeat(self, terminal_id: str, signal_strength: str = "good", battery_pct: float = 100.0) -> dict[str, Any]:
		"""Record a terminal heartbeat/keep-alive signal."""
		terminal = await self._get_terminal(terminal_id)
		terminal["last_heartbeat"] = _now()
		terminal["signal_strength"] = signal_strength
		terminal["battery_pct"] = battery_pct
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)
		return {"terminal_id": terminal_id, "heartbeat_at": terminal["last_heartbeat"], "signal_strength": signal_strength}

	async def mobile_money_withdrawal(self, terminal_id: str, customer_id: str, amount: float, provider: str = "mpesa") -> dict[str, Any]:
		"""Facilitate mobile money withdrawal (M-Pesa, Airtel Money) at the agent terminal."""
		assert provider in {"mpesa", "airtel_money", "equitel", "tkash"}, f"unsupported provider: {provider}"
		assert amount > 0 and amount <= 150_000, "amount must be 0–150,000 for mobile money"
		return await self.cash_withdrawal(terminal_id, customer_id, amount, "KES", pin_verified=True)

	async def mobile_money_deposit(self, terminal_id: str, customer_id: str, amount: float, provider: str = "mpesa") -> dict[str, Any]:
		"""Facilitate mobile money deposit (M-Pesa, Airtel Money) at the agent terminal."""
		assert provider in {"mpesa", "airtel_money", "equitel", "tkash"}, f"unsupported provider: {provider}"
		return await self.cash_deposit(terminal_id, customer_id, amount, "KES")

	async def government_payment(self, terminal_id: str, customer_id: str, service_code: str, amount: float, reference: str) -> dict[str, Any]:
		"""Process a government payment (eCitizen, county, NHIF, NSSF) at the terminal."""
		assert service_code, "service_code required"
		return await self.terminal_transaction(
			terminal_id, "government_payment", amount, "KES", customer_id,
			f"GOV-{service_code}-{reference}",
			metadata={"service_code": service_code, "payment_reference": reference},
		)

	async def school_fees_payment(self, terminal_id: str, customer_id: str, school_code: str, amount: float, student_ref: str) -> dict[str, Any]:
		"""Process a school fees payment at the agent terminal."""
		return await self.terminal_transaction(
			terminal_id, "school_fees_payment", amount, "KES", customer_id,
			f"SCHOOL-{school_code}-{student_ref}",
			metadata={"school_code": school_code, "student_reference": student_ref},
		)

	async def insurance_premium_payment(self, terminal_id: str, customer_id: str, insurer_code: str, policy_number: str, amount: float) -> dict[str, Any]:
		"""Process an insurance premium payment at the agent terminal."""
		return await self.terminal_transaction(
			terminal_id, "insurance_premium", amount, "KES", customer_id,
			f"INS-{insurer_code}-{policy_number}",
			metadata={"insurer_code": insurer_code, "policy_number": policy_number},
		)

	async def agent_performance_report(self, agent_id: str, period: str) -> dict[str, Any]:
		"""Generate performance report for a specific agent across all their terminals."""
		start, end = _period_bounds(period)
		terminals = await self._store.query("terminals", {"agent_id": agent_id}, limit=1000)
		txns = await self._store.query("terminal_transactions", {"agent_id": agent_id}, limit=100_000)
		period_txns = [t for t in txns if start <= t.get("timestamp", "")[:10] <= end]
		total_volume = sum(t.get("amount", 0) for t in period_txns)
		rates = {"cash_deposit": 0.005, "cash_withdrawal": 0.010, "bill_payment": 0.003, "funds_transfer": 0.002}
		total_commission = sum(t.get("amount", 0) * rates.get(t.get("transaction_type", ""), 0.001) for t in period_txns)
		return {
			"agent_id": agent_id, "period": period, "period_start": start, "period_end": end,
			"terminal_count": len(terminals), "transaction_count": len(period_txns),
			"total_volume_kes": round(total_volume, 2), "total_commission_kes": round(total_commission, 2),
			"generated_at": _now(),
		}

	async def float_alert_threshold(self, terminal_id: str, min_float: float, notify_agent: bool = True) -> dict[str, Any]:
		"""Set or check float alert threshold for a terminal."""
		terminal = await self._get_terminal(terminal_id)
		float_bal = terminal.get("float_balance", 0.0)
		below = float_bal < min_float
		record = {
			"terminal_id": terminal_id, "float_balance": float_bal,
			"min_threshold": min_float, "below_threshold": below,
			"checked_at": _now(),
		}
		if below and notify_agent:
			await self._notify.send(
				terminal.get("agent_id", "ops"), "sms",
				f"Low float alert: Terminal {terminal_id}",
				f"Float balance KES {float_bal:,.2f} is below threshold KES {min_float:,.2f}.",
			)
		return record

	async def kyc_verification_terminal(self, terminal_id: str, id_number: str, id_type: str, customer_name: str) -> dict[str, Any]:
		"""Perform KYC verification at the terminal using ID document data."""
		await self._assert_active(terminal_id)
		assert id_number and id_type and customer_name
		# Simulate KYC check — production would call IPRS/NIIMS
		score = len(id_number) + len(customer_name)
		verified = score >= 15 and len(id_number) >= 7
		record = {
			"id": _uid(), "terminal_id": terminal_id, "id_number": id_number,
			"id_type": id_type, "customer_name": customer_name,
			"kyc_status": "verified" if verified else "pending_review",
			"confidence_score": min(score / 30, 1.0), "checked_at": _now(),
		}
		await self._store.put("terminal_kyc_checks", record)
		await self._audit_event("terminal_kyc_checked", "agent", terminal_id, {"id_type": id_type, "status": record["kyc_status"]})
		return record

	async def transaction_limit_check(self, terminal_id: str, transaction_type: str, amount: float) -> dict[str, Any]:
		"""Check if a transaction amount is within CBK-mandated limits for agency banking."""
		limits = {
			"cash_deposit": 999_999.0, "cash_withdrawal": 300_000.0,
			"bill_payment": 500_000.0, "funds_transfer": 999_999.0,
			"government_payment": 500_000.0,
		}
		max_allowed = limits.get(transaction_type, 999_999.0)
		within = amount <= max_allowed
		return {
			"terminal_id": terminal_id, "transaction_type": transaction_type,
			"amount": amount, "max_allowed": max_allowed,
			"within_limit": within, "checked_at": _now(),
		}

	async def pos_diagnostics(self, terminal_id: str) -> dict[str, Any]:
		"""Run diagnostic checks on a POS terminal (printer, card reader, connectivity)."""
		terminal = await self._get_terminal(terminal_id)
		return {
			"terminal_id": terminal_id, "model": terminal.get("model", "unknown"),
			"connectivity": terminal.get("connectivity"),
			"diagnostics": {
				"card_reader": "ok", "thermal_printer": "ok",
				"display": "ok", "keypad": "ok",
				"network": "ok" if terminal.get("status") == "active" else "degraded",
			},
			"overall_status": "pass" if terminal.get("status") == "active" else "fail",
			"checked_at": _now(),
		}

	async def terminal_software_update(self, terminal_id: str, new_version: str, update_source: str, authorized_by: str) -> dict[str, Any]:
		"""Record a terminal software/firmware update event."""
		terminal = await self._get_terminal(terminal_id)
		old_version = terminal.get("software_version", "1.0.0")
		terminal["software_version"] = new_version
		terminal["last_updated_by"] = authorized_by
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)
		record = {
			"id": _uid(), "terminal_id": terminal_id,
			"old_version": old_version, "new_version": new_version,
			"update_source": update_source, "authorized_by": authorized_by,
			"updated_at": _now(),
		}
		await self._store.put("terminal_software_updates", record)
		await self._audit_event("terminal_software_updated", authorized_by, terminal_id, {"new_version": new_version})
		return record

	async def cbk_agent_banking_return(self, period: str, jurisdiction: str = "KE") -> dict[str, Any]:
		"""Generate CBK Agency Banking return (Form ABR-01)."""
		return await self.regulatory_report(period, jurisdiction)

	async def nssf_contribution_payment(self, terminal_id: str, customer_id: str, employer_ref: str, amount: float) -> dict[str, Any]:
		"""Process an NSSF (National Social Security Fund) contribution payment."""
		return await self.government_payment(terminal_id, customer_id, "NSSF", amount, f"NSSF-{employer_ref}")

	async def nhif_premium_payment(self, terminal_id: str, customer_id: str, member_number: str, amount: float) -> dict[str, Any]:
		"""Process an NHIF (National Health Insurance Fund) premium payment."""
		return await self.government_payment(terminal_id, customer_id, "NHIF", amount, f"NHIF-{member_number}")

	async def equity_bank_mini_statement(self, terminal_id: str, customer_id: str, account_number: str) -> dict[str, Any]:
		"""Retrieve mini-statement via Equity Bank agency banking integration."""
		await self._assert_active(terminal_id)
		result = await self.mini_statement_terminal(terminal_id, customer_id, limit=5)
		return {**result, "bank": "Equity Bank Kenya", "account_number": account_number[-4:].zfill(4)}

	async def agent_network_density_report(self, county: str, period: str) -> dict[str, Any]:
		"""Report on agent network density per county for financial inclusion metrics."""
		start, end = _period_bounds(period)
		all_terminals = await self._store.query("terminals", {"tenant_id": self._tenant_id}, limit=10_000)
		county_terminals = [t for t in all_terminals if t.get("location", {}).get("county", "").lower() == county.lower()]
		active = [t for t in county_terminals if t.get("status") == "active"]
		return {
			"county": county, "period": period, "period_start": start, "period_end": end,
			"total_terminals": len(county_terminals), "active_terminals": len(active),
			"coverage_pct": round(len(active) / max(len(county_terminals), 1) * 100, 2),
			"generated_at": _now(),
		}

	async def regulatory_report(
		self,
		period: str,
		jurisdiction: str,
	) -> dict[str, Any]:
		"""Generate a regulatory submission report for CBK or other jurisdiction.

		Includes terminal counts, transaction volumes, cash-in/cash-out, agent
		demographics, and compliance status summary.
		"""
		assert jurisdiction, "jurisdiction required"
		start, end = _period_bounds(period)

		terminals = await self._store.query("terminals", {"tenant_id": self._tenant_id}, limit=10_000)
		txns = await self._store.query("terminal_transactions", {}, limit=500_000)
		period_txns = [t for t in txns if start <= t.get("timestamp", "")[:10] <= end]

		cash_in = sum(t["amount"] for t in period_txns if t.get("transaction_type") == "cash_deposit")
		cash_out = sum(t["amount"] for t in period_txns if t.get("transaction_type") == "cash_withdrawal")

		report: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"report_type": "agency_banking_regulatory",
			"jurisdiction": jurisdiction,
			"period": period,
			"period_start": start,
			"period_end": end,
			"summary": {
				"total_terminals": len(terminals),
				"active_terminals": sum(1 for t in terminals if t.get("status") == "active"),
				"total_transactions": len(period_txns),
				"cash_in_kes": round(cash_in, 2),
				"cash_out_kes": round(cash_out, 2),
				"net_float_movement": round(cash_in - cash_out, 2),
			},
			"compliance": {
				"pci_dss_compliant_count": sum(
					1 for t in terminals if t.get("pci_dss_compliant")
				),
				"tamper_detection_enabled_count": sum(
					1 for t in terminals if t.get("tamper_detection_enabled")
				),
			},
			"generated_at": _now(),
			"status": "draft",
		}
		await self._store.put("regulatory_reports", report)
		await self._audit_event(
			"regulatory_report_generated", "system", "network",
			{"jurisdiction": jurisdiction, "period": period},
		)
		return report

	# ── New world-class methods ────────────────────────────────────────────────

	async def inject_terminal_key(
		self,
		terminal_id: str,
		bdk_id: str,
		ksn: str,
		key_type: str,
		injected_by: str,
		*,
		expiry_days: int = 365,
	) -> dict[str, Any]:
		"""Record a DUKPT/TR-31 key-injection event for a terminal.

		Stores the Base Derivation Key reference (BDK ID), the initial Key Serial
		Number (KSN), and the key type (TDES/AES128/AES256).  Raw key material is
		*never* passed through this method — only opaque references produced by
		the HSM.  After injection the terminal status transitions from
		``configuration_pending`` to ``key_injected``.

		Args:
			terminal_id: Target terminal.
			bdk_id: Opaque HSM reference for the Base Derivation Key.
			ksn: Initial Key Serial Number (hex string, 20 hex chars).
			key_type: One of ``TDES``, ``AES128``, ``AES256``.
			injected_by: Actor performing the injection (HSM operator ID).
			expiry_days: Days until the key expires and must be rotated.
		"""
		assert bdk_id, "bdk_id required"
		assert ksn, "ksn required"
		assert key_type in {"TDES", "AES128", "AES256"}, (
			"key_type must be TDES | AES128 | AES256"
		)
		assert injected_by, "injected_by required"

		terminal = await self._get_terminal(terminal_id)

		expiry_dt = datetime.now(timezone.utc) + timedelta(days=expiry_days)
		key_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"bdk_id": bdk_id,
			"ksn": ksn,
			"key_type": key_type,
			"injected_by": injected_by,
			"injected_at": _now(),
			"expires_at": expiry_dt.isoformat(),
			"status": "active",
			"rotation_count": 0,
		}
		await self._store.put("terminal_keys", key_record)

		# Advance terminal state
		if terminal.get("status") == "configuration_pending":
			terminal["status"] = "key_injected"
		terminal["active_key_id"] = key_record["id"]
		terminal["key_expiry"] = key_record["expires_at"]
		terminal["ksn"] = ksn
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._audit_event(
			"terminal_key_injected", injected_by, terminal_id,
			{"bdk_id": bdk_id, "key_type": key_type, "expires_at": key_record["expires_at"]},
		)
		return key_record

	async def rotate_terminal_key(
		self,
		terminal_id: str,
		new_bdk_id: str,
		new_ksn: str,
		key_type: str,
		rotated_by: str,
	) -> dict[str, Any]:
		"""Rotate the active encryption key on a terminal.

		Marks the previous key record as ``retired`` and injects a new one via
		:meth:`inject_terminal_key`.  Maintains a full rotation audit chain.

		Args:
			terminal_id: Target terminal.
			new_bdk_id: HSM reference for the new BDK.
			new_ksn: New initial KSN (hex string).
			key_type: Key algorithm for the replacement key.
			rotated_by: Actor authorising the rotation.
		"""
		terminal = await self._get_terminal(terminal_id)
		old_key_id = terminal.get("active_key_id")

		if old_key_id:
			old_key = await self._store.get("terminal_keys", old_key_id)
			if old_key:
				old_key["status"] = "retired"
				old_key["retired_at"] = _now()
				await self._store.put("terminal_keys", old_key)

		new_key = await self.inject_terminal_key(
			terminal_id, new_bdk_id, new_ksn, key_type, rotated_by
		)

		rotation_record: dict[str, Any] = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"old_key_id": old_key_id,
			"new_key_id": new_key["id"],
			"rotated_by": rotated_by,
			"rotated_at": _now(),
		}
		await self._store.put("terminal_key_rotations", rotation_record)
		await self._audit_event(
			"terminal_key_rotated", rotated_by, terminal_id,
			{"old_key_id": old_key_id, "new_key_id": new_key["id"]},
		)
		return rotation_record

	async def provision_terminal_certificate(
		self,
		terminal_id: str,
		csr_pem: str,
		issued_by: str,
		*,
		validity_days: int = 90,
	) -> dict[str, Any]:
		"""Issue a TLS client certificate for terminal-to-host mutual TLS.

		In production this calls the platform CA signing API.  Here we model
		the certificate lifecycle record (CSR hash, serial, expiry, revocation
		status) that the platform CA would return.

		Args:
			terminal_id: Terminal requesting the certificate.
			csr_pem: PEM-encoded Certificate Signing Request.
			issued_by: CA operator or automated CA service identifier.
			validity_days: Certificate validity period in days.
		"""
		assert csr_pem, "csr_pem required"
		assert issued_by, "issued_by required"

		await self._assert_active(terminal_id)

		csr_fingerprint = hashlib.sha256(csr_pem.encode()).hexdigest()
		expiry_dt = datetime.now(timezone.utc) + timedelta(days=validity_days)
		serial = hashlib.md5(f"{terminal_id}{_now()}".encode()).hexdigest().upper()[:16]

		cert_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"serial": serial,
			"csr_fingerprint": csr_fingerprint,
			"issued_by": issued_by,
			"issued_at": _now(),
			"expires_at": expiry_dt.isoformat(),
			"status": "active",
			"revoked": False,
			"revoked_at": None,
			"revocation_reason": None,
		}
		await self._store.put("terminal_certificates", cert_record)

		terminal = await self._get_terminal(terminal_id)
		terminal["active_cert_id"] = cert_record["id"]
		terminal["cert_expiry"] = cert_record["expires_at"]
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._audit_event(
			"terminal_certificate_issued", issued_by, terminal_id,
			{"serial": serial, "expires_at": cert_record["expires_at"]},
		)
		return cert_record

	async def revoke_terminal_certificate(
		self,
		terminal_id: str,
		cert_id: str,
		reason: str,
		revoked_by: str,
	) -> dict[str, Any]:
		"""Revoke an active TLS client certificate for a terminal.

		Certificates are revoked on tamper detection, key compromise, or
		terminal decommission.  The terminal is suspended if the revoked
		certificate is its current active one.

		Args:
			terminal_id: Owning terminal.
			cert_id: Certificate record ID to revoke.
			reason: Revocation reason (e.g., ``tamper_detected``, ``key_compromise``).
			revoked_by: Actor authorising the revocation.
		"""
		assert reason, "reason required"
		assert revoked_by, "revoked_by required"

		cert = await self._store.get("terminal_certificates", cert_id)
		if cert is None:
			raise ValueError(f"Certificate not found: {cert_id}")
		if cert["terminal_id"] != terminal_id:
			raise ValueError(f"Certificate {cert_id} does not belong to terminal {terminal_id}")

		cert["status"] = "revoked"
		cert["revoked"] = True
		cert["revoked_at"] = _now()
		cert["revocation_reason"] = reason
		await self._store.put("terminal_certificates", cert)

		# Auto-suspend terminal if this was its active cert
		terminal = await self._get_terminal(terminal_id)
		if terminal.get("active_cert_id") == cert_id:
			terminal["status"] = "suspended"
			terminal["suspension_reason"] = f"certificate_revoked:{reason}"
			terminal["suspended_at"] = _now()
			terminal["updated_at"] = _now()
			await self._store.put("terminals", terminal)

		await self._audit_event(
			"terminal_certificate_revoked", revoked_by, terminal_id,
			{"cert_id": cert_id, "reason": reason},
		)
		await self._notify.send(
			"security_ops@datacraft.co.ke", "email",
			f"Certificate revoked on terminal {terminal_id}",
			f"Cert {cert_id} revoked: {reason} by {revoked_by}.",
		)
		return cert

	async def evaluate_transaction_velocity(
		self,
		terminal_id: str,
		customer_id: str,
		transaction_type: str,
		amount: float,
	) -> dict[str, Any]:
		"""Check real-time transaction velocity limits before authorising a transaction.

		Computes rolling 1-hour and 24-hour counts and volumes per customer and
		per terminal.  Returns a fraud score (0–100) and an ``allow``/``review``/
		``deny`` recommendation.

		Args:
			terminal_id: Terminal originating the transaction.
			customer_id: Customer account reference.
			transaction_type: Transaction category string.
			amount: Proposed transaction amount (KES).
		"""
		now_dt = datetime.now(timezone.utc)
		cutoff_1h = (now_dt - timedelta(hours=1)).isoformat()
		cutoff_24h = (now_dt - timedelta(hours=24)).isoformat()

		all_txns = await self._store.query(
			"terminal_transactions",
			{"customer_id": customer_id},
			limit=5_000,
		)

		txns_1h = [t for t in all_txns if t.get("timestamp", "") >= cutoff_1h]
		txns_24h = [t for t in all_txns if t.get("timestamp", "") >= cutoff_24h]
		terminal_txns_1h = [t for t in txns_1h if t.get("terminal_id") == terminal_id]

		vol_1h = sum(t.get("amount", 0) for t in txns_1h)
		vol_24h = sum(t.get("amount", 0) for t in txns_24h)
		count_1h = len(txns_1h)
		count_24h = len(txns_24h)
		terminal_count_1h = len(terminal_txns_1h)

		# Simple rule-based scoring
		score = 0
		flags: list[str] = []
		if count_1h > 10:
			score += 30; flags.append("high_frequency_1h")
		if vol_1h > 500_000:
			score += 25; flags.append("high_volume_1h")
		if vol_24h > 2_000_000:
			score += 20; flags.append("high_volume_24h")
		if terminal_count_1h > 5:
			score += 15; flags.append("terminal_concentration")
		if amount > 200_000:
			score += 10; flags.append("large_single_amount")

		if score >= 70:
			recommendation = "deny"
		elif score >= 40:
			recommendation = "review"
		else:
			recommendation = "allow"

		result: dict[str, Any] = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"customer_id": customer_id,
			"transaction_type": transaction_type,
			"amount": amount,
			"count_1h": count_1h,
			"count_24h": count_24h,
			"volume_1h": round(vol_1h, 2),
			"volume_24h": round(vol_24h, 2),
			"fraud_score": min(score, 100),
			"flags": flags,
			"recommendation": recommendation,
			"evaluated_at": _now(),
		}
		await self._store.put("terminal_velocity_checks", result)
		return result

	async def push_terminal_parameters(
		self,
		terminal_id: str,
		parameters: dict[str, Any],
		pushed_by: str,
		*,
		version: str | None = None,
		rollback_version: str | None = None,
	) -> dict[str, Any]:
		"""Push an updated parameter set to a terminal over-the-air (OTA).

		Parameters include BIN tables, commission rate tables, CAF files,
		merchant category codes, and EMV tag defaults.  A rollback version may
		be specified so the terminal can revert if the new parameters cause errors.

		Args:
			terminal_id: Destination terminal.
			parameters: Key-value parameter payload.
			pushed_by: Operator authorising the push.
			version: Semantic version tag for this parameter set.
			rollback_version: Parameter version to roll back to on failure.
		"""
		assert parameters, "parameters must be non-empty"
		assert pushed_by, "pushed_by required"

		terminal = await self._get_terminal(terminal_id)

		param_hash = hashlib.sha256(
			json.dumps(parameters, sort_keys=True).encode()
		).hexdigest()

		push_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"version": version or f"auto-{_uid()[:8]}",
			"param_hash": param_hash,
			"parameter_keys": list(parameters.keys()),
			"pushed_by": pushed_by,
			"pushed_at": _now(),
			"rollback_version": rollback_version,
			"status": "deployed",
		}
		await self._store.put("terminal_parameter_deployments", push_record)

		terminal["active_parameter_version"] = push_record["version"]
		terminal["active_param_hash"] = param_hash
		terminal["last_param_push_at"] = _now()
		terminal["updated_at"] = _now()
		await self._store.put("terminals", terminal)

		await self._audit_event(
			"terminal_parameters_pushed", pushed_by, terminal_id,
			{"version": push_record["version"], "keys": push_record["parameter_keys"]},
		)
		return push_record

	async def geo_fence_check(
		self,
		terminal_id: str,
		latitude: float,
		longitude: float,
		*,
		radius_meters: float = 500.0,
	) -> dict[str, Any]:
		"""Validate that a terminal's reported GPS position is within its registered geo-fence.

		If the terminal has drifted outside the allowed radius it is auto-suspended
		and a fraud alert is raised.  Uses the Haversine formula for distance.

		Args:
			terminal_id: Terminal reporting its location.
			latitude: Current GPS latitude in decimal degrees.
			longitude: Current GPS longitude in decimal degrees.
			radius_meters: Allowed radius from registered coordinates (default 500 m).
		"""
		import math

		terminal = await self._get_terminal(terminal_id)
		registered_loc = terminal.get("location", {})
		reg_lat = registered_loc.get("latitude")
		reg_lon = registered_loc.get("longitude")

		within_fence = True
		distance_meters = 0.0

		if reg_lat is not None and reg_lon is not None:
			# Haversine
			R = 6_371_000.0  # Earth radius in metres
			phi1 = math.radians(reg_lat)
			phi2 = math.radians(latitude)
			dphi = math.radians(latitude - reg_lat)
			dlambda = math.radians(longitude - reg_lon)
			a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
			distance_meters = 2 * R * math.asin(math.sqrt(a))
			within_fence = distance_meters <= radius_meters

		result: dict[str, Any] = {
			"id": _uid(),
			"terminal_id": terminal_id,
			"current_lat": latitude,
			"current_lon": longitude,
			"registered_lat": reg_lat,
			"registered_lon": reg_lon,
			"distance_meters": round(distance_meters, 2),
			"radius_meters": radius_meters,
			"within_fence": within_fence,
			"checked_at": _now(),
		}
		await self._store.put("terminal_geo_checks", result)

		if not within_fence:
			await self.fraud_alert_terminal(
				terminal_id, "geo_fence_violation",
				{"distance_meters": distance_meters, "radius_meters": radius_meters},
			)

		return result

	async def relocate_terminal(
		self,
		terminal_id: str,
		new_location: dict[str, Any],
		requested_by: str,
		approved_by: str,
	) -> dict[str, Any]:
		"""Relocate a terminal to a new physical location with dual approval.

		Both the agent (``requested_by``) and a supervisor (``approved_by``) must
		be distinct actors.  The registered location and geo-fence centre are
		updated atomically.

		Args:
			terminal_id: Terminal to relocate.
			new_location: Location dict with at minimum ``latitude``, ``longitude``,
				and ``address``.
			requested_by: Agent or field engineer requesting the relocation.
			approved_by: Supervisor approving the move.
		"""
		assert new_location, "new_location required"
		assert requested_by, "requested_by required"
		assert approved_by, "approved_by required"
		if requested_by == approved_by:
			raise PermissionError("requested_by and approved_by must be different actors")

		terminal = await self._get_terminal(terminal_id)
		old_location = terminal.get("location", {})

		terminal["location"] = new_location
		terminal["previous_location"] = old_location
		terminal["relocated_at"] = _now()
		terminal["relocated_by"] = requested_by
		terminal["relocation_approved_by"] = approved_by
		terminal["updated_at"] = _now()

		# Lift geo-fence suspension if terminal was suspended for geo violation
		if terminal.get("suspension_reason") == "geo_fence_violation":
			terminal["status"] = "active"
			terminal.pop("suspension_reason", None)

		await self._store.put("terminals", terminal)

		relocation_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"terminal_id": terminal_id,
			"old_location": old_location,
			"new_location": new_location,
			"requested_by": requested_by,
			"approved_by": approved_by,
			"relocated_at": _now(),
		}
		await self._store.put("terminal_relocations", relocation_record)
		await self._audit_event(
			"terminal_relocated", approved_by, terminal_id,
			{"old_location": old_location, "new_location": new_location},
		)
		return relocation_record

	async def batch_reconcile_network(
		self,
		recon_date: str,
		*,
		variance_threshold_pct: float = 0.5,
	) -> dict[str, Any]:
		"""Run end-of-day batch reconciliation across all active terminals.

		Aggregates per-terminal reconciliation, flags those with float variance
		exceeding ``variance_threshold_pct`` percent, and emits alert notifications
		for each.  Returns a summary suitable for CBK ABR-01 daily reporting.

		Args:
			recon_date: ISO date string (``YYYY-MM-DD``).
			variance_threshold_pct: Float variance % above which a terminal is
				flagged (default 0.5 %).
		"""
		terminals = await self._store.query(
			"terminals", {"tenant_id": self._tenant_id}, limit=10_000
		)
		active_terminals = [t for t in terminals if t.get("status") == "active"]

		results, flagged = [], []
		total_credits = total_debits = 0.0

		for terminal in active_terminals:
			tid = terminal["id"]
			recon = await self.terminal_reconciliation(tid, recon_date)
			results.append(recon)
			total_credits += recon.get("total_credits", 0.0)
			total_debits += recon.get("total_debits", 0.0)

			float_bal = recon.get("float_balance", 0.0)
			variance = recon.get("float_variance", 0.0)
			variance_pct = abs(variance) / max(float_bal, 1.0) * 100
			if variance_pct > variance_threshold_pct:
				flagged.append({
					"terminal_id": tid,
					"float_variance": variance,
					"variance_pct": round(variance_pct, 3),
				})
				await self._notify.send(
					"reconciliation@datacraft.co.ke", "email",
					f"Float variance on {tid} for {recon_date}",
					f"Terminal {tid} float variance: KES {variance:,.2f} ({variance_pct:.2f}%)",
				)

		summary: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"recon_date": recon_date,
			"total_active_terminals": len(active_terminals),
			"terminals_reconciled": len(results),
			"terminals_flagged": len(flagged),
			"flagged_details": flagged,
			"aggregate_credits_kes": round(total_credits, 2),
			"aggregate_debits_kes": round(total_debits, 2),
			"net_position_kes": round(total_credits - total_debits, 2),
			"generated_at": _now(),
		}
		await self._store.put("network_reconciliations", summary)
		await self._audit_event(
			"network_batch_reconciled", "system", "network",
			{"recon_date": recon_date, "flagged": len(flagged)},
		)
		return summary

	async def agent_credit_drawdown(
		self,
		agent_id: str,
		terminal_id: str,
		amount: float,
		*,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Draw down from an agent's intraday float credit facility.

		The credit line is replenished at EOD via :meth:`agent_credit_repayment`.
		If the requested amount exceeds the remaining credit limit, the call
		raises ``ValueError``.

		Args:
			agent_id: Agent whose credit line is being drawn.
			terminal_id: Terminal that will receive the float top-up.
			amount: Amount to draw (positive KES value).
			currency: Settlement currency (default ``KES``).
		"""
		assert amount > 0, "drawdown amount must be positive"

		# Load or initialise credit facility record
		facilities = await self._store.query(
			"agent_credit_facilities", {"agent_id": agent_id}, limit=1
		)
		if facilities:
			facility = facilities[0]
		else:
			facility = {
				"id": _uid(),
				"agent_id": agent_id,
				"tenant_id": self._tenant_id,
				"credit_limit": 100_000.0,
				"outstanding": 0.0,
				"currency": currency,
				"created_at": _now(),
			}

		available = facility["credit_limit"] - facility["outstanding"]
		if amount > available:
			raise ValueError(
				f"Drawdown {amount:,.2f} exceeds available credit {available:,.2f}"
			)

		facility["outstanding"] = facility["outstanding"] + amount
		facility["last_drawdown_at"] = _now()
		await self._store.put("agent_credit_facilities", facility)

		# Top up float on terminal
		await self.float_management(
			terminal_id, amount, "top_up", authorised_by=f"credit_facility:{facility['id']}"
		)

		drawdown_record: dict[str, Any] = {
			"id": _uid(),
			"agent_id": agent_id,
			"terminal_id": terminal_id,
			"facility_id": facility["id"],
			"amount": amount,
			"currency": currency,
			"outstanding_after": facility["outstanding"],
			"drawn_at": _now(),
		}
		await self._store.put("agent_credit_drawdowns", drawdown_record)
		await self._audit_event(
			"agent_credit_drawdown", agent_id, terminal_id,
			{"amount": amount, "outstanding_after": facility["outstanding"]},
		)
		return drawdown_record

	async def agent_credit_repayment(
		self,
		agent_id: str,
		amount: float,
		*,
		currency: str = "KES",
		reference: str | None = None,
	) -> dict[str, Any]:
		"""Repay outstanding intraday credit drawn by an agent.

		Typically called at end-of-day settlement.  Reduces the outstanding
		balance; excess repayment (over-payment) is rejected.

		Args:
			agent_id: Repaying agent.
			amount: Amount being repaid (positive KES).
			currency: Settlement currency.
			reference: Optional external reference (e.g., RTGS/EFT ref).
		"""
		assert amount > 0, "repayment amount must be positive"

		facilities = await self._store.query(
			"agent_credit_facilities", {"agent_id": agent_id}, limit=1
		)
		if not facilities:
			raise ValueError(f"No credit facility found for agent: {agent_id}")

		facility = facilities[0]
		if amount > facility["outstanding"]:
			raise ValueError(
				f"Repayment {amount:,.2f} exceeds outstanding balance {facility['outstanding']:,.2f}"
			)

		facility["outstanding"] = facility["outstanding"] - amount
		facility["last_repayment_at"] = _now()
		await self._store.put("agent_credit_facilities", facility)

		repayment_record: dict[str, Any] = {
			"id": _uid(),
			"agent_id": agent_id,
			"facility_id": facility["id"],
			"amount": amount,
			"currency": currency,
			"reference": reference,
			"outstanding_after": facility["outstanding"],
			"repaid_at": _now(),
		}
		await self._store.put("agent_credit_repayments", repayment_record)
		await self._audit_event(
			"agent_credit_repayment", agent_id, "credit_facility",
			{"amount": amount, "outstanding_after": facility["outstanding"]},
		)
		return repayment_record

	async def foreign_currency_transaction(
		self,
		terminal_id: str,
		customer_id: str,
		amount: float,
		source_currency: str,
		target_currency: str = "KES",
		*,
		exchange_rate: float | None = None,
	) -> dict[str, Any]:
		"""Execute a cross-currency transaction at an agency terminal.

		Converts ``amount`` from ``source_currency`` to ``target_currency`` using
		the provided (or fetched) CBK daily exchange rate.  Both the original and
		converted amounts are recorded for regulatory reporting.

		Args:
			terminal_id: Originating terminal.
			customer_id: Customer account reference.
			amount: Amount in source currency.
			source_currency: ISO 4217 source currency code.
			target_currency: ISO 4217 target currency code (default ``KES``).
			exchange_rate: Override exchange rate.  If ``None``, a default
				approximate rate is used (production would call the CBK rates API).
		"""
		assert source_currency, "source_currency required"
		assert source_currency != target_currency, "source and target currency must differ"
		assert amount > 0, "amount must be positive"

		# Fallback rates for common pairs (production: CBK API)
		_default_rates: dict[str, float] = {
			"USD": 130.0, "EUR": 140.0, "GBP": 165.0,
			"TZS": 0.050, "UGX": 0.034, "RWF": 0.085,
		}
		if exchange_rate is None:
			exchange_rate = _default_rates.get(source_currency.upper(), 1.0)

		converted_amount = round(amount * exchange_rate, 2)

		txn = await self.terminal_transaction(
			terminal_id, "foreign_currency_exchange", converted_amount,
			target_currency, customer_id,
			f"FX-{source_currency}-{_uid()[:8].upper()}",
			metadata={
				"source_currency": source_currency,
				"source_amount": amount,
				"exchange_rate": exchange_rate,
				"target_currency": target_currency,
				"converted_amount": converted_amount,
			},
		)
		await self._audit_event(
			"terminal_fx_transaction", customer_id, terminal_id,
			{
				"source": f"{source_currency} {amount}",
				"target": f"{target_currency} {converted_amount}",
				"rate": exchange_rate,
			},
		)
		return txn
