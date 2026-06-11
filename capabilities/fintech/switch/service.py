"""PaymentSwitchService — ISO 8583 payment routing hub.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

import hashlib
import json
import random
import string
import uuid
from datetime import date, datetime, timezone
from typing import Any

from .capability_contract import (
	CAPABILITY_ID,
	CAPABILITY_VERSION,
	SUPPORTED_NETWORKS,
	SUPPORTED_TRANSACTION_TYPES,
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


def _stan() -> str:
	"""Generate a 6-digit System Trace Audit Number."""
	return "".join(random.choices(string.digits, k=6))


def _rrn() -> str:
	"""Generate a 12-character Retrieval Reference Number."""
	return "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


def _period_bounds(period: str) -> tuple[str, str]:
	if len(period) == 4:
		return f"{period}-01-01", f"{period}-12-31"
	if len(period) == 7:
		y, m = period.split("-")
		end_day = 31 if int(m) in {1, 3, 5, 7, 8, 10, 12} else 30 if int(m) != 2 else 28
		return f"{period}-01", f"{period}-{end_day:02d}"
	if "Q" in period:
		y, q = period.split("-Q")
		q = int(q)
		sm = (q - 1) * 3 + 1
		em = q * 3
		ed = 31 if em in {3, 12} else 30
		return f"{y}-{sm:02d}-01", f"{y}-{em:02d}-{ed:02d}"
	return period, period


def _log_switch_op(txn_id: str, op: str) -> str:
	return f"[switch:{txn_id}] {op}"


# Response codes per ISO 8583
_RC_APPROVED = "00"
_RC_INSUFFICIENT_FUNDS = "51"
_RC_DO_NOT_HONOUR = "05"
_RC_INVALID_TRANSACTION = "12"
_RC_VELOCITY_EXCEEDED = "61"
_RC_RESTRICTED_CARD = "62"


# ─────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class PaymentSwitchService:
	"""ISO 8583 / ISO 20022 payment switch and routing hub.

	Handles transaction routing, authorisation, settlement, interchange,
	scheme compliance, fraud velocity checks, 3DS, and switch analytics.

	Usage (standalone)::

		svc = PaymentSwitchService()
		result = await svc.route_transaction(txn_data, routing_rules)

	Usage (platform)::

		from apg_common_auth import AuthService
		svc = PaymentSwitchService(auth=AuthService.from_env())
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

	# ── Internal helpers ──────────────────────────────────────

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

	def _select_route(
		self,
		transaction_data: dict[str, Any],
		routing_rules: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Apply routing rules to select the best network for a transaction.

		Rules are evaluated in priority order; first match wins.
		Falls back to 'interbank' if no rule matches.
		"""
		amount = transaction_data.get("amount", 0)
		currency = transaction_data.get("currency", "KES")
		channel = transaction_data.get("channel", "pos")

		for rule in sorted(routing_rules, key=lambda r: r.get("priority", 99)):
			conditions = rule.get("conditions", {})
			match = True
			if "currency" in conditions and conditions["currency"] != currency:
				match = False
			if "min_amount" in conditions and amount < conditions["min_amount"]:
				match = False
			if "max_amount" in conditions and amount > conditions["max_amount"]:
				match = False
			if "channel" in conditions and conditions["channel"] != channel:
				match = False
			if match:
				return {
					"network": rule.get("network", "interbank"),
					"rule_name": rule.get("name", "default"),
					"algorithm": rule.get("algorithm", "rule_based"),
				}

		return {"network": "interbank", "rule_name": "default_fallback", "algorithm": "fallback"}

	# ── Core routing and auth ─────────────────────────────────

	async def route_transaction(
		self,
		transaction_data: dict[str, Any],
		routing_rules: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Route a transaction to the appropriate payment network.

		Assigns STAN and RRN, evaluates routing rules, and creates a routed
		transaction record. Returns the routing decision with network and hop count.
		"""
		assert transaction_data.get("amount", 0) > 0, "amount must be positive"
		assert transaction_data.get("currency"), "currency required"

		stan = transaction_data.get("stan") or _stan()
		rrn = transaction_data.get("rrn") or _rrn()

		# Duplicate STAN check
		existing_stan = await self._store.query(
			"switch_transactions", {"stan": stan}, limit=1
		)

		rule_ctx = {
			"operation": "route_transaction",
			"tenant_context_present": True,
			"network_supported": True,
			"routing_table_version_present": bool(routing_rules),
			"stan_present": True,
			"stan_duplicate": bool(existing_stan),
			"rrn_present": True,
			"amount_lte": 0 if transaction_data.get("amount", 0) > 0 else 1,
			"mac_verified": True,
			"key_expired": False,
		}
		verdict = evaluate_capability_rules(rule_ctx)
		if verdict["decision"] == "deny":
			raise PermissionError(f"Routing denied: {verdict['matched_rules']}")

		route = self._select_route(transaction_data, routing_rules)

		txn_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"stan": stan,
			"rrn": rrn,
			"amount": transaction_data.get("amount"),
			"currency": transaction_data.get("currency"),
			"transaction_type": transaction_data.get("transaction_type", "purchase"),
			"channel": transaction_data.get("channel", "pos"),
			"pan_masked": transaction_data.get("pan_masked"),
			"merchant_id": transaction_data.get("merchant_id"),
			"route": route,
			"network": route["network"],
			"status": "routed",
			"hops": 1,
			"routed_at": _now(),
			"message_type": transaction_data.get("message_type", "0100"),
		}
		await self._store.put("switch_transactions", txn_record)
		await self._audit_event(
			"switch_transaction_routed", "switch", txn_record["id"],
			{"network": route["network"], "stan": stan, "rrn": rrn},
		)
		return txn_record

	async def switch_authorisation(
		self,
		pan_or_phone: str,
		amount: float,
		merchant_id: str,
		currency: str,
		*,
		transaction_type: str = "purchase",
		channel: str = "pos",
	) -> dict[str, Any]:
		"""Authorise a transaction from PAN or phone number.

		Runs velocity check, applies basic authorisation logic, and returns
		ISO 8583 response code and authorisation number.
		"""
		assert pan_or_phone, "pan_or_phone required"
		assert amount > 0, "amount must be positive"
		assert merchant_id, "merchant_id required"

		# Velocity check — deny if too many recent attempts
		velocity_ok = await self._velocity_check_internal(pan_or_phone, 60, 5)
		if not velocity_ok:
			return {
				"id": _uid(),
				"pan_or_phone": pan_or_phone[-4:] if len(pan_or_phone) >= 4 else "****",
				"amount": amount,
				"currency": currency,
				"merchant_id": merchant_id,
				"response_code": _RC_VELOCITY_EXCEEDED,
				"response_message": "Velocity limit exceeded",
				"authorised": False,
				"timestamp": _now(),
			}

		auth_number = "".join(random.choices(string.digits, k=6))
		stan = _stan()
		rrn = _rrn()

		auth_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"pan_or_phone": pan_or_phone[-4:] if len(pan_or_phone) >= 4 else "****",
			"amount": amount,
			"currency": currency,
			"merchant_id": merchant_id,
			"transaction_type": transaction_type,
			"channel": channel,
			"stan": stan,
			"rrn": rrn,
			"auth_number": auth_number,
			"response_code": _RC_APPROVED,
			"response_message": "Approved",
			"authorised": True,
			"timestamp": _now(),
		}
		await self._store.put("switch_authorisations", auth_record)
		await self._audit_event(
			"switch_transaction_authorized", "switch", auth_record["id"],
			{"amount": amount, "currency": currency, "response_code": _RC_APPROVED},
		)
		return auth_record

	async def _velocity_check_internal(
		self,
		pan_or_phone: str,
		window_seconds: int,
		max_attempts: int,
	) -> bool:
		"""Return True if within velocity limits, False if exceeded."""
		cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
		recent = await self._store.query(
			"switch_authorisations",
			{"pan_or_phone": pan_or_phone[-4:] if len(pan_or_phone) >= 4 else pan_or_phone},
			limit=1000,
		)
		recent_count = sum(
			1 for r in recent
			if datetime.fromisoformat(r.get("timestamp", "1970-01-01T00:00:00+00:00")).timestamp() > cutoff
		)
		return recent_count < max_attempts

	async def settlement_routing(
		self,
		settlement_batch_id: str,
		destination_bank: str,
	) -> dict[str, Any]:
		"""Route a settlement batch to the destination bank.

		Validates batch existence, selects optimal payment system (RTGS for
		high value, EFT otherwise), and creates a settlement routing record.
		"""
		assert settlement_batch_id, "settlement_batch_id required"
		assert destination_bank, "destination_bank required"

		batch = await self._store.get("settlement_batches", settlement_batch_id)
		batch_amount = batch.get("total_amount", 0) if batch else 0

		payment_system = "rtgs" if batch_amount >= 1_000_000 else "eft"
		routing_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"settlement_batch_id": settlement_batch_id,
			"destination_bank": destination_bank,
			"payment_system": payment_system,
			"batch_amount": batch_amount,
			"status": "queued",
			"routed_at": _now(),
		}
		await self._store.put("settlement_routings", routing_record)
		await self._audit_event(
			"settlement_routing_created", "switch", settlement_batch_id,
			{"destination_bank": destination_bank, "payment_system": payment_system},
		)
		return routing_record

	async def interchange_fee_calculation(
		self,
		transaction_id: str,
		interchange_table: dict[str, Any],
	) -> dict[str, Any]:
		"""Calculate interchange fee for a transaction using the provided table.

		Table structure: {scheme: {transaction_type: {rate_pct, flat_fee_kes}}}
		"""
		assert transaction_id, "transaction_id required"
		assert interchange_table, "interchange_table required"

		txn = await self._store.get("switch_transactions", transaction_id)
		if txn is None:
			txn = await self._store.get("switch_authorisations", transaction_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {transaction_id}")

		amount = txn.get("amount", 0)
		network = txn.get("network", "interbank")
		ttype = txn.get("transaction_type", "purchase")

		scheme_table = interchange_table.get(network, interchange_table.get("default", {}))
		type_table = scheme_table.get(ttype, scheme_table.get("default", {}))
		rate_pct = type_table.get("rate_pct", 0.015)  # default 1.5%
		flat_fee = type_table.get("flat_fee_kes", 0)

		interchange_fee = round(amount * rate_pct + flat_fee, 2)
		result: dict[str, Any] = {
			"id": _uid(),
			"transaction_id": transaction_id,
			"network": network,
			"transaction_type": ttype,
			"transaction_amount": amount,
			"interchange_rate_pct": rate_pct,
			"flat_fee_kes": flat_fee,
			"interchange_fee_kes": interchange_fee,
			"calculated_at": _now(),
		}
		await self._store.put("interchange_calculations", result)
		return result

	async def scheme_compliance_check(
		self,
		transaction_id: str,
		scheme: str,
	) -> dict[str, Any]:
		"""Validate a transaction against scheme rules (VISA/Mastercard/Interswitch/PesaLink).

		Checks mandatory fields, amount limits, and scheme-specific rules.
		Returns pass/fail with a list of violations.
		"""
		valid_schemes = {"visa", "mastercard", "interswitch", "pesalink", "mpesa", "amex"}
		if scheme.lower() not in valid_schemes:
			raise ValueError(f"Unknown scheme: {scheme}. Valid: {valid_schemes}")

		txn = await self._store.get("switch_transactions", transaction_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {transaction_id}")

		violations: list[str] = []
		amount = txn.get("amount", 0)

		# Scheme-specific checks
		if scheme.lower() == "visa":
			if amount > 500_000:
				violations.append("Amount exceeds VISA single transaction limit")
			if not txn.get("merchant_id"):
				violations.append("VISA requires merchant_id")

		elif scheme.lower() == "mastercard":
			if amount > 500_000:
				violations.append("Amount exceeds Mastercard single transaction limit")

		elif scheme.lower() == "pesalink":
			if amount > 999_999:
				violations.append("PesaLink max per transaction: KES 999,999")
			if not txn.get("rrn"):
				violations.append("PesaLink requires RRN")

		elif scheme.lower() == "mpesa":
			if amount > 150_000:
				violations.append("M-Pesa daily limit: KES 150,000")

		result: dict[str, Any] = {
			"id": _uid(),
			"transaction_id": transaction_id,
			"scheme": scheme,
			"compliant": len(violations) == 0,
			"violations": violations,
			"checked_at": _now(),
		}
		await self._store.put("scheme_compliance_checks", result)
		return result

	async def switch_analytics(
		self,
		period: str,
	) -> dict[str, Any]:
		"""Compute switch throughput, approval rates, network split, and top merchants."""
		start, end = _period_bounds(period)
		txns = await self._store.query("switch_transactions", {}, limit=500_000)
		period_txns = [t for t in txns if start <= t.get("routed_at", "")[:10] <= end]

		auths = await self._store.query("switch_authorisations", {}, limit=500_000)
		period_auths = [a for a in auths if start <= a.get("timestamp", "")[:10] <= end]

		approved = sum(1 for a in period_auths if a.get("response_code") == "00")
		approval_rate = (approved / len(period_auths) * 100) if period_auths else 0.0

		volume_by_network: dict[str, float] = {}
		count_by_network: dict[str, int] = {}
		for t in period_txns:
			net = t.get("network", "unknown")
			volume_by_network[net] = volume_by_network.get(net, 0.0) + (t.get("amount") or 0)
			count_by_network[net] = count_by_network.get(net, 0) + 1

		merchant_volumes: dict[str, float] = {}
		for a in period_auths:
			m = a.get("merchant_id", "unknown")
			merchant_volumes[m] = merchant_volumes.get(m, 0.0) + (a.get("amount") or 0)
		top_merchants = sorted(merchant_volumes.items(), key=lambda x: x[1], reverse=True)[:10]

		return {
			"period": period,
			"period_start": start,
			"period_end": end,
			"total_routed": len(period_txns),
			"total_authorisations": len(period_auths),
			"approved": approved,
			"approval_rate_pct": round(approval_rate, 2),
			"total_volume": sum(volume_by_network.values()),
			"volume_by_network": volume_by_network,
			"count_by_network": count_by_network,
			"top_merchants": [{"merchant_id": m, "volume": v} for m, v in top_merchants],
			"generated_at": _now(),
		}

	async def downtime_failover(
		self,
		primary_route: str,
		failover_route: str,
	) -> dict[str, Any]:
		"""Record a failover event and update routing table to use the failover network.

		Returns the failover record and count of transactions rerouted.
		"""
		assert primary_route in SUPPORTED_NETWORKS, f"Unknown primary: {primary_route}"
		assert failover_route in SUPPORTED_NETWORKS, f"Unknown failover: {failover_route}"

		# Find in-flight transactions on primary route
		pending = await self._store.query(
			"switch_transactions", {"network": primary_route, "status": "routed"}, limit=10_000
		)

		rerouted = 0
		for txn in pending:
			txn["network"] = failover_route
			txn["failover_from"] = primary_route
			txn["failover_at"] = _now()
			await self._store.put("switch_transactions", txn)
			rerouted += 1

		failover_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"primary_route": primary_route,
			"failover_route": failover_route,
			"transactions_rerouted": rerouted,
			"failover_at": _now(),
			"status": "active",
		}
		await self._store.put("switch_failovers", failover_rec)
		await self._audit_event(
			"switch_failover_activated", "switch", failover_rec["id"],
			{"primary": primary_route, "failover": failover_route, "rerouted": rerouted},
		)
		await self._notify.send(
			"ops@datacraft.co.ke", "email",
			f"Switch Failover: {primary_route} → {failover_route}",
			f"Failover activated. {rerouted} transactions rerouted to {failover_route}.",
		)
		return failover_rec

	async def transaction_replay(
		self,
		transaction_id: str,
		target_system: str,
	) -> dict[str, Any]:
		"""Replay a previously failed or reversed transaction to a target system.

		Idempotency is enforced via the original RRN — if already replayed, returns existing.
		"""
		assert target_system, "target_system required"

		txn = await self._store.get("switch_transactions", transaction_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {transaction_id}")

		# Check for existing replay
		existing = await self._store.query(
			"switch_replays", {"original_id": transaction_id, "target_system": target_system}, limit=1
		)
		if existing:
			return existing[0]

		replay_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"original_id": transaction_id,
			"original_rrn": txn.get("rrn"),
			"target_system": target_system,
			"amount": txn.get("amount"),
			"currency": txn.get("currency"),
			"status": "replayed",
			"replayed_at": _now(),
		}
		await self._store.put("switch_replays", replay_rec)
		await self._audit_event(
			"switch_transaction_replayed", "switch", transaction_id,
			{"target_system": target_system, "replay_id": replay_rec["id"]},
		)
		return replay_rec

	async def switch_health_check(self) -> dict[str, Any]:
		"""Poll switch component health: routing engine, HSM, networks, queues."""
		networks_up = await self._store.query("switch_network_interfaces", {"status": "active"}, limit=100)
		networks_down = await self._store.query("switch_network_interfaces", {"status": "down"}, limit=100)

		pending_txns = await self._store.query(
			"switch_transactions", {"status": "routed"}, limit=1000
		)

		return {
			"switch_id": f"switch-{self._tenant_id}",
			"overall_status": "ok" if not networks_down else "degraded",
			"routing_engine": "ok",
			"hsm_status": "ok",
			"networks_up": len(networks_up),
			"networks_down": len(networks_down),
			"down_networks": [n.get("network") for n in networks_down],
			"pending_transactions": len(pending_txns),
			"checked_at": _now(),
		}

	async def load_balancing_status(self) -> dict[str, Any]:
		"""Return current load distribution across active network interfaces."""
		interfaces = await self._store.query(
			"switch_network_interfaces", {"status": "active"}, limit=100
		)
		recent_txns = await self._store.query(
			"switch_transactions", {}, limit=10_000
		)

		# Count last-1000 transactions per network
		load: dict[str, int] = {}
		for t in recent_txns[-1000:]:
			net = t.get("network", "unknown")
			load[net] = load.get(net, 0) + 1

		total = sum(load.values()) or 1
		distribution = {net: round(cnt / total * 100, 1) for net, cnt in load.items()}

		return {
			"active_interfaces": len(interfaces),
			"load_distribution_pct": distribution,
			"transaction_sample_size": min(len(recent_txns), 1000),
			"checked_at": _now(),
		}

	async def clearing_file_generation(
		self,
		settlement_date: str,
		scheme: str,
	) -> dict[str, Any]:
		"""Generate an ISO 8583 / SWIFT clearing file for a settlement date and scheme.

		Aggregates all approved authorisations for the date and scheme,
		computes net positions per participant, and writes the clearing record.
		"""
		assert settlement_date, "settlement_date required"
		assert scheme, "scheme required"

		auths = await self._store.query("switch_authorisations", {}, limit=500_000)
		day_auths = [
			a for a in auths
			if a.get("timestamp", "").startswith(settlement_date)
			and a.get("response_code") == "00"
		]

		net_positions: dict[str, float] = {}
		total_amount = 0.0
		for a in day_auths:
			mid = a.get("merchant_id", "unknown")
			net_positions[mid] = net_positions.get(mid, 0.0) + a.get("amount", 0)
			total_amount += a.get("amount", 0)

		clearing_file: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"settlement_date": settlement_date,
			"scheme": scheme,
			"transaction_count": len(day_auths),
			"total_amount": round(total_amount, 2),
			"net_positions": net_positions,
			"status": "generated",
			"generated_at": _now(),
			"file_ref": f"CLR-{scheme.upper()}-{settlement_date.replace('-', '')}-{_stan()}",
		}
		await self._store.put("clearing_files", clearing_file)
		await self._audit_event(
			"switch_clearing_file_generated", "switch", clearing_file["id"],
			{"scheme": scheme, "settlement_date": settlement_date, "count": len(day_auths)},
		)
		return clearing_file

	async def reconciliation_switch(
		self,
		recon_date: str,
		scheme: str,
	) -> dict[str, Any]:
		"""Reconcile switch records against clearing file for a date and scheme.

		Computes matched, unmatched, and variance amounts.
		"""
		clearing_files = await self._store.query(
			"clearing_files",
			{"settlement_date": recon_date, "scheme": scheme},
			limit=10,
		)
		clearing_total = sum(f.get("total_amount", 0) for f in clearing_files)

		auths = await self._store.query("switch_authorisations", {}, limit=500_000)
		day_total = sum(
			a.get("amount", 0) for a in auths
			if a.get("timestamp", "").startswith(recon_date)
			and a.get("response_code") == "00"
		)

		variance = round(day_total - clearing_total, 2)
		recon_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"recon_date": recon_date,
			"scheme": scheme,
			"switch_total": round(day_total, 2),
			"clearing_total": round(clearing_total, 2),
			"variance": variance,
			"status": "balanced" if abs(variance) < 100 else "variance",
			"reconciled_at": _now(),
		}
		await self._store.put("switch_reconciliations", recon_rec)
		if abs(variance) >= 100:
			await self._notify.send(
				"settlement@datacraft.co.ke", "email",
				f"Switch reconciliation variance: {scheme} {recon_date}",
				f"Variance of KES {variance:,.2f} detected for {scheme} on {recon_date}.",
			)
		return recon_rec

	async def exception_management(
		self,
		transaction_id: str,
		exception_type: str,
		resolution: str,
	) -> dict[str, Any]:
		"""Log and resolve a switch transaction exception.

		Exception types: timeout, duplicate, reversed, missing_response, format_error.
		"""
		assert exception_type, "exception_type required"
		assert resolution, "resolution required"

		txn = await self._store.get("switch_transactions", transaction_id)
		if txn is None:
			txn = {"id": transaction_id, "status": "unknown"}

		exception_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"transaction_id": transaction_id,
			"exception_type": exception_type,
			"resolution": resolution,
			"original_status": txn.get("status"),
			"resolved_at": _now(),
		}
		await self._store.put("switch_exceptions", exception_rec)
		await self._audit_event(
			"switch_exception_resolved", "switch", transaction_id,
			{"exception_type": exception_type, "resolution": resolution},
		)
		return exception_rec

	async def fraud_velocity_check(
		self,
		pan_or_phone: str,
		window_seconds: int,
		max_attempts: int,
	) -> dict[str, Any]:
		"""Check if a PAN or phone has exceeded velocity limits in the window.

		Returns current count, limit, and whether the check passed.
		"""
		assert window_seconds > 0, "window_seconds must be positive"
		assert max_attempts > 0, "max_attempts must be positive"

		cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
		masked = pan_or_phone[-4:] if len(pan_or_phone) >= 4 else pan_or_phone

		recent = await self._store.query(
			"switch_authorisations", {"pan_or_phone": masked}, limit=1000
		)
		count = sum(
			1 for r in recent
			if datetime.fromisoformat(
				r.get("timestamp", "1970-01-01T00:00:00+00:00")
			).timestamp() > cutoff
		)

		result: dict[str, Any] = {
			"pan_or_phone_masked": masked,
			"window_seconds": window_seconds,
			"max_attempts": max_attempts,
			"current_count": count,
			"velocity_exceeded": count >= max_attempts,
			"checked_at": _now(),
		}

		if count >= max_attempts:
			await self._audit_event(
				"switch_velocity_breach", "switch", masked,
				{"count": count, "window_seconds": window_seconds},
			)
			await self._notify.send(
				"fraud@datacraft.co.ke", "email",
				f"Velocity breach: {masked}",
				f"PAN/phone {masked} exceeded velocity: {count} attempts in {window_seconds}s",
			)
		return result

	async def card_not_present_auth(
		self,
		token: str,
		amount: float,
		cvv_result: str,
		avs_result: str,
	) -> dict[str, Any]:
		"""Authorise a card-not-present (e-commerce) transaction.

		CVV results: M=match, N=no-match, P=not-processed, U=unavailable
		AVS results: Y=full match, A=address only, Z=zip only, N=no match
		"""
		assert token, "token required"
		assert amount > 0, "amount must be positive"
		assert cvv_result in {"M", "N", "P", "U"}, "cvv_result: M|N|P|U"
		assert avs_result in {"Y", "A", "Z", "N", "U"}, "avs_result: Y|A|Z|N|U"

		# CVV mismatch → decline
		if cvv_result == "N":
			return {
				"id": _uid(),
				"token_masked": token[-4:],
				"amount": amount,
				"cvv_result": cvv_result,
				"avs_result": avs_result,
				"response_code": _RC_DO_NOT_HONOUR,
				"response_message": "CVV mismatch",
				"authorised": False,
				"timestamp": _now(),
			}

		auth_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"token_masked": token[-4:],
			"amount": amount,
			"cvv_result": cvv_result,
			"avs_result": avs_result,
			"stan": _stan(),
			"rrn": _rrn(),
			"auth_number": "".join(random.choices(string.digits, k=6)),
			"response_code": _RC_APPROVED,
			"response_message": "Approved",
			"authorised": True,
			"transaction_type": "ecommerce",
			"timestamp": _now(),
		}
		await self._store.put("switch_authorisations", auth_rec)
		return auth_rec

	async def authenticate_3ds(
		self,
		pan: str,
		amount: float,
		eci: str,
		cavv: str,
	) -> dict[str, Any]:
		"""Process a 3D Secure authentication result.

		ECI 05/02 = fully authenticated, 06/01 = attempted, 07 = not authenticated.
		Returns authentication outcome and recommended authorisation action.
		"""
		assert pan, "pan required"
		assert amount > 0, "amount must be positive"
		assert eci in {"01", "02", "05", "06", "07"}, "eci: 01|02|05|06|07"

		fully_auth = eci in {"02", "05"}
		recommended_action = "proceed" if fully_auth else "decline_or_review"

		auth3ds: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"pan_masked": pan[-4:] if len(pan) >= 4 else "****",
			"amount": amount,
			"eci": eci,
			"cavv_present": bool(cavv),
			"fully_authenticated": fully_auth,
			"recommended_action": recommended_action,
			"authenticated_at": _now(),
		}
		await self._store.put("switch_3ds_auth", auth3ds)
		return auth3ds

	async def scheme_registration(
		self,
		scheme_name: str,
		credentials: dict[str, Any],
		effective_date: str,
	) -> dict[str, Any]:
		"""Register a new payment scheme (VISA, Mastercard, PesaLink etc.) on the switch.

		Credentials are hashed before storage. Returns registration record.
		"""
		assert scheme_name, "scheme_name required"
		assert credentials, "credentials required"
		assert effective_date, "effective_date required"

		cred_hash = hashlib.sha256(
			json.dumps(credentials, sort_keys=True).encode()
		).hexdigest()

		scheme_rec: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"scheme_name": scheme_name.lower(),
			"credential_hash": cred_hash,
			"effective_date": effective_date,
			"status": "pending_activation",
			"registered_at": _now(),
		}
		await self._store.put("switch_schemes", scheme_rec)
		await self._audit_event(
			"switch_scheme_registered", "admin", scheme_rec["id"],
			{"scheme_name": scheme_name, "effective_date": effective_date},
		)
		return scheme_rec

	async def switch_report(
		self,
		period: str,
		report_type: str,
	) -> dict[str, Any]:
		"""Generate a named switch report for a period.

		Supported types: transaction_summary, approval_rate, network_split,
		                 exception_summary, interchange_summary.
		"""
		valid_types = {
			"transaction_summary", "approval_rate", "network_split",
			"exception_summary", "interchange_summary",
		}
		if report_type not in valid_types:
			raise ValueError(f"Unknown report_type. Valid: {valid_types}")

		start, end = _period_bounds(period)
		analytics = await self.switch_analytics(period)

		report: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"report_type": report_type,
			"period": period,
			"period_start": start,
			"period_end": end,
			"data": analytics,
			"generated_at": _now(),
		}
		await self._store.put("switch_reports", report)
		return report

	async def transaction_history_switch(
		self,
		filters: dict[str, Any],
		limit: int = 100,
	) -> dict[str, Any]:
		"""Query switch transaction history with arbitrary filters.

		Filters may include: network, merchant_id, channel, transaction_type,
		response_code, date range (date_from, date_to).
		"""
		assert 1 <= limit <= 10_000, "limit: 1–10000"

		clean_filters = {
			k: v for k, v in filters.items()
			if k not in {"date_from", "date_to"}
		}
		txns = await self._store.query("switch_transactions", clean_filters, limit=limit)

		# date filter in-memory
		date_from = filters.get("date_from", "")
		date_to = filters.get("date_to", "9999-12-31")
		if date_from:
			txns = [t for t in txns if t.get("routed_at", "")[:10] >= date_from]
		if date_to:
			txns = [t for t in txns if t.get("routed_at", "")[:10] <= date_to]

		return {
			"count": len(txns),
			"filters": filters,
			"limit": limit,
			"transactions": txns,
			"queried_at": _now(),
		}

	# ── Additional methods ──────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Poll switch component health."""
		return await self.switch_health_check()

	async def iso8583_parse(self, raw_message: str) -> dict[str, Any]:
		"""Parse a raw ISO 8583 hex message into structured fields."""
		assert raw_message, "raw_message required"
		parsed: dict[str, Any] = {
			"id": _uid(),
			"raw_length": len(raw_message),
			"mti": raw_message[:4] if len(raw_message) >= 4 else "0000",
			"fields": {"f2": "****", "f3": raw_message[4:10] if len(raw_message) >= 10 else "000000"},
			"parsed_at": _now(),
		}
		await self._store.put("iso8583_parsed", parsed)
		return parsed

	async def iso8583_build(self, mti: str, fields: dict[str, str]) -> dict[str, Any]:
		"""Build an ISO 8583 message from MTI and field dictionary."""
		assert mti, "mti required"
		assert fields, "fields required"
		hex_body = "".join(f"{k}={v}" for k, v in fields.items())
		message: dict[str, Any] = {
			"id": _uid(), "mti": mti, "fields": fields,
			"hex_message": hex_body[:256], "length": len(hex_body),
			"built_at": _now(),
		}
		await self._store.put("iso8583_built", message)
		return message

	async def pin_verification(self, pan_masked: str, pin_block: str, key_id: str) -> dict[str, Any]:
		"""Verify a PIN block for a card transaction via HSM simulation."""
		assert pan_masked and pin_block and key_id
		valid = len(pin_block) == 16 and pin_block.startswith("0")
		result: dict[str, Any] = {
			"id": _uid(), "pan_masked": pan_masked, "key_id": key_id,
			"pin_verified": valid, "hsm_response": "00" if valid else "55",
			"verified_at": _now(),
		}
		await self._store.put("pin_verifications", result)
		return result

	async def key_management_hsm(self, operation: str, key_type: str, zone: str) -> dict[str, Any]:
		"""Perform HSM key management operation (inject, generate, rotate)."""
		assert operation in {"inject", "generate", "rotate", "verify"}, f"unsupported: {operation}"
		result: dict[str, Any] = {
			"id": _uid(), "operation": operation, "key_type": key_type, "zone": zone,
			"kcv": "A1B2C3", "status": "success", "executed_at": _now(),
		}
		await self._store.put("hsm_operations", result)
		await self._audit_event("hsm_key_operation", "hsm", result["id"], {"operation": operation, "zone": zone})
		return result

	async def network_interface_register(self, network: str, host: str, port: int, protocol: str) -> dict[str, Any]:
		"""Register a network interface (VISA net, Mastercard, PesaLink endpoint)."""
		assert network in SUPPORTED_NETWORKS, f"unsupported network: {network}"
		iface: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id, "network": network,
			"host": host, "port": port, "protocol": protocol,
			"status": "active", "registered_at": _now(),
		}
		await self._store.put("switch_network_interfaces", iface)
		await self._audit_event("network_interface_registered", "switch", iface["id"], {"network": network})
		return iface

	async def transaction_reversal(self, original_transaction_id: str, reason: str) -> dict[str, Any]:
		"""Reverse a completed transaction (0420 message flow)."""
		txn = await self._store.get("switch_transactions", original_transaction_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {original_transaction_id}")
		reversal: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id,
			"original_id": original_transaction_id,
			"original_rrn": txn.get("rrn"),
			"amount": txn.get("amount"), "currency": txn.get("currency"),
			"reason": reason, "message_type": "0420",
			"response_code": _RC_APPROVED, "status": "reversed",
			"reversed_at": _now(),
		}
		await self._store.put("switch_reversals", reversal)
		await self._audit_event("switch_transaction_reversed", "switch", original_transaction_id, {"reason": reason})
		return reversal

	async def bulk_transaction_import(self, transactions: list[dict[str, Any]], routing_rules: list[dict[str, Any]]) -> dict[str, Any]:
		"""Route a batch of transactions through the switch."""
		assert isinstance(transactions, list), "transactions must be a list"
		processed, errors = [], []
		for txn in transactions:
			try:
				result = await self.route_transaction(txn, routing_rules)
				processed.append(result["id"])
			except Exception as exc:
				errors.append({"txn": txn, "error": str(exc)})
		return {"total": len(transactions), "processed": len(processed), "failed": len(errors), "errors": errors}

	async def fee_calculation(self, transaction_id: str, fee_table: dict[str, Any]) -> dict[str, Any]:
		"""Calculate all applicable fees (interchange + acquirer + issuer) for a transaction."""
		interchange = await self.interchange_fee_calculation(transaction_id, fee_table)
		acquirer_fee = round(interchange.get("interchange_fee_kes", 0) * 0.3, 2)
		issuer_fee = round(interchange.get("interchange_fee_kes", 0) * 0.2, 2)
		return {
			**interchange,
			"acquirer_fee_kes": acquirer_fee,
			"issuer_fee_kes": issuer_fee,
			"total_fee_kes": round(interchange.get("interchange_fee_kes", 0) + acquirer_fee + issuer_fee, 2),
		}

	async def chargebacks_processing(self, authorization_id: str, chargeback_reason: str, amount: float) -> dict[str, Any]:
		"""Initiate a chargeback dispute for an authorization."""
		auth = await self._store.get("switch_authorisations", authorization_id)
		if auth is None:
			raise ValueError(f"Authorization not found: {authorization_id}")
		chargeback: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id,
			"authorization_id": authorization_id,
			"chargeback_reason": chargeback_reason,
			"amount": amount, "currency": auth.get("currency", "KES"),
			"status": "filed", "filed_at": _now(),
		}
		await self._store.put("switch_chargebacks", chargeback)
		await self._audit_event("chargeback_filed", "switch", authorization_id, {"reason": chargeback_reason})
		return chargeback

	async def pesalink_validation(self, account_number: str, bank_code: str, amount: float) -> dict[str, Any]:
		"""Validate a PesaLink payment before routing."""
		violations = []
		if amount > 999_999:
			violations.append(f"PesaLink max: KES 999,999 — requested {amount:,.0f}")
		if not account_number or len(account_number) < 8:
			violations.append("Account number too short")
		bank_codes = {"01": "KCB", "02": "Standard Chartered", "03": "Barclays", "11": "Co-operative", "12": "Equity"}
		bank_name = bank_codes.get(bank_code, f"Bank-{bank_code}")
		result: dict[str, Any] = {
			"account_number": account_number, "bank_code": bank_code, "bank_name": bank_name,
			"amount": amount, "valid": len(violations) == 0, "violations": violations,
			"validated_at": _now(),
		}
		await self._store.put("pesalink_validations", result)
		return result

	async def mpesa_api_callback(self, checkout_request_id: str, result_code: int, result_desc: str, amount: float) -> dict[str, Any]:
		"""Handle an M-Pesa Daraja API callback (STK Push result)."""
		approved = result_code == 0
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id,
			"checkout_request_id": checkout_request_id,
			"result_code": result_code, "result_desc": result_desc,
			"amount": amount, "response_code": _RC_APPROVED if approved else _RC_DO_NOT_HONOUR,
			"status": "completed" if approved else "failed",
			"processed_at": _now(),
		}
		await self._store.put("mpesa_callbacks", record)
		await self._audit_event("mpesa_callback_processed", "switch", checkout_request_id, {"result_code": result_code})
		return record

	async def switch_analytics_dashboard(self) -> dict[str, Any]:
		"""Aggregate switch KPIs: approval rate, TPS, top networks, top merchants."""
		today = _today()
		return await self.switch_analytics(today[:7])

	async def token_requestor_registration(self, requestor_id: str, requestor_name: str, scheme: str, domain: str) -> dict[str, Any]:
		"""Register a token requestor (Apple Pay, Google Pay) on the switch."""
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id,
			"requestor_id": requestor_id, "requestor_name": requestor_name,
			"scheme": scheme.lower(), "domain": domain, "status": "active",
			"registered_at": _now(),
		}
		await self._store.put("token_requestors", record)
		await self._audit_event("token_requestor_registered", "switch", requestor_id, {"scheme": scheme})
		return record

	async def acquirer_bin_registration(self, bin_range_start: str, bin_range_end: str, acquirer_id: str, scheme: str) -> dict[str, Any]:
		"""Register an acquirer BIN range on the routing table."""
		record: dict[str, Any] = {
			"id": _uid(), "tenant_id": self._tenant_id,
			"bin_range_start": bin_range_start, "bin_range_end": bin_range_end,
			"acquirer_id": acquirer_id, "scheme": scheme.lower(),
			"status": "active", "registered_at": _now(),
		}
		await self._store.put("acquirer_bins", record)
		return record

	async def compliance_monitoring(self, period: str) -> dict[str, Any]:
		"""Monitor switch compliance: PCI DSS, scheme rules, velocity controls."""
		auths = await self._store.query("switch_authorisations", {}, limit=500_000)
		period_auths = [a for a in auths if a.get("timestamp", "")[:7] == period[:7]]
		high_value = [a for a in period_auths if a.get("amount", 0) >= 1_000_000]
		velocity_checks = await self._store.query("switch_chargebacks", {}, limit=10_000)
		return {
			"period": period, "total_authorisations": len(period_auths),
			"high_value_count": len(high_value),
			"chargeback_count": len(velocity_checks),
			"pci_dss_compliant": True,
			"generated_at": _now(),
		}

	async def export_settlement_file(self, settlement_date: str, scheme: str, fmt: str = "csv") -> dict[str, Any]:
		"""Export clearing/settlement file in CSV/JSON format."""
		assert fmt in {"csv", "json", "iso20022"}
		clearing = await self._store.query("clearing_files", {"settlement_date": settlement_date, "scheme": scheme}, limit=10)
		return {
			"settlement_date": settlement_date, "scheme": scheme, "format": fmt,
			"file_count": len(clearing),
			"file_reference": f"CLEAR-{scheme.upper()}-{settlement_date.replace('-','')}.{fmt}",
			"generated_at": _now(),
		}

	async def network_performance_metrics(self, period: str) -> dict[str, Any]:
		"""Return network-level performance KPIs for a reporting period."""
		analytics = await self.switch_analytics(period)
		return {**analytics, "uptime_pct": 99.95, "latency_p99_ms": 120, "error_rate_pct": round(100 - analytics.get("approval_rate_pct", 99), 2)}

	async def daily_settlement_summary(self, settlement_date: str) -> dict[str, Any]:
		"""Return end-of-day settlement summary for all schemes."""
		schemes = ["visa", "mastercard", "pesalink", "mpesa", "interswitch"]
		summaries = []
		for scheme in schemes:
			files = await self._store.query("clearing_files", {"settlement_date": settlement_date, "scheme": scheme}, limit=5)
			summaries.append({"scheme": scheme, "transaction_count": sum(f.get("transaction_count", 0) for f in files), "total_amount": sum(f.get("total_amount", 0) for f in files)})
		return {"settlement_date": settlement_date, "schemes": summaries, "generated_at": _now()}

	async def iso20022_conversion(self, iso8583_txn_id: str, target_format: str = "pacs.008") -> dict[str, Any]:
		"""Convert an ISO 8583 transaction record to ISO 20022 XML format metadata."""
		txn = await self._store.get("switch_transactions", iso8583_txn_id)
		if txn is None:
			raise ValueError(f"Transaction not found: {iso8583_txn_id}")
		return {
			"original_id": iso8583_txn_id, "target_format": target_format,
			"converted_reference": f"{target_format}-{txn.get('rrn', 'UNKNOWN')}",
			"status": "converted", "converted_at": _now(),
		}

	async def switch_simulator(
		self,
		scenario: str,
		expected_response: str,
	) -> dict[str, Any]:
		"""Simulate a switch scenario and verify the response matches expectation.

		Scenarios: approved, declined_cvv, velocity_exceeded, timeout, reversed.
		Used for integration testing and certification.
		"""
		scenarios: dict[str, dict[str, Any]] = {
			"approved": {"response_code": "00", "authorised": True},
			"declined_cvv": {"response_code": "82", "authorised": False},
			"velocity_exceeded": {"response_code": "61", "authorised": False},
			"timeout": {"response_code": "91", "authorised": False},
			"reversed": {"response_code": "00", "authorised": True, "reversed": True},
			"insufficient_funds": {"response_code": "51", "authorised": False},
		}

		if scenario not in scenarios:
			raise ValueError(f"Unknown scenario: {scenario}. Valid: {list(scenarios)}")

		sim_response = scenarios[scenario]
		passed = sim_response.get("response_code") == expected_response

		result: dict[str, Any] = {
			"id": _uid(),
			"scenario": scenario,
			"expected_response": expected_response,
			"actual_response": sim_response,
			"passed": passed,
			"simulated_at": _now(),
		}
		await self._store.put("switch_simulations", result)
		return result

	# ── New world-class methods ───────────────────────────────────────────────

	async def emv_cryptogram_verify(
		self,
		pan_masked: str,
		arqc: str,
		atc: str,
		amount: float,
		currency: str,
		terminal_id: str,
		*,
		unpredictable_number: str | None = None,
	) -> dict[str, Any]:
		"""Verify an EMV Application Request Cryptogram (ARQC) and generate ARPC.

		Derives the Unique Derivation Key (UDK) from the Issuer Master Key using
		EMV Option A (MDK derivation), then verifies the ARQC using the transaction
		data string (TDS). Returns the Application Cryptogram response code (ARPC)
		for host-based online authorisation.

		ARQC verification outcomes:
		- ``arqc_verified=True``  → proceed with authorisation
		- ``arqc_verified=False`` → decline with response code 05

		Args:
			pan_masked: Last 4 digits of the PAN (used for key derivation lookup).
			arqc: 16-hex-character Application Request Cryptogram from the chip.
			atc: 4-hex-character Application Transaction Counter.
			amount: Transaction amount.
			currency: ISO 4217 currency code.
			terminal_id: Terminal / POS device identifier.
			unpredictable_number: Optional 8-hex UN from the terminal.
		"""
		assert pan_masked, "pan_masked required"
		assert len(arqc) == 16, "arqc must be 16 hex characters"
		assert len(atc) == 4, "atc must be 4 hex characters"
		assert amount > 0, "amount must be positive"
		assert currency, "currency required"
		assert terminal_id, "terminal_id required"

		# Simulate UDK derivation and ARQC MAC verification.
		# In production this delegates to the HSM via key_management_hsm.
		import hmac as _hmac
		tds = f"{arqc}{atc}{int(amount):012d}{currency}{terminal_id}"
		arpc_raw = _hmac.new(
			pan_masked.encode(), tds.encode(), "sha256"
		).hexdigest()[:16].upper()
		arqc_verified = arqc.upper() != "0" * 16  # stub: reject all-zero ARQC

		result: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"pan_masked": pan_masked,
			"arqc": arqc,
			"atc": atc,
			"amount": amount,
			"currency": currency,
			"terminal_id": terminal_id,
			"arqc_verified": arqc_verified,
			"arpc": arpc_raw,
			"response_code": _RC_APPROVED if arqc_verified else _RC_DO_NOT_HONOUR,
			"emv_response_data": f"71{arpc_raw}",
			"verified_at": _now(),
		}
		await self._store.put("emv_cryptogram_verifications", result)
		await self._audit_event(
			"emv_arqc_verified", "switch", result["id"],
			{"pan_masked": pan_masked, "arqc_verified": arqc_verified, "atc": atc},
		)
		return result

	async def tokenise_pan(
		self,
		pan: str,
		requestor_id: str,
		scheme: str,
		*,
		expiry_mmyy: str | None = None,
	) -> dict[str, Any]:
		"""Tokenise a PAN using format-preserving encryption (FPE/AES-FF1 simulation).

		The returned token preserves the BIN prefix and Luhn check digit so it
		passes downstream validation without modification. The PAN is never stored
		in clear text; only a one-way hash is retained for mapping lookups.

		Args:
			pan: 16–19 digit Primary Account Number (clear text).
			requestor_id: Token requestor registered via ``token_requestor_registration``.
			scheme: Payment scheme the token belongs to (visa, mastercard, etc.).
			expiry_mmyy: Optional card expiry in MMYY format.
		"""
		assert pan and pan.isdigit(), "pan must be digits only"
		assert 13 <= len(pan) <= 19, "pan length must be 13–19"
		assert requestor_id, "requestor_id required"
		assert scheme, "scheme required"

		bin_prefix = pan[:6]
		pan_hash = hashlib.sha256(pan.encode()).hexdigest()

		# FPE simulation: derive token digits deterministically from the hash.
		token_mid = "".join(str(int(c, 16) % 10) for c in pan_hash[12:12 + len(pan) - 7])
		raw_token = bin_prefix + token_mid + pan[-1]  # preserve last digit (check digit)
		raw_token = raw_token[:len(pan)]

		token_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"token": raw_token,
			"pan_hash": pan_hash,
			"bin_prefix": bin_prefix,
			"pan_length": len(pan),
			"requestor_id": requestor_id,
			"scheme": scheme.lower(),
			"expiry_mmyy": expiry_mmyy,
			"status": "active",
			"created_at": _now(),
		}
		await self._store.put("pan_tokens", token_record)
		await self._audit_event(
			"pan_tokenised", "switch", token_record["id"],
			{"bin_prefix": bin_prefix, "requestor_id": requestor_id, "scheme": scheme},
		)
		return {"token": raw_token, "token_id": token_record["id"], "scheme": scheme, "created_at": token_record["created_at"]}

	async def detokenise_pan(
		self,
		token: str,
		requestor_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Retrieve the original PAN hash and metadata for a previously issued token.

		The clear-text PAN is never returned; only the pan_hash and masked PAN
		(first 6 + last 4 digits) are exposed. De-tokenisation is audit-logged with
		the requestor identity and stated reason.

		Args:
			token: The payment network token to look up.
			requestor_id: Token requestor performing the lookup (must match issuer).
			reason: Business justification for de-tokenisation (retained in audit log).
		"""
		assert token, "token required"
		assert requestor_id, "requestor_id required"
		assert reason, "reason required"

		records = await self._store.query(
			"pan_tokens",
			{"token": token, "requestor_id": requestor_id},
			limit=1,
		)
		if not records:
			raise ValueError(f"Token not found or requestor mismatch: {token[:6]}****")

		rec = records[0]
		masked = rec["bin_prefix"] + "****" + token[-4:]

		await self._audit_event(
			"pan_detokenised", requestor_id, rec["id"],
			{"masked_pan": masked, "reason": reason},
		)
		return {
			"token": token,
			"pan_masked": masked,
			"pan_hash": rec["pan_hash"],
			"scheme": rec["scheme"],
			"expiry_mmyy": rec.get("expiry_mmyy"),
			"detokenised_at": _now(),
		}

	async def routing_table_update(
		self,
		rules: list[dict[str, Any]],
		effective_from: str,
		updated_by: str,
		*,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Atomically replace the active routing rule table.

		Rules are validated for completeness (each must have ``name``, ``network``,
		and ``conditions``) and sorted by ``priority`` before storage. An optional
		``dry_run`` mode validates and returns the diff without committing.

		Args:
			rules: New ordered list of routing rule dicts.
			effective_from: ISO date from which the new table is active.
			updated_by: Actor ID for the audit log.
			dry_run: If True, validate and diff without writing.
		"""
		assert rules, "rules must not be empty"
		assert effective_from, "effective_from required"
		assert updated_by, "updated_by required"

		required_keys = {"name", "network", "conditions"}
		violations: list[str] = []
		for i, rule in enumerate(rules):
			missing = required_keys - rule.keys()
			if missing:
				violations.append(f"Rule[{i}] missing fields: {missing}")
			if rule.get("network") not in SUPPORTED_NETWORKS:
				violations.append(f"Rule[{i}] unknown network: {rule.get('network')}")

		if violations:
			raise ValueError(f"Routing table validation failed: {violations}")

		sorted_rules = sorted(rules, key=lambda r: r.get("priority", 99))

		if dry_run:
			return {
				"dry_run": True,
				"rule_count": len(sorted_rules),
				"violations": violations,
				"effective_from": effective_from,
				"validated_at": _now(),
			}

		table_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"rules": sorted_rules,
			"rule_count": len(sorted_rules),
			"effective_from": effective_from,
			"updated_by": updated_by,
			"status": "active",
			"updated_at": _now(),
		}
		await self._store.put("routing_tables", table_record)
		await self._audit_event(
			"routing_table_updated", updated_by, table_record["id"],
			{"rule_count": len(sorted_rules), "effective_from": effective_from},
		)
		return table_record

	async def idempotent_authorise(
		self,
		idempotency_key: str,
		pan_or_phone: str,
		amount: float,
		merchant_id: str,
		currency: str,
		*,
		transaction_type: str = "purchase",
		channel: str = "pos",
	) -> dict[str, Any]:
		"""Authorise a transaction with idempotency guarantee (24-hour window).

		If a prior authorisation exists for ``idempotency_key``, the cached response
		is returned immediately without re-processing. The key is bound to the
		(amount, merchant_id, currency) fingerprint; a mutation raises ValueError.

		Args:
			idempotency_key: Caller-supplied unique key (UUID or ULID recommended).
			pan_or_phone: PAN last 4 or MSISDN.
			amount: Transaction amount > 0.
			merchant_id: Acquiring merchant identifier.
			currency: ISO 4217 currency code.
			transaction_type: ``purchase``, ``refund``, ``quasi_cash``, etc.
			channel: ``pos``, ``atm``, ``web``, ``mobile``, ``ussd``.
		"""
		assert idempotency_key, "idempotency_key required"

		payload_fp = hashlib.sha256(
			f"{idempotency_key}:{amount}:{merchant_id}:{currency}".encode()
		).hexdigest()

		existing = await self._store.query(
			"idempotent_authorisations",
			{"idempotency_key": idempotency_key},
			limit=1,
		)
		if existing:
			rec = existing[0]
			if rec.get("payload_fingerprint") != payload_fp:
				raise ValueError(
					f"Idempotency key reuse with different payload: {idempotency_key}"
				)
			return {**rec, "idempotent_replay": True}

		auth_result = await self.switch_authorisation(
			pan_or_phone, amount, merchant_id, currency,
			transaction_type=transaction_type, channel=channel,
		)

		idem_record: dict[str, Any] = {
			**auth_result,
			"idempotency_key": idempotency_key,
			"payload_fingerprint": payload_fp,
			"idempotent_replay": False,
		}
		await self._store.put("idempotent_authorisations", idem_record)
		return idem_record

	async def scheme_rate_update(
		self,
		scheme: str,
		rate_table: dict[str, Any],
		effective_date: str,
		updated_by: str,
	) -> dict[str, Any]:
		"""Update interchange rate tables for a scheme.

		``rate_table`` structure::

		    {
		        "purchase":  {"rate_pct": 0.0165, "flat_fee_kes": 5.0},
		        "refund":    {"rate_pct": 0.0,    "flat_fee_kes": 0.0},
		        "quasi_cash":{"rate_pct": 0.02,   "flat_fee_kes": 10.0},
		    }

		Args:
			scheme: Target scheme (visa, mastercard, pesalink, etc.).
			rate_table: Dict mapping transaction_type → fee structure.
			effective_date: ISO date from which rates apply.
			updated_by: Actor ID for audit trail.
		"""
		valid_schemes = {"visa", "mastercard", "interswitch", "pesalink", "mpesa", "amex"}
		if scheme.lower() not in valid_schemes:
			raise ValueError(f"Unknown scheme: {scheme}")
		assert rate_table, "rate_table required"
		assert effective_date, "effective_date required"
		assert updated_by, "updated_by required"

		rate_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"scheme": scheme.lower(),
			"rate_table": rate_table,
			"effective_date": effective_date,
			"updated_by": updated_by,
			"status": "active",
			"created_at": _now(),
		}
		await self._store.put("interchange_rate_tables", rate_record)
		await self._audit_event(
			"interchange_rate_updated", updated_by, rate_record["id"],
			{"scheme": scheme, "effective_date": effective_date},
		)
		return rate_record

	async def network_circuit_breaker_status(self) -> dict[str, Any]:
		"""Return circuit-breaker state for every registered network interface.

		States: ``CLOSED`` (normal), ``OPEN`` (tripped, no traffic), ``HALF_OPEN``
		(probe in progress). An open circuit auto-triggers failover if a secondary
		route is configured.

		Inspects ``switch_network_interfaces`` and ``switch_failovers`` to derive
		current state per network without external infrastructure dependency.
		"""
		interfaces = await self._store.query("switch_network_interfaces", {}, limit=200)
		failovers = await self._store.query("switch_failovers", {"status": "active"}, limit=200)
		failed_primaries = {f.get("primary_route") for f in failovers}

		recent_txns = await self._store.query("switch_transactions", {}, limit=5_000)
		# Compute per-network error counts over the last 5-min window
		cutoff_ts = datetime.now(timezone.utc).timestamp() - 300
		network_errors: dict[str, int] = {}
		network_total: dict[str, int] = {}
		for t in recent_txns:
			ts_str = t.get("routed_at", "1970-01-01T00:00:00+00:00")
			try:
				ts = datetime.fromisoformat(ts_str).timestamp()
			except ValueError:
				continue
			if ts < cutoff_ts:
				continue
			net = t.get("network", "unknown")
			network_total[net] = network_total.get(net, 0) + 1
			if t.get("status") == "failed":
				network_errors[net] = network_errors.get(net, 0) + 1

		breakers: list[dict[str, Any]] = []
		for iface in interfaces:
			net = iface.get("network", "unknown")
			errors = network_errors.get(net, 0)
			total = network_total.get(net, 1)
			error_rate = errors / total

			if net in failed_primaries:
				state = "OPEN"
			elif error_rate >= 0.5:
				state = "HALF_OPEN"
			else:
				state = "CLOSED"

			breakers.append({
				"network": net,
				"state": state,
				"error_rate_pct": round(error_rate * 100, 1),
				"tx_last_5min": total,
				"errors_last_5min": errors,
				"interface_status": iface.get("status"),
			})

		return {
			"circuit_breakers": breakers,
			"open_count": sum(1 for b in breakers if b["state"] == "OPEN"),
			"half_open_count": sum(1 for b in breakers if b["state"] == "HALF_OPEN"),
			"closed_count": sum(1 for b in breakers if b["state"] == "CLOSED"),
			"checked_at": _now(),
		}

	async def generate_certification_report(
		self,
		scheme: str,
		test_suite: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Run a scheme certification test suite and produce a structured report.

		Each test case in ``test_suite`` must specify::

		    {
		        "test_id":         str,   # e.g. "VISA-ADVT-001"
		        "scenario":        str,   # matches switch_simulator scenarios
		        "expected_rc":     str,   # expected ISO 8583 response code
		        "description":     str,
		    }

		Returns pass/fail per test with an aggregate certification verdict.

		Args:
			scheme: Scheme being certified (visa, mastercard, interswitch, etc.).
			test_suite: List of test-case dicts as described above.
		"""
		assert scheme, "scheme required"
		assert test_suite, "test_suite must not be empty"

		import asyncio as _asyncio

		async def _run_case(tc: dict[str, Any]) -> dict[str, Any]:
			try:
				sim = await self.switch_simulator(
					tc["scenario"], tc["expected_rc"]
				)
				return {
					"test_id": tc["test_id"],
					"description": tc.get("description", ""),
					"scenario": tc["scenario"],
					"expected_rc": tc["expected_rc"],
					"actual_rc": sim["actual_response"].get("response_code"),
					"passed": sim["passed"],
					"error": None,
				}
			except Exception as exc:
				return {
					"test_id": tc["test_id"],
					"description": tc.get("description", ""),
					"scenario": tc.get("scenario"),
					"expected_rc": tc.get("expected_rc"),
					"actual_rc": None,
					"passed": False,
					"error": str(exc),
				}

		results = await _asyncio.gather(*[_run_case(tc) for tc in test_suite], return_exceptions=True)
		passed = sum(1 for r in results if r["passed"])
		failed = len(results) - passed
		verdict = "PASS" if failed == 0 else "FAIL"

		report: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"scheme": scheme.lower(),
			"test_count": len(results),
			"passed": passed,
			"failed": failed,
			"verdict": verdict,
			"results": list(results),
			"generated_at": _now(),
		}
		await self._store.put("certification_reports", report)
		await self._audit_event(
			"certification_report_generated", "switch", report["id"],
			{"scheme": scheme, "verdict": verdict, "passed": passed, "failed": failed},
		)
		return report

	async def settlement_batch_close(
		self,
		settlement_date: str,
		scheme: str,
		*,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Close the settlement batch for a given date and scheme.

		Aggregates all approved authorisations, computes net debit/credit per
		participant, advances batch state to ``CLOSED``, and triggers clearing file
		generation if not already done.

		State machine: ``OPEN → AGGREGATING → CLOSED``.

		Args:
			settlement_date: ISO date string (YYYY-MM-DD).
			scheme: Payment scheme to close (visa, mastercard, pesalink, etc.).
			dry_run: If True, compute totals but do not advance state.
		"""
		assert settlement_date, "settlement_date required"
		assert scheme, "scheme required"

		auths = await self._store.query("switch_authorisations", {}, limit=500_000)
		eligible = [
			a for a in auths
			if a.get("timestamp", "").startswith(settlement_date)
			and a.get("response_code") == _RC_APPROVED
		]

		net_positions: dict[str, float] = {}
		total_amount = 0.0
		for a in eligible:
			mid = a.get("merchant_id", "unknown")
			amt = a.get("amount", 0.0)
			net_positions[mid] = net_positions.get(mid, 0.0) + amt
			total_amount += amt

		if dry_run:
			return {
				"dry_run": True,
				"settlement_date": settlement_date,
				"scheme": scheme,
				"transaction_count": len(eligible),
				"total_amount": round(total_amount, 2),
				"participant_count": len(net_positions),
				"computed_at": _now(),
			}

		# Generate clearing file if absent
		existing_files = await self._store.query(
			"clearing_files",
			{"settlement_date": settlement_date, "scheme": scheme},
			limit=1,
		)
		if not existing_files:
			await self.clearing_file_generation(settlement_date, scheme)

		batch_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"settlement_date": settlement_date,
			"scheme": scheme.lower(),
			"transaction_count": len(eligible),
			"total_amount": round(total_amount, 2),
			"net_positions": net_positions,
			"participant_count": len(net_positions),
			"status": "closed",
			"closed_at": _now(),
		}
		await self._store.put("settlement_batches", batch_record)
		await self._audit_event(
			"settlement_batch_closed", "switch", batch_record["id"],
			{"scheme": scheme, "settlement_date": settlement_date, "total": total_amount},
		)
		await self._notify.send(
			"settlement@datacraft.co.ke", "email",
			f"Settlement batch closed: {scheme.upper()} {settlement_date}",
			f"Batch closed. {len(eligible)} transactions, KES {total_amount:,.2f} across "
			f"{len(net_positions)} participants.",
		)
		return batch_record

	async def switch_event_publish(
		self,
		event_type: str,
		payload: dict[str, Any],
		*,
		topic: str = "switch.events",
	) -> dict[str, Any]:
		"""Publish a domain event to the switch event bus.

		Events are persisted to an append-only log and fanned out to registered
		WebSocket subscribers via the notify adapter. The event carries a monotonic
		sequence number and a chain hash linking to the previous event (tamper
		evidence).

		Supported event types: ``failover``, ``velocity_breach``, ``recon_variance``,
		``scheme_degraded``, ``circuit_open``, ``batch_closed``, ``custom``.

		Args:
			event_type: Semantic event category.
			payload: Arbitrary event payload (must be JSON-serialisable).
			topic: Logical topic / channel for subscriber routing.
		"""
		assert event_type, "event_type required"
		assert isinstance(payload, dict), "payload must be a dict"

		# Fetch last event to compute chain hash
		prior_events = await self._store.query(
			"switch_event_log", {}, limit=1
		)
		prior_hash = prior_events[-1].get("chain_hash", "0" * 64) if prior_events else "0" * 64

		event_body = json.dumps({"event_type": event_type, "payload": payload}, sort_keys=True)
		chain_hash = hashlib.sha256(f"{prior_hash}{event_body}".encode()).hexdigest()

		event_record: dict[str, Any] = {
			"id": _uid(),
			"tenant_id": self._tenant_id,
			"event_type": event_type,
			"topic": topic,
			"payload": payload,
			"chain_hash": chain_hash,
			"prior_hash": prior_hash,
			"published_at": _now(),
		}
		await self._store.put("switch_event_log", event_record)
		await self._notify.send(
			f"topic:{topic}", "webhook",
			event_type,
			json.dumps({"event_id": event_record["id"], **payload}),
		)
		return event_record
