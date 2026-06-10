"""Executable service layer for APG Cross-Border Remittance.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import statistics
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FRAUD_DECISIONS,
		SUPPORTED_PAYOUT_METHODS,
		SUPPORTED_PURPOSE_CODES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import RemittanceEvidence, RemittanceQuote, RemittanceRefund, RemittanceTransfer
	from .remittance_runtime import (
		corridor_key,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
		normalize_rate,
		payout_state,
		transfer_band,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FRAUD_DECISIONS,
		SUPPORTED_PAYOUT_METHODS,
		SUPPORTED_PURPOSE_CODES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import RemittanceEvidence, RemittanceQuote, RemittanceRefund, RemittanceTransfer  # type: ignore
	from remittance_runtime import (  # type: ignore
		corridor_key,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
		normalize_rate,
		payout_state,
		transfer_band,
	)


# ---------------------------------------------------------------------------
# FX rate table: (send_currency, receive_currency) -> indicative rate
# ---------------------------------------------------------------------------
_FX_RATES: dict[tuple[str, str], float] = {
	("KES", "USD"): 0.00775,
	("USD", "KES"): 129.0,
	("KES", "UGX"): 28.5,
	("UGX", "KES"): 0.0351,
	("KES", "TZS"): 18.4,
	("TZS", "KES"): 0.0543,
	("USD", "UGX"): 3720.0,
	("UGX", "USD"): 0.000269,
	("USD", "TZS"): 2550.0,
	("TZS", "USD"): 0.000392,
	("KES", "ETB"): 0.735,
	("ETB", "KES"): 1.36,
	("USD", "ETB"): 94.8,
	("ETB", "USD"): 0.01055,
	("GBP", "KES"): 163.5,
	("KES", "GBP"): 0.00612,
	("EUR", "KES"): 140.2,
	("KES", "EUR"): 0.00713,
	("USD", "GBP"): 0.787,
	("GBP", "USD"): 1.270,
	("USD", "EUR"): 0.921,
	("EUR", "USD"): 1.086,
}

# Corridor fee schedule: (send_country, receive_country) -> fee_percent
_CORRIDOR_FEES: dict[tuple[str, str], float] = {
	("KE", "UG"): 1.2,
	("KE", "TZ"): 1.2,
	("KE", "ET"): 1.5,
	("KE", "RW"): 1.3,
	("KE", "NG"): 2.0,
	("KE", "GH"): 2.0,
	("US", "KE"): 2.5,
	("GB", "KE"): 2.5,
	("DE", "KE"): 2.8,
	("US", "UG"): 2.8,
	("US", "TZ"): 2.8,
	("US", "ET"): 3.0,
	("US", "NG"): 3.0,
	("US", "GH"): 2.8,
}

# Partner routing table: corridor_key -> preferred partner
_PARTNER_ROUTING: dict[str, str] = {
	"KE-UG-KES-UGX": "equity_bank_ug",
	"KE-TZ-KES-TZS": "crdb_tz",
	"KE-ET-KES-ETB": "cbe_ethiopia",
	"US-KE-USD-KES": "flutterwave",
	"GB-KE-GBP-KES": "wise",
	"DE-KE-EUR-KES": "wise",
	"US-NG-USD-NGN": "flutterwave",
	"US-GH-USD-GHS": "flutterwave",
}

# CBK reporting thresholds (KES)
_CBK_CTR_THRESHOLD = 1_000_000
_CBK_STR_THRESHOLD = 500_000


class RemittanceService:
	"""Full-featured remittance runtime for generated APG applications.

	Handles FX quoting, compliance, payout routing, tracking, receipts,
	analytics, and CBK regulatory reporting for East African corridors.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		# in-memory stores (replaced by store adapter when injected)
		self.quotes: dict[str, RemittanceQuote] = {}
		self.transfers: dict[str, RemittanceTransfer] = {}
		self.refunds: dict[str, RemittanceRefund] = {}
		self.evidence: dict[str, RemittanceEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# runtime caches
		self._compliance_cache: dict[str, dict[str, Any]] = {}
		self._notification_log: list[dict[str, Any]] = []
		self._cbk_reports: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# FX & quoting
	# ------------------------------------------------------------------

	async def get_fx_quote(
		self,
		send_currency: str,
		receive_currency: str,
		amount: float | int | str,
		*,
		include_fee: bool = True,
		send_country: str = "",
		receive_country: str = "",
	) -> dict[str, Any]:
		"""Return an indicative FX quote with fee for a given corridor."""
		send_currency = normalize_currency(send_currency)
		receive_currency = normalize_currency(receive_currency)
		send_amount = normalize_amount(amount)
		assert send_amount > 0, "amount must be positive"

		pair = (send_currency, receive_currency)
		rate = _FX_RATES.get(pair)
		if rate is None:
			# attempt inverse
			inv_pair = (receive_currency, send_currency)
			inv_rate = _FX_RATES.get(inv_pair)
			if inv_rate is None:
				# cross via USD
				to_usd = _FX_RATES.get((send_currency, "USD"), 1.0)
				from_usd = _FX_RATES.get(("USD", receive_currency), 1.0)
				rate = to_usd * from_usd
			else:
				rate = 1.0 / inv_rate

		fee_pct = 0.0
		if include_fee and send_country and receive_country:
			sc = normalize_country(send_country)
			rc = normalize_country(receive_country)
			fee_pct = _CORRIDOR_FEES.get((sc, rc), 1.8)

		fee_amount = round(send_amount * fee_pct / 100, 4)
		net_send = send_amount - fee_amount
		receive_amount = round(net_send * rate, 4)
		expiry = (
			datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)
		).isoformat()

		self._audit(self.tenant_id, "fx_quote_generated", f"{send_currency}-{receive_currency}")
		return {
			"send_currency": send_currency,
			"receive_currency": receive_currency,
			"send_amount": send_amount,
			"fee_amount": fee_amount,
			"fee_pct": fee_pct,
			"net_send_amount": net_send,
			"fx_rate": rate,
			"receive_amount": receive_amount,
			"quote_expires_at": expiry,
			"rate_source": "indicative",
		}

	async def initiate_remittance(
		self,
		sender_id: str,
		recipient: dict[str, Any],
		amount: float | int | str,
		send_currency: str,
		receive_currency: str,
		corridor: str,
		*,
		purpose_code: str = "family_support",
		payout_method: str = "mobile_money",
		source_of_funds: str = "salary",
		funding_reference: str = "",
		aml_screen_id: str = "",
		fraud_decision: str = "pass",
		human_approval: str = "",
	) -> dict[str, Any]:
		"""High-level entry point: quote + compliance + route + create transfer."""
		send_currency = normalize_currency(send_currency)
		receive_currency = normalize_currency(receive_currency)
		send_amount = normalize_amount(amount)
		parts = corridor.upper().split("-")
		send_country = parts[0] if len(parts) >= 1 else "KE"
		receive_country = parts[1] if len(parts) >= 2 else "UG"

		# 1. Get FX quote
		fx = await self.get_fx_quote(
			send_currency,
			receive_currency,
			send_amount,
			send_country=send_country,
			receive_country=receive_country,
		)

		# 2. Compliance pre-check
		compliance = await self.compliance_check(sender_id, receive_country, send_amount)
		if compliance["blocked"]:
			raise PermissionError(f"compliance_blocked: {compliance['reason']}")

		# 3. Route to partner
		routing = await self.partner_routing(corridor)

		# 4. Build IDs
		import uuid
		quote_id = str(uuid.uuid4())
		transfer_id = str(uuid.uuid4())
		beneficiary_ref = recipient.get("id", str(uuid.uuid4()))

		# 5. Create quote record
		quote_rec = self.create_quote(
			quote_id,
			self.tenant_id,
			send_country,
			receive_country,
			send_currency,
			receive_currency,
			send_amount,
			fx["fx_rate"],
			fx["fee_amount"],
			fx["quote_expires_at"],
		)

		# 6. Create transfer record
		transfer_rec = self.create_transfer(
			transfer_id,
			self.tenant_id,
			quote_id,
			sender_id,
			beneficiary_ref,
			sender_id,  # kyc id same as sender for now
			beneficiary_ref,
			funding_reference or f"fund-{sender_id}",
			payout_method,
			purpose_code,
			source_of_funds,
			aml_screen_id or f"aml-{sender_id}",
			fraud_decision,
			human_approval=human_approval,
		)

		self._audit(self.tenant_id, "remittance_initiated", transfer_id)
		return {
			"transfer_id": transfer_id,
			"quote_id": quote_id,
			"status": transfer_rec["status"],
			"fx_rate": fx["fx_rate"],
			"send_amount": send_amount,
			"send_currency": send_currency,
			"receive_amount": fx["receive_amount"],
			"receive_currency": receive_currency,
			"fee_amount": fx["fee_amount"],
			"partner": routing["partner"],
			"compliance": compliance,
			"recipient": recipient,
			"payout_method": payout_method,
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Compliance
	# ------------------------------------------------------------------

	async def compliance_check(
		self,
		sender_id: str,
		recipient_country: str,
		amount: float | int | str,
		*,
		purpose_code: str = "family_support",
	) -> dict[str, Any]:
		"""AML/CFT/sanctions pre-check for a remittance transaction."""
		send_amount = normalize_amount(amount)
		rc = normalize_country(recipient_country)

		# Sanctions list stub (in production, query OFAC/UN/EU lists)
		sanctioned_countries = {"KP", "IR", "SY", "CU", "RU"}
		country_blocked = rc in sanctioned_countries

		# CTR threshold check
		requires_ctr = send_amount >= _CBK_CTR_THRESHOLD
		requires_str = send_amount >= _CBK_STR_THRESHOLD and purpose_code in {
			"unknown",
			"other",
		}

		blocked = country_blocked
		reason = "sanctioned_country" if country_blocked else ""

		cache_key = f"{sender_id}:{rc}:{int(send_amount)}"
		result: dict[str, Any] = {
			"sender_id": sender_id,
			"recipient_country": rc,
			"amount": send_amount,
			"blocked": blocked,
			"reason": reason,
			"requires_ctr": requires_ctr,
			"requires_str": requires_str,
			"sanctions_checked": True,
			"aml_risk_score": 15 if not blocked else 95,
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._compliance_cache[cache_key] = result
		self._audit(self.tenant_id, "compliance_check_performed", sender_id)
		return result

	# ------------------------------------------------------------------
	# Partner routing
	# ------------------------------------------------------------------

	async def partner_routing(self, corridor: str) -> dict[str, Any]:
		"""Select the optimal payout partner for a given corridor string."""
		key = corridor.upper()
		partner = _PARTNER_ROUTING.get(key, "default_partner")

		# Partner capability matrix
		capabilities: dict[str, list[str]] = {
			"equity_bank_ug": ["bank_transfer", "mobile_money", "cash_pickup"],
			"crdb_tz": ["bank_transfer", "mobile_money"],
			"cbe_ethiopia": ["bank_transfer"],
			"flutterwave": ["mobile_money", "bank_transfer", "card"],
			"wise": ["bank_transfer", "debit_card"],
			"default_partner": ["bank_transfer"],
		}
		sla_hours: dict[str, int] = {
			"equity_bank_ug": 2,
			"crdb_tz": 4,
			"cbe_ethiopia": 24,
			"flutterwave": 1,
			"wise": 2,
			"default_partner": 48,
		}

		self._audit(self.tenant_id, "partner_routed", corridor)
		return {
			"corridor": key,
			"partner": partner,
			"payout_methods": capabilities.get(partner, ["bank_transfer"]),
			"sla_hours": sla_hours.get(partner, 48),
			"failover_partner": "default_partner",
		}

	# ------------------------------------------------------------------
	# Tracking
	# ------------------------------------------------------------------

	async def track_remittance(self, transaction_id: str) -> dict[str, Any]:
		"""Return full lifecycle status of a remittance transfer."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			return {
				"transaction_id": transaction_id,
				"found": False,
				"status": "not_found",
			}

		# Build timeline from audit events
		timeline = [
			ev for ev in self.audit_events
			if ev.get("reference_id") == transaction_id
		]

		self._audit(self.tenant_id, "remittance_tracked", transaction_id)
		return {
			"transaction_id": transaction_id,
			"found": True,
			"status": transfer.status,
			"payout_method": transfer.payout_method,
			"provider_receipt": getattr(transfer, "provider_receipt", None),
			"settlement_reference": getattr(transfer, "settlement_reference", None),
			"timeline": timeline,
			"quote_id": transfer.quote_id,
			"sender_reference": transfer.sender_reference,
			"beneficiary_reference": transfer.beneficiary_reference,
			"tracked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Recipient notifications
	# ------------------------------------------------------------------

	async def recipient_notification(
		self,
		transaction_id: str,
		*,
		channel: str = "sms",
		language: str = "en",
	) -> dict[str, Any]:
		"""Dispatch a recipient notification for a transfer event."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")

		templates: dict[str, dict[str, str]] = {
			"en": {
				"paid": "Your remittance of {amount} has been paid. Ref: {ref}",
				"review_required": "Your remittance is under review. We will update you shortly.",
				"refund_filed": "Your remittance has been refunded. Ref: {ref}",
			},
			"sw": {
				"paid": "Uhamisho wako wa {amount} umelipwa. Kumbukumbu: {ref}",
				"review_required": "Uhamisho wako uko chini ya ukaguzi.",
				"refund_filed": "Uhamisho wako ulirudishwa. Kumbukumbu: {ref}",
			},
		}

		lang_templates = templates.get(language, templates["en"])
		template = lang_templates.get(transfer.status, "Status update: {status}")
		message = template.format(
			amount=transaction_id,
			ref=transaction_id,
			status=transfer.status,
		)

		notification = {
			"transaction_id": transaction_id,
			"channel": channel,
			"language": language,
			"recipient_ref": transfer.beneficiary_reference,
			"message": message,
			"status": transfer.status,
			"dispatched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._notification_log.append(notification)
		self._audit(self.tenant_id, "recipient_notification_sent", transaction_id)
		return notification

	# ------------------------------------------------------------------
	# Payout methods
	# ------------------------------------------------------------------

	async def payout_methods(self, receive_country: str) -> dict[str, Any]:
		"""List available payout methods for a receive country."""
		rc = normalize_country(receive_country)

		country_methods: dict[str, list[dict[str, Any]]] = {
			"UG": [
				{"method": "mobile_money", "providers": ["MTN_UG", "Airtel_UG"], "max_amount": 5_000_000},
				{"method": "bank_transfer", "providers": ["Equity_UG", "Stanbic_UG"], "max_amount": 50_000_000},
				{"method": "cash_pickup", "providers": ["Western_Union", "MoneyGram"], "max_amount": 2_000_000},
			],
			"TZ": [
				{"method": "mobile_money", "providers": ["Vodacom_TZ", "Airtel_TZ", "Tigo_TZ"], "max_amount": 3_000_000},
				{"method": "bank_transfer", "providers": ["CRDB", "NMB"], "max_amount": 50_000_000},
				{"method": "cash_pickup", "providers": ["Western_Union"], "max_amount": 1_500_000},
			],
			"ET": [
				{"method": "bank_transfer", "providers": ["CBE", "Awash_Bank"], "max_amount": 100_000},
				{"method": "mobile_money", "providers": ["Telebirr"], "max_amount": 50_000},
			],
			"NG": [
				{"method": "bank_transfer", "providers": ["GTBank", "Zenith", "UBA"], "max_amount": 5_000_000},
				{"method": "mobile_money", "providers": ["OPay", "PalmPay"], "max_amount": 1_000_000},
			],
			"GH": [
				{"method": "mobile_money", "providers": ["MTN_GH", "AirtelTigo_GH", "Vodafone_GH"], "max_amount": 50_000},
				{"method": "bank_transfer", "providers": ["Ecobank_GH", "GCB"], "max_amount": 500_000},
			],
		}

		methods = country_methods.get(rc, [
			{"method": "bank_transfer", "providers": ["default"], "max_amount": 10_000_000}
		])

		self._audit(self.tenant_id, "payout_methods_queried", rc)
		return {
			"receive_country": rc,
			"methods": methods,
			"currency_supported": rc in SUPPORTED_COUNTRIES,
		}

	async def deliver_to_mobile_money(
		self,
		transaction_id: str,
		phone: str,
		*,
		provider: str = "auto",
		pin_reference: str = "",
	) -> dict[str, Any]:
		"""Initiate mobile money payout for a remittance transfer."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")
		assert transfer.status in {"review_required", "pending", "created"}, (
			f"cannot deliver transfer in status: {transfer.status}"
		)
		assert phone, "phone number required"

		# Normalize phone — strip leading zeros, prepend country code heuristic
		normalized_phone = phone.strip().replace(" ", "").replace("-", "")
		if normalized_phone.startswith("0"):
			normalized_phone = "254" + normalized_phone[1:]

		# Determine network from prefix
		ke_safaricom_prefixes = {"07", "01"}
		prefix = phone[:2] if len(phone) >= 2 else ""
		network = "safaricom" if prefix in ke_safaricom_prefixes else provider

		delivery = {
			"transaction_id": transaction_id,
			"payout_channel": "mobile_money",
			"phone": normalized_phone,
			"network": network,
			"provider": provider,
			"delivery_status": "dispatched",
			"dispatch_reference": f"mmtx-{hashlib.sha256(transaction_id.encode()).hexdigest()[:12]}",
			"dispatched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

		transfer.status = "paid"
		self._audit(self.tenant_id, "mobile_money_payout_dispatched", transaction_id)
		return delivery

	async def bank_payout(
		self,
		transaction_id: str,
		bank_details: dict[str, Any],
	) -> dict[str, Any]:
		"""Initiate bank wire payout for a remittance transfer."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")

		required_fields = {"account_number", "bank_code"}
		missing = required_fields - set(bank_details.keys())
		assert not missing, f"missing bank details: {missing}"

		payout_ref = f"bktx-{hashlib.sha256(transaction_id.encode()).hexdigest()[:12]}"
		result = {
			"transaction_id": transaction_id,
			"payout_channel": "bank_transfer",
			"bank_code": bank_details.get("bank_code"),
			"account_number": bank_details.get("account_number"),
			"account_name": bank_details.get("account_name", ""),
			"swift_code": bank_details.get("swift_code", ""),
			"payout_reference": payout_ref,
			"delivery_status": "submitted",
			"expected_settlement_hours": 24,
			"dispatched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

		transfer.status = "paid"
		self._audit(self.tenant_id, "bank_payout_submitted", transaction_id)
		return result

	async def cash_pickup(
		self,
		transaction_id: str,
		agent_id: str,
		*,
		pickup_code: str = "",
	) -> dict[str, Any]:
		"""Register a cash pickup against a remittance transfer."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")
		assert agent_id, "agent_id required"

		if not pickup_code:
			pickup_code = hashlib.sha256(f"{transaction_id}{agent_id}".encode()).hexdigest()[:8].upper()

		result = {
			"transaction_id": transaction_id,
			"payout_channel": "cash_pickup",
			"agent_id": agent_id,
			"pickup_code": pickup_code,
			"delivery_status": "ready_for_pickup",
			"valid_hours": 72,
			"registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

		transfer.status = "paid"
		self._audit(self.tenant_id, "cash_pickup_registered", transaction_id)
		return result

	# ------------------------------------------------------------------
	# Receipt
	# ------------------------------------------------------------------

	async def remittance_receipt(self, transaction_id: str) -> dict[str, Any]:
		"""Generate a structured receipt document for a completed transfer."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")

		quote = self.quotes.get(transfer.quote_id) if hasattr(transfer, "quote_id") else None

		receipt: dict[str, Any] = {
			"receipt_number": f"RCPT-{transaction_id[:8].upper()}",
			"transaction_id": transaction_id,
			"status": transfer.status,
			"sender_reference": transfer.sender_reference,
			"beneficiary_reference": transfer.beneficiary_reference,
			"payout_method": transfer.payout_method,
			"purpose_code": transfer.purpose_code,
			"provider_receipt": getattr(transfer, "provider_receipt", None),
			"settlement_reference": getattr(transfer, "settlement_reference", None),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

		if quote is not None:
			receipt.update({
				"send_currency": quote.source_currency,
				"receive_currency": quote.destination_currency,
				"send_amount": quote.send_amount,
				"fx_rate": quote.fx_rate,
				"fee_amount": quote.fee_amount,
				"receive_amount": round(
					(quote.send_amount - quote.fee_amount) * quote.fx_rate, 4
				),
				"send_country": quote.source_country,
				"receive_country": quote.destination_country,
			})

		self._audit(self.tenant_id, "remittance_receipt_generated", transaction_id)
		return receipt

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	async def corridor_analytics(
		self,
		period: str,
		*,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute corridor-level volume, value, and fee analytics for a period."""
		tid = tenant_id or self.tenant_id
		transfers = [t for t in self.transfers.values() if t.tenant_id == tid]
		quotes_map = {q.id: q for q in self.quotes.values() if q.tenant_id == tid}

		corridor_stats: dict[str, dict[str, Any]] = {}
		for transfer in transfers:
			quote = quotes_map.get(transfer.quote_id) if hasattr(transfer, "quote_id") else None
			if quote is None:
				continue
			ck = corridor_key(
				quote.source_country,
				quote.destination_country,
				quote.source_currency,
				quote.destination_currency,
			)
			if ck not in corridor_stats:
				corridor_stats[ck] = {
					"corridor": ck,
					"transfer_count": 0,
					"total_send_amount": 0.0,
					"total_fee_amount": 0.0,
					"paid_count": 0,
					"send_amounts": [],
				}
			stats = corridor_stats[ck]
			stats["transfer_count"] += 1
			stats["total_send_amount"] += quote.send_amount
			stats["total_fee_amount"] += quote.fee_amount
			stats["send_amounts"].append(quote.send_amount)
			if transfer.status == "paid":
				stats["paid_count"] += 1

		# Compute derived stats
		for ck, stats in corridor_stats.items():
			amounts = stats.pop("send_amounts", [])
			stats["avg_send_amount"] = statistics.mean(amounts) if amounts else 0.0
			stats["median_send_amount"] = statistics.median(amounts) if amounts else 0.0
			stats["success_rate_pct"] = (
				100 * stats["paid_count"] / stats["transfer_count"]
				if stats["transfer_count"] > 0 else 0.0
			)

		self._audit(tid, "corridor_analytics_computed", period)
		return {
			"tenant_id": tid,
			"period": period,
			"corridor_count": len(corridor_stats),
			"corridors": list(corridor_stats.values()),
			"total_transfers": len(transfers),
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def cbk_reporting(
		self,
		period: str,
		*,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate Central Bank of Kenya regulatory report for a period.

		Covers: Currency Transaction Reports (CTR) and Suspicious Transaction
		Reports (STR) per CBK/PG/01 guidelines.
		"""
		tid = tenant_id or self.tenant_id
		transfers = [t for t in self.transfers.values() if t.tenant_id == tid]
		quotes_map = {q.id: q for q in self.quotes.values() if q.tenant_id == tid}

		ctr_records: list[dict[str, Any]] = []
		str_records: list[dict[str, Any]] = []
		total_volume = 0.0
		total_value_kes = 0.0

		for transfer in transfers:
			quote = quotes_map.get(transfer.quote_id) if hasattr(transfer, "quote_id") else None
			if quote is None:
				continue

			# Convert to KES for reporting
			pair = (quote.source_currency, "KES")
			rate_to_kes = _FX_RATES.get(pair, 1.0)
			amount_kes = quote.send_amount * rate_to_kes
			total_volume += 1
			total_value_kes += amount_kes

			if amount_kes >= _CBK_CTR_THRESHOLD:
				ctr_records.append({
					"transfer_id": transfer.id,
					"sender_reference": transfer.sender_reference,
					"amount_kes": amount_kes,
					"send_currency": quote.source_currency,
					"receive_country": quote.destination_country,
					"purpose_code": transfer.purpose_code,
					"status": transfer.status,
					"report_type": "CTR",
				})

			if transfer.fraud_decision in {"review", "hold", "block"}:
				str_records.append({
					"transfer_id": transfer.id,
					"sender_reference": transfer.sender_reference,
					"amount_kes": amount_kes,
					"fraud_decision": transfer.fraud_decision,
					"aml_screen_id": transfer.aml_screen_id,
					"report_type": "STR",
				})

		report = {
			"tenant_id": tid,
			"period": period,
			"reporting_institution": "Datacraft",
			"regulatory_framework": "CBK/PG/01",
			"report_generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"total_transactions": int(total_volume),
			"total_value_kes": round(total_value_kes, 2),
			"ctr_count": len(ctr_records),
			"str_count": len(str_records),
			"ctr_records": ctr_records,
			"str_records": str_records,
			"ctr_threshold_kes": _CBK_CTR_THRESHOLD,
			"str_threshold_kes": _CBK_STR_THRESHOLD,
		}
		self._cbk_reports.append(report)
		self._audit(tid, "cbk_report_generated", period)
		return report

	# ------------------------------------------------------------------
	# Existing core methods (preserved from original)
	# ------------------------------------------------------------------

	def create_quote(
		self,
		quote_id: str,
		tenant_id: str,
		source_country: str,
		destination_country: str,
		source_currency: str,
		destination_currency: str,
		send_amount: float | int | str,
		fx_rate: float | int | str,
		fee_amount: float | int | str,
		expiry: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		source_country = normalize_country(source_country)
		destination_country = normalize_country(destination_country)
		source_currency = normalize_currency(source_currency)
		destination_currency = normalize_currency(destination_currency)
		amount = normalize_amount(send_amount)
		rate = normalize_rate(fx_rate)
		fee = normalize_amount(fee_amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "quote_transfer",
			"corridor_supported": (
				source_country in SUPPORTED_COUNTRIES
				and destination_country in SUPPORTED_COUNTRIES
			),
			"same_country": source_country == destination_country,
			"source_currency_supported": source_currency in SUPPORTED_CURRENCIES,
			"destination_currency_supported": destination_currency in SUPPORTED_CURRENCIES,
			"positive_amount": amount > 0,
			"positive_fx_rate": rate > 0,
			"fee_non_negative": fee >= 0,
			"expiry_present": bool(expiry),
		})
		if quote_id in self.quotes:
			raise ValueError(f"remittance quote already exists: {quote_id}")
		quote = RemittanceQuote(
			quote_id, tenant_id, source_country, destination_country,
			source_currency, destination_currency, amount, rate, fee, expiry,
		)
		self.quotes[quote_id] = quote
		self._audit(tenant_id, "remittance_quote_created", quote_id)
		return quote.to_dict() | {
			"corridor": corridor_key(
				source_country, destination_country, source_currency, destination_currency
			),
			"transfer_band": transfer_band(amount),
		}

	def create_transfer(
		self,
		transfer_id: str,
		tenant_id: str,
		quote_id: str,
		sender_reference: str,
		beneficiary_reference: str,
		sender_kyc_id: str,
		beneficiary_kyc_id: str,
		funding_reference: str,
		payout_method: str,
		purpose_code: str,
		source_of_funds: str,
		aml_screen_id: str,
		fraud_decision: str,
		aml_review: bool = False,
		sanctions_hit: bool = False,
		human_approval: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		quote = self._tenant_quote_or_none(quote_id, tenant_id)
		payout_method = normalize_code(payout_method)
		purpose_code = normalize_code(purpose_code)
		fraud_decision = normalize_code(fraud_decision)
		high_value = quote.send_amount >= 100000 if quote else False
		fraud_review = fraud_decision in {"review", "hold"}
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_transfer",
			"quote_present": quote is not None,
			"sender_present": bool(sender_reference),
			"beneficiary_present": bool(beneficiary_reference),
			"sender_kyc_present": bool(sender_kyc_id),
			"beneficiary_kyc_present": bool(beneficiary_kyc_id),
			"funding_present": bool(funding_reference),
			"payout_method_supported": payout_method in SUPPORTED_PAYOUT_METHODS,
			"purpose_code_supported": purpose_code in SUPPORTED_PURPOSE_CODES,
			"source_of_funds_present": bool(source_of_funds),
			"aml_screen_present": bool(aml_screen_id),
			"sanctions_hit": sanctions_hit,
			"fraud_decision_supported": fraud_decision in SUPPORTED_FRAUD_DECISIONS,
			"fraud_blocked": fraud_decision == "block",
			"aml_review": aml_review,
			"fraud_review": fraud_review,
			"high_value": high_value,
			"human_approval_recorded": bool(human_approval),
		})
		if transfer_id in self.transfers:
			raise ValueError(f"remittance transfer already exists: {transfer_id}")
		transfer = RemittanceTransfer(
			transfer_id, tenant_id, quote_id, sender_reference, beneficiary_reference,
			sender_kyc_id, beneficiary_kyc_id, funding_reference, payout_method,
			purpose_code, source_of_funds, aml_screen_id, fraud_decision,
			payout_state(fraud_decision, aml_review), human_approval,
		)
		self.transfers[transfer_id] = transfer
		self._audit(tenant_id, "remittance_transfer_created", transfer_id)
		return transfer.to_dict()

	def release_payout(
		self,
		transfer_id: str,
		tenant_id: str,
		provider_receipt: str,
		settlement_reference: str,
	) -> dict[str, Any]:
		transfer = self._tenant_transfer_or_none(transfer_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_payout",
			"transfer_present": transfer is not None,
			"provider_receipt_present": bool(provider_receipt),
			"settlement_reference_present": bool(settlement_reference),
		})
		assert transfer is not None
		transfer.status = "paid"
		transfer.provider_receipt = provider_receipt
		transfer.settlement_reference = settlement_reference
		self._audit(tenant_id, "remittance_payout_released", transfer_id)
		return transfer.to_dict()

	def file_refund(
		self,
		refund_id: str,
		tenant_id: str,
		transfer_id: str,
		reason: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		transfer = self._tenant_transfer_or_none(transfer_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "file_refund",
			"transfer_present": transfer is not None,
			"reason_present": bool(reason),
			"reviewer_present": bool(reviewer_id),
		})
		if refund_id in self.refunds:
			raise ValueError(f"remittance refund already exists: {refund_id}")
		refund = RemittanceRefund(refund_id, tenant_id, transfer_id, reason, reviewer_id)
		self.refunds[refund_id] = refund
		if transfer is not None:
			transfer.status = "refund_filed"
		self._audit(tenant_id, "remittance_refund_filed", refund_id)
		return refund.to_dict()

	def register_remittance_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_remittance_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(
			agent_id, tenant_id, "agent", agent_id, "registered",
			{"name": name, "runtime": runtime, "role": role, "scope": scope},
		)
		self._audit(tenant_id, "remittance_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "remittance_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.remittance.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		quotes = [q for q in self.quotes.values() if q.tenant_id == tenant_id]
		transfers = [t for t in self.transfers.values() if t.tenant_id == tenant_id]
		refunds = [r for r in self.refunds.values() if r.tenant_id == tenant_id]
		total_value = sum(
			self.quotes[t.quote_id].send_amount
			for t in transfers
			if hasattr(t, "quote_id") and t.quote_id in self.quotes
		)
		return {
			"tenant_id": tenant_id,
			"quote_count": len(quotes),
			"transfer_count": len(transfers),
			"paid_count": sum(1 for t in transfers if t.status == "paid"),
			"review_required_count": sum(1 for t in transfers if t.status == "review_required"),
			"refund_count": len(refunds),
			"total_send_value": total_value,
			"notification_count": len(self._notification_log),
			"cbk_report_count": len(self._cbk_reports),
			"audit_event_count": sum(1 for ev in self.audit_events if ev["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_transfers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		transfers = self.transfers.values()
		if tenant_id is not None:
			transfers = [t for t in transfers if t.tenant_id == tenant_id]  # type: ignore[assignment]
		return [t.to_dict() for t in sorted(transfers, key=lambda t: t.id)]

	def list_quotes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		quotes = self.quotes.values()
		if tenant_id is not None:
			quotes = [q for q in quotes if q.tenant_id == tenant_id]  # type: ignore[assignment]
		return [q.to_dict() for q in sorted(quotes, key=lambda q: q.id)]

	async def list_notifications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return notification log, optionally filtered by tenant via transfer lookup."""
		return list(self._notification_log)

	async def cancel_transfer(
		self,
		transaction_id: str,
		reason: str,
		*,
		reviewer_id: str = "system",
	) -> dict[str, Any]:
		"""Cancel a pending transfer before payout is dispatched."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")
		assert transfer.status not in {"paid"}, (
			f"cannot cancel transfer in status: {transfer.status}"
		)
		assert reason, "cancellation reason required"
		transfer.status = "cancelled"
		self._audit(self.tenant_id, "transfer_cancelled", transaction_id)
		return {
			"transaction_id": transaction_id,
			"status": "cancelled",
			"reason": reason,
			"reviewer_id": reviewer_id,
			"cancelled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def retry_payout(
		self,
		transaction_id: str,
		*,
		new_payout_method: str | None = None,
	) -> dict[str, Any]:
		"""Re-queue a failed payout for retry, optionally with a different method."""
		transfer = self.transfers.get(transaction_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transaction_id}")
		assert transfer.status in {"failed", "review_required"}, (
			f"retry not applicable for status: {transfer.status}"
		)
		if new_payout_method:
			transfer.payout_method = normalize_code(new_payout_method)
		transfer.status = "pending"
		self._audit(self.tenant_id, "payout_retry_queued", transaction_id)
		return {
			"transaction_id": transaction_id,
			"status": "pending",
			"payout_method": transfer.payout_method,
			"retry_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def exchange_rate_history(
		self,
		send_currency: str,
		receive_currency: str,
		days: int = 7,
	) -> dict[str, Any]:
		"""Return simulated historical exchange rate series for a currency pair."""
		send_currency = normalize_currency(send_currency)
		receive_currency = normalize_currency(receive_currency)
		pair = (send_currency, receive_currency)
		base_rate = _FX_RATES.get(pair, 1.0)

		import random
		random.seed(42)
		series = []
		for i in range(days):
			date = (
				datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days - i)
			).date().isoformat()
			jitter = random.uniform(-0.005, 0.005)
			series.append({"date": date, "rate": round(base_rate * (1 + jitter), 6)})

		self._audit(self.tenant_id, "rate_history_queried", f"{send_currency}-{receive_currency}")
		return {
			"send_currency": send_currency,
			"receive_currency": receive_currency,
			"days": days,
			"current_rate": base_rate,
			"series": series,
		}

	async def sender_transaction_history(
		self,
		sender_id: str,
		limit: int = 20,
		*,
		tenant_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Retrieve transaction history for a specific sender."""
		tid = tenant_id or self.tenant_id
		results = [
			t.to_dict()
			for t in self.transfers.values()
			if t.tenant_id == tid and t.sender_reference == sender_id
		]
		results.sort(key=lambda r: r.get("id", ""), reverse=True)
		return results[:limit]

	async def fee_schedule(
		self,
		send_country: str,
		receive_country: str,
		send_currency: str,
		receive_currency: str,
	) -> dict[str, Any]:
		"""Return the fee schedule for a corridor."""
		sc = normalize_country(send_country)
		rc = normalize_country(receive_country)
		fee_pct = _CORRIDOR_FEES.get((sc, rc), 1.8)

		tiers = [
			{"min": 0, "max": 10_000, "fee_pct": fee_pct + 0.5},
			{"min": 10_001, "max": 50_000, "fee_pct": fee_pct},
			{"min": 50_001, "max": 200_000, "fee_pct": fee_pct - 0.3},
			{"min": 200_001, "max": None, "fee_pct": fee_pct - 0.5},
		]

		self._audit(self.tenant_id, "fee_schedule_queried", f"{sc}-{rc}")
		return {
			"send_country": sc,
			"receive_country": rc,
			"send_currency": normalize_currency(send_currency),
			"receive_currency": normalize_currency(receive_currency),
			"base_fee_pct": fee_pct,
			"tiers": tiers,
			"fx_spread_pct": 0.5,
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return remittance service health status."""
		return {
			"service": "remittance", "status": "healthy",
			"active_transfers": sum(1 for t in self.transfers.values() if t.status in {"pending", "review_required"}),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def bulk_initiate_remittances(self, remittances: list[dict[str, Any]]) -> dict[str, Any]:
		"""Initiate multiple remittances in bulk."""
		processed, errors = [], []
		for r in remittances:
			try:
				rec = await self.initiate_remittance(
					sender_id=r["sender_id"], recipient=r["recipient"],
					amount=r["amount"], send_currency=r.get("send_currency", "KES"),
					receive_currency=r.get("receive_currency", "KES"), corridor=r["corridor"],
					purpose_code=r.get("purpose_code", "family_support"),
				)
				processed.append(rec["transfer_id"])
			except Exception as exc:
				errors.append({"input": r, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "transfer_ids": processed}

	async def beneficiary_management(self, sender_id: str, beneficiary: dict[str, Any], action: str = "add") -> dict[str, Any]:
		"""Manage beneficiary contacts for a sender (add/update/remove)."""
		assert action in {"add", "update", "remove"}, f"unsupported action: {action}"
		beneficiary_id = f"ben-{sender_id[:8]}-{beneficiary.get('name', '').replace(' ', '_')[:12]}"
		record: dict[str, Any] = {
			"beneficiary_id": beneficiary_id, "sender_id": sender_id,
			"beneficiary": beneficiary, "action": action,
			"status": "active" if action != "remove" else "removed",
			"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, f"beneficiary_{action}", beneficiary_id)
		return record

	async def compliance_screening_batch(self, sender_ids: list[str], country: str) -> list[dict[str, Any]]:
		"""Run compliance checks for multiple senders in batch."""
		results = []
		for sid in sender_ids:
			result = await self.compliance_check(sid, country, 100_000)
			results.append(result)
		return results

	async def exchange_rate_alert_setup(self, sender_id: str, send_currency: str, receive_currency: str, target_rate: float) -> dict[str, Any]:
		"""Set up an exchange rate alert for a sender."""
		pair = (send_currency.upper(), receive_currency.upper())
		current_rate = _FX_RATES.get(pair, 0.0)
		record: dict[str, Any] = {
			"alert_id": f"rate-alert-{sender_id[:8]}",
			"sender_id": sender_id, "currency_pair": f"{send_currency}/{receive_currency}",
			"target_rate": target_rate, "current_rate": current_rate,
			"triggered": current_rate >= target_rate if target_rate > 0 else False,
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, "rate_alert_created", record["alert_id"])
		return record

	async def diaspora_bond_subscription(self, sender_id: str, bond_code: str, amount: float, currency: str, tenor_years: int) -> dict[str, Any]:
		"""Subscribe to a government diaspora bond via the remittance platform."""
		assert sender_id and bond_code and amount > 0 and tenor_years > 0
		subscription: dict[str, Any] = {
			"subscription_id": f"bond-{sender_id[:8]}-{bond_code}",
			"sender_id": sender_id, "bond_code": bond_code,
			"amount": amount, "currency": currency, "tenor_years": tenor_years,
			"coupon_rate_pct": 9.5,
			"maturity_date": datetime.datetime.now(datetime.timezone.utc).replace(year=datetime.datetime.now().year + tenor_years).isoformat()[:10],
			"status": "subscribed", "subscribed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, "diaspora_bond_subscribed", subscription["subscription_id"])
		return subscription

	async def mobile_money_rate(self, send_currency: str, receive_currency: str, provider: str = "mpesa") -> dict[str, Any]:
		"""Return current mobile money transfer rate for a corridor and provider."""
		pair = (send_currency.upper(), receive_currency.upper())
		base_rate = _FX_RATES.get(pair, 0.0)
		provider_spread = {"mpesa": 0.005, "airtel_money": 0.008, "mtn_momo": 0.010}.get(provider.lower(), 0.01)
		effective_rate = base_rate * (1 - provider_spread)
		return {
			"provider": provider, "send_currency": send_currency, "receive_currency": receive_currency,
			"base_rate": base_rate, "provider_spread_pct": provider_spread * 100,
			"effective_rate": round(effective_rate, 6), "source": "indicative",
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def partner_performance_report(self, period: str) -> dict[str, Any]:
		"""Generate a performance report for payout partners."""
		transfers = list(self.transfers.values())
		by_partner: dict[str, dict[str, Any]] = {}
		for t in transfers:
			partner = getattr(t, "payout_method", "unknown")
			if partner not in by_partner:
				by_partner[partner] = {"count": 0, "paid": 0, "failed": 0}
			by_partner[partner]["count"] += 1
			if t.status == "paid":
				by_partner[partner]["paid"] += 1
			elif t.status in {"failed", "cancelled"}:
				by_partner[partner]["failed"] += 1
		self._audit(self.tenant_id, "partner_performance_reported", period)
		return {
			"period": period, "total_transfers": len(transfers),
			"by_partner": by_partner, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def cbk_frc_sar_filing(self, transfer_id: str, reason: str, submitted_by: str) -> dict[str, Any]:
		"""File a Suspicious Activity Report with the Financial Reporting Centre for a remittance."""
		transfer = self.transfers.get(transfer_id)
		if transfer is None:
			raise ValueError(f"transfer not found: {transfer_id}")
		sar: dict[str, Any] = {
			"sar_id": f"SAR-{transfer_id[:8].upper()}", "transfer_id": transfer_id,
			"sender_reference": transfer.sender_reference, "reason": reason,
			"submitted_by": submitted_by, "regulatory_body": "FRC_KENYA",
			"status": "filed", "filed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._cbk_reports.append(sar)
		self._audit(self.tenant_id, "sar_filed_frc", sar["sar_id"])
		return sar

	async def export_remittance_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export remittance data for compliance or analytics."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"transfer_count": len(self.transfers), "quote_count": len(self.quotes),
			"file_reference": f"remittance_{self.tenant_id}_{datetime.datetime.now(datetime.timezone.utc).isoformat()[:10]}.{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_quote_or_none(self, quote_id: str, tenant_id: str) -> RemittanceQuote | None:
		quote = self.quotes.get(quote_id)
		if quote is None or quote.tenant_id != tenant_id:
			return None
		return quote

	def _tenant_transfer_or_none(self, transfer_id: str, tenant_id: str) -> RemittanceTransfer | None:
		transfer = self.transfers.get(transfer_id)
		if transfer is None or transfer.tenant_id != tenant_id:
			return None
		return transfer

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = RemittanceEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "remittance_policy_denied")
			for action in result["actions"]
		)
		raise PermissionError(reasons or "remittance_policy_denied")



	async def ml_remittance_fraud_detect(self, *args, **kwargs):
		"""AI-powered remittance fraud and money laundering detection. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="remittance_fraud_detection")
			return {"fraud_score": round(result.score,3), "flags": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

CrossBorderRemittanceService = RemittanceService
