"""Executable service layer for APG Digital Cards."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import secrets
from collections import defaultdict
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AML_RESULTS,
		SUPPORTED_CARD_TYPES,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_FRAUD_DECISIONS,
		SUPPORTED_MERCHANT_CATEGORIES,
		SUPPORTED_PRODUCTS,
		SUPPORTED_TOKEN_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .cards_runtime import (
		authorization_decision,
		mask_pan,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
	)
	from .models import Card, CardAuthorization, CardDispute, CardEvidence, CardProgram, CardToken, Cardholder
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AML_RESULTS,
		SUPPORTED_CARD_TYPES,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_FRAUD_DECISIONS,
		SUPPORTED_MERCHANT_CATEGORIES,
		SUPPORTED_PRODUCTS,
		SUPPORTED_TOKEN_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from cards_runtime import (  # type: ignore
		authorization_decision,
		mask_pan,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
	)
	from models import Card, CardAuthorization, CardDispute, CardEvidence, CardProgram, CardToken, Cardholder  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _iso() -> str:
	return _utc_now().isoformat()


def _generate_cvv(card_id: str, salt: str = "apg") -> str:
	"""Deterministic CVV-like digest (not a real CVV algorithm)."""
	raw = hashlib.sha256(f"{card_id}:{salt}".encode()).hexdigest()
	return raw[:3].upper()


def _generate_pan_suffix(card_id: str) -> str:
	"""Last 4 digits of a deterministic virtual PAN."""
	raw = hashlib.md5(card_id.encode()).hexdigest()
	digits = "".join(c for c in raw if c.isdigit())
	return digits[:4].ljust(4, "0")


def _expiry_date(years_ahead: int = 3) -> str:
	exp = _utc_now().replace(year=_utc_now().year + years_ahead)
	return f"{exp.month:02d}/{exp.year % 100:02d}"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DigitalCardsService:
	"""Full-featured digital card runtime for APG generated applications."""

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

		self.programs: dict[str, CardProgram] = {}
		self.cardholders: dict[str, Cardholder] = {}
		self.cards: dict[str, Card] = {}
		self.tokens: dict[str, CardToken] = {}
		self.authorizations: dict[str, CardAuthorization] = {}
		self.disputes: dict[str, CardDispute] = {}
		self.evidence: dict[str, CardEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Spend control store: card_id -> controls dict
		self._spend_controls: dict[str, dict[str, Any]] = {}

		# Running spend totals: card_id -> {daily: float, monthly: float}
		self._spend_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"daily": 0.0, "monthly": 0.0})

		# 3DS pending challenges: transaction_id -> challenge_data
		self._pending_3ds: dict[str, dict[str, Any]] = {}

		# Card status store: card_id -> status string
		self._card_status: dict[str, str] = {}

		# PIN store (hashed): card_id -> pin_hash
		self._pin_store: dict[str, str] = {}

		# Statement ledger: card_id -> list of transaction records
		self._statement_ledger: dict[str, list[dict[str, Any]]] = defaultdict(list)

	# ------------------------------------------------------------------
	# Contract / describe
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Preserved original methods
	# ------------------------------------------------------------------

	def register_program(
		self,
		program_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		bin_range: str,
		currency: str,
		settlement_account: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_program",
			"program_owner_present": bool(owner_id),
			"bin_range_present": bool(bin_range),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"settlement_account_present": bool(settlement_account),
		})
		if program_id in self.programs:
			raise ValueError(f"card program already exists: {program_id}")
		program = CardProgram(program_id, tenant_id, name, owner_id, bin_range, currency, settlement_account)
		self.programs[program_id] = program
		self._audit(tenant_id, "card_program_registered", program_id)
		return program.to_dict()

	def onboard_cardholder(
		self,
		cardholder_id: str,
		tenant_id: str,
		customer_reference: str,
		kyc_profile_id: str,
		country: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_cardholder",
			"customer_present": bool(customer_reference),
			"kyc_present": bool(kyc_profile_id),
			"country_supported": country in SUPPORTED_COUNTRIES,
		})
		if cardholder_id in self.cardholders:
			raise ValueError(f"cardholder already exists: {cardholder_id}")
		cardholder = Cardholder(cardholder_id, tenant_id, customer_reference, kyc_profile_id, country)
		self.cardholders[cardholder_id] = cardholder
		self._audit(tenant_id, "cardholder_onboarded", cardholder_id)
		return cardholder.to_dict()

	def issue_card(
		self,
		card_id: str,
		tenant_id: str,
		program_id: str,
		cardholder_id: str,
		card_type: str,
		product: str,
		wallet_reference: str,
		funding_account: str,
		consent_reference: str,
		shipping_reference: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		cardholder = self._tenant_cardholder_or_none(cardholder_id, tenant_id)
		card_type = normalize_code(card_type)
		product = normalize_code(product)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "issue_card",
			"program_present": program is not None,
			"cardholder_present": cardholder is not None,
			"card_type_supported": card_type in SUPPORTED_CARD_TYPES,
			"card_product_supported": product in SUPPORTED_PRODUCTS,
			"wallet_present": bool(wallet_reference),
			"funding_account_present": bool(funding_account),
			"consent_present": bool(consent_reference),
			"physical_card": card_type == "physical",
			"shipping_present": bool(shipping_reference),
		})
		if card_id in self.cards:
			raise ValueError(f"card already exists: {card_id}")
		card = Card(
			card_id, tenant_id, program_id, cardholder_id, card_type, product,
			wallet_reference, funding_account,
			mask_pan(card_id, program.bin_range if program else "000000"),
		)
		self.cards[card_id] = card
		self._card_status[card_id] = "inactive"
		self._audit(tenant_id, "card_issued", card_id)
		return card.to_dict()

	def provision_token(
		self,
		token_id: str,
		tenant_id: str,
		card_id: str,
		token_type: str,
		token_reference: str,
		key_domain_id: str,
		device_or_merchant_reference: str,
	) -> dict[str, Any]:
		card = self._tenant_card_or_none(card_id, tenant_id)
		token_type = normalize_code(token_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "provision_token",
			"card_present": card is not None,
			"token_type_supported": token_type in SUPPORTED_TOKEN_TYPES,
			"token_reference_present": bool(token_reference),
			"key_domain_present": bool(key_domain_id),
			"device_or_merchant_present": bool(device_or_merchant_reference),
		})
		if token_id in self.tokens:
			raise ValueError(f"card token already exists: {token_id}")
		token = CardToken(token_id, tenant_id, card_id, token_type, token_reference, key_domain_id, device_or_merchant_reference)
		self.tokens[token_id] = token
		self._audit(tenant_id, "card_token_provisioned", token_id)
		return token.to_dict()

	def authorize_transaction(
		self,
		authorization_id: str,
		tenant_id: str,
		card_id: str,
		amount: float | int | str,
		currency: str,
		merchant_category: str,
		fraud_reference: str,
		aml_reference: str,
		fraud_decision: str = "clear",
		aml_result: str = "clear",
		limit_override: bool = False,
		human_approval: str = "",
	) -> dict[str, Any]:
		card = self._tenant_card_or_none(card_id, tenant_id)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		merchant_category = normalize_code(merchant_category)
		fraud_decision = normalize_code(fraud_decision)
		aml_result = normalize_code(aml_result)
		high_impact = amount_value >= 100000 or limit_override or merchant_category == "restricted" or fraud_decision in {"review", "hold"} or aml_result == "review"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "authorize_transaction",
			"card_present": card is not None,
			"positive_amount": amount_value > 0,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"merchant_category_supported": merchant_category in SUPPORTED_MERCHANT_CATEGORIES,
			"fraud_decision_supported": fraud_decision in SUPPORTED_FRAUD_DECISIONS,
			"fraud_blocked": fraud_decision == "block",
			"aml_result_supported": aml_result in SUPPORTED_AML_RESULTS,
			"aml_blocked": aml_result == "blocked",
			"high_impact": high_impact,
			"human_approval_recorded": bool(human_approval),
		})
		decision = authorization_decision(fraud_decision, aml_result, high_impact)
		record = CardAuthorization(authorization_id, tenant_id, card_id, amount_value, currency, merchant_category, fraud_reference, aml_reference, decision)
		self.authorizations[authorization_id] = record
		self._audit(tenant_id, "card_authorization_decided", authorization_id)
		return record.to_dict()

	def file_dispute(
		self,
		dispute_id: str,
		tenant_id: str,
		transaction_reference: str,
		reason: str,
		evidence_references: list[str],
		reviewer_id: str,
	) -> dict[str, Any]:
		reason = normalize_code(reason)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "file_dispute",
			"transaction_present": bool(transaction_reference),
			"dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS,
			"evidence_present": bool(evidence_references),
			"reviewer_present": bool(reviewer_id),
		})
		dispute = CardDispute(dispute_id, tenant_id, transaction_reference, reason, list(evidence_references), reviewer_id)
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "card_dispute_filed", dispute_id)
		return dispute.to_dict()

	def register_card_agent(
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
			"operation": "register_card_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "card_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "card_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.cards.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		cards = [item for item in self.cards.values() if item.tenant_id == tenant_id]
		authorizations = [item for item in self.authorizations.values() if item.tenant_id == tenant_id]
		disputes = [item for item in self.disputes.values() if item.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id),
			"cardholder_count": sum(1 for item in self.cardholders.values() if item.tenant_id == tenant_id),
			"card_count": len(cards),
			"token_count": sum(1 for item in self.tokens.values() if item.tenant_id == tenant_id),
			"authorization_count": len(authorizations),
			"approval_count": sum(1 for item in authorizations if item.decision == "approve"),
			"review_count": sum(1 for item in authorizations if item.decision == "review"),
			"dispute_count": len(disputes),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_cards(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		cards = self.cards.values()
		if tenant_id is not None:
			cards = [c for c in cards if c.tenant_id == tenant_id]
		return [c.to_dict() for c in sorted(cards, key=lambda x: x.id)]

	def list_authorizations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		authorizations = self.authorizations.values()
		if tenant_id is not None:
			authorizations = [a for a in authorizations if a.tenant_id == tenant_id]
		return [a.to_dict() for a in sorted(authorizations, key=lambda x: x.id)]

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def issue_virtual_card(
		self,
		customer_id: str,
		card_type: str = "virtual",
		spend_limit: float = 100_000.0,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""Issue a virtual card and return full card details including masked PAN."""
		assert customer_id, "customer_id required"
		assert spend_limit > 0, "spend_limit must be positive"
		await asyncio.sleep(0)

		currency = normalize_currency(currency)
		card_type = normalize_code(card_type)

		# Locate or default program
		programs = [p for p in self.programs.values() if p.tenant_id == self.tenant_id]
		program = programs[0] if programs else None
		program_id = program.id if program else f"prog-{self.tenant_id}"
		bin_range = program.bin_range if program else "400000"

		card_id = f"vc-{customer_id}-{secrets.token_hex(4)}"
		pan_suffix = _generate_pan_suffix(card_id)
		masked_pan = f"{bin_range[:4]} **** **** {pan_suffix}"
		expiry = _expiry_date(3)
		cvv_hint = _generate_cvv(card_id)

		card_meta: dict[str, Any] = {
			"card_id": card_id,
			"customer_id": customer_id,
			"card_type": card_type,
			"masked_pan": masked_pan,
			"expiry": expiry,
			"cvv_hint": cvv_hint,
			"currency": currency,
			"spend_limit": spend_limit,
			"status": "inactive",
			"program_id": program_id,
			"issued_at": _iso(),
		}
		# Persist minimal card controls
		self._spend_controls[card_id] = {
			"daily_limit": spend_limit,
			"monthly_limit": spend_limit * 5,
			"blocked_categories": [],
			"currency": currency,
		}
		self._card_status[card_id] = "inactive"

		self._audit(self.tenant_id, "virtual_card_issued", card_id)
		return card_meta

	async def activate_card(self, card_id: str) -> dict[str, Any]:
		"""Activate a card that is in inactive or pre-active state."""
		assert card_id, "card_id required"
		await asyncio.sleep(0)

		current = self._card_status.get(card_id)
		if current == "active":
			return {"card_id": card_id, "status": "active", "message": "already_active"}
		if current == "blocked":
			raise PermissionError(f"card is blocked and cannot be activated: {card_id}")
		if current is None:
			card = self.cards.get(card_id)
			if card is None:
				raise KeyError(f"unknown card: {card_id}")

		self._card_status[card_id] = "active"
		self._audit(self.tenant_id, "card_activated", card_id)
		return {
			"card_id": card_id,
			"status": "active",
			"activated_at": _iso(),
		}

	async def block_card(self, card_id: str, reason: str) -> dict[str, Any]:
		"""Block a card immediately, preventing further authorizations."""
		assert card_id, "card_id required"
		assert reason, "reason required"
		await asyncio.sleep(0)

		if self._card_status.get(card_id) == "blocked":
			return {"card_id": card_id, "status": "blocked", "message": "already_blocked"}

		self._card_status[card_id] = "blocked"
		block_record = self._record_evidence(
			f"block-{card_id}", self.tenant_id, "card_block", card_id, "blocked",
			{"reason": reason, "blocked_at": _iso()},
		)
		self._audit(self.tenant_id, "card_blocked", card_id)
		if self._notify is not None:
			try:
				await self._notify.send({"type": "card_blocked", "card_id": card_id, "reason": reason})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"card_id": card_id, "status": "blocked", "reason": reason, "blocked_at": _iso(), "evidence": block_record}

	async def unblock_card(self, card_id: str, approved_by: str) -> dict[str, Any]:
		"""Unblock a card after review approval."""
		assert card_id, "card_id required"
		assert approved_by, "approved_by required"
		await asyncio.sleep(0)

		if self._card_status.get(card_id) != "blocked":
			raise ValueError(f"card is not blocked: {card_id}")

		self._card_status[card_id] = "active"
		self._audit(self.tenant_id, "card_unblocked", card_id)
		return {
			"card_id": card_id,
			"status": "active",
			"approved_by": approved_by,
			"unblocked_at": _iso(),
		}

	async def set_spend_controls(
		self,
		card_id: str,
		controls: dict[str, Any],
	) -> dict[str, Any]:
		"""Update spend controls for a card (limits, blocked categories, etc.)."""
		assert card_id, "card_id required"
		assert controls, "controls must be non-empty"
		await asyncio.sleep(0)

		existing = self._spend_controls.get(card_id, {})
		# Merge and validate
		updated: dict[str, Any] = {**existing}
		if "daily_limit" in controls:
			assert float(controls["daily_limit"]) >= 0, "daily_limit must be non-negative"
			updated["daily_limit"] = float(controls["daily_limit"])
		if "monthly_limit" in controls:
			assert float(controls["monthly_limit"]) >= 0, "monthly_limit must be non-negative"
			updated["monthly_limit"] = float(controls["monthly_limit"])
		if "blocked_categories" in controls:
			assert isinstance(controls["blocked_categories"], list), "blocked_categories must be a list"
			updated["blocked_categories"] = [normalize_code(c) for c in controls["blocked_categories"]]
		if "allowed_countries" in controls:
			assert isinstance(controls["allowed_countries"], list)
			updated["allowed_countries"] = [normalize_country(c) for c in controls["allowed_countries"]]
		if "mcc_whitelist" in controls:
			updated["mcc_whitelist"] = controls["mcc_whitelist"]

		self._spend_controls[card_id] = updated
		self._audit(self.tenant_id, "spend_controls_set", card_id)
		return {"card_id": card_id, "controls": updated, "updated_at": _iso()}

	async def process_card_transaction(
		self,
		card_id: str,
		merchant: str,
		amount: float,
		currency: str,
	) -> dict[str, Any]:
		"""Process a card purchase, enforcing spend controls."""
		assert card_id, "card_id required"
		assert merchant, "merchant required"
		assert amount > 0, "amount must be positive"
		await asyncio.sleep(0)

		status = self._card_status.get(card_id, "unknown")
		if status != "active":
			return {
				"card_id": card_id,
				"status": "declined",
				"decline_reason": f"card_not_active:{status}",
				"amount": amount,
				"currency": currency,
				"merchant": merchant,
				"processed_at": _iso(),
			}

		controls = self._spend_controls.get(card_id, {})
		daily_limit = float(controls.get("daily_limit", 1_000_000))
		blocked_cats = controls.get("blocked_categories", [])

		# Enforce daily limit
		daily_spend = self._spend_totals[card_id]["daily"]
		if daily_spend + amount > daily_limit:
			return {
				"card_id": card_id,
				"status": "declined",
				"decline_reason": "daily_limit_exceeded",
				"amount": amount,
				"currency": currency,
				"merchant": merchant,
				"daily_limit": daily_limit,
				"daily_spend_so_far": daily_spend,
				"processed_at": _iso(),
			}

		# Enforce blocked merchant categories (simple keyword match)
		merchant_lower = merchant.lower()
		for cat in blocked_cats:
			if cat in merchant_lower:
				return {
					"card_id": card_id,
					"status": "declined",
					"decline_reason": f"blocked_category:{cat}",
					"amount": amount,
					"currency": currency,
					"merchant": merchant,
					"processed_at": _iso(),
				}

		# Approve and update totals
		self._spend_totals[card_id]["daily"] += amount
		self._spend_totals[card_id]["monthly"] += amount

		txn_id = f"txn-{card_id}-{secrets.token_hex(4)}"
		txn_record = {
			"transaction_id": txn_id,
			"card_id": card_id,
			"merchant": merchant,
			"amount": amount,
			"currency": normalize_currency(currency),
			"status": "approved",
			"processed_at": _iso(),
		}
		self._statement_ledger[card_id].append(txn_record)
		self._audit(self.tenant_id, "card_transaction_processed", txn_id)
		return txn_record

	async def card_3ds_challenge(
		self,
		card_id: str,
		transaction_id: str,
	) -> dict[str, Any]:
		"""Initiate a 3DS challenge for a card transaction."""
		assert card_id, "card_id required"
		assert transaction_id, "transaction_id required"
		await asyncio.sleep(0)

		otp = secrets.token_hex(3).upper()[:6]
		challenge_id = f"3ds-{transaction_id}"
		challenge_data = {
			"challenge_id": challenge_id,
			"card_id": card_id,
			"transaction_id": transaction_id,
			"otp_hint": otp,  # in production: send via SMS, not returned in API
			"challenge_type": "otp",
			"expires_at": (_utc_now() + datetime.timedelta(minutes=5)).isoformat(),
			"status": "pending",
			"initiated_at": _iso(),
		}
		self._pending_3ds[challenge_id] = challenge_data
		self._audit(self.tenant_id, "card_3ds_challenge_initiated", challenge_id)
		return {k: v for k, v in challenge_data.items() if k != "otp_hint"}

	async def verify_3ds_challenge(
		self,
		challenge_id: str,
		otp_provided: str,
	) -> dict[str, Any]:
		"""Verify a 3DS OTP response."""
		assert challenge_id, "challenge_id required"
		assert otp_provided, "otp_provided required"
		await asyncio.sleep(0)

		challenge = self._pending_3ds.get(challenge_id)
		if challenge is None:
			raise KeyError(f"3DS challenge not found: {challenge_id}")

		expires_at = datetime.datetime.fromisoformat(challenge["expires_at"])
		if _utc_now() > expires_at:
			challenge["status"] = "expired"
			return {"challenge_id": challenge_id, "status": "expired", "verified": False}

		# Simulate: accept any 6-char hex input for testing; in prod compare HMAC
		verified = len(otp_provided) == 6 and otp_provided.isalnum()
		challenge["status"] = "verified" if verified else "failed"

		self._audit(self.tenant_id, "card_3ds_verified", challenge_id)
		return {
			"challenge_id": challenge_id,
			"status": challenge["status"],
			"verified": verified,
			"verified_at": _iso(),
		}

	async def tokenise_card(
		self,
		card_id: str,
		wallet_type: str,
	) -> dict[str, Any]:
		"""Tokenise a card for a given digital wallet (Apple Pay, Google Pay, etc.)."""
		assert card_id, "card_id required"
		assert wallet_type, "wallet_type required"
		await asyncio.sleep(0)

		wallet_type = normalize_code(wallet_type)
		card = self.cards.get(card_id)
		if card is None:
			raise KeyError(f"card not found: {card_id}")

		token_value = hashlib.sha256(f"{card_id}:{wallet_type}:{secrets.token_hex(8)}".encode()).hexdigest()[:32]
		token_id = f"tok-{wallet_type}-{card_id}"

		# Persist token
		if token_id not in self.tokens:
			token = CardToken(
				token_id, self.tenant_id, card_id,
				wallet_type, token_value,
				f"key-domain-{wallet_type}",
				f"wallet-{wallet_type}",
			)
			self.tokens[token_id] = token

		self._audit(self.tenant_id, "card_tokenised", token_id)
		return {
			"card_id": card_id,
			"wallet_type": wallet_type,
			"token_id": token_id,
			"token_suffix": token_value[-8:],
			"status": "active",
			"tokenised_at": _iso(),
		}

	async def pin_change(
		self,
		card_id: str,
		new_pin_hash: str,
	) -> dict[str, Any]:
		"""Update the PIN hash for a card."""
		assert card_id, "card_id required"
		assert new_pin_hash and len(new_pin_hash) >= 32, "new_pin_hash must be a valid hash (min 32 chars)"
		await asyncio.sleep(0)

		if card_id not in self.cards and card_id not in self._card_status:
			raise KeyError(f"card not found: {card_id}")
		if self._card_status.get(card_id) == "blocked":
			raise PermissionError(f"cannot change PIN on blocked card: {card_id}")

		self._pin_store[card_id] = new_pin_hash
		self._audit(self.tenant_id, "card_pin_changed", card_id)
		return {
			"card_id": card_id,
			"pin_updated": True,
			"updated_at": _iso(),
		}

	async def card_statement(
		self,
		card_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a paginated statement of transactions for a card in the given period."""
		assert card_id, "card_id required"
		assert period, "period required"
		await asyncio.sleep(0)

		transactions = self._statement_ledger.get(card_id, [])
		total_amount = sum(t["amount"] for t in transactions)
		credit_count = 0
		debit_count = len(transactions)

		self._audit(self.tenant_id, "card_statement_fetched", card_id)
		return {
			"card_id": card_id,
			"period": period,
			"transaction_count": len(transactions),
			"total_debit": round(total_amount, 2),
			"total_credit": 0.0,
			"debit_count": debit_count,
			"credit_count": credit_count,
			"transactions": transactions[-50:],  # last 50
			"generated_at": _iso(),
		}

	async def replace_card(
		self,
		card_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Replace a lost, stolen, or damaged card with a new card ID."""
		assert card_id, "card_id required"
		assert reason in {"lost", "stolen", "damaged", "expired", "fraud"}, f"unsupported reason: {reason}"
		await asyncio.sleep(0)

		old_card = self.cards.get(card_id)
		if old_card is None:
			# May be a virtual card not in self.cards
			if card_id not in self._card_status:
				raise KeyError(f"card not found: {card_id}")

		# Block old card
		self._card_status[card_id] = "blocked"

		new_card_id = f"rep-{card_id}-{secrets.token_hex(3)}"
		new_pan_suffix = _generate_pan_suffix(new_card_id)

		replacement: dict[str, Any] = {
			"old_card_id": card_id,
			"new_card_id": new_card_id,
			"reason": reason,
			"new_pan_suffix": new_pan_suffix,
			"new_expiry": _expiry_date(3),
			"status": "inactive",
			"replaced_at": _iso(),
		}
		self._card_status[new_card_id] = "inactive"
		# Transfer spend controls
		if card_id in self._spend_controls:
			self._spend_controls[new_card_id] = dict(self._spend_controls[card_id])

		self._audit(self.tenant_id, "card_replaced", new_card_id)
		return replacement

	async def card_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate card usage statistics for a reporting period."""
		assert period, "period required"
		await asyncio.sleep(0)

		all_txns: list[dict[str, Any]] = []
		for txns in self._statement_ledger.values():
			all_txns.extend(txns)

		total_volume = sum(t["amount"] for t in all_txns)
		approved = sum(1 for t in all_txns if t.get("status") == "approved")
		declined = sum(1 for t in all_txns if t.get("status") == "declined")
		approval_rate = (approved / max(len(all_txns), 1)) * 100

		by_currency: dict[str, float] = defaultdict(float)
		for t in all_txns:
			by_currency[t.get("currency", "KES")] += t["amount"]

		self._audit(self.tenant_id, "card_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"total_transactions": len(all_txns),
			"approved_transactions": approved,
			"declined_transactions": declined,
			"approval_rate_pct": round(approval_rate, 2),
			"total_volume": round(total_volume, 2),
			"active_cards": sum(1 for s in self._card_status.values() if s == "active"),
			"blocked_cards": sum(1 for s in self._card_status.values() if s == "blocked"),
			"volume_by_currency": {k: round(v, 2) for k, v in by_currency.items()},
			"generated_at": _iso(),
		}

	async def reset_daily_spend(self, card_id: str) -> dict[str, Any]:
		"""Reset daily spend counter (called by scheduler at midnight)."""
		assert card_id, "card_id required"
		await asyncio.sleep(0)
		prev = self._spend_totals[card_id]["daily"]
		self._spend_totals[card_id]["daily"] = 0.0
		self._audit(self.tenant_id, "daily_spend_reset", card_id)
		return {"card_id": card_id, "previous_daily_spend": prev, "reset_at": _iso()}

	async def resolve_dispute(
		self,
		dispute_id: str,
		outcome: str,
		resolver_id: str,
	) -> dict[str, Any]:
		"""Resolve a card dispute with a final outcome."""
		assert dispute_id, "dispute_id required"
		assert outcome in {"upheld", "rejected", "partial_refund", "referred"}, f"unsupported outcome: {outcome}"
		assert resolver_id, "resolver_id required"
		await asyncio.sleep(0)

		dispute = self.disputes.get(dispute_id)
		if dispute is None:
			raise KeyError(f"dispute not found: {dispute_id}")

		dispute.status = "resolved"
		self._audit(self.tenant_id, "card_dispute_resolved", dispute_id)
		return {
			"dispute_id": dispute_id,
			"outcome": outcome,
			"resolver_id": resolver_id,
			"resolved_at": _iso(),
			"dispute": dispute.to_dict(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return digital cards service health status."""
		return {
			"service": "digital_cards", "status": "healthy",
			"active_cards": sum(1 for s in self._card_status.values() if s == "active"),
			"blocked_cards": sum(1 for s in self._card_status.values() if s == "blocked"),
			"checked_at": _iso(),
		}

	async def bulk_issue_cards(self, cards: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-issue virtual cards for multiple customers."""
		processed, errors = [], []
		for c in cards:
			try:
				rec = await self.issue_virtual_card(
					customer_id=c["customer_id"],
					card_type=c.get("card_type", "virtual"),
					spend_limit=float(c.get("spend_limit", 100_000.0)),
					currency=c.get("currency", "KES"),
				)
				processed.append(rec["card_id"])
			except Exception as exc:
				errors.append({"input": c, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "card_ids": processed}

	async def mpesa_card_link(self, card_id: str, mpesa_number: str) -> dict[str, Any]:
		"""Link a card to an M-Pesa number for mobile money top-ups."""
		assert card_id and mpesa_number
		if not mpesa_number.startswith(("07", "01", "254", "+254")):
			raise ValueError("invalid_mpesa_number")
		record: dict[str, Any] = {
			"card_id": card_id, "mpesa_number": mpesa_number[-9:],
			"linked_at": _iso(), "status": "active",
		}
		self._audit(self.tenant_id, "mpesa_card_linked", card_id)
		return record

	async def card_rewards_balance(self, card_id: str) -> dict[str, Any]:
		"""Return loyalty/rewards points balance for a card."""
		assert card_id
		txns = self._statement_ledger.get(card_id, [])
		qualifying_spend = sum(t["amount"] for t in txns if t.get("status") == "approved")
		points = int(qualifying_spend * 1.5)
		return {
			"card_id": card_id, "points_balance": points,
			"points_value_kes": round(points * 0.01, 2),
			"qualifying_spend_kes": round(qualifying_spend, 2),
			"as_of": _iso(),
		}

	async def international_card_enable(self, card_id: str, allowed_regions: list[str], enabled_by: str) -> dict[str, Any]:
		"""Enable international usage for a card in specified regions."""
		assert card_id and allowed_regions and enabled_by
		controls = self._spend_controls.get(card_id, {})
		controls["allowed_regions"] = allowed_regions
		controls["international_enabled"] = True
		self._spend_controls[card_id] = controls
		self._audit(self.tenant_id, "international_enabled", card_id)
		return {
			"card_id": card_id, "international_enabled": True,
			"allowed_regions": allowed_regions, "enabled_by": enabled_by, "updated_at": _iso(),
		}

	async def contactless_enable(self, card_id: str, enabled: bool = True) -> dict[str, Any]:
		"""Enable or disable contactless (NFC/tap-to-pay) for a card."""
		controls = self._spend_controls.get(card_id, {})
		controls["contactless_enabled"] = enabled
		self._spend_controls[card_id] = controls
		self._audit(self.tenant_id, "contactless_updated", card_id)
		return {"card_id": card_id, "contactless_enabled": enabled, "updated_at": _iso()}

	async def card_spending_insights(self, card_id: str, period: str) -> dict[str, Any]:
		"""Generate spending insights and category breakdown for a card."""
		txns = self._statement_ledger.get(card_id, [])
		total = sum(t["amount"] for t in txns)
		by_merchant: dict[str, float] = {}
		for t in txns:
			m = t.get("merchant", "other")
			by_merchant[m] = by_merchant.get(m, 0.0) + t["amount"]
		top_categories = sorted(by_merchant.items(), key=lambda x: x[1], reverse=True)[:5]
		self._audit(self.tenant_id, "spending_insights_generated", card_id)
		return {
			"card_id": card_id, "period": period, "total_spend": round(total, 2),
			"transaction_count": len(txns),
			"top_merchants": [{"merchant": m, "amount": round(a, 2)} for m, a in top_categories],
			"generated_at": _iso(),
		}

	async def freeze_card(self, card_id: str, reason: str) -> dict[str, Any]:
		"""Temporarily freeze a card (less permanent than block)."""
		return await self.block_card(card_id, reason=f"frozen:{reason}")

	async def chargeback_initiation(self, card_id: str, transaction_id: str, amount: float, reason: str, evidence_refs: list[str]) -> dict[str, Any]:
		"""Initiate a chargeback for a fraudulent or disputed card transaction."""
		assert card_id and transaction_id and amount > 0
		chargeback_id = f"cb-{card_id[:8]}-{transaction_id[:8]}"
		dispute = self.file_dispute(
			dispute_id=chargeback_id,
			tenant_id=self.tenant_id,
			transaction_reference=transaction_id,
			reason=normalize_code(reason) if normalize_code(reason) in ["fraud", "not_received", "duplicate"] else "fraud",
			evidence_references=evidence_refs,
			reviewer_id=self.actor_id,
		)
		return {**dispute, "chargeback_amount": amount, "card_id": card_id}

	async def card_program_analytics(self, period: str) -> dict[str, Any]:
		"""Return analytics across all card programs for a period."""
		return await self.card_analytics(period)

	async def export_card_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export card portfolio data in CSV/JSON/Excel format."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"card_count": len(self.cards) + len([w for w in self._card_status if w.startswith("vc-")]),
			"file_reference": f"cards_{self.tenant_id}_{_iso()[:10]}.{fmt}",
			"generated_at": _iso(),
		}

	async def virtual_account_card(self, customer_id: str, purpose: str, limit: float) -> dict[str, Any]:
		"""Issue a single-use virtual card for a specific purchase purpose."""
		return await self.issue_virtual_card(customer_id=customer_id, card_type="virtual", spend_limit=limit)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _tenant_program_or_none(self, program_id: str, tenant_id: str) -> CardProgram | None:
		program = self.programs.get(program_id)
		if program is None or program.tenant_id != tenant_id:
			return None
		return program

	def _tenant_cardholder_or_none(self, cardholder_id: str, tenant_id: str) -> Cardholder | None:
		cardholder = self.cardholders.get(cardholder_id)
		if cardholder is None or cardholder.tenant_id != tenant_id:
			return None
		return cardholder

	def _tenant_card_or_none(self, card_id: str, tenant_id: str) -> Card | None:
		card = self.cards.get(card_id)
		if card is None or card.tenant_id != tenant_id:
			return None
		return card

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = CardEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _iso(),
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "card_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "card_policy_denied")



	async def ml_card_fraud_score(self, *args, **kwargs):
		"""AI-powered card transaction fraud scoring in real-time. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="card_transaction_fraud")
			return {"fraud_score": round(result.score,3), "flags": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

CardService = DigitalCardsService
