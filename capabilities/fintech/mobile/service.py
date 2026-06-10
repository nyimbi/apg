"""Executable service layer for APG Mobile Banking."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import secrets
from collections import defaultdict
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_ACCOUNT_LINK_TYPES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AUTH_FACTORS,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FRAUD_SEVERITIES,
		SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_PAYMENT_TYPES,
		SUPPORTED_PLATFORMS,
		SUPPORTED_SERVICE_REASONS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .mobile_runtime import (
		device_fingerprint_hash,
		is_high_severity,
		normalize_amount,
		normalize_code,
		normalize_codes,
		normalize_country,
		normalize_currency,
		payment_direction,
	)
	from .models import (
		AccountLink,
		AirtimePurchase,
		AuthFactor,
		BillPayment,
		FraudEvent,
		MobileCustomer,
		MobileEvidence,
		MobilePayment,
		MobileProgram,
		NotificationPreference,
		ServiceRequest,
		TrustedDevice,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ACCOUNT_LINK_TYPES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AUTH_FACTORS,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_FRAUD_SEVERITIES,
		SUPPORTED_NOTIFICATION_CHANNELS,
		SUPPORTED_PAYMENT_TYPES,
		SUPPORTED_PLATFORMS,
		SUPPORTED_SERVICE_REASONS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from mobile_runtime import (  # type: ignore
		device_fingerprint_hash,
		is_high_severity,
		normalize_amount,
		normalize_code,
		normalize_codes,
		normalize_country,
		normalize_currency,
		payment_direction,
	)
	from models import (  # type: ignore
		AccountLink,
		AirtimePurchase,
		AuthFactor,
		BillPayment,
		FraudEvent,
		MobileCustomer,
		MobileEvidence,
		MobilePayment,
		MobileProgram,
		NotificationPreference,
		ServiceRequest,
		TrustedDevice,
	)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _iso() -> str:
	return _utc_now().isoformat()


def _mask_msisdn(msisdn: str) -> str:
	if len(msisdn) < 7:
		return msisdn
	return msisdn[:3] + "****" + msisdn[-4:]


def _otp_code(seed: str) -> str:
	digest = hashlib.sha256(seed.encode()).hexdigest()
	digits = "".join(c for c in digest if c.isdigit())
	return digits[:6].ljust(6, "0")


_AIRTIME_PROVIDERS = {"safaricom", "airtel", "telkom", "equitel", "faiba"}
_BILLER_CATEGORIES = {"utility", "insurance", "education", "government", "retail", "telecoms"}

# USSD menu tree (simple state machine)
_USSD_MENU: dict[str, dict[str, Any]] = {
	"root": {
		"text": "Welcome to Mobile Banking\n1. Balance\n2. Transfer\n3. Mini Statement\n4. Buy Airtime\n5. Pay Bill",
		"options": {"1": "balance", "2": "transfer_amount", "3": "mini_statement", "4": "airtime_amount", "5": "bill_biller"},
	},
	"balance": {"text": "Your balance is being fetched...", "options": {}, "terminal": True},
	"transfer_amount": {"text": "Enter amount to transfer:", "options": {}, "terminal": False},
	"mini_statement": {"text": "Fetching last 5 transactions...", "options": {}, "terminal": True},
	"airtime_amount": {"text": "Enter airtime amount:", "options": {}, "terminal": False},
	"bill_biller": {"text": "Enter biller code:", "options": {}, "terminal": False},
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MobileBankingService:
	"""Full-featured mobile banking runtime for APG generated applications."""

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

		self.programs: dict[str, MobileProgram] = {}
		self.customers: dict[str, MobileCustomer] = {}
		self.devices: dict[str, TrustedDevice] = {}
		self.auth_factors: dict[str, AuthFactor] = {}
		self.account_links: dict[str, AccountLink] = {}
		self.payments: dict[str, MobilePayment] = {}
		self.bills: dict[str, BillPayment] = {}
		self.airtime: dict[str, AirtimePurchase] = {}
		self.service_requests: dict[str, ServiceRequest] = {}
		self.notifications: dict[str, NotificationPreference] = {}
		self.fraud_events: dict[str, FraudEvent] = {}
		self.evidence: dict[str, MobileEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Active USSD sessions: session_id -> session_state
		self._ussd_sessions: dict[str, dict[str, Any]] = {}

		# Account balance cache: account_id -> {balance, currency, updated_at}
		self._balance_cache: dict[str, dict[str, Any]] = {}

		# Mini-statement store: account_id -> list of recent transactions
		self._mini_statements: dict[str, list[dict[str, Any]]] = defaultdict(list)

		# KYC store for onboarding: msisdn -> kyc_status
		self._kyc_status: dict[str, str] = {}

		# Loan application store: application_id -> loan dict
		self._loan_applications: dict[str, dict[str, Any]] = {}

		# Push notification preferences cache
		self._push_preferences: dict[str, dict[str, Any]] = {}

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
		country: str,
		currency: str,
		platforms: list[str],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		platforms = normalize_codes(platforms)
		platforms_valid = bool(platforms) and all(p in SUPPORTED_PLATFORMS for p in platforms)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_program",
			"owner_present": bool(owner_id),
			"country_supported": country in SUPPORTED_COUNTRIES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"platforms_valid": platforms_valid,
		})
		if program_id in self.programs:
			raise ValueError(f"mobile program already exists: {program_id}")
		program = MobileProgram(program_id, tenant_id, name, owner_id, country, currency, platforms)
		self.programs[program_id] = program
		self._audit(tenant_id, "mobile_program_registered", program_id)
		return program.to_dict()

	def enroll_customer(
		self,
		customer_id: str,
		tenant_id: str,
		customer_reference: str,
		country: str,
		kyc_reference: str,
		consent_reference: str,
		aml_reference: str,
		fraud_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "enroll_customer",
			"customer_present": bool(customer_reference),
			"country_supported": country in SUPPORTED_COUNTRIES,
			"kyc_present": bool(kyc_reference),
			"consent_present": bool(consent_reference),
			"aml_present": bool(aml_reference),
			"fraud_present": bool(fraud_reference),
		})
		if customer_id in self.customers:
			raise ValueError(f"mobile customer already exists: {customer_id}")
		customer = MobileCustomer(customer_id, tenant_id, customer_reference, country, kyc_reference, consent_reference, aml_reference, fraud_reference)
		self.customers[customer_id] = customer
		self._audit(tenant_id, "mobile_customer_enrolled", customer_id)
		return customer.to_dict()

	def bind_device(
		self,
		device_id: str,
		tenant_id: str,
		customer_id: str,
		platform: str,
		fingerprint: str,
		attestation_reference: str,
		risk_tier: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		platform = normalize_code(platform)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "bind_device",
			"customer_present": customer is not None,
			"platform_supported": platform in SUPPORTED_PLATFORMS,
			"fingerprint_present": bool(fingerprint),
			"attestation_present": bool(attestation_reference),
			"risk_tier_present": bool(risk_tier),
		})
		if device_id in self.devices:
			raise ValueError(f"trusted device already exists: {device_id}")
		device = TrustedDevice(device_id, tenant_id, customer_id, platform, device_fingerprint_hash(fingerprint), attestation_reference, normalize_code(risk_tier))
		self.devices[device_id] = device
		self._audit(tenant_id, "trusted_device_bound", device_id)
		return device.to_dict()

	def register_auth_factor(
		self,
		factor_id: str,
		tenant_id: str,
		customer_id: str,
		device_id: str,
		factor_type: str,
		strength_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		device = self._tenant_device_or_none(device_id, tenant_id)
		factor_type = normalize_code(factor_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_auth_factor",
			"customer_present": customer is not None,
			"device_present": device is not None and (customer is None or device.customer_id == customer.id),
			"factor_type_supported": factor_type in SUPPORTED_AUTH_FACTORS,
			"strength_reference_present": bool(strength_reference),
		})
		factor = AuthFactor(factor_id, tenant_id, customer_id, device_id, factor_type, strength_reference)
		self.auth_factors[factor_id] = factor
		self._audit(tenant_id, "auth_factor_registered", factor_id)
		return factor.to_dict()

	def link_account(
		self,
		link_id: str,
		tenant_id: str,
		customer_id: str,
		link_type: str,
		account_reference: str,
		currency: str,
		provider_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		link_type = normalize_code(link_type)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "link_account",
			"customer_present": customer is not None,
			"link_type_supported": link_type in SUPPORTED_ACCOUNT_LINK_TYPES,
			"account_reference_present": bool(account_reference),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"provider_reference_present": bool(provider_reference),
		})
		link = AccountLink(link_id, tenant_id, customer_id, link_type, account_reference, currency, provider_reference)
		self.account_links[link_id] = link
		self._audit(tenant_id, "account_linked", link_id)
		return link.to_dict()

	def initiate_payment(
		self,
		payment_id: str,
		tenant_id: str,
		customer_id: str,
		device_id: str,
		account_link_id: str,
		payment_type: str,
		amount: float | int | str,
		currency: str,
		recipient_reference: str,
		risk_reference: str,
		human_approval: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		device = self._tenant_device_or_none(device_id, tenant_id)
		link = self._tenant_link_or_none(account_link_id, tenant_id)
		payment_type = normalize_code(payment_type)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		high_value = amount_value >= 100000
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "initiate_payment",
			"customer_present": customer is not None,
			"device_present": device is not None and (customer is None or device.customer_id == customer.id),
			"account_link_present": link is not None and (customer is None or link.customer_id == customer.id),
			"payment_type_supported": payment_type in SUPPORTED_PAYMENT_TYPES,
			"positive_amount": amount_value > 0,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"currency_matches_link": link is not None and link.currency == currency,
			"recipient_present": bool(recipient_reference),
			"risk_reference_present": bool(risk_reference),
			"high_value": high_value,
			"human_approval_recorded": bool(human_approval),
		})
		payment = MobilePayment(payment_id, tenant_id, customer_id, device_id, account_link_id, payment_type, amount_value, currency, recipient_reference, risk_reference, human_approval)
		self.payments[payment_id] = payment
		self._audit(tenant_id, "mobile_payment_initiated", payment_id)
		return payment.to_dict() | {"direction": payment_direction(payment_type)}

	def record_bill_payment(
		self,
		bill_id: str,
		tenant_id: str,
		payment_id: str,
		biller_reference: str,
		bill_account_reference: str,
	) -> dict[str, Any]:
		payment = self._tenant_payment_or_none(payment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_bill_payment",
			"payment_present": payment is not None,
			"payment_type_matches": payment is not None and payment.payment_type == "bill_payment",
			"biller_reference_present": bool(biller_reference),
		})
		bill = BillPayment(bill_id, tenant_id, payment_id, biller_reference, bill_account_reference)
		self.bills[bill_id] = bill
		self._audit(tenant_id, "bill_payment_recorded", bill_id)
		return bill.to_dict()

	def purchase_airtime(
		self,
		airtime_id: str,
		tenant_id: str,
		payment_id: str,
		operator_reference: str,
		phone_reference: str,
	) -> dict[str, Any]:
		payment = self._tenant_payment_or_none(payment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "purchase_airtime",
			"payment_present": payment is not None,
			"payment_type_matches": payment is not None and payment.payment_type == "airtime",
			"operator_reference_present": bool(operator_reference),
			"phone_reference_present": bool(phone_reference),
		})
		airtime = AirtimePurchase(airtime_id, tenant_id, payment_id, operator_reference, phone_reference)
		self.airtime[airtime_id] = airtime
		self._audit(tenant_id, "airtime_purchased", airtime_id)
		return airtime.to_dict()

	def open_service_request(
		self,
		request_id: str,
		tenant_id: str,
		customer_id: str,
		reason: str,
		reviewer_id: str,
		evidence_references: list[str],
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_service_request",
			"customer_present": customer is not None,
			"service_reason_supported": reason in SUPPORTED_SERVICE_REASONS,
			"evidence_present": bool(evidence_references),
			"reviewer_present": bool(reviewer_id),
		})
		request = ServiceRequest(request_id, tenant_id, customer_id, reason, reviewer_id, list(evidence_references))
		self.service_requests[request_id] = request
		self._audit(tenant_id, "service_request_opened", request_id)
		return request.to_dict()

	def set_notification_preference(
		self,
		preference_id: str,
		tenant_id: str,
		customer_id: str,
		channel: str,
		consent_reference: str,
		enabled: bool = True,
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		channel = normalize_code(channel)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "set_notification_preference",
			"customer_present": customer is not None,
			"notification_channel_supported": channel in SUPPORTED_NOTIFICATION_CHANNELS,
			"consent_present": bool(consent_reference),
		})
		preference = NotificationPreference(preference_id, tenant_id, customer_id, channel, consent_reference, bool(enabled))
		self.notifications[preference_id] = preference
		self._audit(tenant_id, "notification_preference_set", preference_id)
		return preference.to_dict()

	def record_fraud_event(
		self,
		event_id: str,
		tenant_id: str,
		customer_id: str,
		severity: str,
		evidence_references: list[str],
		human_approval: str = "",
	) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		severity = normalize_code(severity)
		high_severity = is_high_severity(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_fraud_event",
			"customer_present": customer is not None,
			"severity_supported": severity in SUPPORTED_FRAUD_SEVERITIES,
			"evidence_present": bool(evidence_references),
			"high_severity": high_severity,
			"human_approval_recorded": bool(human_approval),
		})
		event = FraudEvent(event_id, tenant_id, customer_id, severity, list(evidence_references), human_approval)
		self.fraud_events[event_id] = event
		self._audit(tenant_id, "fraud_event_recorded", event_id)
		return event.to_dict()

	def register_mobile_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_mobile_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "mobile_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "mobile_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.mobile.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id),
			"customer_count": sum(1 for item in self.customers.values() if item.tenant_id == tenant_id),
			"device_count": sum(1 for item in self.devices.values() if item.tenant_id == tenant_id),
			"auth_factor_count": sum(1 for item in self.auth_factors.values() if item.tenant_id == tenant_id),
			"account_link_count": sum(1 for item in self.account_links.values() if item.tenant_id == tenant_id),
			"payment_count": sum(1 for item in self.payments.values() if item.tenant_id == tenant_id),
			"bill_count": sum(1 for item in self.bills.values() if item.tenant_id == tenant_id),
			"airtime_count": sum(1 for item in self.airtime.values() if item.tenant_id == tenant_id),
			"service_request_count": sum(1 for item in self.service_requests.values() if item.tenant_id == tenant_id),
			"notification_count": sum(1 for item in self.notifications.values() if item.tenant_id == tenant_id),
			"fraud_event_count": sum(1 for item in self.fraud_events.values() if item.tenant_id == tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_payments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.payments.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda x: x.id)]

	def list_devices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.devices.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda x: x.id)]

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def mobile_onboarding(
		self,
		msisdn: str,
		id_number: str,
		kyc_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Full mobile onboarding: KYC verification, consent capture, customer creation."""
		assert msisdn, "msisdn required"
		assert id_number, "id_number required"
		assert kyc_data, "kyc_data must be non-empty"
		await asyncio.sleep(0)

		masked_msisdn = _mask_msisdn(msisdn)
		country = str(kyc_data.get("country", "KE"))
		first_name = str(kyc_data.get("first_name", ""))
		last_name = str(kyc_data.get("last_name", ""))
		dob = str(kyc_data.get("date_of_birth", ""))

		# Validate minimum KYC fields
		if not (first_name and last_name and dob):
			raise ValueError("kyc_data must include first_name, last_name, date_of_birth")

		# Simulate KYC check
		kyc_score = len(first_name) + len(last_name) + len(id_number)
		kyc_passed = kyc_score >= 10 and len(id_number) >= 6
		kyc_status = "verified" if kyc_passed else "pending_review"
		self._kyc_status[msisdn] = kyc_status

		# Generate reference IDs
		customer_id = f"mob-{hashlib.md5(msisdn.encode()).hexdigest()[:12]}"
		kyc_reference = f"kyc-{id_number[:4]}-{secrets.token_hex(4)}"
		consent_reference = f"consent-{secrets.token_hex(6)}"
		aml_reference = f"aml-{secrets.token_hex(6)}"
		fraud_reference = f"fraud-{secrets.token_hex(6)}"

		# Enroll if not already enrolled
		if customer_id not in self.customers:
			self.enroll_customer(
				customer_id=customer_id,
				tenant_id=self.tenant_id,
				customer_reference=msisdn,
				country=normalize_country(country),
				kyc_reference=kyc_reference,
				consent_reference=consent_reference,
				aml_reference=aml_reference,
				fraud_reference=fraud_reference,
			)

		self._audit(self.tenant_id, "mobile_onboarding_completed", customer_id)
		return {
			"customer_id": customer_id,
			"msisdn": masked_msisdn,
			"kyc_status": kyc_status,
			"kyc_reference": kyc_reference,
			"consent_reference": consent_reference,
			"country": country,
			"onboarded_at": _iso(),
		}

	async def account_balance_inquiry(
		self,
		account_id: str,
		channel: str = "mobile",
	) -> dict[str, Any]:
		"""Return the current balance for an account, optionally cached."""
		assert account_id, "account_id required"
		await asyncio.sleep(0)

		channel = normalize_code(channel)
		cached = self._balance_cache.get(account_id)
		# Use cache if fresher than 30 seconds
		if cached:
			age_secs = (_utc_now() - datetime.datetime.fromisoformat(cached["updated_at"])).total_seconds()
			if age_secs < 30:
				return cached | {"channel": channel, "from_cache": True}

		# Simulate balance fetch
		balance_seed = int(hashlib.md5(account_id.encode()).hexdigest()[:8], 16) % 1_000_000
		balance = float(balance_seed) / 100
		currency = "KES"

		balance_data: dict[str, Any] = {
			"account_id": account_id,
			"balance": balance,
			"available_balance": balance * 0.95,
			"currency": currency,
			"channel": channel,
			"from_cache": False,
			"updated_at": _iso(),
		}
		self._balance_cache[account_id] = balance_data
		self._audit(self.tenant_id, "balance_inquired", account_id)
		return balance_data

	async def mini_statement(
		self,
		account_id: str,
		limit: int = 5,
	) -> dict[str, Any]:
		"""Return the last N transactions for an account."""
		assert account_id, "account_id required"
		assert 1 <= limit <= 20, "limit must be between 1 and 20"
		await asyncio.sleep(0)

		transactions = self._mini_statements.get(account_id, [])
		# Return most recent first
		recent = sorted(transactions, key=lambda t: t.get("timestamp", ""), reverse=True)[:limit]

		self._audit(self.tenant_id, "mini_statement_fetched", account_id)
		return {
			"account_id": account_id,
			"limit": limit,
			"transaction_count": len(recent),
			"transactions": recent,
			"fetched_at": _iso(),
		}

	async def funds_transfer(
		self,
		from_account: str,
		to_account: str,
		amount: float,
		reference: str,
	) -> dict[str, Any]:
		"""Initiate a funds transfer between accounts."""
		assert from_account, "from_account required"
		assert to_account, "to_account required"
		assert amount > 0, "amount must be positive"
		assert reference, "reference required"
		assert from_account != to_account, "source and destination must differ"
		await asyncio.sleep(0)

		# Check source balance
		balance_data = await self.account_balance_inquiry(from_account)
		available = float(balance_data.get("available_balance", 0))
		if available < amount:
			return {
				"status": "declined",
				"reason": "insufficient_funds",
				"from_account": from_account,
				"to_account": to_account,
				"amount": amount,
				"available_balance": available,
				"reference": reference,
				"processed_at": _iso(),
			}

		transfer_id = f"ft-{reference}-{secrets.token_hex(4)}"
		# Debit source
		src_balance = self._balance_cache.get(from_account, {})
		if src_balance:
			src_balance["balance"] = float(src_balance.get("balance", 0)) - amount
			src_balance["available_balance"] = float(src_balance.get("available_balance", 0)) - amount
			src_balance["updated_at"] = _iso()

		# Record in mini-statements
		txn_record = {
			"transaction_id": transfer_id,
			"type": "debit",
			"amount": amount,
			"currency": "KES",
			"counterparty": to_account,
			"reference": reference,
			"timestamp": _iso(),
		}
		self._mini_statements[from_account].append(txn_record)
		self._mini_statements[to_account].append({
			**txn_record,
			"type": "credit",
			"counterparty": from_account,
		})

		self._audit(self.tenant_id, "funds_transferred", transfer_id)
		return {
			"transfer_id": transfer_id,
			"from_account": from_account,
			"to_account": to_account,
			"amount": amount,
			"currency": "KES",
			"reference": reference,
			"status": "completed",
			"processed_at": _iso(),
		}

	async def bill_payment(
		self,
		account_id: str,
		biller_code: str,
		account_number: str,
		amount: float,
	) -> dict[str, Any]:
		"""Pay a bill from an account to a registered biller."""
		assert account_id, "account_id required"
		assert biller_code, "biller_code required"
		assert account_number, "account_number required"
		assert amount > 0, "amount must be positive"
		await asyncio.sleep(0)

		# Validate biller (prefix-based)
		biller_category = biller_code.split("-")[0].lower() if "-" in biller_code else "utility"
		if biller_category not in _BILLER_CATEGORIES:
			biller_category = "utility"

		balance_data = await self.account_balance_inquiry(account_id)
		available = float(balance_data.get("available_balance", 0))
		if available < amount:
			return {
				"status": "declined",
				"reason": "insufficient_funds",
				"account_id": account_id,
				"biller_code": biller_code,
				"amount": amount,
				"processed_at": _iso(),
			}

		bill_id = f"bp-{biller_code}-{secrets.token_hex(4)}"
		# Deduct from balance cache
		bal = self._balance_cache.get(account_id, {})
		if bal:
			bal["balance"] = float(bal.get("balance", 0)) - amount
			bal["available_balance"] = float(bal.get("available_balance", 0)) - amount
			bal["updated_at"] = _iso()

		txn_record = {
			"transaction_id": bill_id,
			"type": "debit",
			"amount": amount,
			"currency": "KES",
			"counterparty": biller_code,
			"reference": account_number,
			"timestamp": _iso(),
		}
		self._mini_statements[account_id].append(txn_record)

		self._audit(self.tenant_id, "bill_paid", bill_id)
		return {
			"bill_id": bill_id,
			"account_id": account_id,
			"biller_code": biller_code,
			"biller_category": biller_category,
			"account_number": account_number,
			"amount": amount,
			"currency": "KES",
			"status": "completed",
			"processed_at": _iso(),
		}

	async def airtime_purchase(
		self,
		account_id: str,
		phone: str,
		amount: float,
		provider: str,
	) -> dict[str, Any]:
		"""Purchase airtime for a phone number from a mobile provider."""
		assert account_id, "account_id required"
		assert phone, "phone required"
		assert amount > 0, "amount must be positive"
		provider = provider.lower()
		if provider not in _AIRTIME_PROVIDERS:
			raise ValueError(f"unsupported provider: {provider}. Supported: {_AIRTIME_PROVIDERS}")
		await asyncio.sleep(0)

		balance_data = await self.account_balance_inquiry(account_id)
		available = float(balance_data.get("available_balance", 0))
		if available < amount:
			return {
				"status": "declined",
				"reason": "insufficient_funds",
				"amount": amount,
				"available": available,
				"processed_at": _iso(),
			}

		airtime_id = f"at-{provider}-{secrets.token_hex(4)}"
		bal = self._balance_cache.get(account_id, {})
		if bal:
			bal["balance"] = float(bal.get("balance", 0)) - amount
			bal["available_balance"] = float(bal.get("available_balance", 0)) - amount
			bal["updated_at"] = _iso()

		self._mini_statements[account_id].append({
			"transaction_id": airtime_id,
			"type": "debit",
			"amount": amount,
			"currency": "KES",
			"counterparty": f"{provider}:{_mask_msisdn(phone)}",
			"reference": airtime_id,
			"timestamp": _iso(),
		})

		self._audit(self.tenant_id, "airtime_purchased_direct", airtime_id)
		return {
			"airtime_id": airtime_id,
			"account_id": account_id,
			"phone": _mask_msisdn(phone),
			"amount": amount,
			"currency": "KES",
			"provider": provider,
			"status": "completed",
			"processed_at": _iso(),
		}

	async def loan_application_mobile(
		self,
		account_id: str,
		amount: float,
		tenor: int,
	) -> dict[str, Any]:
		"""Submit a mobile loan application and return a preliminary decision."""
		assert account_id, "account_id required"
		assert amount > 0, "amount must be positive"
		assert 1 <= tenor <= 36, "tenor must be between 1 and 36 months"
		await asyncio.sleep(0)

		# Credit score proxy using account transaction history
		txn_count = len(self._mini_statements.get(account_id, []))
		balance_data = await self.account_balance_inquiry(account_id)
		avg_balance = float(balance_data.get("balance", 0))

		# Simple scoring: more transactions and higher balance = better score
		credit_score = min(int(txn_count * 5 + avg_balance / 1000), 850)
		approved = credit_score >= 300 and amount <= avg_balance * 3
		interest_rate = 0.18 if credit_score >= 600 else 0.24  # annual
		monthly_rate = interest_rate / 12
		monthly_payment = (amount * monthly_rate) / (1 - (1 + monthly_rate) ** (-tenor)) if approved else 0.0

		app_id = f"loan-{account_id}-{secrets.token_hex(4)}"
		application = {
			"application_id": app_id,
			"account_id": account_id,
			"amount_requested": amount,
			"tenor_months": tenor,
			"credit_score": credit_score,
			"decision": "approved" if approved else "declined",
			"decline_reason": None if approved else "insufficient_credit_score",
			"interest_rate_annual": interest_rate if approved else None,
			"monthly_payment": round(monthly_payment, 2) if approved else None,
			"total_repayable": round(monthly_payment * tenor, 2) if approved else None,
			"applied_at": _iso(),
		}
		self._loan_applications[app_id] = application
		self._audit(self.tenant_id, "loan_application_submitted", app_id)
		return application

	async def ussd_session(
		self,
		msisdn: str,
		session_id: str,
		input_text: str,
	) -> dict[str, Any]:
		"""Process a USSD session step and return the next menu prompt."""
		assert msisdn, "msisdn required"
		assert session_id, "session_id required"
		await asyncio.sleep(0)

		session = self._ussd_sessions.get(session_id)
		if session is None:
			# New session — serve root menu
			session = {
				"session_id": session_id,
				"msisdn": _mask_msisdn(msisdn),
				"current_state": "root",
				"history": [],
				"started_at": _iso(),
			}
			self._ussd_sessions[session_id] = session

		current_state = session["current_state"]
		menu = _USSD_MENU.get(current_state, _USSD_MENU["root"])

		response_text = menu["text"]
		is_terminal = menu.get("terminal", False)
		next_state = current_state

		if input_text and not is_terminal:
			chosen = menu["options"].get(input_text.strip())
			if chosen:
				next_state = chosen
				next_menu = _USSD_MENU.get(chosen, {})
				response_text = next_menu.get("text", "Invalid option. Please try again.")
				is_terminal = next_menu.get("terminal", False)
				session["current_state"] = next_state
			else:
				response_text = "Invalid option. Please try again.\n" + menu["text"]

		session["history"].append({"input": input_text, "state": next_state, "at": _iso()})

		if is_terminal:
			# Inject real data for terminal states
			if next_state == "balance":
				bal = await self.account_balance_inquiry(f"acc-{msisdn}")
				response_text = f"Balance: KES {bal['balance']:.2f}\nAvailable: KES {bal['available_balance']:.2f}"
			elif next_state == "mini_statement":
				stmt = await self.mini_statement(f"acc-{msisdn}", limit=3)
				lines = [f"{t.get('type','').upper()}: KES {t.get('amount', 0):.2f}" for t in stmt["transactions"]]
				response_text = "Last transactions:\n" + ("\n".join(lines) or "No transactions found.")
			del self._ussd_sessions[session_id]

		self._audit(self.tenant_id, "ussd_session_step", session_id)
		return {
			"session_id": session_id,
			"msisdn": _mask_msisdn(msisdn),
			"response_text": response_text,
			"current_state": next_state,
			"is_terminal": is_terminal,
			"processed_at": _iso(),
		}

	async def push_notification_settings(
		self,
		customer_id: str,
		preferences: dict[str, Any],
	) -> dict[str, Any]:
		"""Update push notification settings for a customer."""
		assert customer_id, "customer_id required"
		assert preferences, "preferences must be non-empty"
		await asyncio.sleep(0)

		current = self._push_preferences.get(customer_id, {})
		updated = {**current, **preferences, "updated_at": _iso()}
		self._push_preferences[customer_id] = updated

		# Persist as notification preferences for each channel
		for channel, enabled in preferences.items():
			if channel in SUPPORTED_NOTIFICATION_CHANNELS:
				pref_id = f"pref-{customer_id}-{channel}"
				if customer_id in self.customers:
					try:
						self.set_notification_preference(
							pref_id, self.tenant_id, customer_id,
							channel, f"consent-{customer_id}", bool(enabled),
						)
					except Exception as _exc:
						_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._audit(self.tenant_id, "push_notification_settings_updated", customer_id)
		return {
			"customer_id": customer_id,
			"preferences": updated,
			"channels_configured": list(preferences.keys()),
			"updated_at": _iso(),
		}

	async def mobile_analytics(self, period: str) -> dict[str, Any]:
		"""Generate mobile banking analytics for a reporting period."""
		assert period, "period required"
		await asyncio.sleep(0)

		all_payments = [p for p in self.payments.values() if p.tenant_id == self.tenant_id]
		all_bills = [b for b in self.bills.values() if b.tenant_id == self.tenant_id]
		all_airtime = [a for a in self.airtime.values() if a.tenant_id == self.tenant_id]
		all_customers = [c for c in self.customers.values() if c.tenant_id == self.tenant_id]
		all_devices = [d for d in self.devices.values() if d.tenant_id == self.tenant_id]

		payment_volume = sum(getattr(p, "amount", 0) for p in all_payments)
		active_sessions = len(self._ussd_sessions)
		fraud_events = sum(1 for f in self.fraud_events.values() if f.tenant_id == self.tenant_id)

		by_payment_type: dict[str, int] = defaultdict(int)
		for p in all_payments:
			by_payment_type[getattr(p, "payment_type", "other")] += 1

		by_platform: dict[str, int] = defaultdict(int)
		for d in all_devices:
			by_platform[d.platform] += 1

		loan_count = len(self._loan_applications)
		approved_loans = sum(1 for l in self._loan_applications.values() if l.get("decision") == "approved")

		self._audit(self.tenant_id, "mobile_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"total_customers": len(all_customers),
			"active_devices": len(all_devices),
			"total_payments": len(all_payments),
			"payment_volume": round(payment_volume, 2),
			"bill_payments": len(all_bills),
			"airtime_purchases": len(all_airtime),
			"active_ussd_sessions": active_sessions,
			"fraud_events": fraud_events,
			"loan_applications": loan_count,
			"approved_loans": approved_loans,
			"by_payment_type": dict(by_payment_type),
			"by_platform": dict(by_platform),
			"generated_at": _iso(),
		}

	async def send_otp(self, msisdn: str, purpose: str) -> dict[str, Any]:
		"""Generate and dispatch an OTP for authentication or transaction confirmation."""
		assert msisdn, "msisdn required"
		assert purpose, "purpose required"
		await asyncio.sleep(0)

		otp = _otp_code(f"{msisdn}:{purpose}:{_utc_now().timestamp():.0f}")
		otp_id = f"otp-{secrets.token_hex(6)}"
		expires_at = (_utc_now() + datetime.timedelta(minutes=5)).isoformat()

		# In production, dispatch via SMS gateway
		if self._notify is not None:
			try:
				await self._notify.send({"type": "otp", "msisdn": msisdn, "otp": otp, "expires_at": expires_at})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		self._audit(self.tenant_id, "otp_sent", otp_id)
		return {
			"otp_id": otp_id,
			"msisdn": _mask_msisdn(msisdn),
			"purpose": purpose,
			"expires_at": expires_at,
			"sent_at": _iso(),
			# otp value intentionally omitted from production response
		}

	async def close_account(self, account_id: str, reason: str, approved_by: str) -> dict[str, Any]:
		"""Close a mobile banking account after balance zero confirmation."""
		assert account_id, "account_id required"
		assert reason, "reason required"
		assert approved_by, "approved_by required"
		await asyncio.sleep(0)

		balance_data = await self.account_balance_inquiry(account_id)
		if float(balance_data.get("available_balance", 1)) > 0:
			raise PermissionError("account_has_outstanding_balance: cannot close")

		# Invalidate cache
		self._balance_cache.pop(account_id, None)
		self._audit(self.tenant_id, "account_closed", account_id)
		return {
			"account_id": account_id,
			"reason": reason,
			"approved_by": approved_by,
			"status": "closed",
			"closed_at": _iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return mobile banking service health status."""
		return {
			"service": "mobile_banking", "status": "healthy",
			"enrolled_customers": len(self.customers),
			"active_ussd_sessions": len(self._ussd_sessions),
			"checked_at": _iso(),
		}

	async def bulk_enroll_customers(self, customers: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-enroll mobile banking customers."""
		processed, errors = [], []
		for c in customers:
			try:
				rec = self.enroll_customer(
					customer_id=c.get("customer_id", f"mob-{_iso()[:10]}-{len(processed):03d}"),
					tenant_id=self.tenant_id,
					customer_reference=c["customer_reference"],
					country=c.get("country", "KE"),
					kyc_reference=c.get("kyc_reference", f"kyc-{len(processed)}"),
					consent_reference=c.get("consent_reference", f"consent-{len(processed)}"),
					aml_reference=c.get("aml_reference", f"aml-{len(processed)}"),
					fraud_reference=c.get("fraud_reference", f"fraud-{len(processed)}"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": c, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "customer_ids": processed}

	async def device_revocation(self, device_id: str, reason: str) -> dict[str, Any]:
		"""Revoke a trusted device (e.g., lost phone, fraud)."""
		device = self._tenant_device_or_none(device_id, self.tenant_id)
		if device is None:
			raise KeyError(f"device not found: {device_id}")
		device.status = "revoked"  # type: ignore[attr-defined]
		self._audit(self.tenant_id, "device_revoked", device_id)
		return {"device_id": device_id, "status": "revoked", "reason": reason, "revoked_at": _iso()}

	async def number_portability_check(self, msisdn: str) -> dict[str, Any]:
		"""Check mobile number portability status for a MSISDN."""
		assert msisdn, "msisdn required"
		prefix = msisdn[:4] if len(msisdn) >= 4 else msisdn
		network_map = {"0722": "Safaricom", "0711": "Airtel", "0733": "Airtel", "0700": "Telkom", "0765": "Faiba"}
		current_network = network_map.get(prefix, "Unknown")
		return {
			"msisdn": _mask_msisdn(msisdn), "current_network": current_network,
			"ported": False, "original_network": current_network,
			"checked_at": _iso(),
		}

	async def mpesa_c2b_payment(self, account_id: str, amount: float, mpesa_ref: str) -> dict[str, Any]:
		"""Process an M-Pesa C2B (Customer-to-Business) payment."""
		assert mpesa_ref and amount > 0
		return await self.funds_transfer(account_id, f"business-{account_id}", amount, mpesa_ref)

	async def bulk_payment_upload(self, account_id: str, payments: list[dict[str, Any]]) -> dict[str, Any]:
		"""Upload and process a batch of mobile payments."""
		processed, errors = [], []
		for p in payments:
			try:
				rec = await self.funds_transfer(account_id, p["to_account"], float(p["amount"]), p.get("reference", _uuid()))
				processed.append(rec["transfer_id"])
			except Exception as exc:
				errors.append({"payment": p, "error": str(exc)})
		return {"total": len(payments), "processed": len(processed), "failed": len(errors), "transfer_ids": processed}

	async def account_statement_mobile(self, account_id: str, period: str) -> dict[str, Any]:
		"""Generate account statement from mobile banking channel."""
		txns = self._mini_statements.get(account_id, [])
		balance = await self.account_balance_inquiry(account_id)
		return {
			"account_id": account_id, "period": period,
			"transaction_count": len(txns), "transactions": txns[-20:],
			"closing_balance": balance.get("balance", 0), "currency": "KES",
			"generated_at": _iso(),
		}

	async def fraud_report_mobile(self, customer_id: str, incident_description: str, amount: float) -> dict[str, Any]:
		"""Report a fraud incident from the mobile banking channel."""
		assert customer_id and incident_description
		record = self.record_fraud_event(
			event_id=f"mob-fraud-{customer_id}-{_iso()[:10]}",
			tenant_id=self.tenant_id,
			customer_id=customer_id,
			severity="high",
			evidence_references=[f"mobile_report_{customer_id}"],
			human_approval="",
		)
		return {**record, "incident_description": incident_description, "amount": amount}

	async def cbk_mobile_banking_return(self, period: str) -> dict[str, Any]:
		"""Generate a CBK Mobile Banking regulatory return."""
		return {
			"report_type": "CBK_MOBILE_BANKING_RETURN", "period": period,
			"enrolled_customers": sum(1 for c in self.customers.values() if c.tenant_id == self.tenant_id),
			"registered_devices": sum(1 for d in self.devices.values() if d.tenant_id == self.tenant_id),
			"payments": sum(1 for p in self.payments.values() if p.tenant_id == self.tenant_id),
			"status": "draft", "generated_at": _iso(),
		}

	async def export_mobile_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export mobile banking data."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"customers": sum(1 for c in self.customers.values() if c.tenant_id == self.tenant_id),
			"file_reference": f"mobile_{self.tenant_id}_{_iso()[:10]}.{fmt}", "generated_at": _iso(),
		}

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _tenant_customer_or_none(self, customer_id: str, tenant_id: str) -> MobileCustomer | None:
		customer = self.customers.get(customer_id)
		return customer if customer is not None and customer.tenant_id == tenant_id else None

	def _tenant_device_or_none(self, device_id: str, tenant_id: str) -> TrustedDevice | None:
		device = self.devices.get(device_id)
		return device if device is not None and device.tenant_id == tenant_id else None

	def _tenant_link_or_none(self, link_id: str, tenant_id: str) -> AccountLink | None:
		link = self.account_links.get(link_id)
		return link if link is not None and link.tenant_id == tenant_id else None

	def _tenant_payment_or_none(self, payment_id: str, tenant_id: str) -> MobilePayment | None:
		payment = self.payments.get(payment_id)
		return payment if payment is not None and payment.tenant_id == tenant_id else None

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = MobileEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
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
		reasons = ", ".join(action.get("reason", "mobile_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "mobile_policy_denied")



	async def ml_mobile_security_score(self, *args, **kwargs):
		"""AI-powered mobile banking session security risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="mobile_banking_security")
			return {"security_risk": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

MobileService = MobileBankingService
