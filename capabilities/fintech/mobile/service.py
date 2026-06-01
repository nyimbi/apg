"""Executable service layer for APG Mobile Banking."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_ACCOUNT_LINK_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTH_FACTORS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_FRAUD_SEVERITIES, SUPPORTED_NOTIFICATION_CHANNELS, SUPPORTED_PAYMENT_TYPES, SUPPORTED_PLATFORMS, SUPPORTED_SERVICE_REASONS, evaluate_capability_rules, get_capability_contract
	from .mobile_runtime import device_fingerprint_hash, is_high_severity, normalize_amount, normalize_code, normalize_codes, normalize_country, normalize_currency, payment_direction
	from .models import AccountLink, AirtimePurchase, AuthFactor, BillPayment, FraudEvent, MobileCustomer, MobileEvidence, MobilePayment, MobileProgram, NotificationPreference, ServiceRequest, TrustedDevice
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_ACCOUNT_LINK_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTH_FACTORS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_FRAUD_SEVERITIES, SUPPORTED_NOTIFICATION_CHANNELS, SUPPORTED_PAYMENT_TYPES, SUPPORTED_PLATFORMS, SUPPORTED_SERVICE_REASONS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from mobile_runtime import device_fingerprint_hash, is_high_severity, normalize_amount, normalize_code, normalize_codes, normalize_country, normalize_currency, payment_direction  # type: ignore
	from models import AccountLink, AirtimePurchase, AuthFactor, BillPayment, FraudEvent, MobileCustomer, MobileEvidence, MobilePayment, MobileProgram, NotificationPreference, ServiceRequest, TrustedDevice  # type: ignore


class MobileBankingService:
	"""Dependency-light mobile banking runtime for generated applications."""

	def __init__(self) -> None:
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

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_program(self, program_id: str, tenant_id: str, name: str, owner_id: str, country: str, currency: str, platforms: list[str], policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		platforms = normalize_codes(platforms)
		platforms_valid = bool(platforms) and all(platform in SUPPORTED_PLATFORMS for platform in platforms)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_program", "owner_present": bool(owner_id), "country_supported": country in SUPPORTED_COUNTRIES, "currency_supported": currency in SUPPORTED_CURRENCIES, "platforms_valid": platforms_valid})
		if program_id in self.programs:
			raise ValueError(f"mobile program already exists: {program_id}")
		program = MobileProgram(program_id, tenant_id, name, owner_id, country, currency, platforms)
		self.programs[program_id] = program
		self._audit(tenant_id, "mobile_program_registered", program_id)
		return program.to_dict()

	def enroll_customer(self, customer_id: str, tenant_id: str, customer_reference: str, country: str, kyc_reference: str, consent_reference: str, aml_reference: str, fraud_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "enroll_customer", "customer_present": bool(customer_reference), "country_supported": country in SUPPORTED_COUNTRIES, "kyc_present": bool(kyc_reference), "consent_present": bool(consent_reference), "aml_present": bool(aml_reference), "fraud_present": bool(fraud_reference)})
		if customer_id in self.customers:
			raise ValueError(f"mobile customer already exists: {customer_id}")
		customer = MobileCustomer(customer_id, tenant_id, customer_reference, country, kyc_reference, consent_reference, aml_reference, fraud_reference)
		self.customers[customer_id] = customer
		self._audit(tenant_id, "mobile_customer_enrolled", customer_id)
		return customer.to_dict()

	def bind_device(self, device_id: str, tenant_id: str, customer_id: str, platform: str, fingerprint: str, attestation_reference: str, risk_tier: str, policy_attached: bool = True) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		platform = normalize_code(platform)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "bind_device", "customer_present": customer is not None, "platform_supported": platform in SUPPORTED_PLATFORMS, "fingerprint_present": bool(fingerprint), "attestation_present": bool(attestation_reference), "risk_tier_present": bool(risk_tier)})
		if device_id in self.devices:
			raise ValueError(f"trusted device already exists: {device_id}")
		device = TrustedDevice(device_id, tenant_id, customer_id, platform, device_fingerprint_hash(fingerprint), attestation_reference, normalize_code(risk_tier))
		self.devices[device_id] = device
		self._audit(tenant_id, "trusted_device_bound", device_id)
		return device.to_dict()

	def register_auth_factor(self, factor_id: str, tenant_id: str, customer_id: str, device_id: str, factor_type: str, strength_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		device = self._tenant_device_or_none(device_id, tenant_id)
		factor_type = normalize_code(factor_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_auth_factor", "customer_present": customer is not None, "device_present": device is not None and (customer is None or device.customer_id == customer.id), "factor_type_supported": factor_type in SUPPORTED_AUTH_FACTORS, "strength_reference_present": bool(strength_reference)})
		factor = AuthFactor(factor_id, tenant_id, customer_id, device_id, factor_type, strength_reference)
		self.auth_factors[factor_id] = factor
		self._audit(tenant_id, "auth_factor_registered", factor_id)
		return factor.to_dict()

	def link_account(self, link_id: str, tenant_id: str, customer_id: str, link_type: str, account_reference: str, currency: str, provider_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		link_type = normalize_code(link_type)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "link_account", "customer_present": customer is not None, "link_type_supported": link_type in SUPPORTED_ACCOUNT_LINK_TYPES, "account_reference_present": bool(account_reference), "currency_supported": currency in SUPPORTED_CURRENCIES, "provider_reference_present": bool(provider_reference)})
		link = AccountLink(link_id, tenant_id, customer_id, link_type, account_reference, currency, provider_reference)
		self.account_links[link_id] = link
		self._audit(tenant_id, "account_linked", link_id)
		return link.to_dict()

	def initiate_payment(self, payment_id: str, tenant_id: str, customer_id: str, device_id: str, account_link_id: str, payment_type: str, amount: float | int | str, currency: str, recipient_reference: str, risk_reference: str, human_approval: str = "", policy_attached: bool = True) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		device = self._tenant_device_or_none(device_id, tenant_id)
		link = self._tenant_link_or_none(account_link_id, tenant_id)
		payment_type = normalize_code(payment_type)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		high_value = amount_value >= 100000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "initiate_payment", "customer_present": customer is not None, "device_present": device is not None and (customer is None or device.customer_id == customer.id), "account_link_present": link is not None and (customer is None or link.customer_id == customer.id), "payment_type_supported": payment_type in SUPPORTED_PAYMENT_TYPES, "positive_amount": amount_value > 0, "currency_supported": currency in SUPPORTED_CURRENCIES, "currency_matches_link": link is not None and link.currency == currency, "recipient_present": bool(recipient_reference), "risk_reference_present": bool(risk_reference), "high_value": high_value, "human_approval_recorded": bool(human_approval)})
		payment = MobilePayment(payment_id, tenant_id, customer_id, device_id, account_link_id, payment_type, amount_value, currency, recipient_reference, risk_reference, human_approval)
		self.payments[payment_id] = payment
		self._audit(tenant_id, "mobile_payment_initiated", payment_id)
		return payment.to_dict() | {"direction": payment_direction(payment_type)}

	def record_bill_payment(self, bill_id: str, tenant_id: str, payment_id: str, biller_reference: str, bill_account_reference: str) -> dict[str, Any]:
		payment = self._tenant_payment_or_none(payment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_bill_payment", "payment_present": payment is not None, "payment_type_matches": payment is not None and payment.payment_type == "bill_payment", "biller_reference_present": bool(biller_reference)})
		bill = BillPayment(bill_id, tenant_id, payment_id, biller_reference, bill_account_reference)
		self.bills[bill_id] = bill
		self._audit(tenant_id, "bill_payment_recorded", bill_id)
		return bill.to_dict()

	def purchase_airtime(self, airtime_id: str, tenant_id: str, payment_id: str, operator_reference: str, phone_reference: str) -> dict[str, Any]:
		payment = self._tenant_payment_or_none(payment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "purchase_airtime", "payment_present": payment is not None, "payment_type_matches": payment is not None and payment.payment_type == "airtime", "operator_reference_present": bool(operator_reference), "phone_reference_present": bool(phone_reference)})
		airtime = AirtimePurchase(airtime_id, tenant_id, payment_id, operator_reference, phone_reference)
		self.airtime[airtime_id] = airtime
		self._audit(tenant_id, "airtime_purchased", airtime_id)
		return airtime.to_dict()

	def open_service_request(self, request_id: str, tenant_id: str, customer_id: str, reason: str, reviewer_id: str, evidence_references: list[str]) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_service_request", "customer_present": customer is not None, "service_reason_supported": reason in SUPPORTED_SERVICE_REASONS, "evidence_present": bool(evidence_references), "reviewer_present": bool(reviewer_id)})
		request = ServiceRequest(request_id, tenant_id, customer_id, reason, reviewer_id, list(evidence_references))
		self.service_requests[request_id] = request
		self._audit(tenant_id, "service_request_opened", request_id)
		return request.to_dict()

	def set_notification_preference(self, preference_id: str, tenant_id: str, customer_id: str, channel: str, consent_reference: str, enabled: bool = True) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		channel = normalize_code(channel)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "set_notification_preference", "customer_present": customer is not None, "notification_channel_supported": channel in SUPPORTED_NOTIFICATION_CHANNELS, "consent_present": bool(consent_reference)})
		preference = NotificationPreference(preference_id, tenant_id, customer_id, channel, consent_reference, bool(enabled))
		self.notifications[preference_id] = preference
		self._audit(tenant_id, "notification_preference_set", preference_id)
		return preference.to_dict()

	def record_fraud_event(self, event_id: str, tenant_id: str, customer_id: str, severity: str, evidence_references: list[str], human_approval: str = "") -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		severity = normalize_code(severity)
		high_severity = is_high_severity(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_fraud_event", "customer_present": customer is not None, "severity_supported": severity in SUPPORTED_FRAUD_SEVERITIES, "evidence_present": bool(evidence_references), "high_severity": high_severity, "human_approval_recorded": bool(human_approval)})
		event = FraudEvent(event_id, tenant_id, customer_id, severity, list(evidence_references), human_approval)
		self.fraud_events[event_id] = event
		self._audit(tenant_id, "fraud_event_recorded", event_id)
		return event.to_dict()

	def register_mobile_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_mobile_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "mobile_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "mobile_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.mobile.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id), "customer_count": sum(1 for item in self.customers.values() if item.tenant_id == tenant_id), "device_count": sum(1 for item in self.devices.values() if item.tenant_id == tenant_id), "auth_factor_count": sum(1 for item in self.auth_factors.values() if item.tenant_id == tenant_id), "account_link_count": sum(1 for item in self.account_links.values() if item.tenant_id == tenant_id), "payment_count": sum(1 for item in self.payments.values() if item.tenant_id == tenant_id), "bill_count": sum(1 for item in self.bills.values() if item.tenant_id == tenant_id), "airtime_count": sum(1 for item in self.airtime.values() if item.tenant_id == tenant_id), "service_request_count": sum(1 for item in self.service_requests.values() if item.tenant_id == tenant_id), "notification_count": sum(1 for item in self.notifications.values() if item.tenant_id == tenant_id), "fraud_event_count": sum(1 for item in self.fraud_events.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_payments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.payments.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_devices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.devices.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

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

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = MobileEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "mobile_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "mobile_policy_denied")


MobileService = MobileBankingService
