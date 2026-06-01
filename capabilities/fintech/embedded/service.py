"""Executable service layer for APG Embedded Finance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANNELS, SUPPORTED_ENVIRONMENTS, SUPPORTED_PRODUCTS, evaluate_capability_rules, get_capability_contract
	from .embedded_runtime import normalize_code, normalize_codes, normalize_domain, percent_bounded, public_reference
	from .models import CustomerConsent, EmbeddedAccount, EmbeddedCardOffer, EmbeddedEvidence, EmbeddedLendingOffer, EmbeddedPayment, HostApplication, PartnerProgram, ProductPlacement, RevenueShare, SettlementBatch
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANNELS, SUPPORTED_ENVIRONMENTS, SUPPORTED_PRODUCTS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from embedded_runtime import normalize_code, normalize_codes, normalize_domain, percent_bounded, public_reference  # type: ignore
	from models import CustomerConsent, EmbeddedAccount, EmbeddedCardOffer, EmbeddedEvidence, EmbeddedLendingOffer, EmbeddedPayment, HostApplication, PartnerProgram, ProductPlacement, RevenueShare, SettlementBatch  # type: ignore


class EmbeddedFinanceService:
	"""In-memory Embedded Finance runtime for generated APG applications."""

	def __init__(self) -> None:
		self.programs: dict[str, PartnerProgram] = {}
		self.applications: dict[str, HostApplication] = {}
		self.placements: dict[str, ProductPlacement] = {}
		self.consents: dict[str, CustomerConsent] = {}
		self.accounts: dict[str, EmbeddedAccount] = {}
		self.payments: dict[str, EmbeddedPayment] = {}
		self.cards: dict[str, EmbeddedCardOffer] = {}
		self.lending: dict[str, EmbeddedLendingOffer] = {}
		self.settlements: dict[str, SettlementBatch] = {}
		self.revenue_shares: dict[str, RevenueShare] = {}
		self.evidence: dict[str, EmbeddedEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_partner_program(self, program_id: str, tenant_id: str, name: str, kyb_reference: str, contract_reference: str, risk_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_partner_program", "kyb_present": bool(kyb_reference), "contract_present": bool(contract_reference), "risk_present": bool(risk_reference)})
		program = PartnerProgram(program_id, tenant_id, name, kyb_reference, contract_reference, risk_reference)
		self.programs[program_id] = program
		self._audit(tenant_id, "partner_program_registered", program_id)
		return program.to_dict()

	def register_host_application(self, application_id: str, tenant_id: str, program_id: str, name: str, environment: str, domain: str, terms_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		environment = normalize_code(environment)
		domain = normalize_domain(domain)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_host_application", "program_present": program is not None, "environment_supported": environment in SUPPORTED_ENVIRONMENTS, "domain_present": bool(domain), "terms_present": bool(terms_reference)})
		application = HostApplication(application_id, tenant_id, program_id, name, environment, domain, terms_reference)
		self.applications[application_id] = application
		self._audit(tenant_id, "host_application_registered", application_id)
		return application.to_dict()

	def publish_product_placement(self, placement_id: str, tenant_id: str, application_id: str, product_type: str, channel: str, scopes: list[str], risk_policy_reference: str) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		product_type = normalize_code(product_type)
		channel = normalize_code(channel)
		scopes = normalize_codes(scopes)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_product_placement", "application_present": application is not None, "product_supported": product_type in SUPPORTED_PRODUCTS, "channel_supported": channel in SUPPORTED_CHANNELS, "scopes_present": bool(scopes), "risk_policy_present": bool(risk_policy_reference)})
		placement = ProductPlacement(placement_id, tenant_id, application_id, product_type, channel, scopes, risk_policy_reference)
		self.placements[placement_id] = placement
		self._audit(tenant_id, "product_placement_published", placement_id)
		return placement.to_dict()

	def capture_customer_consent(self, consent_id: str, tenant_id: str, application_id: str, customer_reference: str, scopes: list[str], expiry_date: str, policy_attached: bool = True) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		scopes = normalize_codes(scopes)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "capture_customer_consent", "application_present": application is not None, "customer_present": bool(customer_reference), "scopes_present": bool(scopes), "expiry_present": bool(expiry_date)})
		consent = CustomerConsent(consent_id, tenant_id, application_id, customer_reference, scopes, expiry_date)
		self.consents[consent_id] = consent
		self._audit(tenant_id, "customer_consent_captured", consent_id)
		return consent.to_dict()

	def open_embedded_account(self, account_id: str, tenant_id: str, application_id: str, customer_reference: str, wallet_reference: str, kyc_reference: str) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_embedded_account", "application_present": application is not None, "kyc_present": bool(kyc_reference), "wallet_present": bool(wallet_reference)})
		account = EmbeddedAccount(account_id, tenant_id, application_id, customer_reference, wallet_reference, kyc_reference)
		self.accounts[account_id] = account
		self._audit(tenant_id, "embedded_account_opened", account_id)
		return account.to_dict() | {"public_account_reference": public_reference("acct", application_id, customer_reference)}

	def initiate_embedded_payment(self, payment_id: str, tenant_id: str, application_id: str, placement_id: str, consent_id: str, source_reference: str, destination_reference: str, amount_minor: int, currency: str, risk_reference: str) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		placement = self._tenant_placement_or_none(placement_id, tenant_id)
		consent = self._tenant_consent_or_none(consent_id, tenant_id)
		currency = currency.strip().upper()
		consent_covers_scope = consent is not None and consent.application_id == application_id and "payments.write" in consent.scopes and consent.status == "active"
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "initiate_embedded_payment", "application_present": application is not None, "placement_present": placement is not None, "placement_matches_application": placement is not None and placement.application_id == application_id, "consent_present": consent is not None, "consent_covers_scope": consent_covers_scope, "positive_amount": int(amount_minor) > 0, "currency_supported": currency in get_capability_contract(tenant_id)["configuration"]["payments"]["supported_currencies"], "risk_reference_present": bool(risk_reference)})
		payment = EmbeddedPayment(payment_id, tenant_id, application_id, placement_id, consent_id, source_reference, destination_reference, int(amount_minor), currency, risk_reference)
		self.payments[payment_id] = payment
		self._audit(tenant_id, "embedded_payment_initiated", payment_id)
		return payment.to_dict()

	def offer_embedded_card(self, card_id: str, tenant_id: str, application_id: str, customer_reference: str, limit_minor: int, risk_reference: str) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "offer_embedded_card", "application_present": application is not None, "positive_limit": int(limit_minor) > 0, "risk_reference_present": bool(risk_reference)})
		card = EmbeddedCardOffer(card_id, tenant_id, application_id, customer_reference, int(limit_minor), risk_reference)
		self.cards[card_id] = card
		self._audit(tenant_id, "embedded_card_offered", card_id)
		return card.to_dict()

	def create_lending_offer(self, offer_id: str, tenant_id: str, application_id: str, customer_reference: str, amount_minor: int, affordability_reference: str, underwriting_reference: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_lending_offer", "affordability_present": bool(affordability_reference), "underwriting_present": bool(underwriting_reference)})
		offer = EmbeddedLendingOffer(offer_id, tenant_id, application_id, customer_reference, int(amount_minor), affordability_reference, underwriting_reference)
		self.lending[offer_id] = offer
		self._audit(tenant_id, "embedded_lending_offer_created", offer_id)
		return offer.to_dict()

	def close_settlement_batch(self, batch_id: str, tenant_id: str, program_id: str, amount_minor: int, currency: str, reconciliation_reference: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "close_settlement_batch", "reconciled": bool(reconciliation_reference), "positive_amount": int(amount_minor) > 0})
		batch = SettlementBatch(batch_id, tenant_id, program_id, int(amount_minor), currency.strip().upper(), reconciliation_reference)
		self.settlements[batch_id] = batch
		self._audit(tenant_id, "settlement_batch_closed", batch_id)
		return batch.to_dict()

	def record_revenue_share(self, share_id: str, tenant_id: str, program_id: str, percent: float, contract_reference: str) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_revenue_share", "program_present": program is not None, "percent_bounded": percent_bounded(float(percent)), "contract_present": bool(contract_reference)})
		share = RevenueShare(share_id, tenant_id, program_id, float(percent), contract_reference)
		self.revenue_shares[share_id] = share
		self._audit(tenant_id, "revenue_share_recorded", share_id)
		return share.to_dict()

	def register_embedded_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_embedded_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = EmbeddedEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "embedded_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "embedded_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.embedded.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "program_count": self._count(self.programs, tenant_id), "application_count": self._count(self.applications, tenant_id), "placement_count": self._count(self.placements, tenant_id), "consent_count": self._count(self.consents, tenant_id), "account_count": self._count(self.accounts, tenant_id), "payment_count": self._count(self.payments, tenant_id), "card_count": self._count(self.cards, tenant_id), "lending_count": self._count(self.lending, tenant_id), "settlement_count": self._count(self.settlements, tenant_id), "revenue_share_count": self._count(self.revenue_shares, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_program_or_none(self, item_id: str, tenant_id: str) -> PartnerProgram | None:
		item = self.programs.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_application_or_none(self, item_id: str, tenant_id: str) -> HostApplication | None:
		item = self.applications.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_placement_or_none(self, item_id: str, tenant_id: str) -> ProductPlacement | None:
		item = self.placements.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_consent_or_none(self, item_id: str, tenant_id: str) -> CustomerConsent | None:
		item = self.consents.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "embedded_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "embedded_policy_denied")


EmbeddedService = EmbeddedFinanceService
