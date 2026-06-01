"""Executable service layer for APG Digital Cards."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AML_RESULTS, SUPPORTED_CARD_TYPES, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISPUTE_REASONS, SUPPORTED_FRAUD_DECISIONS, SUPPORTED_MERCHANT_CATEGORIES, SUPPORTED_PRODUCTS, SUPPORTED_TOKEN_TYPES, evaluate_capability_rules, get_capability_contract
	from .cards_runtime import authorization_decision, mask_pan, normalize_amount, normalize_code, normalize_country, normalize_currency
	from .models import Card, CardAuthorization, CardDispute, CardEvidence, CardProgram, CardToken, Cardholder
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AML_RESULTS, SUPPORTED_CARD_TYPES, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISPUTE_REASONS, SUPPORTED_FRAUD_DECISIONS, SUPPORTED_MERCHANT_CATEGORIES, SUPPORTED_PRODUCTS, SUPPORTED_TOKEN_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from cards_runtime import authorization_decision, mask_pan, normalize_amount, normalize_code, normalize_country, normalize_currency  # type: ignore
	from models import Card, CardAuthorization, CardDispute, CardEvidence, CardProgram, CardToken, Cardholder  # type: ignore


class CardService:
	"""Dependency-light card runtime for generated applications."""

	def __init__(self) -> None:
		self.programs: dict[str, CardProgram] = {}
		self.cardholders: dict[str, Cardholder] = {}
		self.cards: dict[str, Card] = {}
		self.tokens: dict[str, CardToken] = {}
		self.authorizations: dict[str, CardAuthorization] = {}
		self.disputes: dict[str, CardDispute] = {}
		self.evidence: dict[str, CardEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_program(self, program_id: str, tenant_id: str, name: str, owner_id: str, bin_range: str, currency: str, settlement_account: str, policy_attached: bool = True) -> dict[str, Any]:
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_program", "program_owner_present": bool(owner_id), "bin_range_present": bool(bin_range), "currency_supported": currency in SUPPORTED_CURRENCIES, "settlement_account_present": bool(settlement_account)})
		if program_id in self.programs:
			raise ValueError(f"card program already exists: {program_id}")
		program = CardProgram(program_id, tenant_id, name, owner_id, bin_range, currency, settlement_account)
		self.programs[program_id] = program
		self._audit(tenant_id, "card_program_registered", program_id)
		return program.to_dict()

	def onboard_cardholder(self, cardholder_id: str, tenant_id: str, customer_reference: str, kyc_profile_id: str, country: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_cardholder", "customer_present": bool(customer_reference), "kyc_present": bool(kyc_profile_id), "country_supported": country in SUPPORTED_COUNTRIES})
		if cardholder_id in self.cardholders:
			raise ValueError(f"cardholder already exists: {cardholder_id}")
		cardholder = Cardholder(cardholder_id, tenant_id, customer_reference, kyc_profile_id, country)
		self.cardholders[cardholder_id] = cardholder
		self._audit(tenant_id, "cardholder_onboarded", cardholder_id)
		return cardholder.to_dict()

	def issue_card(self, card_id: str, tenant_id: str, program_id: str, cardholder_id: str, card_type: str, product: str, wallet_reference: str, funding_account: str, consent_reference: str, shipping_reference: str = "", policy_attached: bool = True) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		cardholder = self._tenant_cardholder_or_none(cardholder_id, tenant_id)
		card_type = normalize_code(card_type)
		product = normalize_code(product)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "issue_card", "program_present": program is not None, "cardholder_present": cardholder is not None, "card_type_supported": card_type in SUPPORTED_CARD_TYPES, "card_product_supported": product in SUPPORTED_PRODUCTS, "wallet_present": bool(wallet_reference), "funding_account_present": bool(funding_account), "consent_present": bool(consent_reference), "physical_card": card_type == "physical", "shipping_present": bool(shipping_reference)})
		if card_id in self.cards:
			raise ValueError(f"card already exists: {card_id}")
		card = Card(card_id, tenant_id, program_id, cardholder_id, card_type, product, wallet_reference, funding_account, mask_pan(card_id, program.bin_range if program else "000000"))
		self.cards[card_id] = card
		self._audit(tenant_id, "card_issued", card_id)
		return card.to_dict()

	def provision_token(self, token_id: str, tenant_id: str, card_id: str, token_type: str, token_reference: str, key_domain_id: str, device_or_merchant_reference: str) -> dict[str, Any]:
		card = self._tenant_card_or_none(card_id, tenant_id)
		token_type = normalize_code(token_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "provision_token", "card_present": card is not None, "token_type_supported": token_type in SUPPORTED_TOKEN_TYPES, "token_reference_present": bool(token_reference), "key_domain_present": bool(key_domain_id), "device_or_merchant_present": bool(device_or_merchant_reference)})
		if token_id in self.tokens:
			raise ValueError(f"card token already exists: {token_id}")
		token = CardToken(token_id, tenant_id, card_id, token_type, token_reference, key_domain_id, device_or_merchant_reference)
		self.tokens[token_id] = token
		self._audit(tenant_id, "card_token_provisioned", token_id)
		return token.to_dict()

	def authorize_transaction(self, authorization_id: str, tenant_id: str, card_id: str, amount: float | int | str, currency: str, merchant_category: str, fraud_reference: str, aml_reference: str, fraud_decision: str = "clear", aml_result: str = "clear", limit_override: bool = False, human_approval: str = "") -> dict[str, Any]:
		card = self._tenant_card_or_none(card_id, tenant_id)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		merchant_category = normalize_code(merchant_category)
		fraud_decision = normalize_code(fraud_decision)
		aml_result = normalize_code(aml_result)
		high_impact = amount_value >= 100000 or limit_override or merchant_category == "restricted" or fraud_decision in {"review", "hold"} or aml_result == "review"
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "authorize_transaction", "card_present": card is not None, "positive_amount": amount_value > 0, "currency_supported": currency in SUPPORTED_CURRENCIES, "merchant_category_supported": merchant_category in SUPPORTED_MERCHANT_CATEGORIES, "fraud_decision_supported": fraud_decision in SUPPORTED_FRAUD_DECISIONS, "fraud_blocked": fraud_decision == "block", "aml_result_supported": aml_result in SUPPORTED_AML_RESULTS, "aml_blocked": aml_result == "blocked", "high_impact": high_impact, "human_approval_recorded": bool(human_approval)})
		decision = authorization_decision(fraud_decision, aml_result, high_impact)
		record = CardAuthorization(authorization_id, tenant_id, card_id, amount_value, currency, merchant_category, fraud_reference, aml_reference, decision)
		self.authorizations[authorization_id] = record
		self._audit(tenant_id, "card_authorization_decided", authorization_id)
		return record.to_dict()

	def file_dispute(self, dispute_id: str, tenant_id: str, transaction_reference: str, reason: str, evidence_references: list[str], reviewer_id: str) -> dict[str, Any]:
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "file_dispute", "transaction_present": bool(transaction_reference), "dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS, "evidence_present": bool(evidence_references), "reviewer_present": bool(reviewer_id)})
		dispute = CardDispute(dispute_id, tenant_id, transaction_reference, reason, list(evidence_references), reviewer_id)
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "card_dispute_filed", dispute_id)
		return dispute.to_dict()

	def register_card_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_card_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "card_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "card_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.cards.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		cards = [item for item in self.cards.values() if item.tenant_id == tenant_id]
		authorizations = [item for item in self.authorizations.values() if item.tenant_id == tenant_id]
		disputes = [item for item in self.disputes.values() if item.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id), "cardholder_count": sum(1 for item in self.cardholders.values() if item.tenant_id == tenant_id), "card_count": len(cards), "token_count": sum(1 for item in self.tokens.values() if item.tenant_id == tenant_id), "authorization_count": len(authorizations), "approval_count": sum(1 for item in authorizations if item.decision == "approve"), "review_count": sum(1 for item in authorizations if item.decision == "review"), "dispute_count": len(disputes), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_cards(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		cards = self.cards.values()
		if tenant_id is not None:
			cards = [card for card in cards if card.tenant_id == tenant_id]
		return [card.to_dict() for card in sorted(cards, key=lambda item: item.id)]

	def list_authorizations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		authorizations = self.authorizations.values()
		if tenant_id is not None:
			authorizations = [authorization for authorization in authorizations if authorization.tenant_id == tenant_id]
		return [authorization.to_dict() for authorization in sorted(authorizations, key=lambda item: item.id)]

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

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = CardEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "card_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "card_policy_denied")


DigitalCardsService = CardService
