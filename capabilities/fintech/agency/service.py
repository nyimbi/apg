"""Executable service layer for APG Agency Banking."""

from __future__ import annotations

from typing import Any

try:
	from .agency_runtime import apply_float_delta, estimate_commission, normalize_amount, normalize_code, normalize_codes, normalize_country, normalize_currency, service_requires_float
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASH_MOVEMENT_TYPES, SUPPORTED_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_CUSTOMER_TIERS, SUPPORTED_DISPUTE_REASONS, SUPPORTED_OUTLET_TYPES, SUPPORTED_SERVICES, SUPPORTED_SETTLEMENT_MODELS, SUPPORTED_SUPERVISION_OUTCOMES, evaluate_capability_rules, get_capability_contract
	from .models import AccreditedAgent, AgencyCustomer, AgencyDispute, AgencyEvidence, AgencyOutlet, AgencyProgram, AgencyTransaction, CashMovement, CommissionSettlement, FloatAccount, SupervisionVisit
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from agency_runtime import apply_float_delta, estimate_commission, normalize_amount, normalize_code, normalize_codes, normalize_country, normalize_currency, service_requires_float  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASH_MOVEMENT_TYPES, SUPPORTED_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_CUSTOMER_TIERS, SUPPORTED_DISPUTE_REASONS, SUPPORTED_OUTLET_TYPES, SUPPORTED_SERVICES, SUPPORTED_SETTLEMENT_MODELS, SUPPORTED_SUPERVISION_OUTCOMES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AccreditedAgent, AgencyCustomer, AgencyDispute, AgencyEvidence, AgencyOutlet, AgencyProgram, AgencyTransaction, CashMovement, CommissionSettlement, FloatAccount, SupervisionVisit  # type: ignore


class AgencyBankingService:
	"""Dependency-light agency banking runtime for generated applications."""

	def __init__(self) -> None:
		self.programs: dict[str, AgencyProgram] = {}
		self.outlets: dict[str, AgencyOutlet] = {}
		self.agents: dict[str, AccreditedAgent] = {}
		self.float_accounts: dict[str, FloatAccount] = {}
		self.customers: dict[str, AgencyCustomer] = {}
		self.transactions: dict[str, AgencyTransaction] = {}
		self.cash_movements: dict[str, CashMovement] = {}
		self.commissions: dict[str, CommissionSettlement] = {}
		self.disputes: dict[str, AgencyDispute] = {}
		self.supervision_visits: dict[str, SupervisionVisit] = {}
		self.evidence: dict[str, AgencyEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_program(self, program_id: str, tenant_id: str, name: str, owner_id: str, country: str, currency: str, settlement_model: str, services: list[str], policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		settlement_model = normalize_code(settlement_model)
		services = normalize_codes(services)
		services_valid = bool(services) and all(service in SUPPORTED_SERVICES for service in services)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_program", "owner_present": bool(owner_id), "country_supported": country in SUPPORTED_COUNTRIES, "currency_supported": currency in SUPPORTED_CURRENCIES, "settlement_model_supported": settlement_model in SUPPORTED_SETTLEMENT_MODELS, "services_valid": services_valid})
		if program_id in self.programs:
			raise ValueError(f"agency program already exists: {program_id}")
		program = AgencyProgram(program_id, tenant_id, name, owner_id, country, currency, settlement_model, services)
		self.programs[program_id] = program
		self._audit(tenant_id, "agency_program_registered", program_id)
		return program.to_dict()

	def onboard_outlet(self, outlet_id: str, tenant_id: str, program_id: str, name: str, outlet_type: str, country: str, license_reference: str, location_reference: str, security_plan_reference: str, primary_channel: str, initial_float: float | int | str, policy_attached: bool = True) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		outlet_type = normalize_code(outlet_type)
		country = normalize_country(country)
		primary_channel = normalize_code(primary_channel)
		initial_float_value = normalize_amount(initial_float)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_outlet", "program_present": program is not None, "outlet_type_supported": outlet_type in SUPPORTED_OUTLET_TYPES, "country_supported": country in SUPPORTED_COUNTRIES, "license_present": bool(license_reference), "location_present": bool(location_reference), "security_plan_present": bool(security_plan_reference), "channel_supported": primary_channel in SUPPORTED_CHANNELS, "initial_float_valid": initial_float_value >= 500})
		if outlet_id in self.outlets:
			raise ValueError(f"agency outlet already exists: {outlet_id}")
		outlet = AgencyOutlet(outlet_id, tenant_id, program_id, name, outlet_type, country, license_reference, location_reference, security_plan_reference, primary_channel, initial_float_value)
		self.outlets[outlet_id] = outlet
		self._audit(tenant_id, "agency_outlet_onboarded", outlet_id)
		return outlet.to_dict()

	def accredit_agent(self, agent_id: str, tenant_id: str, outlet_id: str, name: str, identity_reference: str, training_reference: str, background_check_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "accredit_agent", "outlet_present": outlet is not None, "identity_present": bool(identity_reference), "training_present": bool(training_reference), "background_check_present": bool(background_check_reference)})
		if agent_id in self.agents:
			raise ValueError(f"agency agent already exists: {agent_id}")
		agent = AccreditedAgent(agent_id, tenant_id, outlet_id, name, identity_reference, training_reference, background_check_reference)
		self.agents[agent_id] = agent
		self._audit(tenant_id, "agency_agent_accredited", agent_id)
		return agent.to_dict()

	def open_float_account(self, float_account_id: str, tenant_id: str, outlet_id: str, currency: str, opening_balance: float | int | str, ledger_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		currency = normalize_currency(currency)
		balance = normalize_amount(opening_balance)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "open_float_account", "outlet_present": outlet is not None, "currency_supported": currency in SUPPORTED_CURRENCIES, "balance_non_negative": balance >= 0, "ledger_reference_present": bool(ledger_reference)})
		if float_account_id in self.float_accounts:
			raise ValueError(f"float account already exists: {float_account_id}")
		account = FloatAccount(float_account_id, tenant_id, outlet_id, currency, balance, ledger_reference)
		self.float_accounts[float_account_id] = account
		self._audit(tenant_id, "float_account_opened", float_account_id)
		return account.to_dict()

	def onboard_customer(self, customer_id: str, tenant_id: str, customer_reference: str, tier: str, kyc_reference: str, consent_reference: str, aml_reference: str, fraud_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		tier = normalize_code(tier)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_customer", "customer_present": bool(customer_reference), "customer_tier_supported": tier in SUPPORTED_CUSTOMER_TIERS, "kyc_present": bool(kyc_reference), "consent_present": bool(consent_reference), "aml_present": bool(aml_reference), "fraud_present": bool(fraud_reference)})
		if customer_id in self.customers:
			raise ValueError(f"agency customer already exists: {customer_id}")
		customer = AgencyCustomer(customer_id, tenant_id, customer_reference, tier, kyc_reference, consent_reference, aml_reference, fraud_reference)
		self.customers[customer_id] = customer
		self._audit(tenant_id, "agency_customer_onboarded", customer_id)
		return customer.to_dict()

	def record_transaction(self, transaction_id: str, tenant_id: str, outlet_id: str, agent_id: str, customer_id: str, float_account_id: str, service: str, amount: float | int | str, currency: str, channel: str, customer_reference: str, risk_reference: str, human_approval: str = "", policy_attached: bool = True) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		float_account = self._tenant_float_or_none(float_account_id, tenant_id)
		program = self._tenant_program_or_none(outlet.program_id, tenant_id) if outlet is not None else None
		service = normalize_code(service)
		channel = normalize_code(channel)
		currency = normalize_currency(currency)
		amount_value = normalize_amount(amount)
		high_value = amount_value >= 100000
		float_sufficient = not service_requires_float(service) or (float_account is not None and float_account.available_balance >= amount_value)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_transaction", "outlet_present": outlet is not None, "agent_present": agent is not None and (outlet is None or agent.outlet_id == outlet.id), "customer_present": customer is not None, "float_account_present": float_account is not None and (outlet is None or float_account.outlet_id == outlet.id), "service_supported": service in SUPPORTED_SERVICES, "service_allowed_by_program": program is not None and service in program.services, "channel_supported": channel in SUPPORTED_CHANNELS, "currency_supported": currency in SUPPORTED_CURRENCIES, "currency_matches_float": float_account is not None and float_account.currency == currency, "positive_amount": amount_value > 0, "within_limit": amount_value <= 200000, "float_sufficient": float_sufficient, "customer_reference_present": bool(customer_reference), "risk_reference_present": bool(risk_reference), "high_value": high_value, "human_approval_recorded": bool(human_approval)})
		if transaction_id in self.transactions:
			raise ValueError(f"agency transaction already exists: {transaction_id}")
		transaction = AgencyTransaction(transaction_id, tenant_id, outlet_id, agent_id, customer_id, float_account_id, service, amount_value, currency, channel, customer_reference, risk_reference)
		self.transactions[transaction_id] = transaction
		if float_account is not None:
			float_account.available_balance = apply_float_delta(float_account.available_balance, service, amount_value)
		self._audit(tenant_id, "agency_transaction_recorded", transaction_id)
		return transaction.to_dict()

	def record_cash_movement(self, movement_id: str, tenant_id: str, outlet_id: str, movement_type: str, amount: float | int | str, currency: str, custodian_reference: str, human_approval: str = "") -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		movement_type = normalize_code(movement_type)
		currency = normalize_currency(currency)
		amount_value = normalize_amount(amount)
		high_value = amount_value >= 100000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_cash_movement", "outlet_present": outlet is not None, "movement_type_supported": movement_type in SUPPORTED_CASH_MOVEMENT_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES, "positive_amount": amount_value > 0, "custodian_present": bool(custodian_reference), "high_value": high_value, "human_approval_recorded": bool(human_approval)})
		movement = CashMovement(movement_id, tenant_id, outlet_id, movement_type, amount_value, currency, custodian_reference, human_approval)
		self.cash_movements[movement_id] = movement
		self._audit(tenant_id, "cash_movement_recorded", movement_id)
		return movement.to_dict()

	def settle_commission(self, settlement_id: str, tenant_id: str, outlet_id: str, period: str, amount: float | int | str, currency: str, reconciliation_reference: str, payment_reference: str) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		currency = normalize_currency(currency)
		amount_value = normalize_amount(amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "settle_commission", "outlet_present": outlet is not None, "currency_supported": currency in SUPPORTED_CURRENCIES, "positive_amount": amount_value > 0, "reconciliation_present": bool(reconciliation_reference), "payment_reference_present": bool(payment_reference)})
		settlement = CommissionSettlement(settlement_id, tenant_id, outlet_id, period, amount_value, currency, reconciliation_reference, payment_reference)
		self.commissions[settlement_id] = settlement
		self._audit(tenant_id, "commission_settlement_recorded", settlement_id)
		return settlement.to_dict()

	def open_dispute(self, dispute_id: str, tenant_id: str, transaction_id: str, reason: str, reviewer_id: str, evidence_references: list[str]) -> dict[str, Any]:
		transaction = self._tenant_transaction_or_none(transaction_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_dispute", "transaction_present": transaction is not None, "dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS, "evidence_present": bool(evidence_references), "reviewer_present": bool(reviewer_id)})
		dispute = AgencyDispute(dispute_id, tenant_id, transaction_id, reason, reviewer_id, list(evidence_references))
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "agency_dispute_opened", dispute_id)
		return dispute.to_dict()

	def record_supervision_visit(self, visit_id: str, tenant_id: str, outlet_id: str, supervisor_id: str, outcome: str, evidence_references: list[str], findings: list[str] | None = None, remediation_plan_reference: str = "") -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		outcome = normalize_code(outcome)
		findings = list(findings or [])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_supervision_visit", "outlet_present": outlet is not None, "supervisor_present": bool(supervisor_id), "outcome_supported": outcome in SUPPORTED_SUPERVISION_OUTCOMES, "evidence_present": bool(evidence_references), "findings_present": bool(findings), "remediation_plan_present": bool(remediation_plan_reference)})
		visit = SupervisionVisit(visit_id, tenant_id, outlet_id, supervisor_id, outcome, list(evidence_references), findings, remediation_plan_reference)
		self.supervision_visits[visit_id] = visit
		self._audit(tenant_id, "supervision_visit_recorded", visit_id)
		return visit.to_dict()

	def register_agency_ai_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_agency_ai_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "ai_agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "agency_ai_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "agency_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.agency.lifecycle", "accepted": True}

	def estimate_transaction_commission(self, transaction_id: str, tenant_id: str) -> dict[str, Any]:
		transaction = self._tenant_transaction_or_none(transaction_id, tenant_id)
		if transaction is None:
			raise KeyError(f"agency transaction not found: {transaction_id}")
		return {"transaction_id": transaction_id, "tenant_id": tenant_id, "commission": estimate_commission(transaction.amount, transaction.service), "currency": transaction.currency}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id), "outlet_count": sum(1 for item in self.outlets.values() if item.tenant_id == tenant_id), "agent_count": sum(1 for item in self.agents.values() if item.tenant_id == tenant_id), "float_account_count": sum(1 for item in self.float_accounts.values() if item.tenant_id == tenant_id), "customer_count": sum(1 for item in self.customers.values() if item.tenant_id == tenant_id), "transaction_count": sum(1 for item in self.transactions.values() if item.tenant_id == tenant_id), "cash_movement_count": sum(1 for item in self.cash_movements.values() if item.tenant_id == tenant_id), "commission_count": sum(1 for item in self.commissions.values() if item.tenant_id == tenant_id), "dispute_count": sum(1 for item in self.disputes.values() if item.tenant_id == tenant_id), "supervision_count": sum(1 for item in self.supervision_visits.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.transactions.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_outlets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.outlets.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_float_accounts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.float_accounts.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _tenant_program_or_none(self, program_id: str, tenant_id: str) -> AgencyProgram | None:
		program = self.programs.get(program_id)
		return program if program is not None and program.tenant_id == tenant_id else None

	def _tenant_outlet_or_none(self, outlet_id: str, tenant_id: str) -> AgencyOutlet | None:
		outlet = self.outlets.get(outlet_id)
		return outlet if outlet is not None and outlet.tenant_id == tenant_id else None

	def _tenant_agent_or_none(self, agent_id: str, tenant_id: str) -> AccreditedAgent | None:
		agent = self.agents.get(agent_id)
		return agent if agent is not None and agent.tenant_id == tenant_id else None

	def _tenant_float_or_none(self, float_account_id: str, tenant_id: str) -> FloatAccount | None:
		account = self.float_accounts.get(float_account_id)
		return account if account is not None and account.tenant_id == tenant_id else None

	def _tenant_customer_or_none(self, customer_id: str, tenant_id: str) -> AgencyCustomer | None:
		customer = self.customers.get(customer_id)
		return customer if customer is not None and customer.tenant_id == tenant_id else None

	def _tenant_transaction_or_none(self, transaction_id: str, tenant_id: str) -> AgencyTransaction | None:
		transaction = self.transactions.get(transaction_id)
		return transaction if transaction is not None and transaction.tenant_id == tenant_id else None

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = AgencyEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "agency_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "agency_policy_denied")


AgencyService = AgencyBankingService
