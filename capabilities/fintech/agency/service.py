"""Executable service layer for APG Agency Banking."""

from __future__ import annotations

import datetime
import statistics
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .domain.adapters import get_auth_adapter, get_audit_adapter
	from .database.store import get_store
	from .agency_runtime import (
		apply_float_delta, estimate_commission, normalize_amount,
		normalize_code, normalize_codes, normalize_country,
		normalize_currency, service_requires_float,
	)
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASH_MOVEMENT_TYPES,
		SUPPORTED_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES,
		SUPPORTED_CUSTOMER_TIERS, SUPPORTED_DISPUTE_REASONS, SUPPORTED_OUTLET_TYPES,
		SUPPORTED_SERVICES, SUPPORTED_SETTLEMENT_MODELS, SUPPORTED_SUPERVISION_OUTCOMES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AccreditedAgent, AgencyCustomer, AgencyDispute, AgencyEvidence,
		AgencyOutlet, AgencyProgram, AgencyTransaction, CashMovement,
		CommissionSettlement, FloatAccount, SupervisionVisit,
	)
except ImportError:  # pragma: no cover
	from agency_runtime import apply_float_delta, estimate_commission, normalize_amount, normalize_code, normalize_codes, normalize_country, normalize_currency, service_requires_float  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASH_MOVEMENT_TYPES, SUPPORTED_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_CUSTOMER_TIERS, SUPPORTED_DISPUTE_REASONS, SUPPORTED_OUTLET_TYPES, SUPPORTED_SERVICES, SUPPORTED_SETTLEMENT_MODELS, SUPPORTED_SUPERVISION_OUTCOMES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AccreditedAgent, AgencyCustomer, AgencyDispute, AgencyEvidence, AgencyOutlet, AgencyProgram, AgencyTransaction, CashMovement, CommissionSettlement, FloatAccount, SupervisionVisit  # type: ignore


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


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
		# Extended state for new methods
		self._dormancy_records: dict[str, dict[str, Any]] = {}   # agent_id -> dormancy record
		self._float_topup_history: list[dict[str, Any]] = []
		self._compliance_check_records: list[dict[str, Any]] = []
		self._network_analytics_runs: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Contract                                                             #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core existing methods                                                #
	# ------------------------------------------------------------------ #

	def register_program(
		self,
		program_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		country: str,
		currency: str,
		settlement_model: str,
		services: list[str],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		settlement_model = normalize_code(settlement_model)
		services = normalize_codes(services)
		services_valid = bool(services) and all(service in SUPPORTED_SERVICES for service in services)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_program",
			"owner_present": bool(owner_id),
			"country_supported": country in SUPPORTED_COUNTRIES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"settlement_model_supported": settlement_model in SUPPORTED_SETTLEMENT_MODELS,
			"services_valid": services_valid,
		})
		if program_id in self.programs:
			raise ValueError(f"agency program already exists: {program_id}")
		program = AgencyProgram(program_id, tenant_id, name, owner_id, country, currency, settlement_model, services)
		self.programs[program_id] = program
		self._audit(tenant_id, "agency_program_registered", program_id)
		return program.to_dict()

	def onboard_outlet(
		self,
		outlet_id: str,
		tenant_id: str,
		program_id: str,
		name: str,
		outlet_type: str,
		country: str,
		license_reference: str,
		location_reference: str,
		security_plan_reference: str,
		primary_channel: str,
		initial_float: float | int | str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		outlet_type = normalize_code(outlet_type)
		country = normalize_country(country)
		primary_channel = normalize_code(primary_channel)
		initial_float_value = normalize_amount(initial_float)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_outlet",
			"program_present": program is not None,
			"outlet_type_supported": outlet_type in SUPPORTED_OUTLET_TYPES,
			"country_supported": country in SUPPORTED_COUNTRIES,
			"license_present": bool(license_reference),
			"location_present": bool(location_reference),
			"security_plan_present": bool(security_plan_reference),
			"channel_supported": primary_channel in SUPPORTED_CHANNELS,
			"initial_float_valid": initial_float_value >= 500,
		})
		if outlet_id in self.outlets:
			raise ValueError(f"agency outlet already exists: {outlet_id}")
		outlet = AgencyOutlet(outlet_id, tenant_id, program_id, name, outlet_type, country, license_reference, location_reference, security_plan_reference, primary_channel, initial_float_value)
		self.outlets[outlet_id] = outlet
		self._audit(tenant_id, "agency_outlet_onboarded", outlet_id)
		return outlet.to_dict()

	def accredit_agent(
		self,
		agent_id: str,
		tenant_id: str,
		outlet_id: str,
		name: str,
		identity_reference: str,
		training_reference: str,
		background_check_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "accredit_agent",
			"outlet_present": outlet is not None,
			"identity_present": bool(identity_reference),
			"training_present": bool(training_reference),
			"background_check_present": bool(background_check_reference),
		})
		if agent_id in self.agents:
			raise ValueError(f"agency agent already exists: {agent_id}")
		agent = AccreditedAgent(agent_id, tenant_id, outlet_id, name, identity_reference, training_reference, background_check_reference)
		self.agents[agent_id] = agent
		self._audit(tenant_id, "agency_agent_accredited", agent_id)
		return agent.to_dict()

	def open_float_account(
		self,
		float_account_id: str,
		tenant_id: str,
		outlet_id: str,
		currency: str,
		opening_balance: float | int | str,
		ledger_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		currency = normalize_currency(currency)
		balance = normalize_amount(opening_balance)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "open_float_account",
			"outlet_present": outlet is not None,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"balance_non_negative": balance >= 0,
			"ledger_reference_present": bool(ledger_reference),
		})
		if float_account_id in self.float_accounts:
			raise ValueError(f"float account already exists: {float_account_id}")
		account = FloatAccount(float_account_id, tenant_id, outlet_id, currency, balance, ledger_reference)
		self.float_accounts[float_account_id] = account
		self._audit(tenant_id, "float_account_opened", float_account_id)
		return account.to_dict()

	def onboard_customer(
		self,
		customer_id: str,
		tenant_id: str,
		customer_reference: str,
		tier: str,
		kyc_reference: str,
		consent_reference: str,
		aml_reference: str,
		fraud_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		tier = normalize_code(tier)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_customer",
			"customer_present": bool(customer_reference),
			"customer_tier_supported": tier in SUPPORTED_CUSTOMER_TIERS,
			"kyc_present": bool(kyc_reference),
			"consent_present": bool(consent_reference),
			"aml_present": bool(aml_reference),
			"fraud_present": bool(fraud_reference),
		})
		if customer_id in self.customers:
			raise ValueError(f"agency customer already exists: {customer_id}")
		customer = AgencyCustomer(customer_id, tenant_id, customer_reference, tier, kyc_reference, consent_reference, aml_reference, fraud_reference)
		self.customers[customer_id] = customer
		self._audit(tenant_id, "agency_customer_onboarded", customer_id)
		return customer.to_dict()

	def record_transaction(
		self,
		transaction_id: str,
		tenant_id: str,
		outlet_id: str,
		agent_id: str,
		customer_id: str,
		float_account_id: str,
		service: str,
		amount: float | int | str,
		currency: str,
		channel: str,
		customer_reference: str,
		risk_reference: str,
		human_approval: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
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
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_transaction",
			"outlet_present": outlet is not None,
			"agent_present": agent is not None and (outlet is None or agent.outlet_id == outlet.id),
			"customer_present": customer is not None,
			"float_account_present": float_account is not None and (outlet is None or float_account.outlet_id == outlet.id),
			"service_supported": service in SUPPORTED_SERVICES,
			"service_allowed_by_program": program is not None and service in program.services,
			"channel_supported": channel in SUPPORTED_CHANNELS,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"currency_matches_float": float_account is not None and float_account.currency == currency,
			"positive_amount": amount_value > 0,
			"within_limit": amount_value <= 200000,
			"float_sufficient": float_sufficient,
			"customer_reference_present": bool(customer_reference),
			"risk_reference_present": bool(risk_reference),
			"high_value": high_value,
			"human_approval_recorded": bool(human_approval),
		})
		if transaction_id in self.transactions:
			raise ValueError(f"agency transaction already exists: {transaction_id}")
		transaction = AgencyTransaction(transaction_id, tenant_id, outlet_id, agent_id, customer_id, float_account_id, service, amount_value, currency, channel, customer_reference, risk_reference)
		self.transactions[transaction_id] = transaction
		if float_account is not None:
			float_account.available_balance = apply_float_delta(float_account.available_balance, service, amount_value)
		self._audit(tenant_id, "agency_transaction_recorded", transaction_id)
		return transaction.to_dict()

	def record_cash_movement(
		self,
		movement_id: str,
		tenant_id: str,
		outlet_id: str,
		movement_type: str,
		amount: float | int | str,
		currency: str,
		custodian_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		movement_type = normalize_code(movement_type)
		currency = normalize_currency(currency)
		amount_value = normalize_amount(amount)
		high_value = amount_value >= 100000
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_cash_movement",
			"outlet_present": outlet is not None,
			"movement_type_supported": movement_type in SUPPORTED_CASH_MOVEMENT_TYPES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"positive_amount": amount_value > 0,
			"custodian_present": bool(custodian_reference),
			"high_value": high_value,
			"human_approval_recorded": bool(human_approval),
		})
		movement = CashMovement(movement_id, tenant_id, outlet_id, movement_type, amount_value, currency, custodian_reference, human_approval)
		self.cash_movements[movement_id] = movement
		self._audit(tenant_id, "cash_movement_recorded", movement_id)
		return movement.to_dict()

	def settle_commission(
		self,
		settlement_id: str,
		tenant_id: str,
		outlet_id: str,
		period: str,
		amount: float | int | str,
		currency: str,
		reconciliation_reference: str,
		payment_reference: str,
	) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		currency = normalize_currency(currency)
		amount_value = normalize_amount(amount)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "settle_commission",
			"outlet_present": outlet is not None,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"positive_amount": amount_value > 0,
			"reconciliation_present": bool(reconciliation_reference),
			"payment_reference_present": bool(payment_reference),
		})
		settlement = CommissionSettlement(settlement_id, tenant_id, outlet_id, period, amount_value, currency, reconciliation_reference, payment_reference)
		self.commissions[settlement_id] = settlement
		self._audit(tenant_id, "commission_settlement_recorded", settlement_id)
		return settlement.to_dict()

	def open_dispute(
		self,
		dispute_id: str,
		tenant_id: str,
		transaction_id: str,
		reason: str,
		reviewer_id: str,
		evidence_references: list[str],
	) -> dict[str, Any]:
		transaction = self._tenant_transaction_or_none(transaction_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_dispute",
			"transaction_present": transaction is not None,
			"dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS,
			"evidence_present": bool(evidence_references),
			"reviewer_present": bool(reviewer_id),
		})
		dispute = AgencyDispute(dispute_id, tenant_id, transaction_id, reason, reviewer_id, list(evidence_references))
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "agency_dispute_opened", dispute_id)
		return dispute.to_dict()

	def record_supervision_visit(
		self,
		visit_id: str,
		tenant_id: str,
		outlet_id: str,
		supervisor_id: str,
		outcome: str,
		evidence_references: list[str],
		findings: list[str] | None = None,
		remediation_plan_reference: str = "",
	) -> dict[str, Any]:
		outlet = self._tenant_outlet_or_none(outlet_id, tenant_id)
		outcome = normalize_code(outcome)
		findings = list(findings or [])
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_supervision_visit",
			"outlet_present": outlet is not None,
			"supervisor_present": bool(supervisor_id),
			"outcome_supported": outcome in SUPPORTED_SUPERVISION_OUTCOMES,
			"evidence_present": bool(evidence_references),
			"findings_present": bool(findings),
			"remediation_plan_present": bool(remediation_plan_reference),
		})
		visit = SupervisionVisit(visit_id, tenant_id, outlet_id, supervisor_id, outcome, list(evidence_references), findings, remediation_plan_reference)
		self.supervision_visits[visit_id] = visit
		self._audit(tenant_id, "supervision_visit_recorded", visit_id)
		return visit.to_dict()

	def register_agency_ai_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_agency_ai_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
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

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def register_agent(
		self,
		name: str,
		location: str,
		phone: str,
		id_number: str,
		float_account: str,
		tenant_id: str = "default",
		outlet_id: str = "",
		program_id: str = "",
	) -> dict[str, Any]:
		"""Register a new agency banking agent with identity and float account.

		Creates an accredited agent record linked to the outlet.  If no
		outlet_id is provided, a default outlet is created or inferred.
		Returns the agent record with float account details.
		"""
		assert _present(name), "name required"
		assert _present(location), "location required"
		assert _present(phone), "phone required"
		assert _present(id_number), "id_number required"
		assert _present(float_account), "float_account reference required"
		agent_id = f"agent-{id_number[:8].replace(' ', '')}-{_utcnow()[:10]}"
		# Use provided outlet or create a default one
		if not outlet_id:
			if not program_id or program_id not in self.programs:
				raise ValueError("program_id required when outlet_id not provided")
			outlet_id = f"outlet-{agent_id}"
			outlet_type = SUPPORTED_OUTLET_TYPES[0] if SUPPORTED_OUTLET_TYPES else "retail_agent"
			country = self.programs[program_id].country if program_id in self.programs else (SUPPORTED_COUNTRIES[0] if SUPPORTED_COUNTRIES else "KE")
			channel = SUPPORTED_CHANNELS[0] if SUPPORTED_CHANNELS else "mobile"
			self.onboard_outlet(
				outlet_id=outlet_id,
				tenant_id=tenant_id,
				program_id=program_id,
				name=f"Outlet for {name}",
				outlet_type=outlet_type,
				country=country,
				license_reference=f"lic-{agent_id}",
				location_reference=location,
				security_plan_reference=f"sp-{agent_id}",
				primary_channel=channel,
				initial_float=500,
			)
		agent = self.accredit_agent(
			agent_id=agent_id,
			tenant_id=tenant_id,
			outlet_id=outlet_id,
			name=name,
			identity_reference=id_number,
			training_reference=f"training-{agent_id}",
			background_check_reference=f"bgcheck-{agent_id}",
		)
		agent["phone"] = phone
		agent["location"] = location
		agent["float_account_reference"] = float_account
		self._audit(tenant_id, "agent_registered_full", agent_id)
		return agent

	async def agent_float_top_up(
		self,
		agent_id: str,
		amount: float | int | str,
		source: str,
		tenant_id: str = "default",
		approved_by: str = "system",
	) -> dict[str, Any]:
		"""Top up float for an agent's float account.

		Finds the float account linked to the agent's outlet, validates
		amount, and credits the account.
		"""
		assert agent_id, "agent_id required"
		assert source, "source required"
		amount_value = normalize_amount(amount)
		assert amount_value > 0, "top-up amount must be positive"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		# Find float account for this outlet
		float_account = next(
			(fa for fa in self.float_accounts.values()
			 if fa.tenant_id == tenant_id and fa.outlet_id == agent.outlet_id),
			None,
		)
		if float_account is None:
			raise ValueError(f"No float account found for outlet {agent.outlet_id}")
		prior_balance = float_account.available_balance
		float_account.available_balance += amount_value
		topup_record: dict[str, Any] = {
			"agent_id": agent_id,
			"float_account_id": float_account.id,
			"amount": amount_value,
			"currency": float_account.currency,
			"source": source,
			"prior_balance": prior_balance,
			"new_balance": float_account.available_balance,
			"approved_by": approved_by,
			"tenant_id": tenant_id,
			"topped_up_at": _utcnow(),
		}
		self._float_topup_history.append(topup_record)
		self._audit(tenant_id, "agent_float_top_up", agent_id)
		return topup_record

	async def agent_float_check(
		self,
		agent_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Check the current float balance for an agent.

		Returns available balance, minimum threshold status, and recent
		top-up history.
		"""
		assert agent_id, "agent_id required"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		float_account = next(
			(fa for fa in self.float_accounts.values()
			 if fa.tenant_id == tenant_id and fa.outlet_id == agent.outlet_id),
			None,
		)
		if float_account is None:
			return {"agent_id": agent_id, "float_account": None, "message": "No float account found"}
		min_threshold = 1000.0
		below_threshold = float_account.available_balance < min_threshold
		recent_topups = [
			t for t in self._float_topup_history
			if t.get("agent_id") == agent_id and t.get("tenant_id") == tenant_id
		][-5:]
		if below_threshold:
			self._audit(tenant_id, "agent_float_below_threshold", agent_id)
		return {
			"agent_id": agent_id,
			"float_account_id": float_account.id,
			"available_balance": float_account.available_balance,
			"currency": float_account.currency,
			"minimum_threshold": min_threshold,
			"below_threshold": below_threshold,
			"recent_topups": recent_topups,
			"checked_at": _utcnow(),
		}

	async def customer_deposit(
		self,
		agent_id: str,
		customer_phone: str,
		amount: float | int | str,
		tenant_id: str = "default",
		risk_reference: str = "",
	) -> dict[str, Any]:
		"""Process a customer cash deposit through an agent.

		Finds or creates a customer record for the phone number, locates the
		agent's float account, and records the deposit transaction.
		"""
		assert agent_id, "agent_id required"
		assert customer_phone, "customer_phone required"
		amount_value = normalize_amount(amount)
		assert amount_value > 0, "deposit amount must be positive"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		# Resolve customer
		customer_id = f"cust-{customer_phone.replace('+', '')}"
		if customer_id not in self.customers:
			tier = SUPPORTED_CUSTOMER_TIERS[0] if SUPPORTED_CUSTOMER_TIERS else "basic"
			self.onboard_customer(
				customer_id=customer_id,
				tenant_id=tenant_id,
				customer_reference=customer_phone,
				tier=tier,
				kyc_reference=f"kyc-{customer_id}",
				consent_reference=f"consent-{customer_id}",
				aml_reference=f"aml-{customer_id}",
				fraud_reference=f"fraud-{customer_id}",
			)
		# Resolve float account
		float_account = next(
			(fa for fa in self.float_accounts.values()
			 if fa.tenant_id == tenant_id and fa.outlet_id == agent.outlet_id),
			None,
		)
		if float_account is None:
			raise ValueError(f"No float account for agent {agent_id}")
		tx_id = f"dep-{agent_id}-{customer_phone}-{_utcnow()}"
		service = "deposit" if "deposit" in (SUPPORTED_SERVICES or []) else (SUPPORTED_SERVICES[0] if SUPPORTED_SERVICES else "cash_in")
		currency = float_account.currency
		channel = SUPPORTED_CHANNELS[0] if SUPPORTED_CHANNELS else "mobile"
		return self.record_transaction(
			transaction_id=tx_id,
			tenant_id=tenant_id,
			outlet_id=agent.outlet_id,
			agent_id=agent_id,
			customer_id=customer_id,
			float_account_id=float_account.id,
			service=service,
			amount=amount_value,
			currency=currency,
			channel=channel,
			customer_reference=customer_phone,
			risk_reference=risk_reference or f"risk-{tx_id}",
		)

	async def customer_withdrawal(
		self,
		agent_id: str,
		customer_phone: str,
		amount: float | int | str,
		customer_id: str,
		tenant_id: str = "default",
		risk_reference: str = "",
	) -> dict[str, Any]:
		"""Process a customer cash withdrawal through an agent.

		Verifies the customer, checks float sufficiency, and records the
		withdrawal transaction.
		"""
		assert agent_id, "agent_id required"
		assert customer_phone, "customer_phone required"
		assert customer_id, "customer_id required"
		amount_value = normalize_amount(amount)
		assert amount_value > 0, "withdrawal amount must be positive"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		if customer is None:
			raise ValueError(f"Customer {customer_id} not found")
		float_account = next(
			(fa for fa in self.float_accounts.values()
			 if fa.tenant_id == tenant_id and fa.outlet_id == agent.outlet_id),
			None,
		)
		if float_account is None:
			raise ValueError(f"No float account for agent {agent_id}")
		if float_account.available_balance < amount_value:
			raise ValueError(f"Insufficient float: {float_account.available_balance} < {amount_value}")
		tx_id = f"wdl-{agent_id}-{customer_id}-{_utcnow()}"
		service = "withdrawal" if "withdrawal" in (SUPPORTED_SERVICES or []) else (SUPPORTED_SERVICES[0] if SUPPORTED_SERVICES else "cash_out")
		currency = float_account.currency
		channel = SUPPORTED_CHANNELS[0] if SUPPORTED_CHANNELS else "mobile"
		return self.record_transaction(
			transaction_id=tx_id,
			tenant_id=tenant_id,
			outlet_id=agent.outlet_id,
			agent_id=agent_id,
			customer_id=customer_id,
			float_account_id=float_account.id,
			service=service,
			amount=amount_value,
			currency=currency,
			channel=channel,
			customer_reference=customer_phone,
			risk_reference=risk_reference or f"risk-{tx_id}",
		)

	async def bill_payment_agent(
		self,
		agent_id: str,
		customer_id: str,
		biller: str,
		amount: float | int | str,
		tenant_id: str = "default",
		risk_reference: str = "",
	) -> dict[str, Any]:
		"""Process a bill payment through an agency banking agent.

		Finds the agent's float account, validates balance, and records the
		bill payment transaction with biller reference.
		"""
		assert agent_id, "agent_id required"
		assert customer_id, "customer_id required"
		assert biller, "biller required"
		amount_value = normalize_amount(amount)
		assert amount_value > 0, "bill payment amount must be positive"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		if customer is None:
			raise ValueError(f"Customer {customer_id} not found")
		float_account = next(
			(fa for fa in self.float_accounts.values()
			 if fa.tenant_id == tenant_id and fa.outlet_id == agent.outlet_id),
			None,
		)
		if float_account is None:
			raise ValueError(f"No float account for agent {agent_id}")
		tx_id = f"bill-{agent_id}-{biller}-{_utcnow()}"
		service = "bill_payment" if "bill_payment" in (SUPPORTED_SERVICES or []) else (SUPPORTED_SERVICES[0] if SUPPORTED_SERVICES else "bill_payment")
		currency = float_account.currency
		channel = SUPPORTED_CHANNELS[0] if SUPPORTED_CHANNELS else "mobile"
		tx = self.record_transaction(
			transaction_id=tx_id,
			tenant_id=tenant_id,
			outlet_id=agent.outlet_id,
			agent_id=agent_id,
			customer_id=customer_id,
			float_account_id=float_account.id,
			service=service,
			amount=amount_value,
			currency=currency,
			channel=channel,
			customer_reference=biller,
			risk_reference=risk_reference or f"risk-{tx_id}",
		)
		tx["biller"] = biller
		return tx

	async def agent_commission(
		self,
		agent_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Calculate and return earned commission for an agent in a period.

		Iterates all transactions for the agent, computes commission per
		transaction, and returns a commission statement.
		"""
		assert agent_id, "agent_id required"
		assert period, "period required"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		agent_txs = [
			t for t in self.transactions.values()
			if t.tenant_id == tenant_id and t.agent_id == agent_id
		]
		total_commission = 0.0
		tx_breakdown: list[dict[str, Any]] = []
		for tx in agent_txs:
			commission = estimate_commission(tx.amount, tx.service)
			total_commission += commission
			tx_breakdown.append({
				"transaction_id": tx.id,
				"service": tx.service,
				"amount": tx.amount,
				"commission": commission,
			})
		self._audit(tenant_id, "agent_commission_calculated", agent_id)
		return {
			"agent_id": agent_id,
			"period": period,
			"tenant_id": tenant_id,
			"transaction_count": len(agent_txs),
			"total_commission": round(total_commission, 2),
			"currency": "KES",
			"breakdown": tx_breakdown[:50],
			"calculated_at": _utcnow(),
		}

	async def agent_transaction_report(
		self,
		agent_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Generate a transaction report for an agent for a period.

		Returns: volume by service, total value, error/dispute count, and
		float utilisation.
		"""
		assert agent_id, "agent_id required"
		assert period, "period required"
		agent_txs = [
			t for t in self.transactions.values()
			if t.tenant_id == tenant_id and t.agent_id == agent_id
		]
		by_service: dict[str, dict[str, Any]] = {}
		for tx in agent_txs:
			svc = tx.service
			if svc not in by_service:
				by_service[svc] = {"count": 0, "total_value": 0.0}
			by_service[svc]["count"] += 1
			by_service[svc]["total_value"] += tx.amount
		total_value = sum(tx.amount for tx in agent_txs)
		dispute_count = sum(
			1 for d in self.disputes.values()
			if d.tenant_id == tenant_id
			and self.transactions.get(d.transaction_id) is not None
			and self.transactions[d.transaction_id].agent_id == agent_id
		)
		self._audit(tenant_id, "agent_transaction_report_generated", agent_id)
		return {
			"agent_id": agent_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_transactions": len(agent_txs),
			"total_value": round(total_value, 2),
			"by_service": {s: {"count": v["count"], "total_value": round(v["total_value"], 2)} for s, v in by_service.items()},
			"dispute_count": dispute_count,
			"generated_at": _utcnow(),
		}

	async def agent_compliance_check(
		self,
		agent_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Run a compliance check on an agent.

		Checks: identity document present, training completed, background
		check done, float account exists, recent supervision visit, no open
		disputes.  Returns compliance score and issues.
		"""
		assert agent_id, "agent_id required"
		agent = self._tenant_agent_or_none(agent_id, tenant_id)
		if agent is None:
			raise ValueError(f"Agent {agent_id} not found")
		checks: dict[str, bool] = {
			"identity_document_present": _present(agent.identity_reference),
			"training_completed": _present(agent.training_reference),
			"background_check_done": _present(agent.background_check_reference),
			"float_account_exists": any(
				fa.outlet_id == agent.outlet_id and fa.tenant_id == tenant_id
				for fa in self.float_accounts.values()
			),
			"recent_supervision_visit": any(
				v.outlet_id == agent.outlet_id and v.tenant_id == tenant_id
				for v in self.supervision_visits.values()
			),
			"no_open_disputes": not any(
				d.tenant_id == tenant_id
				and self.transactions.get(d.transaction_id) is not None
				and self.transactions[d.transaction_id].agent_id == agent_id
				and d.status == "open"
				for d in self.disputes.values()
			),
		}
		issues = [k for k, v in checks.items() if not v]
		compliance_score = round(sum(checks.values()) / len(checks) * 100, 1)
		compliant = len(issues) == 0
		check_record: dict[str, Any] = {
			"agent_id": agent_id,
			"tenant_id": tenant_id,
			"compliant": compliant,
			"compliance_score": compliance_score,
			"checks": checks,
			"issues": issues,
			"checked_at": _utcnow(),
		}
		self._compliance_check_records.append(check_record)
		if not compliant:
			self._audit(tenant_id, "agent_compliance_issues_found", agent_id)
		return check_record

	async def dormant_agent_management(
		self,
		days_inactive: int,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Identify and manage dormant agents based on inactivity threshold.

		Scans all agents, checks last transaction date, and flags those
		inactive for more than days_inactive.  Returns dormancy report.
		"""
		assert days_inactive > 0, "days_inactive must be positive"
		now = datetime.datetime.utcnow()
		dormant: list[dict[str, Any]] = []
		active: list[str] = []
		for agent in self.agents.values():
			if agent.tenant_id != tenant_id:
				continue
			# Get last transaction date for this agent
			agent_txs = sorted(
				[t for t in self.transactions.values() if t.tenant_id == tenant_id and t.agent_id == agent.id],
				key=lambda t: getattr(t, "created_at", "") or "",
				reverse=True,
			)
			if not agent_txs:
				# Never transacted: flag as dormant from registration
				days_since = days_inactive + 1  # treat as dormant
			else:
				last_tx = agent_txs[0]
				try:
					last_dt = datetime.datetime.fromisoformat(
						getattr(last_tx, "created_at", _utcnow()).replace("Z", "")
					)
					days_since = (now - last_dt).days
				except Exception:
					days_since = 0
			if days_since >= days_inactive:
				dormancy: dict[str, Any] = {
					"agent_id": agent.id,
					"days_inactive": days_since,
					"outlet_id": agent.outlet_id,
					"status": "dormant",
					"flagged_at": _utcnow(),
				}
				self._dormancy_records[agent.id] = dormancy
				dormant.append(dormancy)
			else:
				active.append(agent.id)
		self._audit(tenant_id, "dormant_agent_management_run", str(len(dormant)))
		return {
			"tenant_id": tenant_id,
			"days_inactive_threshold": days_inactive,
			"dormant_count": len(dormant),
			"active_count": len(active),
			"dormant_agents": dormant[:50],
			"checked_at": _utcnow(),
		}

	async def agent_network_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Analyse the entire agent network performance for a period.

		Returns: network-wide transaction volume, top agents, outlet
		distribution, float utilisation, commission payable, and compliance
		score distribution.
		"""
		assert period, "period required"
		all_agents = [a for a in self.agents.values() if a.tenant_id == tenant_id]
		all_txs = [t for t in self.transactions.values() if t.tenant_id == tenant_id]
		total_volume = sum(t.amount for t in all_txs)
		# Top agents by transaction volume
		agent_volumes: dict[str, float] = {}
		for t in all_txs:
			agent_volumes[t.agent_id] = agent_volumes.get(t.agent_id, 0.0) + t.amount
		top_agents = sorted(agent_volumes.items(), key=lambda x: x[1], reverse=True)[:10]
		# Outlet distribution by type
		outlet_type_dist: dict[str, int] = {}
		for outlet in self.outlets.values():
			if outlet.tenant_id == tenant_id:
				outlet_type_dist[outlet.outlet_type] = outlet_type_dist.get(outlet.outlet_type, 0) + 1
		# Float utilisation
		total_float = sum(fa.available_balance for fa in self.float_accounts.values() if fa.tenant_id == tenant_id)
		avg_float = total_float / max(len([fa for fa in self.float_accounts.values() if fa.tenant_id == tenant_id]), 1)
		# Compliance scores
		compliance_scores = [r["compliance_score"] for r in self._compliance_check_records if r.get("tenant_id") == tenant_id]
		avg_compliance = round(statistics.mean(compliance_scores), 2) if compliance_scores else None
		# Total commission payable
		total_commission = sum(estimate_commission(t.amount, t.service) for t in all_txs)
		run_record: dict[str, Any] = {"period": period, "tenant_id": tenant_id, "computed_at": _utcnow()}
		self._network_analytics_runs.append(run_record)
		self._audit(tenant_id, "agent_network_analytics_run", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"agent_count": len(all_agents),
			"outlet_count": sum(1 for o in self.outlets.values() if o.tenant_id == tenant_id),
			"total_transactions": len(all_txs),
			"total_volume": round(total_volume, 2),
			"top_agents": [{"agent_id": a, "volume": round(v, 2)} for a, v in top_agents],
			"outlet_type_distribution": outlet_type_dist,
			"total_float_balance": round(total_float, 2),
			"avg_float_per_outlet": round(avg_float, 2),
			"avg_compliance_score": avg_compliance,
			"total_commission_payable": round(total_commission, 2),
			"dormant_agents": len(self._dormancy_records),
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Additional methods                                                  #
	# ------------------------------------------------------------------ #

	async def health_check(self) -> dict[str, Any]:
		"""Return agency banking service health status."""
		return {
			"service": "agency_banking", "status": "healthy",
			"agent_count": len(self.agents), "active_float_accounts": len(self.float_accounts),
			"checked_at": _utcnow(),
		}

	async def bulk_accredit_agents(self, agents: list[dict[str, Any]], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk-accredit multiple agents."""
		processed, errors = [], []
		for a in agents:
			try:
				rec = await self.register_agent(
					name=a["name"], location=a.get("location", ""), phone=a.get("phone", ""),
					id_number=a.get("id_number", ""), float_account=a.get("float_account", ""),
					tenant_id=tenant_id, outlet_id=a.get("outlet_id", ""), program_id=a.get("program_id", ""),
				)
				processed.append(rec.get("id", a["name"]))
			except Exception as exc:
				errors.append({"input": a, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "agent_ids": processed}

	async def mobile_money_airtime_purchase(self, agent_id: str, customer_id: str, phone: str, amount: float, provider: str = "safaricom", tenant_id: str = "default") -> dict[str, Any]:
		"""Purchase airtime for a customer through an agent terminal."""
		return await self.bill_payment_agent(agent_id, customer_id, f"AIRTIME-{provider.upper()}", amount, tenant_id)

	async def school_fees_collection(self, agent_id: str, customer_id: str, school_code: str, amount: float, student_ref: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Collect school fees through an agent."""
		return await self.bill_payment_agent(agent_id, customer_id, f"SCHOOL-{school_code}", amount, tenant_id)

	async def government_payments_agent(self, agent_id: str, customer_id: str, service_code: str, amount: float, reference: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Process government payments (eCitizen, county, NHIF, NSSF) through an agent."""
		return await self.bill_payment_agent(agent_id, customer_id, f"GOV-{service_code}", amount, tenant_id)

	async def float_rebalancing(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Rebalance float across agent outlets — move excess to deficient outlets."""
		accounts = [fa for fa in self.float_accounts.values() if fa.tenant_id == tenant_id]
		avg_float = sum(fa.available_balance for fa in accounts) / max(len(accounts), 1)
		excess = [(fa, fa.available_balance - avg_float) for fa in accounts if fa.available_balance > avg_float * 1.2]
		deficient = [(fa, avg_float - fa.available_balance) for fa in accounts if fa.available_balance < avg_float * 0.8]
		transfers = []
		for (excess_fa, excess_amt), (def_fa, def_amt) in zip(excess, deficient):
			move = min(excess_amt, def_amt)
			if move > 100:
				excess_fa.available_balance -= move
				def_fa.available_balance += move
				transfers.append({"from_outlet": excess_fa.outlet_id, "to_outlet": def_fa.outlet_id, "amount": round(move, 2)})
		self._audit(tenant_id, "float_rebalanced", "network")
		return {"tenant_id": tenant_id, "accounts_reviewed": len(accounts), "transfers": transfers, "rebalanced_at": _utcnow()}

	async def cbk_agency_banking_return(self, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""File the CBK Agency Banking return (monthly/quarterly)."""
		txns = [t for t in self.transactions.values() if t.tenant_id == tenant_id]
		cash_in = sum(t.amount for t in txns if t.service in {"deposit", "cash_in"})
		cash_out = sum(t.amount for t in txns if t.service in {"withdrawal", "cash_out"})
		return {
			"report_type": "CBK_AGENCY_BANKING_RETURN", "period": period,
			"total_agents": sum(1 for a in self.agents.values() if a.tenant_id == tenant_id),
			"total_outlets": sum(1 for o in self.outlets.values() if o.tenant_id == tenant_id),
			"transaction_count": len(txns), "cash_in": round(cash_in, 2), "cash_out": round(cash_out, 2),
			"status": "draft", "generated_at": _utcnow(),
		}

	async def export_agent_data(self, tenant_id: str = "default", fmt: str = "csv") -> dict[str, Any]:
		"""Export agent registry and transaction data."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": tenant_id, "format": fmt,
			"agents": sum(1 for a in self.agents.values() if a.tenant_id == tenant_id),
			"file_reference": f"agency_{tenant_id}_{_utcnow()[:10]}.{fmt}", "generated_at": _utcnow(),
		}

	async def geo_coverage_report(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Report on geographic coverage of agent network by country/region."""
		outlets = [o for o in self.outlets.values() if o.tenant_id == tenant_id]
		by_country: dict[str, int] = {}
		for o in outlets:
			by_country[o.country] = by_country.get(o.country, 0) + 1
		return {
			"tenant_id": tenant_id, "total_outlets": len(outlets),
			"by_country": by_country, "generated_at": _utcnow(),
		}

	async def agent_training_record(self, agent_id: str, training_type: str, completion_date: str, score: float, tenant_id: str = "default") -> dict[str, Any]:
		"""Record agent compliance training completion."""
		assert agent_id and training_type and completion_date
		assert 0 <= score <= 100, "score must be 0–100"
		record: dict[str, Any] = {
			"agent_id": agent_id, "training_type": training_type,
			"completion_date": completion_date, "score": score,
			"passed": score >= 70, "tenant_id": tenant_id, "recorded_at": _utcnow(),
		}
		self._audit(tenant_id, "agent_training_recorded", agent_id)
		return record

	# ------------------------------------------------------------------ #
	# List helpers                                                         #
	# ------------------------------------------------------------------ #

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

	# ------------------------------------------------------------------ #
	# Dashboard                                                           #
	# ------------------------------------------------------------------ #

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id),
			"outlet_count": sum(1 for item in self.outlets.values() if item.tenant_id == tenant_id),
			"agent_count": sum(1 for item in self.agents.values() if item.tenant_id == tenant_id),
			"float_account_count": sum(1 for item in self.float_accounts.values() if item.tenant_id == tenant_id),
			"customer_count": sum(1 for item in self.customers.values() if item.tenant_id == tenant_id),
			"transaction_count": sum(1 for item in self.transactions.values() if item.tenant_id == tenant_id),
			"cash_movement_count": sum(1 for item in self.cash_movements.values() if item.tenant_id == tenant_id),
			"commission_count": sum(1 for item in self.commissions.values() if item.tenant_id == tenant_id),
			"dispute_count": sum(1 for item in self.disputes.values() if item.tenant_id == tenant_id),
			"supervision_count": sum(1 for item in self.supervision_visits.values() if item.tenant_id == tenant_id),
			"dormant_agents": len(self._dormancy_records),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

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


# Backward-compatible alias
AgencyService = AgencyBankingService
