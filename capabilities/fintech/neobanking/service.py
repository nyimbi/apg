"""Executable service layer for APG Digital Neobanking."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_ACCOUNT_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASE_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_PAYMENT_RAILS, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import AccountTransaction, BankProgram, CustomerProfile, DepositAccount, NeobankingEvidence, PaymentRailLink, SavingsPot, ServiceCase, StatementRecord
	from .neobanking_runtime import account_number, normalize_amount, normalize_code, normalize_country, normalize_currency, today_iso, transaction_direction
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_ACCOUNT_TYPES, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CASE_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_PAYMENT_RAILS, SUPPORTED_TRANSACTION_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AccountTransaction, BankProgram, CustomerProfile, DepositAccount, NeobankingEvidence, PaymentRailLink, SavingsPot, ServiceCase, StatementRecord  # type: ignore
	from neobanking_runtime import account_number, normalize_amount, normalize_code, normalize_country, normalize_currency, today_iso, transaction_direction  # type: ignore


class NeobankingService:
	"""Dependency-light neobanking runtime for generated applications."""

	def __init__(self) -> None:
		self.programs: dict[str, BankProgram] = {}
		self.customers: dict[str, CustomerProfile] = {}
		self.accounts: dict[str, DepositAccount] = {}
		self.rails: dict[str, PaymentRailLink] = {}
		self.transactions: dict[str, AccountTransaction] = {}
		self.savings_pots: dict[str, SavingsPot] = {}
		self.statements: dict[str, StatementRecord] = {}
		self.cases: dict[str, ServiceCase] = {}
		self.evidence: dict[str, NeobankingEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_program(self, program_id: str, tenant_id: str, name: str, owner_id: str, country: str, base_currency: str, settlement_account: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(base_currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_program", "owner_present": bool(owner_id), "country_supported": country in SUPPORTED_COUNTRIES, "currency_supported": currency in SUPPORTED_CURRENCIES, "settlement_account_present": bool(settlement_account)})
		if program_id in self.programs:
			raise ValueError(f"bank program already exists: {program_id}")
		program = BankProgram(program_id, tenant_id, name, owner_id, country, currency, settlement_account)
		self.programs[program_id] = program
		self._audit(tenant_id, "bank_program_registered", program_id)
		return program.to_dict()

	def onboard_customer(self, customer_id: str, tenant_id: str, customer_reference: str, kyc_profile_id: str, country: str, consent_reference: str, aml_reference: str, fraud_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_customer", "customer_present": bool(customer_reference), "kyc_present": bool(kyc_profile_id), "aml_present": bool(aml_reference), "fraud_present": bool(fraud_reference), "country_supported": country in SUPPORTED_COUNTRIES, "consent_present": bool(consent_reference)})
		if customer_id in self.customers:
			raise ValueError(f"digital customer already exists: {customer_id}")
		customer = CustomerProfile(customer_id, tenant_id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference)
		self.customers[customer_id] = customer
		self._audit(tenant_id, "digital_customer_onboarded", customer_id)
		return customer.to_dict()

	def open_account(self, account_id: str, tenant_id: str, program_id: str, customer_id: str, account_type: str, currency: str, initial_balance: float | int | str = 0, policy_attached: bool = True) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		account_type = normalize_code(account_type)
		currency = normalize_currency(currency)
		balance = normalize_amount(initial_balance)
		country = program.country if program else ""
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "open_account", "program_present": program is not None, "customer_present": customer is not None, "account_type_supported": account_type in SUPPORTED_ACCOUNT_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES, "initial_balance_non_negative": balance >= 0})
		if account_id in self.accounts:
			raise ValueError(f"deposit account already exists: {account_id}")
		account = DepositAccount(account_id, tenant_id, program_id, customer_id, account_type, currency, account_number(account_id, country), balance)
		self.accounts[account_id] = account
		self._audit(tenant_id, "deposit_account_opened", account_id)
		return account.to_dict()

	def link_payment_rail(self, link_id: str, tenant_id: str, account_id: str, rail: str, provider_reference: str, wallet_reference: str = "", card_reference: str = "") -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		rail = normalize_code(rail)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "link_payment_rail", "account_present": account is not None, "rail_supported": rail in SUPPORTED_PAYMENT_RAILS, "provider_reference_present": bool(provider_reference)})
		link = PaymentRailLink(link_id, tenant_id, account_id, rail, provider_reference, wallet_reference, card_reference)
		self.rails[link_id] = link
		self._audit(tenant_id, "payment_rail_linked", link_id)
		return link.to_dict()

	def post_transaction(self, transaction_id: str, tenant_id: str, account_id: str, kind: str, amount: float | int | str, currency: str, reference: str, risk_reference: str, human_approval: str = "") -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		kind = normalize_code(kind)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		direction = transaction_direction(kind)
		high_impact = amount_value >= 100000 or kind in {"withdrawal", "transfer_out"} and amount_value >= 50000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "post_transaction", "account_present": account is not None, "transaction_type_supported": kind in SUPPORTED_TRANSACTION_TYPES, "positive_amount": amount_value > 0, "currency_matches_account": account is not None and currency == account.currency, "risk_reference_present": bool(risk_reference), "high_impact": high_impact, "human_approval_recorded": bool(human_approval)})
		if account is not None:
			account.balance = round(account.balance + amount_value, 2) if direction == "credit" else round(account.balance - amount_value, 2)
		record = AccountTransaction(transaction_id, tenant_id, account_id, kind, amount_value, currency, direction, reference, risk_reference)
		self.transactions[transaction_id] = record
		self._audit(tenant_id, "account_transaction_posted", transaction_id)
		return record.to_dict()

	def create_savings_pot(self, pot_id: str, tenant_id: str, account_id: str, name: str, target_amount: float | int | str) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		target = normalize_amount(target_amount)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "create_savings_pot", "account_present": account is not None, "name_present": bool(name), "positive_target": target > 0})
		pot = SavingsPot(pot_id, tenant_id, account_id, name, target, account.currency if account else "")
		self.savings_pots[pot_id] = pot
		self._audit(tenant_id, "savings_pot_created", pot_id)
		return pot.to_dict()

	def issue_statement(self, statement_id: str, tenant_id: str, account_id: str, period_start: str, period_end: str) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		period_present = bool(period_start and period_end)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "issue_statement", "account_present": account is not None, "period_present": period_present})
		transactions = [item for item in self.transactions.values() if item.tenant_id == tenant_id and item.account_id == account_id]
		statement = StatementRecord(statement_id, tenant_id, account_id, period_start, period_end, len(transactions), account.balance if account else 0)
		self.statements[statement_id] = statement
		self._audit(tenant_id, "statement_issued", statement_id)
		return statement.to_dict()

	def open_service_case(self, case_id: str, tenant_id: str, customer_id: str, account_id: str, reason: str, reviewer_id: str, evidence_references: list[str]) -> dict[str, Any]:
		customer = self._tenant_customer_or_none(customer_id, tenant_id)
		account = self._tenant_account_or_none(account_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_service_case", "customer_present": customer is not None, "account_present": account is not None, "case_reason_supported": reason in SUPPORTED_CASE_REASONS, "reviewer_present": bool(reviewer_id), "evidence_present": bool(evidence_references)})
		case = ServiceCase(case_id, tenant_id, customer_id, account_id, reason, reviewer_id, list(evidence_references))
		self.cases[case_id] = case
		self._audit(tenant_id, "service_case_opened", case_id)
		return case.to_dict()

	def register_neobanking_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_neobanking_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "neobanking_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "neobanking_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.neobanking.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id), "customer_count": sum(1 for item in self.customers.values() if item.tenant_id == tenant_id), "account_count": sum(1 for item in self.accounts.values() if item.tenant_id == tenant_id), "rail_count": sum(1 for item in self.rails.values() if item.tenant_id == tenant_id), "transaction_count": sum(1 for item in self.transactions.values() if item.tenant_id == tenant_id), "savings_pot_count": sum(1 for item in self.savings_pots.values() if item.tenant_id == tenant_id), "statement_count": sum(1 for item in self.statements.values() if item.tenant_id == tenant_id), "case_count": sum(1 for item in self.cases.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "as_of": today_iso(), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_accounts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.accounts.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_transactions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.transactions.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _tenant_program_or_none(self, program_id: str, tenant_id: str) -> BankProgram | None:
		item = self.programs.get(program_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_customer_or_none(self, customer_id: str, tenant_id: str) -> CustomerProfile | None:
		item = self.customers.get(customer_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_account_or_none(self, account_id: str, tenant_id: str) -> DepositAccount | None:
		item = self.accounts.get(account_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = NeobankingEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "neobanking_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "neobanking_policy_denied")


DigitalNeobankingService = NeobankingService
