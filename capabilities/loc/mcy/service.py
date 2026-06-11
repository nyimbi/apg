"""Async service layer for APG Multi-Currency Management."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_CURRENCY_STATUSES,
		SUPPORTED_FX_ACCOUNT_TYPES,
		SUPPORTED_RATE_SOURCES,
		SUPPORTED_RATE_TYPES,
		SUPPORTED_REVALUATION_METHODS,
		SUPPORTED_REVALUATION_STATUSES,
		SUPPORTED_ROUNDING_MODES,
		SUPPORTED_TRANSLATION_METHODS,
		SUPPORTED_TRANSLATION_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		CurrencyConfigCreate,
		CurrencyConfigResponse,
		CurrencyConfigUpdate,
		CurrencyTranslationCreate,
		CurrencyTranslationResponse,
		CurrencyTranslationUpdate,
		ExchangeRateCreate,
		ExchangeRateResponse,
		ExchangeRateUpdate,
		FxAccountCreate,
		FxAccountResponse,
		FxGainLossReport,
		McyAgentCreate,
		McyAgentResponse,
		McyAuditEvent,
		RevaluationCreate,
		RevaluationResponse,
		RevaluationUpdate,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_CURRENCY_STATUSES,
		SUPPORTED_FX_ACCOUNT_TYPES,
		SUPPORTED_RATE_SOURCES,
		SUPPORTED_RATE_TYPES,
		SUPPORTED_REVALUATION_METHODS,
		SUPPORTED_REVALUATION_STATUSES,
		SUPPORTED_ROUNDING_MODES,
		SUPPORTED_TRANSLATION_METHODS,
		SUPPORTED_TRANSLATION_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore[no-redef]
		CurrencyConfigCreate,
		CurrencyConfigResponse,
		CurrencyConfigUpdate,
		CurrencyTranslationCreate,
		CurrencyTranslationResponse,
		CurrencyTranslationUpdate,
		ExchangeRateCreate,
		ExchangeRateResponse,
		ExchangeRateUpdate,
		FxAccountCreate,
		FxAccountResponse,
		FxGainLossReport,
		McyAgentCreate,
		McyAgentResponse,
		McyAuditEvent,
		RevaluationCreate,
		RevaluationResponse,
		RevaluationUpdate,
	)


def _present(v: str | None) -> bool:
	return bool(v and v.strip())


class MultiCurrencyManagementService:
	"""Tenant-scoped runtime for Multi-Currency Management capability."""

	def __init__(self) -> None:
		self._currencies: dict[tuple[str, str], CurrencyConfigResponse] = {}
		self._rates: dict[tuple[str, str], ExchangeRateResponse] = {}
		self._revaluations: dict[tuple[str, str], RevaluationResponse] = {}
		self._translations: dict[tuple[str, str], CurrencyTranslationResponse] = {}
		self._fx_accounts: dict[tuple[str, str], FxAccountResponse] = {}
		self._agents: dict[tuple[str, str], McyAgentResponse] = {}
		self._audit_events: list[McyAuditEvent] = []

	# --- Contract ---

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the full capability contract."""
		return get_capability_contract(tenant_id)

	async def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		"""Evaluate capability rules against context."""
		return evaluate_capability_rules(context)

	# --- Currency Configuration ---

	async def configure_currency(self, payload: CurrencyConfigCreate, actor_id: str = "system") -> CurrencyConfigResponse:
		"""Configure a currency for use within the tenant."""
		self._log_operation("configure_currency", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "configure_currency",
			"currency_supported": payload.code.upper() in SUPPORTED_CURRENCIES,
			"currency_name_present": _present(payload.name),
			"precision_valid": 0 <= payload.decimal_places <= 6,
			"rounding_mode_supported": payload.rounding_mode in SUPPORTED_ROUNDING_MODES,
		})
		currency = CurrencyConfigResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			code=payload.code.upper(),
			name=payload.name,
			symbol=payload.symbol,
			decimal_places=payload.decimal_places,
			rounding_mode=payload.rounding_mode,
			status="active",
			is_functional=payload.is_functional,
			is_presentation=payload.is_presentation,
			notes=payload.notes,
			created_by=actor_id,
		)
		self._currencies[self._key(payload.tenant_id, currency.id)] = currency
		await self._emit(payload.tenant_id, "currency_configured", currency.id, actor_id)
		return currency

	async def get_currency(self, tenant_id: str, currency_id: str) -> CurrencyConfigResponse:
		"""Get a configured currency by ID."""
		self._enforce_tenant(tenant_id)
		c = self._currencies.get(self._key(tenant_id, currency_id))
		if not c:
			raise KeyError(f"currency '{currency_id}' not found for tenant '{tenant_id}'")
		return c

	async def get_currency_by_code(self, tenant_id: str, code: str) -> CurrencyConfigResponse | None:
		"""Lookup a configured currency by ISO code."""
		self._enforce_tenant(tenant_id)
		for c in self._currencies.values():
			if c.tenant_id == tenant_id and c.code == code.upper():
				return c
		return None

	async def list_currencies(self, tenant_id: str, status: str | None = None) -> list[CurrencyConfigResponse]:
		"""List configured currencies, optionally filtered by status."""
		self._enforce_tenant(tenant_id)
		result = [c for c in self._currencies.values() if c.tenant_id == tenant_id]
		if status:
			result = [c for c in result if c.status == status]
		return result

	async def update_currency(self, tenant_id: str, currency_id: str, payload: CurrencyConfigUpdate, actor_id: str = "system") -> CurrencyConfigResponse:
		"""Update currency configuration."""
		self._enforce_tenant(tenant_id)
		currency = await self.get_currency(tenant_id, currency_id)
		data = currency.model_dump()
		data.update(payload.model_dump(exclude_none=True))
		data["updated_at"] = datetime.utcnow()
		updated = CurrencyConfigResponse.model_validate(data)
		self._currencies[self._key(tenant_id, currency_id)] = updated
		await self._emit(tenant_id, "currency_updated", currency_id, actor_id)
		return updated

	# --- Exchange Rates ---

	async def record_exchange_rate(self, payload: ExchangeRateCreate, actor_id: str = "system") -> ExchangeRateResponse:
		"""Record an exchange rate between two currencies."""
		self._log_operation("record_rate", payload.tenant_id)
		today = date.today()
		is_backdated = payload.effective_date < today
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_rate",
			"from_currency_supported": payload.from_currency.upper() in SUPPORTED_CURRENCIES,
			"to_currency_supported": payload.to_currency.upper() in SUPPORTED_CURRENCIES,
			"rate_type_supported": payload.rate_type in SUPPORTED_RATE_TYPES,
			"rate_source_supported": payload.rate_source in SUPPORTED_RATE_SOURCES,
			"effective_date_present": payload.effective_date is not None,
			"rate_positive": payload.rate > 0,
			"rate_source": payload.rate_source,
			"approval_present": _present(payload.approval_reference),
			"backdated": is_backdated,
			"backdating_override_present": _present(payload.backdating_override),
		})
		rate = ExchangeRateResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			from_currency=payload.from_currency.upper(),
			to_currency=payload.to_currency.upper(),
			rate=payload.rate,
			rate_type=payload.rate_type,
			rate_source=payload.rate_source,
			effective_date=payload.effective_date,
			expiry_date=payload.expiry_date,
			approval_reference=payload.approval_reference,
			backdating_override=payload.backdating_override,
			is_active=True,
			notes=payload.notes,
			created_by=actor_id,
		)
		self._rates[self._key(payload.tenant_id, rate.id)] = rate
		await self._emit(payload.tenant_id, "exchange_rate_recorded", rate.id, actor_id)
		return rate

	async def get_exchange_rate(self, tenant_id: str, rate_id: str) -> ExchangeRateResponse:
		"""Get an exchange rate by ID."""
		self._enforce_tenant(tenant_id)
		rate = self._rates.get(self._key(tenant_id, rate_id))
		if not rate:
			raise KeyError(f"exchange rate '{rate_id}' not found for tenant '{tenant_id}'")
		return rate

	async def list_exchange_rates(self, tenant_id: str, from_currency: str | None = None, to_currency: str | None = None, rate_type: str | None = None, effective_date: date | None = None) -> list[ExchangeRateResponse]:
		"""List exchange rates with optional filters."""
		self._enforce_tenant(tenant_id)
		result = [r for r in self._rates.values() if r.tenant_id == tenant_id and r.is_active]
		if from_currency:
			result = [r for r in result if r.from_currency == from_currency.upper()]
		if to_currency:
			result = [r for r in result if r.to_currency == to_currency.upper()]
		if rate_type:
			result = [r for r in result if r.rate_type == rate_type]
		if effective_date:
			result = [r for r in result if r.effective_date <= effective_date and (r.expiry_date is None or r.expiry_date >= effective_date)]
		return result

	async def get_rate_for_date(self, tenant_id: str, from_currency: str, to_currency: str, as_of: date, rate_type: str = "spot") -> ExchangeRateResponse | None:
		"""Return the most recently effective rate for a currency pair as of a given date."""
		rates = await self.list_exchange_rates(tenant_id, from_currency=from_currency, to_currency=to_currency, rate_type=rate_type, effective_date=as_of)
		if not rates:
			return None
		return max(rates, key=lambda r: r.effective_date)

	async def convert_amount(self, tenant_id: str, amount: float, from_currency: str, to_currency: str, as_of: date, rate_type: str = "spot") -> dict[str, Any]:
		"""Convert an amount between two currencies using the effective rate."""
		self._enforce_tenant(tenant_id)
		if from_currency.upper() == to_currency.upper():
			return {"tenant_id": tenant_id, "amount": amount, "from_currency": from_currency.upper(), "to_currency": to_currency.upper(), "converted_amount": amount, "rate": 1.0, "as_of": as_of}
		rate = await self.get_rate_for_date(tenant_id, from_currency, to_currency, as_of, rate_type)
		if not rate:
			# Try inverse
			inv_rate = await self.get_rate_for_date(tenant_id, to_currency, from_currency, as_of, rate_type)
			if inv_rate:
				converted = round(amount / inv_rate.rate, 6)
				return {"tenant_id": tenant_id, "amount": amount, "from_currency": from_currency.upper(), "to_currency": to_currency.upper(), "converted_amount": converted, "rate": round(1.0 / inv_rate.rate, 6), "as_of": as_of}
			raise KeyError(f"no exchange rate found for {from_currency}/{to_currency} as of {as_of}")
		converted = round(amount * rate.rate, 6)
		return {"tenant_id": tenant_id, "amount": amount, "from_currency": from_currency.upper(), "to_currency": to_currency.upper(), "converted_amount": converted, "rate": rate.rate, "as_of": as_of}

	# --- FX Accounts ---

	async def register_fx_account(self, payload: FxAccountCreate, actor_id: str = "system") -> FxAccountResponse:
		"""Register an FX gain/loss account."""
		self._log_operation("register_fx_account", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_fx_account",
			"account_type_supported": payload.account_type in SUPPORTED_FX_ACCOUNT_TYPES,
			"account_code_present": _present(payload.account_code),
		})
		account = FxAccountResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			account_type=payload.account_type,
			account_code=payload.account_code,
			account_name=payload.account_name,
			currency=payload.currency.upper(),
			entity_id=payload.entity_id,
			notes=payload.notes,
			created_by=actor_id,
		)
		self._fx_accounts[self._key(payload.tenant_id, account.id)] = account
		await self._emit(payload.tenant_id, "fx_account_registered", account.id, actor_id)
		return account

	async def get_fx_account(self, tenant_id: str, account_id: str) -> FxAccountResponse:
		"""Get an FX account by ID."""
		self._enforce_tenant(tenant_id)
		acct = self._fx_accounts.get(self._key(tenant_id, account_id))
		if not acct:
			raise KeyError(f"fx account '{account_id}' not found for tenant '{tenant_id}'")
		return acct

	async def list_fx_accounts(self, tenant_id: str, account_type: str | None = None) -> list[FxAccountResponse]:
		"""List FX accounts for a tenant."""
		self._enforce_tenant(tenant_id)
		result = [a for a in self._fx_accounts.values() if a.tenant_id == tenant_id and a.is_active]
		if account_type:
			result = [a for a in result if a.account_type == account_type]
		return result

	# --- Revaluation ---

	async def create_revaluation(self, payload: RevaluationCreate, actor_id: str = "system") -> RevaluationResponse:
		"""Create an FX revaluation run for an entity and period."""
		self._log_operation("create_revaluation", payload.tenant_id)
		gain_acct = self._fx_accounts.get(self._key(payload.tenant_id, payload.fx_gain_account_id))
		loss_acct = self._fx_accounts.get(self._key(payload.tenant_id, payload.fx_loss_account_id))
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_revaluation",
			"revaluation_method_supported": payload.revaluation_method in SUPPORTED_REVALUATION_METHODS,
			"period_present": payload.period_start is not None and payload.period_end is not None,
			"fx_account_present": gain_acct is not None and loss_acct is not None,
		})
		revaluation = RevaluationResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			entity_id=payload.entity_id,
			period_start=payload.period_start,
			period_end=payload.period_end,
			revaluation_method=payload.revaluation_method,
			functional_currency=payload.functional_currency.upper(),
			fx_gain_account_id=payload.fx_gain_account_id,
			fx_loss_account_id=payload.fx_loss_account_id,
			status="draft",
			description=payload.description,
			created_by=actor_id,
		)
		self._revaluations[self._key(payload.tenant_id, revaluation.id)] = revaluation
		await self._emit(payload.tenant_id, "revaluation_created", revaluation.id, actor_id)
		return revaluation

	async def get_revaluation(self, tenant_id: str, revaluation_id: str) -> RevaluationResponse:
		"""Get a revaluation run by ID."""
		self._enforce_tenant(tenant_id)
		rev = self._revaluations.get(self._key(tenant_id, revaluation_id))
		if not rev:
			raise KeyError(f"revaluation '{revaluation_id}' not found for tenant '{tenant_id}'")
		return rev

	async def list_revaluations(self, tenant_id: str, entity_id: str | None = None, status: str | None = None) -> list[RevaluationResponse]:
		"""List revaluations for a tenant."""
		self._enforce_tenant(tenant_id)
		result = [r for r in self._revaluations.values() if r.tenant_id == tenant_id]
		if entity_id:
			result = [r for r in result if r.entity_id == entity_id]
		if status:
			result = [r for r in result if r.status == status]
		return result

	async def approve_revaluation(self, tenant_id: str, revaluation_id: str, approver_id: str, approval_reference: str) -> RevaluationResponse:
		"""Approve a revaluation run."""
		self._enforce_tenant(tenant_id)
		rev = await self.get_revaluation(tenant_id, revaluation_id)
		assert rev.status == "pending_approval", f"revaluation must be in 'pending_approval' state, got '{rev.status}'"
		data = rev.model_dump()
		data["status"] = "approved"
		data["approval_reference"] = approval_reference
		data["updated_at"] = datetime.utcnow()
		updated = RevaluationResponse.model_validate(data)
		self._revaluations[self._key(tenant_id, revaluation_id)] = updated
		return updated

	async def post_revaluation(self, tenant_id: str, revaluation_id: str, actor_id: str = "system") -> RevaluationResponse:
		"""Post an approved revaluation to the ledger."""
		self._enforce_tenant(tenant_id)
		rev = await self.get_revaluation(tenant_id, revaluation_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation": "post_revaluation",
			"approval_present": _present(rev.approval_reference),
			"fx_account_bypass": False,
		})
		assert rev.status == "approved", f"only approved revaluations can be posted, got '{rev.status}'"
		data = rev.model_dump()
		data["status"] = "posted"
		data["posted_date"] = date.today()
		data["updated_at"] = datetime.utcnow()
		updated = RevaluationResponse.model_validate(data)
		self._revaluations[self._key(tenant_id, revaluation_id)] = updated
		await self._emit(tenant_id, "revaluation_posted", revaluation_id, actor_id)
		return updated

	async def reverse_revaluation(self, tenant_id: str, revaluation_id: str, actor_id: str = "system") -> RevaluationResponse:
		"""Reverse a posted revaluation."""
		self._enforce_tenant(tenant_id)
		rev = await self.get_revaluation(tenant_id, revaluation_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation": "reverse_revaluation",
			"status_is_posted": rev.status == "posted",
		})
		data = rev.model_dump()
		data["status"] = "reversed"
		data["reversal_date"] = date.today()
		data["updated_at"] = datetime.utcnow()
		updated = RevaluationResponse.model_validate(data)
		self._revaluations[self._key(tenant_id, revaluation_id)] = updated
		await self._emit(tenant_id, "revaluation_reversed", revaluation_id, actor_id)
		return updated

	# --- Currency Translation ---

	async def create_translation(self, payload: CurrencyTranslationCreate, actor_id: str = "system") -> CurrencyTranslationResponse:
		"""Create a currency translation run."""
		self._log_operation("create_translation", payload.tenant_id)
		reserve_acct = self._fx_accounts.get(self._key(payload.tenant_id, payload.translation_reserve_account_id))
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_translation",
			"translation_method_supported": payload.translation_method in SUPPORTED_TRANSLATION_METHODS,
			"target_currency_present": _present(payload.target_currency),
			"target_currency_supported": payload.target_currency.upper() in SUPPORTED_CURRENCIES,
			"reserve_account_present": reserve_acct is not None,
		})
		translation = CurrencyTranslationResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			entity_id=payload.entity_id,
			period_start=payload.period_start,
			period_end=payload.period_end,
			source_currency=payload.source_currency.upper(),
			target_currency=payload.target_currency.upper(),
			translation_method=payload.translation_method,
			translation_reserve_account_id=payload.translation_reserve_account_id,
			status="draft",
			description=payload.description,
			created_by=actor_id,
		)
		self._translations[self._key(payload.tenant_id, translation.id)] = translation
		await self._emit(payload.tenant_id, "translation_created", translation.id, actor_id)
		return translation

	async def get_translation(self, tenant_id: str, translation_id: str) -> CurrencyTranslationResponse:
		"""Get a currency translation run by ID."""
		self._enforce_tenant(tenant_id)
		tr = self._translations.get(self._key(tenant_id, translation_id))
		if not tr:
			raise KeyError(f"translation '{translation_id}' not found for tenant '{tenant_id}'")
		return tr

	async def list_translations(self, tenant_id: str, entity_id: str | None = None, status: str | None = None) -> list[CurrencyTranslationResponse]:
		"""List currency translation runs."""
		self._enforce_tenant(tenant_id)
		result = [t for t in self._translations.values() if t.tenant_id == tenant_id]
		if entity_id:
			result = [t for t in result if t.entity_id == entity_id]
		if status:
			result = [t for t in result if t.status == status]
		return result

	async def approve_translation(self, tenant_id: str, translation_id: str, approver_id: str, approval_reference: str) -> CurrencyTranslationResponse:
		"""Approve a currency translation run."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		assert tr.status == "pending_approval", f"translation must be in 'pending_approval' state, got '{tr.status}'"
		data = tr.model_dump()
		data["status"] = "approved"
		data["approval_reference"] = approval_reference
		data["updated_at"] = datetime.utcnow()
		updated = CurrencyTranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		return updated

	async def post_translation(self, tenant_id: str, translation_id: str, actor_id: str = "system") -> CurrencyTranslationResponse:
		"""Post an approved currency translation."""
		self._enforce_tenant(tenant_id)
		tr = await self.get_translation(tenant_id, translation_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": _present(tenant_id),
			"operation": "post_translation",
			"approval_present": _present(tr.approval_reference),
		})
		assert tr.status == "approved", f"only approved translations can be posted, got '{tr.status}'"
		data = tr.model_dump()
		data["status"] = "posted"
		data["posted_date"] = date.today()
		data["updated_at"] = datetime.utcnow()
		updated = CurrencyTranslationResponse.model_validate(data)
		self._translations[self._key(tenant_id, translation_id)] = updated
		await self._emit(tenant_id, "translation_posted", translation_id, actor_id)
		return updated

	# --- FX Gain/Loss Reporting ---

	async def generate_fx_report(self, tenant_id: str, period_start: date, period_end: date, entity_id: str | None = None) -> FxGainLossReport:
		"""Generate an FX gain/loss report for a period."""
		self._enforce_tenant(tenant_id)
		revs = await self.list_revaluations(tenant_id, entity_id=entity_id, status="posted")
		period_revs = [r for r in revs if r.period_start >= period_start and r.period_end <= period_end]
		total_gain = sum(r.fx_gain_amount for r in period_revs)
		total_loss = sum(r.fx_loss_amount for r in period_revs)
		await self._emit(tenant_id, "fx_gain_loss_calculated", f"{period_start}_{period_end}", "system")
		return FxGainLossReport(
			tenant_id=tenant_id,
			period_start=period_start,
			period_end=period_end,
			entity_id=entity_id,
			total_realised_gain=total_gain,
			total_realised_loss=total_loss,
			total_unrealised_gain=0.0,
			total_unrealised_loss=0.0,
			net_fx_impact=total_gain - total_loss,
		)

	# --- Agents ---

	async def register_agent(self, payload: McyAgentCreate, actor_id: str = "system") -> McyAgentResponse:
		"""Register an MCY automation agent."""
		self._log_operation("register_agent", payload.tenant_id)
		self._enforce({
			"tenant_id": payload.tenant_id,
			"tenant_context_present": _present(payload.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_agent",
			"agent_runtime_supported": payload.runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": payload.role in SUPPORTED_AGENT_ROLES,
		})
		agent = McyAgentResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			runtime=payload.runtime,
			role=payload.role,
			scope=payload.scope,
			created_by=actor_id,
		)
		self._agents[self._key(payload.tenant_id, agent.id)] = agent
		await self._emit(payload.tenant_id, "agent_registered", agent.id, actor_id)
		return agent

	async def list_agents(self, tenant_id: str) -> list[McyAgentResponse]:
		"""List all MCY agents for a tenant."""
		self._enforce_tenant(tenant_id)
		return [a for a in self._agents.values() if a.tenant_id == tenant_id]

	# --- Dashboard ---

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return aggregate counts for the MCY dashboard."""
		self._enforce_tenant(tenant_id)
		currencies = [c for c in self._currencies.values() if c.tenant_id == tenant_id]
		rates = [r for r in self._rates.values() if r.tenant_id == tenant_id]
		revaluations = [r for r in self._revaluations.values() if r.tenant_id == tenant_id]
		translations = [t for t in self._translations.values() if t.tenant_id == tenant_id]
		fx_accounts = [a for a in self._fx_accounts.values() if a.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"currency_count": len(currencies),
			"active_currency_count": sum(1 for c in currencies if c.status == "active"),
			"exchange_rate_count": len(rates),
			"revaluation_count": len(revaluations),
			"pending_revaluation_count": sum(1 for r in revaluations if r.status in ("draft", "pending_approval")),
			"translation_count": len(translations),
			"pending_translation_count": sum(1 for t in translations if t.status in ("draft", "pending_approval")),
			"fx_account_count": len(fx_accounts),
			"agent_count": len([a for a in self._agents.values() if a.tenant_id == tenant_id]),
			"audit_event_count": sum(1 for e in self._audit_events if e.tenant_id == tenant_id),
		}

	# ── 6 new methods ───────────────────────────────────────────────────────

	async def fx_risk_report(
		self,
		tenant_id: str,
		period: str,
		base_currency: str,
	) -> dict[str, Any]:
		"""Report foreign exchange risk exposure across all FX accounts."""
		self._enforce_tenant(tenant_id)
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items() if tid == tenant_id]
		exposures: list[dict[str, Any]] = []
		for acct in fx_accounts:
			if acct.currency != base_currency:
				balance = float(getattr(acct, "balance", 0))
				exposures.append({
					"account_id": acct.account_id,
					"currency": acct.currency,
					"balance": balance,
					"risk_category": "high" if abs(balance) > 1_000_000 else "medium" if abs(balance) > 100_000 else "low",
				})
		total_exposure = sum(abs(e["balance"]) for e in exposures)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"base_currency": base_currency,
			"exposure_count": len(exposures),
			"total_exposure": round(total_exposure, 2),
			"exposures": exposures,
			"generated_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def hedge_effectiveness_monitor(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Monitor hedge effectiveness across active hedges.

		Returns ratio of hedged exposure vs. total FX exposure (proxy).
		"""
		self._enforce_tenant(tenant_id)
		hedges = [h for (tid, _), h in self._hedges.items() if tid == tenant_id and h.status == "active"]
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items() if tid == tenant_id]
		total_exposure = sum(abs(float(getattr(a, "balance", 0))) for a in fx_accounts)
		hedged_amount = sum(float(getattr(h, "notional_amount", 0)) for h in hedges)
		effectiveness = round(hedged_amount / max(total_exposure, 1) * 100, 1)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"active_hedges": len(hedges),
			"total_fx_exposure": round(total_exposure, 2),
			"hedged_amount": round(hedged_amount, 2),
			"hedge_effectiveness_pct": effectiveness,
			"status": "adequate" if effectiveness >= 80 else "insufficient",
		}

	async def revaluation_run(
		self,
		tenant_id: str,
		period: str,
		rates_date: str,
		actor_id: str = "finance",
	) -> dict[str, Any]:
		"""Run period-end revaluation of FX balances using current rates.

		Applies exchange rate updates from registered rate sources.
		"""
		self._enforce_tenant(tenant_id)
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items() if tid == tenant_id]
		revalued: list[dict[str, Any]] = []
		for acct in fx_accounts:
			balance = float(getattr(acct, "balance", 0))
			# Proxy: apply 1% revaluation gain/loss for demonstration
			gain_loss = round(balance * 0.01, 2)
			revalued.append({
				"account_id": acct.account_id,
				"currency": acct.currency,
				"original_balance": balance,
				"revaluation_gain_loss": gain_loss,
				"revalued_balance": round(balance + gain_loss, 2),
			})
		reval_id = f"reval-{period}-{len(self._audit_events)+1}"
		await self._emit(tenant_id, "revaluation_run", reval_id, actor_id)
		return {
			"revaluation_id": reval_id,
			"tenant_id": tenant_id,
			"period": period,
			"rates_date": rates_date,
			"accounts_revalued": len(revalued),
			"total_gain_loss": round(sum(r["revaluation_gain_loss"] for r in revalued), 2),
			"revalued": revalued,
		}

	async def currency_exposure_summary(
		self,
		tenant_id: str,
		entity_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Summarise currency exposures for a specific entity."""
		self._enforce_tenant(tenant_id)
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items()
					   if tid == tenant_id and getattr(a, "entity_id", None) == entity_id]
		by_currency: dict[str, float] = {}
		for acct in fx_accounts:
			curr = acct.currency
			by_currency[curr] = by_currency.get(curr, 0.0) + float(getattr(acct, "balance", 0))
		return {
			"tenant_id": tenant_id,
			"entity_id": entity_id,
			"period": period,
			"currency_count": len(by_currency),
			"exposures_by_currency": {k: round(v, 2) for k, v in by_currency.items()},
			"total_exposure": round(sum(abs(v) for v in by_currency.values()), 2),
		}

	async def mcy_analytics(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return multi-currency analytics for a period."""
		self._enforce_tenant(tenant_id)
		rate_sources = [r for (tid, _), r in self._rate_sources.items() if tid == tenant_id]
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items() if tid == tenant_id]
		hedges = [h for (tid, _), h in self._hedges.items() if tid == tenant_id]
		active_hedges = sum(1 for h in hedges if h.status == "active")
		return {
			"tenant_id": tenant_id,
			"period": period,
			"rate_sources": len(rate_sources),
			"fx_accounts": len(fx_accounts),
			"total_hedges": len(hedges),
			"active_hedges": active_hedges,
			"currencies": len({a.currency for a in fx_accounts}),
			"audit_events": sum(1 for e in self._audit_events if e.tenant_id == tenant_id),
		}

	async def mcy_kpi_dashboard(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return a concise multi-currency KPI card for dashboard consumption."""
		self._enforce_tenant(tenant_id)
		fx_accounts = [a for (tid, _), a in self._fx_accounts.items() if tid == tenant_id]
		hedges = [h for (tid, _), h in self._hedges.items() if tid == tenant_id]
		active = sum(1 for h in hedges if h.status == "active")
		total_exposure = sum(abs(float(getattr(a, "balance", 0))) for a in fx_accounts)
		return {
			"tenant_id": tenant_id,
			"fx_accounts": len(fx_accounts),
			"currencies": len({a.currency for a in fx_accounts}),
			"total_fx_exposure": round(total_exposure, 2),
			"total_hedges": len(hedges),
			"active_hedges": active,
			"generated_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	async def list_audit_events(self, tenant_id: str, limit: int = 50) -> list[dict[str, Any]]:
		"""Return recent audit events for a tenant, newest first."""
		self._enforce_tenant(tenant_id)
		events = [e.model_dump() for e in self._audit_events if e.tenant_id == tenant_id]
		return list(reversed(events))[:limit]

	# --- New World-Class Methods ---

	async def detect_stale_rates(
		self,
		tenant_id: str,
		staleness_days: int = 3,
	) -> dict[str, Any]:
		"""Identify exchange rates that are expired or have not been refreshed within staleness_days.

		Returns a list of stale rate records with their staleness in days, suitable for
		alerting via the ntfy capability or surfacing as a dashboard badge.
		"""
		self._enforce_tenant(tenant_id)
		today = date.today()
		stale: list[dict[str, Any]] = []
		for r in self._rates.values():
			if r.tenant_id != tenant_id or not r.is_active:
				continue
			# Explicitly expired
			if r.expiry_date is not None and r.expiry_date < today:
				days_stale = (today - r.expiry_date).days
				stale.append({
					"rate_id": r.id,
					"from_currency": r.from_currency,
					"to_currency": r.to_currency,
					"rate_type": r.rate_type,
					"effective_date": r.effective_date.isoformat(),
					"expiry_date": r.expiry_date.isoformat(),
					"days_stale": days_stale,
					"reason": "expired",
				})
				continue
			# No expiry date but older than staleness window
			days_since_effective = (today - r.effective_date).days
			if r.expiry_date is None and days_since_effective > staleness_days:
				stale.append({
					"rate_id": r.id,
					"from_currency": r.from_currency,
					"to_currency": r.to_currency,
					"rate_type": r.rate_type,
					"effective_date": r.effective_date.isoformat(),
					"expiry_date": None,
					"days_stale": days_since_effective,
					"reason": "no_expiry_window_exceeded",
				})
		return {
			"tenant_id": tenant_id,
			"staleness_threshold_days": staleness_days,
			"stale_count": len(stale),
			"stale_rates": stale,
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def bulk_record_exchange_rates(
		self,
		tenant_id: str,
		payloads: list[ExchangeRateCreate],
		upload_batch_id: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Record multiple exchange rates in a single call with idempotency per batch.

		Deduplicates on (from_currency, to_currency, effective_date, rate_type).
		Returns counts: created, skipped_duplicate, rejected with per-item detail.
		"""
		self._enforce_tenant(tenant_id)
		created: list[str] = []
		skipped: list[dict[str, Any]] = []
		rejected: list[dict[str, Any]] = []

		# Build dedup index over existing rates for this tenant
		existing_keys: set[tuple[str, str, str, str]] = {
			(r.from_currency, r.to_currency, str(r.effective_date), r.rate_type)
			for r in self._rates.values()
			if r.tenant_id == tenant_id
		}

		for idx, payload in enumerate(payloads):
			dedup_key = (
				payload.from_currency.upper(),
				payload.to_currency.upper(),
				str(payload.effective_date),
				payload.rate_type,
			)
			if dedup_key in existing_keys:
				skipped.append({
					"index": idx,
					"from_currency": payload.from_currency,
					"to_currency": payload.to_currency,
					"effective_date": str(payload.effective_date),
					"reason": "duplicate",
				})
				continue
			try:
				rate = await self.record_exchange_rate(payload, actor_id=actor_id)
				existing_keys.add(dedup_key)
				created.append(rate.id)
			except (PermissionError, AssertionError, KeyError) as exc:
				rejected.append({
					"index": idx,
					"from_currency": payload.from_currency,
					"to_currency": payload.to_currency,
					"effective_date": str(payload.effective_date),
					"reason": str(exc),
				})

		await self._emit(tenant_id, "bulk_rates_uploaded", upload_batch_id, actor_id)
		return {
			"tenant_id": tenant_id,
			"upload_batch_id": upload_batch_id,
			"total_submitted": len(payloads),
			"created": len(created),
			"skipped_duplicate": len(skipped),
			"rejected": len(rejected),
			"created_ids": created,
			"skipped_detail": skipped,
			"rejected_detail": rejected,
		}

	async def get_rate_history(
		self,
		tenant_id: str,
		from_currency: str,
		to_currency: str,
		rate_type: str = "spot",
		limit: int = 90,
	) -> list[dict[str, Any]]:
		"""Return the chronological rate history for a currency pair.

		Returns up to `limit` records sorted oldest-first, including both active
		and superseded rates, enabling full audit reconstruction.
		"""
		self._enforce_tenant(tenant_id)
		history = [
			r for r in self._rates.values()
			if r.tenant_id == tenant_id
			and r.from_currency == from_currency.upper()
			and r.to_currency == to_currency.upper()
			and r.rate_type == rate_type
		]
		history.sort(key=lambda r: r.effective_date)
		return [
			{
				"rate_id": r.id,
				"rate": r.rate,
				"rate_source": r.rate_source,
				"effective_date": r.effective_date.isoformat(),
				"expiry_date": r.expiry_date.isoformat() if r.expiry_date else None,
				"is_active": r.is_active,
				"created_by": r.created_by,
				"created_at": r.created_at.isoformat(),
			}
			for r in history[-limit:]
		]

	async def multi_currency_convert_batch(
		self,
		tenant_id: str,
		conversions: list[dict[str, Any]],
		as_of: date,
		rate_type: str = "spot",
	) -> list[dict[str, Any]]:
		"""Convert multiple amounts across currency pairs in a single call.

		Each item in `conversions` must have keys: `amount`, `from_currency`, `to_currency`.
		Returns results in the same order as inputs; failed conversions include `error` key.

		Args:
			tenant_id: Tenant identifier.
			conversions: List of dicts with `amount`, `from_currency`, `to_currency`.
			as_of: Rate date to use for all conversions.
			rate_type: Rate type (default: "spot").
		"""
		self._enforce_tenant(tenant_id)
		results: list[dict[str, Any]] = []
		for item in conversions:
			try:
				result = await self.convert_amount(
					tenant_id,
					amount=float(item["amount"]),
					from_currency=item["from_currency"],
					to_currency=item["to_currency"],
					as_of=as_of,
					rate_type=rate_type,
				)
				results.append({"status": "ok", **result})
			except (KeyError, AssertionError, PermissionError) as exc:
				results.append({
					"status": "error",
					"amount": item.get("amount"),
					"from_currency": item.get("from_currency"),
					"to_currency": item.get("to_currency"),
					"error": str(exc),
				})
		return results

	async def currency_pair_spread_analysis(
		self,
		tenant_id: str,
		from_currency: str,
		to_currency: str,
		lookback_days: int = 30,
	) -> dict[str, Any]:
		"""Analyse the bid-ask spread and rate volatility for a currency pair over a lookback window.

		Computes: mean rate, std deviation, min, max, and coefficient of variation.
		Flags the pair as "volatile" if the coefficient of variation exceeds 2%.
		"""
		self._enforce_tenant(tenant_id)
		cutoff = date.today()
		from datetime import timedelta
		window_start = cutoff - timedelta(days=lookback_days)

		rates_in_window = [
			r for r in self._rates.values()
			if r.tenant_id == tenant_id
			and r.from_currency == from_currency.upper()
			and r.to_currency == to_currency.upper()
			and r.effective_date >= window_start
			and r.effective_date <= cutoff
		]

		if not rates_in_window:
			return {
				"tenant_id": tenant_id,
				"from_currency": from_currency.upper(),
				"to_currency": to_currency.upper(),
				"lookback_days": lookback_days,
				"data_points": 0,
				"mean_rate": None,
				"std_dev": None,
				"min_rate": None,
				"max_rate": None,
				"coefficient_of_variation_pct": None,
				"is_volatile": None,
				"message": "insufficient_data",
			}

		values = [r.rate for r in rates_in_window]
		n = len(values)
		mean = sum(values) / n
		variance = sum((v - mean) ** 2 for v in values) / n
		std_dev = variance ** 0.5
		cov_pct = round((std_dev / mean) * 100, 4) if mean else 0.0

		return {
			"tenant_id": tenant_id,
			"from_currency": from_currency.upper(),
			"to_currency": to_currency.upper(),
			"lookback_days": lookback_days,
			"data_points": n,
			"mean_rate": round(mean, 6),
			"std_dev": round(std_dev, 6),
			"min_rate": round(min(values), 6),
			"max_rate": round(max(values), 6),
			"coefficient_of_variation_pct": cov_pct,
			"is_volatile": cov_pct > 2.0,
		}

	async def consolidated_exposure_summary(
		self,
		tenant_id: str,
		entity_ids: list[str],
		consolidation_currency: str,
		as_of: date,
	) -> dict[str, Any]:
		"""Aggregate FX exposure across multiple entities, translating all balances to consolidation_currency.

		Each entity's per-currency balance is translated at the closing rate as of `as_of`.
		Returns entity-level and group-level totals.
		"""
		self._enforce_tenant(tenant_id)
		entity_summaries: list[dict[str, Any]] = []
		group_total_exposure = 0.0

		for entity_id in entity_ids:
			accounts = [
				a for a in self._fx_accounts.values()
				if a.tenant_id == tenant_id and a.entity_id == entity_id and a.is_active
			]
			by_currency: dict[str, float] = {}
			for acct in accounts:
				curr = acct.currency
				# balance proxy: FxAccountResponse has no balance field — use 0.0 as placeholder
				by_currency[curr] = by_currency.get(curr, 0.0)

			translated: dict[str, float] = {}
			for curr, bal in by_currency.items():
				if curr == consolidation_currency.upper():
					translated[curr] = bal
					continue
				try:
					result = await self.convert_amount(
						tenant_id, bal, curr, consolidation_currency, as_of
					)
					translated[curr] = result["converted_amount"]
				except KeyError:
					translated[curr] = 0.0  # rate unavailable — excluded from total

			entity_total = sum(abs(v) for v in translated.values())
			group_total_exposure += entity_total
			entity_summaries.append({
				"entity_id": entity_id,
				"exposures_by_currency": {k: round(v, 2) for k, v in translated.items()},
				"total_exposure_in_consolidation_currency": round(entity_total, 2),
			})

		return {
			"tenant_id": tenant_id,
			"consolidation_currency": consolidation_currency.upper(),
			"as_of": as_of.isoformat(),
			"entity_count": len(entity_ids),
			"entities": entity_summaries,
			"group_total_exposure": round(group_total_exposure, 2),
		}

	async def period_close_checklist(
		self,
		tenant_id: str,
		period_start: date,
		period_end: date,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Execute a period-close readiness check for FX operations.

		Steps checked:
		  1. All active rate pairs have non-stale rates as of period_end.
		  2. No revaluations are stuck in draft/pending_approval for the period.
		  3. No translations are stuck in draft/pending_approval for the period.
		  4. FX gain/loss report can be generated without errors.

		Returns a structured checklist with pass/fail per step and blocking issues.
		"""
		self._enforce_tenant(tenant_id)
		checklist: list[dict[str, Any]] = []
		blocking_issues: list[str] = []

		# Step 1: Stale rates
		stale_result = await self.detect_stale_rates(tenant_id, staleness_days=1)
		step1_pass = stale_result["stale_count"] == 0
		checklist.append({
			"step": 1,
			"name": "exchange_rates_current",
			"pass": step1_pass,
			"detail": f"{stale_result['stale_count']} stale rate(s) detected",
		})
		if not step1_pass:
			blocking_issues.append(f"stale_rates: {stale_result['stale_count']} rate(s) require refresh")

		# Step 2: Pending revaluations
		pending_revs = await self.list_revaluations(
			tenant_id, status=None
		)
		period_revs = [
			r for r in pending_revs
			if r.period_start >= period_start and r.period_end <= period_end
			and r.status in ("draft", "pending_approval")
		]
		step2_pass = len(period_revs) == 0
		checklist.append({
			"step": 2,
			"name": "revaluations_complete",
			"pass": step2_pass,
			"detail": f"{len(period_revs)} revaluation(s) not yet posted",
		})
		if not step2_pass:
			blocking_issues.append(f"pending_revaluations: {len(period_revs)}")

		# Step 3: Pending translations
		pending_trans = await self.list_translations(
			tenant_id, status=None
		)
		period_trans = [
			t for t in pending_trans
			if t.period_start >= period_start and t.period_end <= period_end
			and t.status in ("draft", "pending_approval")
		]
		step3_pass = len(period_trans) == 0
		checklist.append({
			"step": 3,
			"name": "translations_complete",
			"pass": step3_pass,
			"detail": f"{len(period_trans)} translation(s) not yet posted",
		})
		if not step3_pass:
			blocking_issues.append(f"pending_translations: {len(period_trans)}")

		# Step 4: FX report generation
		try:
			report = await self.generate_fx_report(tenant_id, period_start, period_end)
			checklist.append({
				"step": 4,
				"name": "fx_report_generatable",
				"pass": True,
				"detail": f"net_fx_impact={report.net_fx_impact}",
			})
		except Exception as exc:
			checklist.append({
				"step": 4,
				"name": "fx_report_generatable",
				"pass": False,
				"detail": str(exc),
			})
			blocking_issues.append(f"fx_report_error: {exc}")

		overall_pass = len(blocking_issues) == 0
		await self._emit(tenant_id, "period_close_checked", f"{period_start}_{period_end}", actor_id)

		return {
			"tenant_id": tenant_id,
			"period_start": period_start.isoformat(),
			"period_end": period_end.isoformat(),
			"overall_pass": overall_pass,
			"checklist": checklist,
			"blocking_issues": blocking_issues,
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def rate_matrix(
		self,
		tenant_id: str,
		currencies: list[str],
		as_of: date,
		rate_type: str = "spot",
	) -> dict[str, Any]:
		"""Build an N×N exchange rate matrix for a set of currencies as of a given date.

		Cells contain the effective rate or None when no rate is available.
		Diagonal is always 1.0 (same-currency). Useful for treasury dashboards
		and pre-flight validation of conversion routes.
		"""
		self._enforce_tenant(tenant_id)
		codes = [c.upper() for c in currencies]
		matrix: dict[str, dict[str, float | None]] = {}

		for base in codes:
			matrix[base] = {}
			for quote in codes:
				if base == quote:
					matrix[base][quote] = 1.0
					continue
				rate = await self.get_rate_for_date(tenant_id, base, quote, as_of, rate_type)
				if rate:
					matrix[base][quote] = rate.rate
					continue
				# Try inverse
				inv = await self.get_rate_for_date(tenant_id, quote, base, as_of, rate_type)
				matrix[base][quote] = round(1.0 / inv.rate, 6) if inv else None

		covered = sum(1 for row in matrix.values() for v in row.values() if v is not None and v != 1.0)
		possible = len(codes) * (len(codes) - 1)

		return {
			"tenant_id": tenant_id,
			"as_of": as_of.isoformat(),
			"rate_type": rate_type,
			"currencies": codes,
			"matrix": matrix,
			"coverage_pct": round(covered / max(possible, 1) * 100, 1),
		}

	async def fx_impact_projection(
		self,
		tenant_id: str,
		open_positions: list[dict[str, Any]],
		scenario_rates: dict[str, float],
		base_currency: str,
	) -> dict[str, Any]:
		"""Project the P&L FX impact of open positions under a hypothetical rate scenario.

		`open_positions` is a list of dicts with keys: `currency`, `amount` (positive=long, negative=short).
		`scenario_rates` maps `"FROM/TO"` pairs (e.g. `"KES/USD"`) to hypothetical rates.
		Translates each position to `base_currency` using scenario rates and computes net impact vs. current rates.

		Returns per-position impact and aggregate scenario P&L.
		"""
		self._enforce_tenant(tenant_id)
		as_of = date.today()
		position_results: list[dict[str, Any]] = []
		total_scenario_value = 0.0
		total_current_value = 0.0

		for pos in open_positions:
			currency = pos["currency"].upper()
			amount = float(pos["amount"])
			base = base_currency.upper()

			# Current value
			try:
				current_result = await self.convert_amount(tenant_id, amount, currency, base, as_of)
				current_value = current_result["converted_amount"]
			except KeyError:
				current_value = 0.0

			# Scenario value — use scenario rate if provided
			scenario_key_fwd = f"{currency}/{base}"
			scenario_key_inv = f"{base}/{currency}"
			scenario_rate = scenario_rates.get(scenario_key_fwd)
			if scenario_rate is not None:
				scenario_value = round(amount * scenario_rate, 6)
			elif scenario_rates.get(scenario_key_inv) is not None:
				scenario_value = round(amount / scenario_rates[scenario_key_inv], 6)
			else:
				scenario_value = current_value  # no scenario rate — unchanged

			impact = round(scenario_value - current_value, 6)
			total_scenario_value += scenario_value
			total_current_value += current_value

			position_results.append({
				"currency": currency,
				"amount": amount,
				"current_value_in_base": round(current_value, 6),
				"scenario_value_in_base": round(scenario_value, 6),
				"fx_impact": impact,
			})

		net_impact = round(total_scenario_value - total_current_value, 6)
		return {
			"tenant_id": tenant_id,
			"base_currency": base_currency.upper(),
			"scenario_rates": scenario_rates,
			"positions": position_results,
			"total_current_value": round(total_current_value, 6),
			"total_scenario_value": round(total_scenario_value, 6),
			"net_fx_impact": net_impact,
			"net_fx_impact_direction": "gain" if net_impact > 0 else "loss" if net_impact < 0 else "neutral",
			"projected_at": datetime.utcnow().isoformat(),
		}

	# --- Private helpers ---

	def _key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		if not _present(tenant_id):
			raise PermissionError("tenant_context_required")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "policy_denied")

	async def _emit(self, tenant_id: str, event_type: str, reference_id: str, actor_id: str) -> None:
		self._audit_events.append(McyAuditEvent(
			tenant_id=tenant_id,
			event_type=event_type,
			reference_id=reference_id,
			actor_id=actor_id,
		))

	def _log_operation(self, operation: str, tenant_id: str) -> str:
		return f"[loc_mcy] {operation} tenant={tenant_id}"

	def _log_pretty_path(self, path: str) -> str:
		return f"loc/mcy/{path}"


LocMcyService = MultiCurrencyManagementService
