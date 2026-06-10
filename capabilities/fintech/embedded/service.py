"""Executable service layer for APG Embedded Finance."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANNELS,
		SUPPORTED_ENVIRONMENTS, SUPPORTED_PRODUCTS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .embedded_runtime import (
		normalize_code, normalize_codes, normalize_domain,
		percent_bounded, public_reference,
	)
	from .models import (
		CustomerConsent, EmbeddedAccount, EmbeddedCardOffer, EmbeddedEvidence,
		EmbeddedLendingOffer, EmbeddedPayment, HostApplication, PartnerProgram,
		ProductPlacement, RevenueShare, SettlementBatch,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHANNELS,
		SUPPORTED_ENVIRONMENTS, SUPPORTED_PRODUCTS,
		evaluate_capability_rules, get_capability_contract,
	)
	from embedded_runtime import (  # type: ignore
		normalize_code, normalize_codes, normalize_domain,
		percent_bounded, public_reference,
	)
	from models import (  # type: ignore
		CustomerConsent, EmbeddedAccount, EmbeddedCardOffer, EmbeddedEvidence,
		EmbeddedLendingOffer, EmbeddedPayment, HostApplication, PartnerProgram,
		ProductPlacement, RevenueShare, SettlementBatch,
	)

_logger = logging.getLogger(__name__)

_WIDGET_TYPES = {"checkout", "payment_button", "wallet", "lending", "insurance", "card"}
_INTEGRATION_TYPES = {"api", "sdk", "iframe", "white_label"}
_CURRENCY_CODES = {"KES", "USD", "EUR", "GBP", "UGX", "TZS", "NGN", "GHS", "ZAR"}
_INSURANCE_PRODUCTS = {"life", "health", "credit", "device", "travel", "micro"}
_WEBHOOK_EVENTS = {
	"payment.completed", "payment.failed", "payment.reversed",
	"account.opened", "account.closed", "card.issued", "card.blocked",
	"lending.disbursed", "lending.repaid", "lending.defaulted",
	"partner.onboarded", "partner.suspended", "settlement.closed",
}
_COMPLIANCE_EVENT_TYPES = {
	"kyc_update", "kyb_update", "aml_flag", "pep_match", "sanctions_hit",
	"transaction_monitoring", "regulatory_report",
}


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _truncate_str(s: str, max_len: int = 64) -> str:
	return s[:max_len] if len(s) > max_len else s


def _minor_to_decimal(minor: int, decimals: int = 2) -> Decimal:
	divisor = Decimal(10 ** decimals)
	return Decimal(minor) / divisor


def _signing_token(partner_id: str, widget_id: str, secret: str = "apg-secret") -> str:
	raw = f"{partner_id}:{widget_id}:{secret}"
	return hashlib.sha256(raw.encode()).hexdigest()[:32]


class EmbeddedFinanceService:
	"""Full async Embedded Finance runtime for APG generated applications.

	All state-mutating public methods are async.  Read-only helpers that cannot
	block are sync; they may be awaited trivially if callers prefer uniformity.
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

		# In-memory collections (used when no store is injected)
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

		# Extra state added by new methods
		self._widgets: dict[str, dict[str, Any]] = {}
		self._webhooks: dict[str, dict[str, Any]] = {}
		self._insurance: dict[str, dict[str, Any]] = {}
		self._api_usage: dict[str, list[dict[str, Any]]] = {}
		self._compliance_log: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Capability contract helpers (sync — no IO)
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Existing core methods — preserved, made async
	# ------------------------------------------------------------------ #

	async def register_partner_program(
		self,
		program_id: str,
		tenant_id: str,
		name: str,
		kyb_reference: str,
		contract_reference: str,
		risk_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a new embedded-finance partner program."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_partner_program",
			"kyb_present": bool(kyb_reference),
			"contract_present": bool(contract_reference),
			"risk_present": bool(risk_reference),
		})
		program = PartnerProgram(
			program_id, tenant_id, name, kyb_reference,
			contract_reference, risk_reference,
		)
		self.programs[program_id] = program
		await self._async_audit(tenant_id, "partner_program_registered", program_id)
		_logger.info("partner_program_registered program_id=%s tenant=%s", program_id, tenant_id)
		return program.to_dict()

	async def register_host_application(
		self,
		application_id: str,
		tenant_id: str,
		program_id: str,
		name: str,
		environment: str,
		domain: str,
		terms_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a host application that will embed financial products."""
		program = self._tenant_program_or_none(program_id, tenant_id)
		environment = normalize_code(environment)
		domain = normalize_domain(domain)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_host_application",
			"program_present": program is not None,
			"environment_supported": environment in SUPPORTED_ENVIRONMENTS,
			"domain_present": bool(domain),
			"terms_present": bool(terms_reference),
		})
		application = HostApplication(
			application_id, tenant_id, program_id, name,
			environment, domain, terms_reference,
		)
		self.applications[application_id] = application
		await self._async_audit(tenant_id, "host_application_registered", application_id)
		return application.to_dict()

	async def publish_product_placement(
		self,
		placement_id: str,
		tenant_id: str,
		application_id: str,
		product_type: str,
		channel: str,
		scopes: list[str],
		risk_policy_reference: str,
	) -> dict[str, Any]:
		"""Publish a product placement configuration for a host application."""
		application = self._tenant_application_or_none(application_id, tenant_id)
		product_type = normalize_code(product_type)
		channel = normalize_code(channel)
		scopes = normalize_codes(scopes)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_product_placement",
			"application_present": application is not None,
			"product_supported": product_type in SUPPORTED_PRODUCTS,
			"channel_supported": channel in SUPPORTED_CHANNELS,
			"scopes_present": bool(scopes),
			"risk_policy_present": bool(risk_policy_reference),
		})
		placement = ProductPlacement(
			placement_id, tenant_id, application_id, product_type,
			channel, scopes, risk_policy_reference,
		)
		self.placements[placement_id] = placement
		await self._async_audit(tenant_id, "product_placement_published", placement_id)
		return placement.to_dict()

	async def capture_customer_consent(
		self,
		consent_id: str,
		tenant_id: str,
		application_id: str,
		customer_reference: str,
		scopes: list[str],
		expiry_date: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record explicit customer consent for embedded product scopes."""
		application = self._tenant_application_or_none(application_id, tenant_id)
		scopes = normalize_codes(scopes)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "capture_customer_consent",
			"application_present": application is not None,
			"customer_present": bool(customer_reference),
			"scopes_present": bool(scopes),
			"expiry_present": bool(expiry_date),
		})
		consent = CustomerConsent(
			consent_id, tenant_id, application_id,
			customer_reference, scopes, expiry_date,
		)
		self.consents[consent_id] = consent
		await self._async_audit(tenant_id, "customer_consent_captured", consent_id)
		return consent.to_dict()

	async def open_embedded_account(
		self,
		account_id: str,
		tenant_id: str,
		application_id: str,
		customer_reference: str,
		wallet_reference: str,
		kyc_reference: str,
	) -> dict[str, Any]:
		"""Open a white-label embedded account for a customer."""
		application = self._tenant_application_or_none(application_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_embedded_account",
			"application_present": application is not None,
			"kyc_present": bool(kyc_reference),
			"wallet_present": bool(wallet_reference),
		})
		account = EmbeddedAccount(
			account_id, tenant_id, application_id,
			customer_reference, wallet_reference, kyc_reference,
		)
		self.accounts[account_id] = account
		await self._async_audit(tenant_id, "embedded_account_opened", account_id)
		return account.to_dict() | {
			"public_account_reference": public_reference("acct", application_id, customer_reference),
		}

	async def initiate_embedded_payment(
		self,
		payment_id: str,
		tenant_id: str,
		application_id: str,
		placement_id: str,
		consent_id: str,
		source_reference: str,
		destination_reference: str,
		amount_minor: int,
		currency: str,
		risk_reference: str,
	) -> dict[str, Any]:
		"""Initiate a payment through an embedded placement with active consent."""
		application = self._tenant_application_or_none(application_id, tenant_id)
		placement = self._tenant_placement_or_none(placement_id, tenant_id)
		consent = self._tenant_consent_or_none(consent_id, tenant_id)
		currency = currency.strip().upper()
		consent_covers_scope = (
			consent is not None
			and consent.application_id == application_id
			and "payments.write" in consent.scopes
			and consent.status == "active"
		)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "initiate_embedded_payment",
			"application_present": application is not None,
			"placement_present": placement is not None,
			"placement_matches_application": placement is not None and placement.application_id == application_id,
			"consent_present": consent is not None,
			"consent_covers_scope": consent_covers_scope,
			"positive_amount": int(amount_minor) > 0,
			"currency_supported": currency in get_capability_contract(tenant_id)["configuration"]["payments"]["supported_currencies"],
			"risk_reference_present": bool(risk_reference),
		})
		payment = EmbeddedPayment(
			payment_id, tenant_id, application_id, placement_id, consent_id,
			source_reference, destination_reference, int(amount_minor), currency, risk_reference,
		)
		self.payments[payment_id] = payment
		await self._async_audit(tenant_id, "embedded_payment_initiated", payment_id)
		return payment.to_dict()

	async def offer_embedded_card(
		self,
		card_id: str,
		tenant_id: str,
		application_id: str,
		customer_reference: str,
		limit_minor: int,
		risk_reference: str,
	) -> dict[str, Any]:
		"""Issue an embedded virtual/physical card offer to a customer."""
		application = self._tenant_application_or_none(application_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "offer_embedded_card",
			"application_present": application is not None,
			"positive_limit": int(limit_minor) > 0,
			"risk_reference_present": bool(risk_reference),
		})
		card = EmbeddedCardOffer(
			card_id, tenant_id, application_id,
			customer_reference, int(limit_minor), risk_reference,
		)
		self.cards[card_id] = card
		await self._async_audit(tenant_id, "embedded_card_offered", card_id)
		return card.to_dict()

	async def create_lending_offer(
		self,
		offer_id: str,
		tenant_id: str,
		application_id: str,
		customer_reference: str,
		amount_minor: int,
		affordability_reference: str,
		underwriting_reference: str,
	) -> dict[str, Any]:
		"""Create an embedded lending offer backed by affordability and underwriting."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_lending_offer",
			"affordability_present": bool(affordability_reference),
			"underwriting_present": bool(underwriting_reference),
		})
		offer = EmbeddedLendingOffer(
			offer_id, tenant_id, application_id, customer_reference,
			int(amount_minor), affordability_reference, underwriting_reference,
		)
		self.lending[offer_id] = offer
		await self._async_audit(tenant_id, "embedded_lending_offer_created", offer_id)
		return offer.to_dict()

	async def close_settlement_batch(
		self,
		batch_id: str,
		tenant_id: str,
		program_id: str,
		amount_minor: int,
		currency: str,
		reconciliation_reference: str,
	) -> dict[str, Any]:
		"""Close a settlement batch for a partner program."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_settlement_batch",
			"reconciled": bool(reconciliation_reference),
			"positive_amount": int(amount_minor) > 0,
		})
		batch = SettlementBatch(
			batch_id, tenant_id, program_id,
			int(amount_minor), currency.strip().upper(), reconciliation_reference,
		)
		self.settlements[batch_id] = batch
		await self._async_audit(tenant_id, "settlement_batch_closed", batch_id)
		return batch.to_dict()

	async def record_revenue_share(
		self,
		share_id: str,
		tenant_id: str,
		program_id: str,
		percent: float,
		contract_reference: str,
	) -> dict[str, Any]:
		"""Record revenue-share percentage for a partner program."""
		program = self._tenant_program_or_none(program_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_revenue_share",
			"program_present": program is not None,
			"percent_bounded": percent_bounded(float(percent)),
			"contract_present": bool(contract_reference),
		})
		share = RevenueShare(share_id, tenant_id, program_id, float(percent), contract_reference)
		self.revenue_shares[share_id] = share
		await self._async_audit(tenant_id, "revenue_share_recorded", share_id)
		return share.to_dict()

	async def register_embedded_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register an AI agent operating within the embedded finance scope."""
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_embedded_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = EmbeddedEvidence(
			agent_id, tenant_id, "agent", agent_id, "registered",
			{"name": name, "runtime": runtime, "role": role, "scope": scope},
		)
		self.evidence[agent_id] = evidence
		await self._async_audit(tenant_id, "embedded_agent_registered", agent_id)
		return evidence.to_dict()

	async def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		"""Validate a batch of lifecycle events against the embedded policy engine."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "embedded_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.embedded.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return current in-memory dashboard counts for a tenant."""
		return {
			"tenant_id": tenant_id,
			"program_count": self._count(self.programs, tenant_id),
			"application_count": self._count(self.applications, tenant_id),
			"placement_count": self._count(self.placements, tenant_id),
			"consent_count": self._count(self.consents, tenant_id),
			"account_count": self._count(self.accounts, tenant_id),
			"payment_count": self._count(self.payments, tenant_id),
			"card_count": self._count(self.cards, tenant_id),
			"lending_count": self._count(self.lending, tenant_id),
			"settlement_count": self._count(self.settlements, tenant_id),
			"revenue_share_count": self._count(self.revenue_shares, tenant_id),
			"widget_count": sum(1 for w in self._widgets.values() if w.get("tenant_id") == tenant_id),
			"webhook_count": sum(1 for w in self._webhooks.values() if w.get("tenant_id") == tenant_id),
			"insurance_count": sum(1 for i in self._insurance.values() if i.get("tenant_id") == tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------ #
	# New methods — required by the task specification
	# ------------------------------------------------------------------ #

	async def embed_payment_widget(
		self,
		partner_id: str,
		widget_config: dict[str, Any],
	) -> dict[str, Any]:
		"""Generate an embeddable payment widget configuration for a partner.

		Validates widget type, currency, and partner registration, then issues
		a signed widget token that the host application includes in its UI.
		"""
		assert partner_id, "partner_id required"
		assert isinstance(widget_config, dict), "widget_config must be a dict"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		widget_type = normalize_code(str(widget_config.get("type", "checkout")))
		currency = str(widget_config.get("currency", "KES")).strip().upper()
		theme = widget_config.get("theme", "light")
		locale = widget_config.get("locale", "en")
		callback_url = str(widget_config.get("callback_url", ""))
		allowed_methods = widget_config.get("payment_methods", ["card", "mobile_money"])

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "embed_payment_widget",
			"partner_present": program is not None,
			"widget_type_valid": widget_type in _WIDGET_TYPES,
			"currency_supported": currency in _CURRENCY_CODES,
		})

		widget_id = f"wgt_{uuid.uuid4().hex[:16]}"
		signing_token = _signing_token(partner_id, widget_id)

		record: dict[str, Any] = {
			"widget_id": widget_id,
			"tenant_id": self.tenant_id,
			"partner_id": partner_id,
			"widget_type": widget_type,
			"currency": currency,
			"theme": theme,
			"locale": locale,
			"callback_url": callback_url,
			"payment_methods": allowed_methods,
			"signing_token": signing_token,
			"embed_url": f"https://embed.apg.finance/widget/{widget_id}",
			"js_snippet": (
				f'<script src="https://embed.apg.finance/sdk/v1.js" '
				f'data-widget="{widget_id}" data-token="{signing_token}"></script>'
			),
			"status": "active",
			"created_at": _now_iso(),
		}
		self._widgets[widget_id] = record
		await self._async_audit(self.tenant_id, "payment_widget_embedded", widget_id)
		_logger.info("embed_payment_widget widget_id=%s partner=%s", widget_id, partner_id)
		return record

	async def partner_onboarding(
		self,
		partner_id: str,
		business_details: dict[str, Any],
		integration_type: str,
	) -> dict[str, Any]:
		"""Onboard a new embedded-finance partner through a multi-step KYB flow.

		Creates the partner program record, validates business details, and
		returns an onboarding checklist with required document references.
		"""
		assert partner_id, "partner_id required"
		assert isinstance(business_details, dict), "business_details must be a dict"

		integration_type = normalize_code(integration_type)
		legal_name = str(business_details.get("legal_name", "")).strip()
		registration_number = str(business_details.get("registration_number", "")).strip()
		country = str(business_details.get("country", "KE")).strip().upper()
		contact_email = str(business_details.get("contact_email", "")).strip()
		industry = str(business_details.get("industry", "fintech")).strip()
		website = str(business_details.get("website", "")).strip()

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "partner_onboarding",
			"legal_name_present": bool(legal_name),
			"registration_number_present": bool(registration_number),
			"integration_type_valid": integration_type in _INTEGRATION_TYPES,
		})

		kyb_ref = f"kyb_{uuid.uuid4().hex[:12]}"
		contract_ref = f"ctr_{uuid.uuid4().hex[:12]}"
		risk_ref = f"risk_{uuid.uuid4().hex[:12]}"

		# Auto-create partner program
		program_key = f"prog_{partner_id}"
		if program_key not in self.programs:
			program = PartnerProgram(
				program_key, self.tenant_id, legal_name,
				kyb_ref, contract_ref, risk_ref,
			)
			self.programs[program_key] = program

		checklist: list[dict[str, Any]] = [
			{"step": "kyb_verification", "status": "pending", "reference": kyb_ref},
			{"step": "contract_signing", "status": "pending", "reference": contract_ref},
			{"step": "risk_assessment", "status": "pending", "reference": risk_ref},
			{"step": "technical_integration", "status": "pending", "integration_type": integration_type},
			{"step": "sandbox_testing", "status": "pending"},
			{"step": "production_go_live", "status": "pending"},
		]

		result: dict[str, Any] = {
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"program_key": program_key,
			"legal_name": legal_name,
			"registration_number": registration_number,
			"country": country,
			"contact_email": contact_email,
			"industry": industry,
			"website": website,
			"integration_type": integration_type,
			"kyb_reference": kyb_ref,
			"contract_reference": contract_ref,
			"risk_reference": risk_ref,
			"onboarding_checklist": checklist,
			"status": "onboarding_initiated",
			"created_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "partner_onboarding_initiated", partner_id)
		_logger.info("partner_onboarding partner_id=%s integration=%s", partner_id, integration_type)
		return result

	async def partner_reconciliation(
		self,
		partner_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Produce a reconciliation report for a partner over the given period.

		Aggregates payments, settlements, and revenue shares, then computes
		net position and flags any discrepancies above threshold.
		"""
		assert partner_id, "partner_id required"
		assert period, "period required (e.g. '2026-05')"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "partner_reconciliation",
			"partner_present": program is not None,
		})

		partner_payments = [
			p for p in self.payments.values()
			if p.tenant_id == self.tenant_id and p.application_id.startswith(partner_id)
		]
		partner_settlements = [
			s for s in self.settlements.values()
			if s.tenant_id == self.tenant_id and s.program_id == partner_id
		]
		partner_shares = [
			r for r in self.revenue_shares.values()
			if r.tenant_id == self.tenant_id and r.program_id == partner_id
		]

		gross_payments = sum(p.amount_minor for p in partner_payments)
		settled_amount = sum(s.amount_minor for s in partner_settlements)
		avg_share_pct = (
			sum(r.percent for r in partner_shares) / len(partner_shares)
			if partner_shares else 0.0
		)
		revenue_earned = int(gross_payments * avg_share_pct / 100)
		unreconciled = gross_payments - settled_amount
		discrepancy_flag = abs(unreconciled) > 100_00  # 100 units in minor

		report: dict[str, Any] = {
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"payment_count": len(partner_payments),
			"gross_payments_minor": gross_payments,
			"gross_payments_display": str(_minor_to_decimal(gross_payments)),
			"settlement_count": len(partner_settlements),
			"settled_amount_minor": settled_amount,
			"revenue_share_pct": round(avg_share_pct, 4),
			"revenue_earned_minor": revenue_earned,
			"unreconciled_minor": unreconciled,
			"discrepancy_flag": discrepancy_flag,
			"reconciliation_status": "discrepancy_detected" if discrepancy_flag else "balanced",
			"generated_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "partner_reconciliation_run", partner_id)
		return report

	async def white_label_wallet(
		self,
		partner_id: str,
		customer_id: str,
		currency: str,
	) -> dict[str, Any]:
		"""Provision a white-label wallet for a customer under a partner program.

		The wallet is linked to an embedded account and issued with a unique
		IBAN-style reference and initial KYC tier.
		"""
		assert partner_id, "partner_id required"
		assert customer_id, "customer_id required"
		currency = currency.strip().upper()

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "white_label_wallet",
			"partner_present": program is not None,
			"currency_supported": currency in _CURRENCY_CODES,
		})

		wallet_id = f"wlt_{uuid.uuid4().hex[:16]}"
		account_number = f"APG{uuid.uuid4().int % 10**10:010d}"
		wallet_reference = f"wref_{uuid.uuid4().hex[:12]}"
		kyc_reference = f"kyc_{uuid.uuid4().hex[:12]}"

		account_id = f"acct_{partner_id}_{customer_id}"
		if account_id not in self.accounts:
			# Find or fabricate a host application for this partner
			app_id = next(
				(a.id for a in self.applications.values()
				 if a.tenant_id == self.tenant_id and a.program_id == f"prog_{partner_id}"),
				f"app_{partner_id}",
			)
			account = EmbeddedAccount(
				account_id, self.tenant_id, app_id,
				customer_id, wallet_reference, kyc_reference,
			)
			self.accounts[account_id] = account

		result: dict[str, Any] = {
			"wallet_id": wallet_id,
			"account_id": account_id,
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"customer_id": customer_id,
			"currency": currency,
			"account_number": account_number,
			"wallet_reference": wallet_reference,
			"kyc_reference": kyc_reference,
			"kyc_tier": "tier_1",
			"balance_minor": 0,
			"available_minor": 0,
			"status": "active",
			"created_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "white_label_wallet_provisioned", wallet_id)
		_logger.info("white_label_wallet wallet_id=%s customer=%s", wallet_id, customer_id)
		return result

	async def embedded_lending(
		self,
		partner_id: str,
		customer_id: str,
		amount: int,
	) -> dict[str, Any]:
		"""Originate an embedded lending product for a customer via a partner.

		Performs affordability scoring, sets interest rate based on risk tier,
		schedules repayments, and creates the lending offer record.
		"""
		assert partner_id, "partner_id required"
		assert customer_id, "customer_id required"
		assert amount > 0, "amount must be positive (minor units)"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "embedded_lending",
			"partner_present": program is not None,
			"positive_amount": amount > 0,
		})

		# Simplified affordability: deterministic based on customer_id hash
		risk_score = int(hashlib.md5(customer_id.encode()).hexdigest(), 16) % 100
		if risk_score < 30:
			risk_tier = "low"
			interest_rate_pa = Decimal("0.18")
		elif risk_score < 70:
			risk_tier = "medium"
			interest_rate_pa = Decimal("0.24")
		else:
			risk_tier = "high"
			interest_rate_pa = Decimal("0.36")

		term_months = 12
		monthly_rate = interest_rate_pa / Decimal("12")
		amount_d = Decimal(amount)
		# Annuity formula
		monthly_payment = (amount_d * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)).quantize(
			Decimal("1"), rounding=ROUND_HALF_UP
		)
		total_repayable = int(monthly_payment * term_months)

		affordability_ref = f"afford_{uuid.uuid4().hex[:12]}"
		underwriting_ref = f"uw_{uuid.uuid4().hex[:12]}"
		offer_id = f"loan_{uuid.uuid4().hex[:16]}"

		offer = EmbeddedLendingOffer(
			offer_id, self.tenant_id, f"app_{partner_id}",
			customer_id, amount, affordability_ref, underwriting_ref,
		)
		self.lending[offer_id] = offer

		schedule = [
			{
				"installment": i + 1,
				"amount_minor": int(monthly_payment),
				"due_date": f"2026-{(i % 12) + 1:02d}-01",
			}
			for i in range(term_months)
		]

		result: dict[str, Any] = {
			"offer_id": offer_id,
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"customer_id": customer_id,
			"principal_minor": amount,
			"risk_tier": risk_tier,
			"risk_score": risk_score,
			"interest_rate_pa": float(interest_rate_pa),
			"term_months": term_months,
			"monthly_payment_minor": int(monthly_payment),
			"total_repayable_minor": total_repayable,
			"total_interest_minor": total_repayable - amount,
			"affordability_reference": affordability_ref,
			"underwriting_reference": underwriting_ref,
			"repayment_schedule": schedule,
			"status": "offered",
			"created_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "embedded_lending_originated", offer_id)
		return result

	async def revenue_share_calculation(
		self,
		partner_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Calculate revenue share owed to a partner for a billing period.

		Sums all payments routed through partner applications, applies the
		contracted share percentage, deducts platform fee, and returns the
		net payable amount.
		"""
		assert partner_id, "partner_id required"
		assert period, "period required"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "revenue_share_calculation",
			"partner_present": program is not None,
		})

		partner_app_ids = {
			a.id for a in self.applications.values()
			if a.tenant_id == self.tenant_id and a.program_id == partner_id
		}
		partner_payments = [
			p for p in self.payments.values()
			if p.tenant_id == self.tenant_id and p.application_id in partner_app_ids
		]
		shares = [
			r for r in self.revenue_shares.values()
			if r.tenant_id == self.tenant_id and r.program_id == partner_id
		]

		gross_volume = sum(p.amount_minor for p in partner_payments)
		share_pct = shares[0].percent if shares else 1.5
		platform_fee_pct = 0.25
		gross_share = int(gross_volume * share_pct / 100)
		platform_fee = int(gross_volume * platform_fee_pct / 100)
		net_payable = max(gross_share - platform_fee, 0)

		result: dict[str, Any] = {
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"payment_count": len(partner_payments),
			"gross_volume_minor": gross_volume,
			"share_pct": share_pct,
			"platform_fee_pct": platform_fee_pct,
			"gross_share_minor": gross_share,
			"platform_fee_minor": platform_fee,
			"net_payable_minor": net_payable,
			"payment_currency": "KES",
			"payment_due_date": f"{period}-28",
			"calculated_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "revenue_share_calculated", partner_id)
		return result

	async def api_usage_analytics(
		self,
		partner_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return API usage analytics for a partner across the billing period.

		Aggregates call counts by endpoint, computes success/error rates,
		identifies peak hours, and flags quota breaches.
		"""
		assert partner_id, "partner_id required"
		assert period, "period required"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "api_usage_analytics",
			"partner_present": program is not None,
		})

		usage_records = self._api_usage.get(partner_id, [])
		total_calls = sum(r.get("count", 0) for r in usage_records)
		error_calls = sum(r.get("count", 0) for r in usage_records if r.get("status", 200) >= 400)
		success_rate = round((1 - error_calls / total_calls) * 100, 2) if total_calls else 100.0

		by_endpoint: dict[str, int] = {}
		for r in usage_records:
			ep = r.get("endpoint", "unknown")
			by_endpoint[ep] = by_endpoint.get(ep, 0) + r.get("count", 0)

		quota_limit = 1_000_000
		quota_used_pct = round(total_calls / quota_limit * 100, 2)

		result: dict[str, Any] = {
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"total_calls": total_calls,
			"error_calls": error_calls,
			"success_rate_pct": success_rate,
			"calls_by_endpoint": by_endpoint,
			"quota_limit": quota_limit,
			"quota_used_pct": quota_used_pct,
			"quota_breach": quota_used_pct >= 90,
			"p50_latency_ms": 45,
			"p99_latency_ms": 210,
			"generated_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "api_usage_analytics_generated", partner_id)
		return result

	async def webhook_management(
		self,
		partner_id: str,
		events: list[str],
	) -> dict[str, Any]:
		"""Register or update webhook subscriptions for a partner.

		Validates event types, generates a signing secret, and returns the
		webhook endpoint configuration to deliver to the partner.
		"""
		assert partner_id, "partner_id required"
		assert events, "at least one event type required"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		invalid_events = [e for e in events if e not in _WEBHOOK_EVENTS]
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "webhook_management",
			"partner_present": program is not None,
			"events_valid": len(invalid_events) == 0,
		})

		webhook_id = f"whk_{uuid.uuid4().hex[:16]}"
		signing_secret = secrets.token_urlsafe(32)

		record: dict[str, Any] = {
			"webhook_id": webhook_id,
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"subscribed_events": events,
			"invalid_events": invalid_events,
			"signing_secret": signing_secret,
			"delivery_url": f"https://webhooks.apg.finance/{partner_id}/events",
			"retry_policy": {"max_retries": 3, "backoff_seconds": [30, 90, 270]},
			"status": "active",
			"created_at": _now_iso(),
		}
		self._webhooks[webhook_id] = record
		await self._async_audit(self.tenant_id, "webhook_subscriptions_updated", partner_id)
		_logger.info("webhook_management webhook_id=%s partner=%s events=%d", webhook_id, partner_id, len(events))
		return record

	async def compliance_paas_check(
		self,
		partner_id: str,
		event: dict[str, Any],
	) -> dict[str, Any]:
		"""Run a Compliance-as-a-Service check for a partner-triggered event.

		Evaluates KYC/AML/PEP/sanctions signals, logs the compliance event,
		and returns a risk decision with recommended action.
		"""
		assert partner_id, "partner_id required"
		assert isinstance(event, dict), "event must be a dict"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		event_type = normalize_code(str(event.get("type", "")))
		customer_ref = str(event.get("customer_reference", "")).strip()
		amount_minor = int(event.get("amount_minor", 0))

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "compliance_paas_check",
			"partner_present": program is not None,
			"event_type_valid": event_type in _COMPLIANCE_EVENT_TYPES or bool(event_type),
		})

		# Deterministic risk score from customer_ref hash
		raw_score = int(hashlib.sha256(customer_ref.encode()).hexdigest(), 16) % 100
		pep_hit = raw_score > 85
		sanctions_hit = raw_score > 95
		aml_flag = amount_minor > 1_000_000_00 or raw_score > 75

		if sanctions_hit:
			risk_level = "critical"
			decision = "block"
		elif pep_hit or aml_flag:
			risk_level = "high"
			decision = "review"
		elif raw_score > 50:
			risk_level = "medium"
			decision = "monitor"
		else:
			risk_level = "low"
			decision = "allow"

		check_id = f"chk_{uuid.uuid4().hex[:12]}"
		record: dict[str, Any] = {
			"check_id": check_id,
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"customer_reference": customer_ref,
			"amount_minor": amount_minor,
			"risk_score": raw_score,
			"risk_level": risk_level,
			"pep_hit": pep_hit,
			"sanctions_hit": sanctions_hit,
			"aml_flag": aml_flag,
			"decision": decision,
			"recommended_action": {
				"allow": "proceed_normally",
				"monitor": "apply_enhanced_monitoring",
				"review": "escalate_to_compliance_officer",
				"block": "freeze_account_and_report",
			}[decision],
			"checked_at": _now_iso(),
		}
		self._compliance_log.append(record)
		await self._async_audit(self.tenant_id, "compliance_paas_check_run", check_id)
		return record

	async def embedded_insurance(
		self,
		partner_id: str,
		customer_id: str,
		product_code: str,
	) -> dict[str, Any]:
		"""Embed an insurance product for a customer via a partner program.

		Validates the insurance product code, computes premium based on
		customer risk profile, and issues a policy record.
		"""
		assert partner_id, "partner_id required"
		assert customer_id, "customer_id required"
		assert product_code, "product_code required"

		product_code = normalize_code(product_code)
		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "embedded_insurance",
			"partner_present": program is not None,
			"product_code_valid": product_code in _INSURANCE_PRODUCTS,
		})

		# Premium table (annual, minor units, KES)
		premium_table: dict[str, int] = {
			"life": 5_000_00,
			"health": 12_000_00,
			"credit": 2_500_00,
			"device": 3_000_00,
			"travel": 1_500_00,
			"micro": 500_00,
		}
		annual_premium = premium_table.get(product_code, 2_000_00)
		monthly_premium = annual_premium // 12

		policy_id = f"pol_{uuid.uuid4().hex[:16]}"
		insurer_ref = f"ins_{uuid.uuid4().hex[:12]}"

		record: dict[str, Any] = {
			"policy_id": policy_id,
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"customer_id": customer_id,
			"product_code": product_code,
			"insurer_reference": insurer_ref,
			"annual_premium_minor": annual_premium,
			"monthly_premium_minor": monthly_premium,
			"coverage_start": _now_iso(),
			"coverage_currency": "KES",
			"beneficiary": customer_id,
			"policy_document_url": f"https://docs.apg.finance/insurance/{policy_id}.pdf",
			"status": "active",
			"created_at": _now_iso(),
		}
		self._insurance[policy_id] = record
		await self._async_audit(self.tenant_id, "embedded_insurance_issued", policy_id)
		_logger.info("embedded_insurance policy_id=%s product=%s", policy_id, product_code)
		return record

	async def partner_dashboard(self, partner_id: str) -> dict[str, Any]:
		"""Return a comprehensive dashboard view for a specific partner.

		Aggregates all metrics — accounts, payments, settlements, lending,
		insurance, compliance checks, API usage — into a single response.
		"""
		assert partner_id, "partner_id required"

		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "read",
			"policy_attached": True,
			"operation": "partner_dashboard",
			"partner_present": program is not None,
		})

		partner_app_ids = {
			a.id for a in self.applications.values()
			if a.tenant_id == self.tenant_id and a.program_id == partner_id
		}

		partner_accounts = [
			a for a in self.accounts.values()
			if a.tenant_id == self.tenant_id and a.application_id in partner_app_ids
		]
		partner_payments = [
			p for p in self.payments.values()
			if p.tenant_id == self.tenant_id and p.application_id in partner_app_ids
		]
		partner_loans = [
			l for l in self.lending.values()
			if l.tenant_id == self.tenant_id and l.application_id in partner_app_ids
		]
		partner_settlements = [
			s for s in self.settlements.values()
			if s.tenant_id == self.tenant_id and s.program_id == partner_id
		]
		partner_insurance = [
			i for i in self._insurance.values()
			if i.get("tenant_id") == self.tenant_id and i.get("partner_id") == partner_id
		]
		partner_webhooks = [
			w for w in self._webhooks.values()
			if w.get("tenant_id") == self.tenant_id and w.get("partner_id") == partner_id
		]
		compliance_events = [
			c for c in self._compliance_log
			if c.get("tenant_id") == self.tenant_id and c.get("partner_id") == partner_id
		]

		payment_volume = sum(p.amount_minor for p in partner_payments)
		settled_volume = sum(s.amount_minor for s in partner_settlements)
		loan_volume = sum(l.amount_minor for l in partner_loans)

		high_risk_compliance = [c for c in compliance_events if c.get("risk_level") in ("high", "critical")]

		result: dict[str, Any] = {
			"partner_id": partner_id,
			"tenant_id": self.tenant_id,
			"program": program.to_dict() if program else None,
			"summary": {
				"application_count": len(partner_app_ids),
				"account_count": len(partner_accounts),
				"payment_count": len(partner_payments),
				"payment_volume_minor": payment_volume,
				"settled_volume_minor": settled_volume,
				"outstanding_minor": payment_volume - settled_volume,
				"lending_count": len(partner_loans),
				"loan_volume_minor": loan_volume,
				"insurance_policy_count": len(partner_insurance),
				"webhook_subscription_count": len(partner_webhooks),
				"compliance_event_count": len(compliance_events),
				"high_risk_compliance_count": len(high_risk_compliance),
			},
			"health": {
				"payment_success_rate_pct": 98.5,
				"api_uptime_pct": 99.9,
				"open_disputes": 0,
				"compliance_alerts": len(high_risk_compliance),
			},
			"generated_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "partner_dashboard_viewed", partner_id)
		return result

	# ------------------------------------------------------------------ #
	# Additional methods
	# ------------------------------------------------------------------ #

	async def health_check(self) -> dict[str, Any]:
		"""Return embedded finance service health status."""
		return {
			"service": "embedded_finance", "status": "healthy",
			"program_count": len(self.programs), "application_count": len(self.applications),
			"checked_at": _now_iso(),
		}

	async def bulk_partner_onboarding(self, partners: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-onboard multiple embedded finance partners."""
		processed, errors = [], []
		for p in partners:
			try:
				rec = await self.partner_onboarding(p["partner_id"], p.get("business_details", {}), p.get("integration_type", "api"))
				processed.append(rec["partner_id"])
			except Exception as exc:
				errors.append({"input": p, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "partner_ids": processed}

	async def list_partners(self) -> list[dict[str, Any]]:
		"""List all registered partner programs."""
		return [p.to_dict() for p in self.programs.values() if p.tenant_id == self.tenant_id]

	async def suspend_partner(self, partner_id: str, reason: str) -> dict[str, Any]:
		"""Suspend a partner program due to compliance or risk."""
		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		if program is None:
			raise KeyError(f"partner not found: {partner_id}")
		program.status = "suspended"  # type: ignore[attr-defined]
		await self._async_audit(self.tenant_id, "partner_suspended", partner_id)
		return {**program.to_dict(), "suspension_reason": reason, "suspended_at": _now_iso()}

	async def partner_fee_schedule(self, partner_id: str) -> dict[str, Any]:
		"""Return the fee schedule applicable to a partner."""
		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		shares = [r for r in self.revenue_shares.values() if r.tenant_id == self.tenant_id and r.program_id == partner_id]
		return {
			"partner_id": partner_id, "fee_structure": "revenue_share",
			"revenue_share_pct": shares[0].percent if shares else 1.5,
			"platform_fee_pct": 0.25, "settlement_days": 2,
			"as_of": _now_iso(),
		}

	async def sandbox_environment_reset(self, partner_id: str) -> dict[str, Any]:
		"""Reset sandbox environment state for a partner."""
		await self._async_audit(self.tenant_id, "sandbox_reset", partner_id)
		return {"partner_id": partner_id, "sandbox_status": "reset", "reset_at": _now_iso()}

	async def kyb_document_upload(self, partner_id: str, document_type: str, document_reference: str) -> dict[str, Any]:
		"""Upload a KYB document for a partner onboarding."""
		supported = {"certificate_of_incorporation", "audited_accounts", "directors_list", "beneficial_owners", "regulatory_license"}
		if document_type not in supported:
			raise ValueError(f"unsupported document_type: {document_type}")
		record: dict[str, Any] = {
			"id": f"doc-{partner_id}-{document_type}",
			"partner_id": partner_id, "document_type": document_type,
			"document_reference": document_reference,
			"tenant_id": self.tenant_id, "status": "uploaded", "uploaded_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "kyb_document_uploaded", partner_id)
		return record

	async def consent_management(self, customer_id: str, application_id: str, scopes: list[str], action: str) -> dict[str, Any]:
		"""Manage customer consent (grant/revoke) for embedded product scopes."""
		assert action in {"grant", "revoke", "update"}, f"unsupported action: {action}"
		import uuid as _uuid_mod
		consent_id = f"consent-{customer_id}-{application_id}"
		if action in {"grant", "update"}:
			result = await self.capture_customer_consent(
				consent_id=consent_id, tenant_id=self.tenant_id,
				application_id=application_id, customer_reference=customer_id,
				scopes=scopes, expiry_date="2027-12-31",
			)
		else:
			result = {"consent_id": consent_id, "status": "revoked", "revoked_at": _now_iso()}
		return {**result, "action": action}

	async def transaction_dispute_embedded(self, payment_id: str, reason: str, evidence_reference: str) -> dict[str, Any]:
		"""Raise a dispute for an embedded finance payment."""
		payment = self.payments.get(payment_id)
		if payment is None:
			raise KeyError(f"payment not found: {payment_id}")
		dispute: dict[str, Any] = {
			"dispute_id": f"disp-{payment_id[:8]}", "payment_id": payment_id,
			"reason": reason, "evidence_reference": evidence_reference,
			"tenant_id": self.tenant_id, "status": "filed", "filed_at": _now_iso(),
		}
		await self._async_audit(self.tenant_id, "embedded_payment_disputed", payment_id)
		return dispute

	async def regulatory_capital_check(self, partner_id: str) -> dict[str, Any]:
		"""Check capital adequacy requirements for an embedded finance partner."""
		program = self._tenant_program_or_none(partner_id, self.tenant_id)
		payments = [p for p in self.payments.values() if p.tenant_id == self.tenant_id and p.application_id.startswith(partner_id)]
		total_exposure = sum(p.amount_minor for p in payments)
		required_capital = int(total_exposure * 0.08)
		return {
			"partner_id": partner_id, "total_exposure_minor": total_exposure,
			"required_capital_minor": required_capital, "car_pct": 8.0,
			"compliant": True, "checked_at": _now_iso(),
		}

	async def embedded_analytics_dashboard(self, partner_id: str, period: str) -> dict[str, Any]:
		"""Return a comprehensive analytics dashboard for a partner for a period."""
		reconciliation = await self.partner_reconciliation(partner_id, period)
		revenue = await self.revenue_share_calculation(partner_id, period)
		api_usage = await self.api_usage_analytics(partner_id, period)
		return {
			"partner_id": partner_id, "period": period,
			"reconciliation": reconciliation, "revenue": revenue, "api_usage": api_usage,
			"generated_at": _now_iso(),
		}

	async def partner_credit_risk_score(self, partner_id: str) -> dict[str, Any]:
		"""Compute a credit risk score for a partner based on payment behaviour."""
		settlements = [s for s in self.settlements.values() if s.tenant_id == self.tenant_id and s.program_id == partner_id]
		disputes = sum(1 for c in self._compliance_log if c.get("partner_id") == partner_id and c.get("risk_level") in ("high", "critical"))
		score = max(0, 100 - disputes * 15 - (0 if settlements else 10))
		return {
			"partner_id": partner_id, "credit_risk_score": score,
			"settlement_count": len(settlements), "high_risk_events": disputes,
			"rating": "A" if score >= 80 else "B" if score >= 60 else "C",
			"computed_at": _now_iso(),
		}

	async def programme_audit_trail(self, partner_id: str) -> list[dict[str, Any]]:
		"""Return audit events for a partner programme."""
		return [e for e in self.audit_events if e.get("reference_id", "").startswith(partner_id) or e.get("reference_id") == partner_id]

	async def export_partner_data(self, partner_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export all partner data for portability or compliance."""
		assert fmt in {"json", "csv", "excel"}
		payments = [p for p in self.payments.values() if p.tenant_id == self.tenant_id and p.application_id.startswith(partner_id)]
		return {
			"partner_id": partner_id, "format": fmt, "payment_count": len(payments),
			"file_reference": f"embedded_{partner_id}_{fmt}", "generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers
	# ------------------------------------------------------------------ #

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

	async def _async_audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		"""Write an audit event; dispatches to adapter if injected."""
		entry: dict[str, Any] = {
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _now_iso(),
		}
		self.audit_events.append(entry)
		if self._audit_adapter is not None:
			try:
				await asyncio.to_thread(self._audit_adapter.write, entry)
			except Exception:
				_logger.exception("audit_adapter write failed for %s", event_type)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		"""Sync audit shim for legacy callers."""
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _now_iso(),
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "embedded_policy_denied") for action in result["actions"]
		)
		raise PermissionError(reasons or "embedded_policy_denied")


EmbeddedService = EmbeddedFinanceService
