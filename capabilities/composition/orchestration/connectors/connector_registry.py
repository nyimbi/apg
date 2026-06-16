"""APG Connector Registry.

Provides discovery and instantiation of all available APG connectors.
Connectors are declared in CONNECTORS_MANIFEST and can be installed/listed
via the APG CLI: `apg connector list`, `apg connector install mpesa`.

Usage::

    registry = ConnectorRegistry()
    connector = registry.get("mpesa", tenant_id="my-tenant", user_id="admin")
    await connector.initialize()
    result = await connector.stk_push(100, "254712345678", "ORDER-123")
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# Connector manifest — add new connectors here
CONNECTORS_MANIFEST: dict[str, dict[str, Any]] = {
	"mpesa": {
		"display_name": "MPESA (Safaricom Daraja 2.0)",
		"description": "East Africa's dominant mobile money platform. Kenya, Tanzania, Uganda, Ghana.",
		"category": "payment",
		"regions": ["KE", "TZ", "UG", "GH", "RW", "MZ", "ET", "LS"],
		"module": "capabilities.composition.orchestration.connectors.africa.mpesa_connector",
		"class": "MPESAConnector",
		"config_class": "MPESAConfiguration",
		"env_factory": "mpesa_connector_from_env",
		"required_env": ["MPESA_CONSUMER_KEY", "MPESA_CONSUMER_SECRET", "MPESA_SHORTCODE"],
		"optional_env": ["MPESA_PASSKEY", "MPESA_ENV", "MPESA_INITIATOR_NAME", "MPESA_CALLBACK_URL_BASE"],
		"docs_url": "https://developer.safaricom.co.ke/APIs",
	},
	"stripe": {
		"display_name": "Stripe",
		"description": "Global payment processing. Cards, subscriptions, payouts, refunds, disputes.",
		"category": "payment",
		"regions": ["global"],
		"module": "capabilities.composition.orchestration.connectors.stripe_connector",
		"class": "StripeConnector",
		"config_class": "StripeConfiguration",
		"env_factory": "stripe_connector_from_env",
		"required_env": ["STRIPE_SECRET_KEY"],
		"optional_env": ["STRIPE_WEBHOOK_SECRET", "STRIPE_API_VERSION"],
		"docs_url": "https://stripe.com/docs/api",
	},
	"equity_bank": {
		"display_name": "Equity Bank",
		"description": "Equity Bank APIs — East Africa's largest bank by customer count. Account inquiry, PesaLink, MPESA↔Equity transfers, standing orders.",
		"category": "banking",
		"regions": ["KE", "UG", "TZ", "RW", "CD", "SS"],
		"module": "capabilities.composition.orchestration.connectors.africa.equity_connector",
		"class": "EquityBankConnector",
		"config_class": "EquityBankConfiguration",
		"env_factory": "equity_connector_from_env",
		"required_env": ["EQUITY_CLIENT_ID", "EQUITY_CLIENT_SECRET"],
		"optional_env": ["EQUITY_ENV", "EQUITY_MERCHANT_CODE"],
		"docs_url": "https://developer.equitybankgroup.com/",
	},
	"kcb": {
		"display_name": "KCB Bank",
		"description": "Kenya Commercial Bank APIs — corporate banking, bulk payroll, MPESA↔KCB transfers.",
		"category": "banking",
		"regions": ["KE", "UG", "TZ", "RW", "ET", "SS", "BI"],
		"module": "capabilities.composition.orchestration.connectors.africa.kcb_connector",
		"class": "KCBConnector",
		"config_class": "KCBConfiguration",
		"env_factory": "kcb_connector_from_env",
		"required_env": ["KCB_CONSUMER_KEY", "KCB_CONSUMER_SECRET"],
		"optional_env": ["KCB_ENV", "KCB_SHORTCODE"],
		"docs_url": "https://developer.kcbgroup.com/",
	},
	"mtn_momo": {
		"display_name": "MTN Mobile Money (MoMo)",
		"description": "MTN MoMo API — collections, disbursements, KYC. West & Central Africa's largest telco. NG, GH, UG, CM, CI, ZM.",
		"category": "payment",
		"regions": ["NG", "GH", "UG", "CM", "CI", "ZM", "RW", "ZW", "SZ", "BJ", "GN", "LR", "MG"],
		"module": "capabilities.composition.orchestration.connectors.africa.mtn_connector",
		"class": "MTNConnector",
		"config_class": "MTNConfiguration",
		"env_factory": "mtn_connector_from_env",
		"required_env": ["MTN_API_USER_ID", "MTN_API_KEY", "MTN_SUBSCRIPTION_KEY"],
		"optional_env": ["MTN_ENVIRONMENT", "MTN_TARGET_ENVIRONMENT", "MTN_CALLBACK_URL_BASE"],
		"docs_url": "https://momodeveloper.mtn.com/",
	},
	"airtel_money": {
		"display_name": "Airtel Money",
		"description": "Airtel Africa Payment API — C2B collections, B2C disbursements, balance, KYC. KE, UG, TZ, RW, ZM, MG, CD.",
		"category": "payment",
		"regions": ["KE", "UG", "TZ", "RW", "ZM", "MG", "CD", "MW", "NE", "TD", "GH", "CG", "SL"],
		"module": "capabilities.composition.orchestration.connectors.africa.airtel_connector",
		"class": "AirtelConnector",
		"config_class": "AirtelConfiguration",
		"env_factory": "airtel_connector_from_env",
		"required_env": ["AIRTEL_CLIENT_ID", "AIRTEL_CLIENT_SECRET", "AIRTEL_COUNTRY", "AIRTEL_CURRENCY"],
		"optional_env": ["AIRTEL_ENV", "AIRTEL_CALLBACK_URL_BASE"],
		"docs_url": "https://developers.airtel.africa/documentation",
	},
	"airtel_money_v2": {
		"display_name": "Airtel Money v2 (AirtelMoneyConnector)",
		"description": (
			"Production-quality Airtel Africa connector — send_money (B2C), request_payment (C2B), "
			"check_balance, transaction_status. Markets: KE, UG, TZ, RW, ZM. "
			"All calls wrapped in ConnectorError; OAuth2 token auto-refreshed."
		),
		"category": "payment",
		"regions": ["KE", "UG", "TZ", "RW", "ZM"],
		"module": "capabilities.composition.orchestration.connectors.airtel_connector",
		"class": "AirtelMoneyConnector",
		"config_class": "AirtelMoneyConfiguration",
		"env_factory": "airtel_money_connector_from_env",
		"required_env": ["AIRTEL_CLIENT_ID", "AIRTEL_CLIENT_SECRET", "AIRTEL_COUNTRY", "AIRTEL_CURRENCY"],
		"optional_env": ["AIRTEL_ENV", "AIRTEL_CALLBACK_URL_BASE"],
		"docs_url": "https://developers.airtel.africa/documentation",
	},
	"orange_money": {
		"display_name": "Orange Money",
		"description": "Orange Money Web Payment & Cashout — Francophone West Africa. CI, SN, CM, ML, BF, MG, NE.",
		"category": "payment",
		"regions": ["CI", "SN", "CM", "ML", "BF", "MG", "NE", "GN", "LR", "SL"],
		"module": "capabilities.composition.orchestration.connectors.africa.orange_connector",
		"class": "OrangeConnector",
		"config_class": "OrangeConfiguration",
		"env_factory": "orange_connector_from_env",
		"required_env": ["ORANGE_CLIENT_ID", "ORANGE_CLIENT_SECRET", "ORANGE_MERCHANT_KEY", "ORANGE_COUNTRY"],
		"optional_env": ["ORANGE_ENV", "ORANGE_CALLBACK_URL_BASE"],
		"docs_url": "https://developer.orange.com/apis/om-webpay-prod/getting-started",
	},
	"orange_money_v2": {
		"display_name": "Orange Money (OrangeMoneyConnector)",
		"description": "Orange Money send_money, request_payment, check_balance, transaction_status. Francophone West Africa: CI, SN, CM, ML, BF.",
		"category": "payment",
		"regions": ["CI", "SN", "CM", "ML", "BF"],
		"module": "capabilities.composition.orchestration.connectors.orange_connector",
		"class": "OrangeMoneyConnector",
		"config_class": "OrangeMoneyConfiguration",
		"env_factory": "orange_money_connector_from_env",
		"required_env": ["ORANGE_MONEY_CLIENT_ID", "ORANGE_MONEY_CLIENT_SECRET", "ORANGE_MONEY_MERCHANT_KEY", "ORANGE_MONEY_COUNTRY"],
		"optional_env": ["ORANGE_MONEY_ENV", "ORANGE_MONEY_CALLBACK_URL"],
		"docs_url": "https://developer.orange.com/apis/om-webpay-prod/getting-started",
	},
	"mtn_momo_v2": {
		"display_name": "MTN MoMo (MTNMoMoConnector)",
		"description": (
			"Production-quality MTN Mobile Money connector — send_money (B2C), request_payment (C2B), "
			"check_balance, transaction_status. Markets: NG, GH, UG, CM, CI, ZM. "
			"Three-part credential scheme (subscription key + API user + API key). "
			"OAuth2 Basic Auth token auto-refreshed."
		),
		"category": "payment",
		"regions": ["NG", "GH", "UG", "CM", "CI", "ZM"],
		"module": "capabilities.composition.orchestration.connectors.mtn_connector",
		"class": "MTNMoMoConnector",
		"config_class": "MTNMoMoConfiguration",
		"env_factory": "mtn_momo_connector_from_env",
		"required_env": ["MTN_SUBSCRIPTION_KEY", "MTN_API_USER", "MTN_API_KEY", "MTN_COUNTRY"],
		"optional_env": ["MTN_ENV", "MTN_CALLBACK_URL_BASE", "MTN_CURRENCY"],
		"docs_url": "https://momodeveloper.mtn.com/docs/services/collection",
	},
	"wave_v2": {
		"display_name": "Wave Mobile Money (WaveConnector)",
		"description": (
			"Production-quality Wave connector — send_money (B2C payout), request_payment (C2B checkout), "
			"check_balance, transaction_status. Markets: SN, CI, ML, BF, GN. API-key auth, no OAuth2 flow."
		),
		"category": "payment",
		"regions": ["SN", "CI", "ML", "BF", "GN"],
		"module": "capabilities.composition.orchestration.connectors.wave_connector",
		"class": "WaveConnector",
		"config_class": "WaveConfiguration",
		"env_factory": "wave_connector_from_env",
		"required_env": ["WAVE_API_KEY", "WAVE_COUNTRY"],
		"optional_env": ["WAVE_ENV", "WAVE_CALLBACK_URL_BASE", "WAVE_CURRENCY"],
		"docs_url": "https://docs.wave.com/",
	},
	"mshwari_v2": {
		"display_name": "M-Shwari (MShwariConnector)",
		"description": (
			"Production-quality M-Shwari connector (CBA/Safaricom) — lock_savings, loan_apply, loan_repay, "
			"check_balance. Kenya only. Uses Daraja B2C + STK Push APIs."
		),
		"category": "payment",
		"regions": ["KE"],
		"module": "capabilities.composition.orchestration.connectors.mshwari_connector",
		"class": "MShwariConnector",
		"config_class": "MShwariConfiguration",
		"env_factory": "mshwari_connector_from_env",
		"required_env": [
			"MSHWARI_CONSUMER_KEY", "MSHWARI_CONSUMER_SECRET", "MSHWARI_SHORTCODE",
			"MSHWARI_INITIATOR_NAME", "MSHWARI_SECURITY_CREDENTIAL",
		],
		"optional_env": ["MSHWARI_ENV", "MSHWARI_CALLBACK_URL_BASE"],
		"docs_url": "https://developer.safaricom.co.ke/APIs",
	},
	"wave": {
		"display_name": "Wave Mobile Money",
		"description": "Wave Business Payments API — fast-growing mobile money in Francophone West Africa. SN, CI, ML, BF, GN.",
		"category": "payment",
		"regions": ["SN", "CI", "ML", "BF", "GN"],
		"module": "capabilities.composition.orchestration.connectors.africa.wave_connector",
		"class": "WaveConnector",
		"config_class": "WaveConfiguration",
		"env_factory": "wave_connector_from_env",
		"required_env": ["WAVE_API_KEY"],
		"optional_env": ["WAVE_ENV", "WAVE_CALLBACK_URL_BASE"],
		"docs_url": "https://www.wave.com/en/business/api/",
	},
	"mshwari": {
		"display_name": "M-Shwari (CBA + Safaricom)",
		"description": "M-Shwari savings lock/unlock and micro-loan products via Safaricom Daraja API. Kenya only.",
		"category": "payment",
		"regions": ["KE"],
		"module": "capabilities.composition.orchestration.connectors.africa.mshwari_connector",
		"class": "MShwariConnector",
		"config_class": "MShwariConfiguration",
		"env_factory": "mshwari_connector_from_env",
		"required_env": ["MSHWARI_CONSUMER_KEY", "MSHWARI_CONSUMER_SECRET", "MSHWARI_SHORTCODE"],
		"optional_env": ["MSHWARI_PASSKEY", "MSHWARI_INITIATOR_NAME", "MSHWARI_INITIATOR_PASSWORD", "MSHWARI_ENV", "MSHWARI_CALLBACK_URL_BASE"],
		"docs_url": "https://developer.safaricom.co.ke/APIs",
	},
	"cbk_rtgs": {
		"display_name": "CBK Kenya RTGS (KEPSS)",
		"description": "Kenya Electronic Payments and Settlement System — CBK-operated large-value interbank RTGS. Same-day KES settlement for licensed financial institutions.",
		"category": "interbank",
		"regions": ["KE"],
		"module": "capabilities.composition.orchestration.connectors.africa.cbk_rtgs_connector",
		"class": "CBKRTGSConnector",
		"config_class": "CBKRTGSConfiguration",
		"env_factory": "cbk_rtgs_connector_from_env",
		"required_env": ["KEPSS_PARTICIPANT_CODE", "KEPSS_CERTIFICATE_PATH", "KEPSS_BIC_CODE"],
		"optional_env": ["KEPSS_CERTIFICATE_PASSWORD", "KEPSS_ENV"],
		"docs_url": "https://kepss.centralbank.go.ke",
	},
	"nibss": {
		"display_name": "NIBSS Nigeria (NIP + NEFT)",
		"description": "Nigeria Interbank Settlement System — NIP instant real-time transfers and NEFT batch clearing. OAuth2 + HMAC-SHA256 signed requests.",
		"category": "interbank",
		"regions": ["NG"],
		"module": "capabilities.composition.orchestration.connectors.africa.nibss_connector",
		"class": "NIBSSConnector",
		"config_class": "NIBSSConfiguration",
		"env_factory": "nibss_connector_from_env",
		"required_env": ["NIBSS_INSTITUTION_CODE", "NIBSS_CLIENT_ID", "NIBSS_CLIENT_SECRET", "NIBSS_HMAC_KEY"],
		"optional_env": ["NIBSS_ENV"],
		"docs_url": "https://nibss-plc.com.ng/developer",
	},
	"bceao": {
		"display_name": "BCEAO STAR-UEMOA (West Africa)",
		"description": "Banque Centrale des États de l'Afrique de l'Ouest — STAR-UEMOA interbank system for 8 West African countries (CI, SN, ML, BF, BJ, NE, TG, GW). Currency: XOF.",
		"category": "interbank",
		"regions": ["CI", "SN", "ML", "BF", "BJ", "NE", "TG", "GW"],
		"module": "capabilities.composition.orchestration.connectors.africa.bceao_connector",
		"class": "BCEAOConnector",
		"config_class": "BCEAOConfiguration",
		"env_factory": "bceao_connector_from_env",
		"required_env": ["BCEAO_PARTICIPANT_CODE", "BCEAO_API_KEY", "BCEAO_INSTITUTION_BIC", "BCEAO_COUNTRY_CODE"],
		"optional_env": ["BCEAO_ENV"],
		"docs_url": "https://star-uemoa.bceao.int",
	},
	"salesforce": {
		"display_name": "Salesforce",
		"description": "Salesforce REST API — contacts, leads, opportunities, accounts, cases, SOQL queries.",
		"category": "crm",
		"regions": ["global"],
		"module": "capabilities.composition.orchestration.connectors.salesforce_connector",
		"class": "SalesforceConnector",
		"config_class": "SalesforceConfiguration",
		"env_factory": "salesforce_connector_from_env",
		"required_env": ["SFDC_CLIENT_ID", "SFDC_CLIENT_SECRET", "SFDC_USERNAME", "SFDC_PASSWORD"],
		"optional_env": ["SFDC_ENV"],
		"docs_url": "https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/",
	},
	"whatsapp": {
		"display_name": "WhatsApp Business API",
		"description": "WhatsApp Business Cloud API — text, templates, interactive buttons, media. 2B+ users in Africa.",
		"category": "messaging",
		"regions": ["global"],
		"module": "capabilities.composition.orchestration.connectors.whatsapp_connector",
		"class": "WhatsAppConnector",
		"config_class": "WhatsAppConfiguration",
		"env_factory": "whatsapp_connector_from_env",
		"required_env": ["WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID"],
		"optional_env": ["WHATSAPP_BUSINESS_ACCOUNT_ID", "WHATSAPP_API_VERSION"],
		"docs_url": "https://developers.facebook.com/docs/whatsapp/cloud-api",
	},
}


class ConnectorRegistry:
	"""Registry of available and installed APG connectors."""

	def list_available(self) -> list[dict[str, Any]]:
		"""Return all available connectors with their metadata."""
		return [
			{
				"id": connector_id,
				"display_name": meta.get("display_name", connector_id),
				"description": meta.get("description", ""),
				"category": meta.get("category", ""),
				"regions": meta.get("regions", []),
				"status": meta.get("status", "available"),
			}
			for connector_id, meta in CONNECTORS_MANIFEST.items()
		]

	def list_installed(self) -> list[str]:
		"""Return IDs of connectors whose Python module is importable."""
		installed = []
		for connector_id, meta in CONNECTORS_MANIFEST.items():
			if meta.get("status") == "planned":
				continue
			module_path = meta.get("module", "")
			try:
				__import__(module_path)
				installed.append(connector_id)
			except ImportError:
				pass
		return installed

	def get_metadata(self, connector_id: str) -> dict[str, Any] | None:
		"""Return metadata for a connector by ID."""
		return CONNECTORS_MANIFEST.get(connector_id)

	def get(
		self,
		connector_id: str,
		tenant_id: str,
		user_id: str = "system",
		**config_kwargs: Any,
	) -> Any:
		"""Instantiate a connector by ID.

		Loads the connector class from the manifest and constructs it.
		For connectors with an env_factory, prefer mpesa_connector_from_env()
		when environment variables are already set.

		Args:
			connector_id: Connector identifier (e.g. "mpesa")
			tenant_id: APG tenant ID
			user_id: User configuring the connector
			**config_kwargs: Additional configuration fields passed to the
			                  connector's Configuration model.

		Returns:
			Instantiated (but not yet initialized) connector instance.

		Raises:
			KeyError: if connector_id is not in the manifest
			ImportError: if the connector's module is not installed
		"""
		meta = CONNECTORS_MANIFEST.get(connector_id)
		if meta is None:
			available = list(CONNECTORS_MANIFEST)
			raise KeyError(f"Unknown connector: {connector_id!r}. Available: {available}")

		if meta.get("status") == "planned":
			raise ImportError(
				f"Connector {connector_id!r} is planned but not yet implemented. "
				"Check the APG roadmap for the implementation timeline."
			)

		module = __import__(meta["module"], fromlist=[meta["class"], meta.get("config_class", "")])
		connector_cls = getattr(module, meta["class"])
		config_cls = getattr(module, meta["config_class"])
		config = config_cls(tenant_id=tenant_id, user_id=user_id, **config_kwargs)
		return connector_cls(config)

	def from_env(self, connector_id: str, tenant_id: str, user_id: str = "system") -> Any:
		"""Construct a connector from environment variables using its env_factory.

		Raises KeyError if connector not found, ImportError if not installed,
		KeyError/ValueError if required environment variables are missing.
		"""
		meta = CONNECTORS_MANIFEST.get(connector_id)
		if meta is None:
			raise KeyError(f"Unknown connector: {connector_id!r}")

		env_factory_name = meta.get("env_factory")
		if not env_factory_name:
			raise NotImplementedError(f"Connector {connector_id!r} has no env_factory defined")

		module = __import__(meta["module"], fromlist=[env_factory_name])
		factory = getattr(module, env_factory_name)
		return factory(tenant_id=tenant_id, user_id=user_id)


# Module-level singleton
_registry: ConnectorRegistry | None = None


def get_connector_registry() -> ConnectorRegistry:
	"""Return the module-level ConnectorRegistry singleton."""
	global _registry
	if _registry is None:
		_registry = ConnectorRegistry()
	return _registry
