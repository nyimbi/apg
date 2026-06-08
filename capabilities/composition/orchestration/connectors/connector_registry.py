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
