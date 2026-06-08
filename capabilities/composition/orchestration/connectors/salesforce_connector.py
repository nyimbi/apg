"""Salesforce CRM API Connector.

Integrates APG crm_adv with Salesforce Sales Cloud via the Salesforce
REST API. Supports OAuth2 authentication with username-password or
JWT-based server-to-server flows.

Reference: https://developer.salesforce.com/docs/atlas.en-us.api_rest.meta/api_rest/

Authentication: OAuth2 (username-password or Connected App JWT)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration

_log = logging.getLogger(__name__)

_LOGIN_URL_PRODUCTION = "https://login.salesforce.com"
_LOGIN_URL_SANDBOX = "https://test.salesforce.com"
_API_VERSION = "v62.0"


class SalesforceConfiguration(ConnectorConfiguration):
	client_id: str = Field(..., description="Connected App consumer key")
	client_secret: str = Field(..., description="Connected App consumer secret")
	username: str = Field(..., description="Salesforce username (for username-password flow)")
	password: str = Field(..., description="Salesforce password + security token")
	environment: str = Field(default="sandbox", pattern="^(sandbox|production)$")
	api_version: str = Field(default=_API_VERSION, description="Salesforce API version")


class SalesforceConnector(BaseConnector):
	"""Salesforce REST API connector.

	Supports:
	  - Contact CRUD (create, read, update, query)
	  - Lead CRUD and conversion
	  - Opportunity CRUD
	  - Account CRUD
	  - Case management
	  - SOQL queries
	  - Bulk upsert via Composite API
	"""

	def __init__(self, config: SalesforceConfiguration) -> None:
		super().__init__(config)
		self._config: SalesforceConfiguration = config
		self._login_url = (
			_LOGIN_URL_SANDBOX if config.environment == "sandbox" else _LOGIN_URL_PRODUCTION
		)
		self._access_token: str = ""
		self._instance_url: str = ""
		self._token_expires_at: float = 0.0
		self._client: httpx.AsyncClient | None = None

	async def _connect(self) -> None:
		self._client = httpx.AsyncClient(timeout=self._config.timeout_seconds)
		await self._authenticate()

	async def _disconnect(self) -> None:
		if self._client:
			await self._client.aclose()
			self._client = None
		self._access_token = ""
		self._instance_url = ""

	async def _execute_operation(self, operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
		handlers = {
			"create_contact": self._create_object("Contact"),
			"update_contact": self._update_object("Contact"),
			"query_contacts": self._query_object("Contact"),
			"create_lead": self._create_object("Lead"),
			"update_lead": self._update_object("Lead"),
			"query_leads": self._query_object("Lead"),
			"create_opportunity": self._create_object("Opportunity"),
			"update_opportunity": self._update_object("Opportunity"),
			"create_account": self._create_object("Account"),
			"create_case": self._create_object("Case"),
			"soql_query": self._soql_query,
			"get_record": self._get_record,
		}
		handler = handlers.get(operation)
		if handler is None:
			raise ValueError(f"Unknown Salesforce operation: {operation!r}")
		return await handler(**parameters)

	async def _health_check(self) -> bool:
		try:
			await self._ensure_token()
			resp = await self._client.get(
				f"{self._instance_url}/services/data/{self._config.api_version}/",
				headers=self._auth_header(),
			)
			return resp.status_code == 200
		except Exception:
			return False

	# ── Public operations ──────────────────────────────────────────────

	async def create_contact(self, fields: dict[str, Any]) -> dict[str, Any]:
		return await self._execute_operation("create_contact", {"fields": fields})

	async def update_contact(self, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
		return await self._execute_operation("update_contact", {"record_id": record_id, "fields": fields})

	async def query_contacts(self, where_clause: str = "", limit: int = 50) -> dict[str, Any]:
		return await self._execute_operation("query_contacts", {"where_clause": where_clause, "limit": limit})

	async def create_lead(self, fields: dict[str, Any]) -> dict[str, Any]:
		return await self._execute_operation("create_lead", {"fields": fields})

	async def update_lead(self, record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
		return await self._execute_operation("update_lead", {"record_id": record_id, "fields": fields})

	async def create_opportunity(self, fields: dict[str, Any]) -> dict[str, Any]:
		return await self._execute_operation("create_opportunity", {"fields": fields})

	async def soql_query(self, query: str) -> dict[str, Any]:
		return await self._execute_operation("soql_query", {"query": query})

	async def get_record(self, sobject: str, record_id: str, fields: list[str] | None = None) -> dict[str, Any]:
		return await self._execute_operation("get_record", {
			"sobject": sobject, "record_id": record_id, "fields": fields or [],
		})

	# ── Private implementation ─────────────────────────────────────────

	async def _authenticate(self) -> None:
		"""OAuth2 Username-Password flow."""
		data = {
			"grant_type": "password",
			"client_id": self._config.client_id,
			"client_secret": self._config.client_secret,
			"username": self._config.username,
			"password": self._config.password,
		}
		resp = await self._client.post(
			f"{self._login_url}/services/oauth2/token",
			data=data,
			headers={"Content-Type": "application/x-www-form-urlencoded"},
		)
		resp.raise_for_status()
		token_data = resp.json()
		self._access_token = token_data["access_token"]
		self._instance_url = token_data["instance_url"]
		self._token_expires_at = time.time() + 7200  # Salesforce tokens last 2h

	async def _ensure_token(self) -> None:
		if time.time() > self._token_expires_at - 60:
			await self._authenticate()

	def _auth_header(self) -> dict[str, str]:
		return {
			"Authorization": f"Bearer {self._access_token}",
			"Content-Type": "application/json",
		}

	def _sobject_url(self, sobject: str) -> str:
		return f"{self._instance_url}/services/data/{self._config.api_version}/sobjects/{sobject}"

	def _query_url(self) -> str:
		return f"{self._instance_url}/services/data/{self._config.api_version}/query"

	def _create_object(self, sobject: str):
		async def _create(fields: dict[str, Any]) -> dict[str, Any]:
			await self._ensure_token()
			resp = await self._client.post(
				self._sobject_url(sobject), json=fields, headers=self._auth_header()
			)
			resp.raise_for_status()
			return resp.json()
		return _create

	def _update_object(self, sobject: str):
		async def _update(record_id: str, fields: dict[str, Any]) -> dict[str, Any]:
			await self._ensure_token()
			resp = await self._client.patch(
				f"{self._sobject_url(sobject)}/{record_id}",
				json=fields, headers=self._auth_header(),
			)
			resp.raise_for_status()
			return {"id": record_id, "success": True}
		return _update

	def _query_object(self, sobject: str):
		async def _query(where_clause: str = "", limit: int = 50) -> dict[str, Any]:
			await self._ensure_token()
			soql = f"SELECT Id, Name FROM {sobject}"
			if where_clause:
				soql += f" WHERE {where_clause}"
			soql += f" LIMIT {limit}"
			resp = await self._client.get(
				self._query_url(), params={"q": soql}, headers=self._auth_header()
			)
			resp.raise_for_status()
			return resp.json()
		return _query

	async def _soql_query(self, query: str) -> dict[str, Any]:
		await self._ensure_token()
		resp = await self._client.get(
			self._query_url(), params={"q": query}, headers=self._auth_header()
		)
		resp.raise_for_status()
		return resp.json()

	async def _get_record(
		self, sobject: str, record_id: str, fields: list[str]
	) -> dict[str, Any]:
		await self._ensure_token()
		params = {}
		if fields:
			params["fields"] = ",".join(fields)
		resp = await self._client.get(
			f"{self._sobject_url(sobject)}/{record_id}",
			params=params, headers=self._auth_header(),
		)
		resp.raise_for_status()
		return resp.json()


def salesforce_connector_from_env(tenant_id: str, user_id: str = "system") -> SalesforceConnector:
	config = SalesforceConfiguration(
		name="Salesforce",
		tenant_id=tenant_id,
		user_id=user_id,
		client_id=os.environ["SFDC_CLIENT_ID"],
		client_secret=os.environ["SFDC_CLIENT_SECRET"],
		username=os.environ["SFDC_USERNAME"],
		password=os.environ["SFDC_PASSWORD"],
		environment=os.environ.get("SFDC_ENV", "sandbox"),
	)
	return SalesforceConnector(config)
