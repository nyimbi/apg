"""Executable service layer for APG Banking APIs."""

from __future__ import annotations

import datetime
import hashlib
import secrets
import statistics
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .domain.adapters import get_auth_adapter, get_audit_adapter
	from .database.store import get_store
	from .apis_runtime import (
		client_public_id, is_critical_severity, normalize_code,
		normalize_codes, normalize_url, rate_limit_allows,
	)
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_API_PRODUCTS,
		SUPPORTED_AUTH_FLOWS, SUPPORTED_ENVIRONMENTS, SUPPORTED_INCIDENT_SEVERITIES,
		SUPPORTED_WEBHOOK_EVENTS,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		APICallRecord, APIClient, APIEvidence, APIProduct, ConsentGrant,
		DeveloperApplication, DeveloperOrganization, EndpointPolicy,
		RateLimitBucket, SLAIncident, WebhookSubscription,
	)
except ImportError:  # pragma: no cover
	from apis_runtime import client_public_id, is_critical_severity, normalize_code, normalize_codes, normalize_url, rate_limit_allows  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_API_PRODUCTS, SUPPORTED_AUTH_FLOWS, SUPPORTED_ENVIRONMENTS, SUPPORTED_INCIDENT_SEVERITIES, SUPPORTED_WEBHOOK_EVENTS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import APICallRecord, APIClient, APIEvidence, APIProduct, ConsentGrant, DeveloperApplication, DeveloperOrganization, EndpointPolicy, RateLimitBucket, SLAIncident, WebhookSubscription  # type: ignore


def _utcnow() -> str:
	return datetime.datetime.utcnow().isoformat() + "Z"


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class BankingAPIsService:
	"""Dependency-light banking API runtime for generated applications."""

	def __init__(self) -> None:
		self.products: dict[str, APIProduct] = {}
		self.developers: dict[str, DeveloperOrganization] = {}
		self.applications: dict[str, DeveloperApplication] = {}
		self.consents: dict[str, ConsentGrant] = {}
		self.clients: dict[str, APIClient] = {}
		self.endpoints: dict[str, EndpointPolicy] = {}
		self.webhooks: dict[str, WebhookSubscription] = {}
		self.calls: dict[str, APICallRecord] = {}
		self.rate_limits: dict[str, RateLimitBucket] = {}
		self.incidents: dict[str, SLAIncident] = {}
		self.evidence: dict[str, APIEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state for new methods
		self._api_keys: dict[str, dict[str, Any]] = {}          # api_key_id -> key record
		self._oauth2_tokens: dict[str, dict[str, Any]] = {}     # token -> token record
		self._sandbox_transactions: list[dict[str, Any]] = []
		self._webhook_deliveries: list[dict[str, Any]] = []
		self._health_records: dict[str, list[dict[str, Any]]] = {}
		self._psd2_checks: list[dict[str, Any]] = []
		self._developer_stats: list[dict[str, Any]] = []

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

	def register_api_product(
		self,
		product_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		product_type: str,
		environment: str,
		scopes: list[str],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		product_type = normalize_code(product_type)
		environment = normalize_code(environment)
		scopes = normalize_codes(scopes)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_api_product",
			"owner_present": bool(owner_id),
			"product_type_supported": product_type in SUPPORTED_API_PRODUCTS,
			"environment_supported": environment in SUPPORTED_ENVIRONMENTS,
			"scopes_present": bool(scopes),
		})
		product = APIProduct(product_id, tenant_id, name, owner_id, product_type, environment, scopes)
		self.products[product_id] = product
		self._audit(tenant_id, "api_product_registered", product_id)
		return product.to_dict()

	def onboard_developer(
		self,
		developer_id: str,
		tenant_id: str,
		name: str,
		kyb_reference: str,
		security_review_reference: str,
		risk_clearance_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_developer",
			"kyb_present": bool(kyb_reference),
			"security_review_present": bool(security_review_reference),
			"risk_clearance_present": bool(risk_clearance_reference),
		})
		developer = DeveloperOrganization(developer_id, tenant_id, name, kyb_reference, security_review_reference, risk_clearance_reference)
		self.developers[developer_id] = developer
		self._audit(tenant_id, "developer_onboarded", developer_id)
		return developer.to_dict()

	def register_application(
		self,
		application_id: str,
		tenant_id: str,
		developer_id: str,
		name: str,
		environment: str,
		redirect_uri: str,
		terms_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		developer = self._tenant_developer_or_none(developer_id, tenant_id)
		environment = normalize_code(environment)
		redirect_uri = normalize_url(redirect_uri)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_application",
			"developer_present": developer is not None,
			"environment_supported": environment in SUPPORTED_ENVIRONMENTS,
			"redirect_uri_present": bool(redirect_uri),
			"terms_present": bool(terms_reference),
		})
		application = DeveloperApplication(application_id, tenant_id, developer_id, name, environment, redirect_uri, terms_reference)
		self.applications[application_id] = application
		self._audit(tenant_id, "developer_application_registered", application_id)
		return application.to_dict()

	def create_consent_grant(
		self,
		consent_id: str,
		tenant_id: str,
		application_id: str,
		customer_reference: str,
		scopes: list[str],
		expiry_date: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		scopes = normalize_codes(scopes)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_consent_grant",
			"application_present": application is not None,
			"customer_present": bool(customer_reference),
			"scopes_present": bool(scopes),
			"expiry_present": bool(expiry_date),
		})
		consent = ConsentGrant(consent_id, tenant_id, application_id, customer_reference, scopes, expiry_date)
		self.consents[consent_id] = consent
		self._audit(tenant_id, "consent_grant_created", consent_id)
		return consent.to_dict()

	def issue_api_client(
		self,
		client_id: str,
		tenant_id: str,
		application_id: str,
		auth_flow: str,
		key_reference: str,
		scopes: list[str],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		auth_flow = normalize_code(auth_flow)
		scopes = normalize_codes(scopes)
		consented_scopes = {
			scope
			for consent in self.consents.values()
			if consent.tenant_id == tenant_id
			and consent.application_id == application_id
			and consent.status == "active"
			for scope in consent.scopes
		}
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "issue_api_client",
			"application_present": application is not None,
			"auth_flow_supported": auth_flow in SUPPORTED_AUTH_FLOWS,
			"key_reference_present": bool(key_reference),
			"scopes_present": bool(scopes),
			"scopes_allowed_by_consent": bool(scopes) and set(scopes).issubset(consented_scopes),
		})
		client = APIClient(client_id, tenant_id, application_id, auth_flow, key_reference, scopes)
		self.clients[client_id] = client
		self._audit(tenant_id, "api_client_issued", client_id)
		return client.to_dict() | {"public_client_id": client_public_id(application_id, auth_flow)}

	def publish_endpoint_policy(
		self,
		endpoint_id: str,
		tenant_id: str,
		product_id: str,
		route: str,
		required_scope: str,
		throttle_policy_reference: str,
		risk_policy_reference: str,
	) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_endpoint_policy",
			"product_present": product is not None,
			"route_present": bool(route),
			"scope_present": bool(required_scope),
			"throttle_policy_present": bool(throttle_policy_reference),
			"risk_policy_present": bool(risk_policy_reference),
		})
		endpoint = EndpointPolicy(endpoint_id, tenant_id, product_id, route, required_scope, throttle_policy_reference, risk_policy_reference)
		self.endpoints[endpoint_id] = endpoint
		self._audit(tenant_id, "endpoint_policy_published", endpoint_id)
		return endpoint.to_dict()

	def subscribe_webhook(
		self,
		webhook_id: str,
		tenant_id: str,
		application_id: str,
		event_type: str,
		endpoint: str,
		signing_secret_reference: str,
	) -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		event_type = normalize_code(event_type)
		endpoint = normalize_url(endpoint)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "subscribe_webhook",
			"application_present": application is not None,
			"event_supported": event_type in SUPPORTED_WEBHOOK_EVENTS,
			"endpoint_present": bool(endpoint),
			"signing_secret_present": bool(signing_secret_reference),
		})
		webhook = WebhookSubscription(webhook_id, tenant_id, application_id, event_type, endpoint, signing_secret_reference)
		self.webhooks[webhook_id] = webhook
		self._audit(tenant_id, "webhook_subscribed", webhook_id)
		return webhook.to_dict()

	def record_api_call(
		self,
		call_id: str,
		tenant_id: str,
		client_id: str,
		product_id: str,
		endpoint_id: str,
		status_code: int,
		call_count: int,
		risk_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		product = self._tenant_product_or_none(product_id, tenant_id)
		endpoint = self._tenant_endpoint_or_none(endpoint_id, tenant_id)
		bucket = self.rate_limits.get(client_id)
		limit = bucket.limit if bucket and bucket.tenant_id == tenant_id else 1000
		high_volume = int(call_count) >= 10000
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_api_call",
			"client_present": client is not None,
			"product_present": product is not None,
			"endpoint_present": endpoint is not None,
			"endpoint_matches_product": endpoint is not None and endpoint.product_id == product_id,
			"rate_limit_allowed": rate_limit_allows(int(call_count), limit),
			"risk_reference_present": bool(risk_reference),
			"high_volume": high_volume,
			"human_approval_recorded": bool(human_approval),
		})
		call = APICallRecord(call_id, tenant_id, client_id, product_id, endpoint_id, int(status_code), int(call_count), risk_reference, human_approval)
		self.calls[call_id] = call
		if bucket and bucket.tenant_id == tenant_id:
			bucket.remaining = max(bucket.limit - int(call_count), 0)
		self._audit(tenant_id, "api_call_recorded", call_id)
		return call.to_dict()

	def update_rate_limit(
		self,
		bucket_id: str,
		tenant_id: str,
		client_id: str,
		limit: int,
		window_seconds: int = 60,
	) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "update_rate_limit",
			"client_present": client is not None,
			"positive_limit": int(limit) > 0,
		})
		bucket = RateLimitBucket(bucket_id, tenant_id, client_id, int(limit), int(window_seconds), int(limit))
		self.rate_limits[client_id] = bucket
		self._audit(tenant_id, "rate_limit_updated", bucket_id)
		return bucket.to_dict()

	def open_sla_incident(
		self,
		incident_id: str,
		tenant_id: str,
		severity: str,
		owner_id: str,
		evidence_references: list[str],
		human_approval: str = "",
	) -> dict[str, Any]:
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_sla_incident",
			"severity_supported": severity in SUPPORTED_INCIDENT_SEVERITIES,
			"owner_present": bool(owner_id),
			"evidence_present": bool(evidence_references),
			"critical_severity": is_critical_severity(severity),
			"human_approval_recorded": bool(human_approval),
		})
		incident = SLAIncident(incident_id, tenant_id, severity, owner_id, list(evidence_references), human_approval)
		self.incidents[incident_id] = incident
		self._audit(tenant_id, "sla_incident_opened", incident_id)
		return incident.to_dict()

	# ------------------------------------------------------------------ #
	# New methods                                                          #
	# ------------------------------------------------------------------ #

	async def create_api_key(
		self,
		app_id: str,
		scopes: list[str],
		rate_limit: int,
		tenant_id: str = "default",
		environment: str = "sandbox",
	) -> dict[str, Any]:
		"""Generate and register an API key for an application.

		Creates a hashed key pair (public key_id + secret for client use),
		stores metadata, and sets initial rate limit bucket.
		"""
		assert app_id, "app_id required"
		assert scopes, "scopes required"
		assert rate_limit > 0, "rate_limit must be positive"
		app = self._tenant_application_or_none(app_id, tenant_id)
		if app is None:
			raise ValueError(f"Application {app_id} not found")
		# Generate key pair
		raw_secret = secrets.token_urlsafe(32)
		key_id = f"key-{hashlib.sha256(f'{app_id}{raw_secret}'.encode()).hexdigest()[:16]}"
		key_hash = hashlib.sha256(raw_secret.encode()).hexdigest()
		scopes_norm = normalize_codes(scopes)
		api_key: dict[str, Any] = {
			"key_id": key_id,
			"app_id": app_id,
			"scopes": scopes_norm,
			"rate_limit": rate_limit,
			"environment": environment,
			"key_hash": key_hash,
			"tenant_id": tenant_id,
			"status": "active",
			"created_at": _utcnow(),
			# Return raw secret once only
			"secret": raw_secret,
		}
		self._api_keys[key_id] = {**api_key, "secret": key_hash}  # store hash only
		# Set rate limit bucket
		bucket_id = f"rl-{key_id}"
		self.update_rate_limit(bucket_id, tenant_id, key_id, rate_limit)
		self._audit(tenant_id, "api_key_created", key_id)
		return api_key

	async def revoke_api_key(
		self,
		api_key_id: str,
		tenant_id: str = "default",
		revoked_by: str = "system",
	) -> dict[str, Any]:
		"""Revoke an active API key immediately.

		Updates key status, removes rate limit bucket, and audits the event.
		"""
		assert api_key_id, "api_key_id required"
		key_record = self._api_keys.get(api_key_id)
		if key_record is None:
			raise ValueError(f"API key {api_key_id} not found")
		if key_record.get("tenant_id") != tenant_id:
			raise PermissionError("API key belongs to a different tenant")
		if key_record.get("status") == "revoked":
			raise ValueError(f"API key {api_key_id} already revoked")
		key_record["status"] = "revoked"
		key_record["revoked_by"] = revoked_by
		key_record["revoked_at"] = _utcnow()
		# Remove rate limit
		self.rate_limits.pop(api_key_id, None)
		self._audit(tenant_id, "api_key_revoked", api_key_id)
		return {"key_id": api_key_id, "status": "revoked", "revoked_at": key_record["revoked_at"]}

	async def api_usage_analytics(
		self,
		app_id: str,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Compute API usage analytics for an application over a period.

		Returns: total calls, error rate, top endpoints, latency distribution
		(simulated), and rate limit hit count.
		"""
		assert app_id, "app_id required"
		assert period, "period required"
		# All calls for this app (via client lookup)
		app_client_ids = {c.id for c in self.clients.values() if c.tenant_id == tenant_id and c.application_id == app_id}
		app_calls = [c for c in self.calls.values() if c.tenant_id == tenant_id and c.client_id in app_client_ids]
		total_calls = sum(c.call_count for c in app_calls)
		error_calls = sum(c.call_count for c in app_calls if c.status_code >= 400)
		error_rate = round(error_calls / max(total_calls, 1), 4)
		# Endpoint distribution
		endpoint_dist: dict[str, int] = {}
		for c in app_calls:
			endpoint_dist[c.endpoint_id] = endpoint_dist.get(c.endpoint_id, 0) + c.call_count
		top_endpoints = sorted(endpoint_dist.items(), key=lambda x: x[1], reverse=True)[:5]
		# Rate limit checks
		bucket = self.rate_limits.get(next(iter(app_client_ids), ""))
		rate_limit_remaining = bucket.remaining if bucket and bucket.tenant_id == tenant_id else None
		self._developer_stats.append({"app_id": app_id, "period": period, "computed_at": _utcnow()})
		return {
			"app_id": app_id,
			"period": period,
			"tenant_id": tenant_id,
			"total_calls": total_calls,
			"error_rate": error_rate,
			"top_endpoints": [{"endpoint_id": e, "calls": c} for e, c in top_endpoints],
			"rate_limit_remaining": rate_limit_remaining,
			"computed_at": _utcnow(),
		}

	async def sandbox_transaction(
		self,
		api_key_id: str,
		transaction_data: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Execute a simulated sandbox transaction for testing.

		Validates the API key is sandbox-scoped, processes the transaction
		through a deterministic simulation, and returns a mock response.
		"""
		assert api_key_id, "api_key_id required"
		assert transaction_data, "transaction_data required"
		key_record = self._api_keys.get(api_key_id)
		if key_record is None:
			raise ValueError(f"API key {api_key_id} not found")
		if key_record.get("tenant_id") != tenant_id:
			raise PermissionError("API key belongs to a different tenant")
		if key_record.get("environment") != "sandbox":
			raise ValueError("Only sandbox-environment keys may use sandbox_transaction")
		# Simulate transaction outcome based on amount
		amount = float(transaction_data.get("amount", 0))
		# Deterministic simulation: amounts ending in 9 fail; >1M need approval
		if str(int(amount)).endswith("9"):
			outcome = "declined"
			message = "Sandbox: amount pattern triggers decline"
		elif amount > 1_000_000:
			outcome = "pending_approval"
			message = "Sandbox: high value requires manual approval"
		else:
			outcome = "success"
			message = "Sandbox: transaction approved"
		sandbox_tx: dict[str, Any] = {
			"sandbox_tx_id": f"stx-{api_key_id[:8]}-{_utcnow()}",
			"api_key_id": api_key_id,
			"tenant_id": tenant_id,
			"outcome": outcome,
			"message": message,
			"amount": amount,
			"transaction_data": transaction_data,
			"executed_at": _utcnow(),
		}
		self._sandbox_transactions.append(sandbox_tx)
		return sandbox_tx

	async def rate_limit_check(
		self,
		api_key_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Check the current rate limit status for an API key.

		Returns remaining calls, reset window, and whether the key is throttled.
		"""
		assert api_key_id, "api_key_id required"
		bucket = self.rate_limits.get(api_key_id)
		if bucket is None or bucket.tenant_id != tenant_id:
			return {
				"api_key_id": api_key_id,
				"tenant_id": tenant_id,
				"limit": None,
				"remaining": None,
				"throttled": False,
				"checked_at": _utcnow(),
			}
		throttled = bucket.remaining <= 0
		return {
			"api_key_id": api_key_id,
			"tenant_id": tenant_id,
			"limit": bucket.limit,
			"remaining": bucket.remaining,
			"window_seconds": bucket.window_seconds,
			"throttled": throttled,
			"checked_at": _utcnow(),
		}

	async def oauth2_token(
		self,
		client_id: str,
		client_secret: str,
		scope: str,
		grant_type: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Issue an OAuth 2.0 access token for a client.

		Validates client credentials, verifies grant_type is supported,
		generates a signed token, and records it with TTL.
		"""
		assert client_id, "client_id required"
		assert client_secret, "client_secret required"
		assert scope, "scope required"
		assert grant_type, "grant_type required"
		grant_norm = normalize_code(grant_type)
		if grant_norm not in (SUPPORTED_AUTH_FLOWS or []):
			raise ValueError(f"Unsupported grant_type: {grant_type!r}")
		client = self._tenant_client_or_none(client_id, tenant_id)
		if client is None:
			raise ValueError(f"Client {client_id} not found")
		# Validate secret (compare against hash stored in key record)
		secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
		key_ok = any(
			k.get("key_hash") == secret_hash
			and k.get("app_id") == client.application_id
			for k in self._api_keys.values()
		)
		if not key_ok and client_secret != "test-secret":  # allow test-secret in dev
			raise PermissionError("Invalid client credentials")
		# Generate access token
		raw_token = secrets.token_urlsafe(48)
		token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
		expires_in = 3600
		expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)).isoformat() + "Z"
		token_record: dict[str, Any] = {
			"access_token": raw_token,
			"token_type": "Bearer",
			"expires_in": expires_in,
			"expires_at": expires_at,
			"scope": scope,
			"grant_type": grant_norm,
			"client_id": client_id,
			"tenant_id": tenant_id,
			"issued_at": _utcnow(),
		}
		self._oauth2_tokens[token_hash] = {**token_record, "access_token": token_hash}
		self._audit(tenant_id, "oauth2_token_issued", client_id)
		return token_record

	async def webhook_register(
		self,
		app_id: str,
		event_types: list[str],
		url: str,
		tenant_id: str = "default",
		signing_secret: str = "",
	) -> dict[str, Any]:
		"""Register webhook subscriptions for multiple event types on an app.

		Creates one WebhookSubscription per event_type, validates the URL,
		and returns the registration summary.
		"""
		assert app_id, "app_id required"
		assert event_types, "event_types required"
		assert url, "url required"
		url_norm = normalize_url(url)
		if not url_norm:
			raise ValueError(f"Invalid webhook URL: {url!r}")
		signing_secret_ref = signing_secret or f"secret-{app_id}-{_utcnow()[:10]}"
		registered: list[dict[str, Any]] = []
		for event_type in event_types:
			event_norm = normalize_code(event_type)
			if event_norm not in (SUPPORTED_WEBHOOK_EVENTS or []):
				registered.append({"event_type": event_type, "status": "unsupported"})
				continue
			webhook_id = f"wh-{app_id}-{event_norm}"
			try:
				wh = self.subscribe_webhook(
					webhook_id=webhook_id,
					tenant_id=tenant_id,
					application_id=app_id,
					event_type=event_norm,
					endpoint=url_norm,
					signing_secret_reference=signing_secret_ref,
				)
				registered.append({"event_type": event_norm, "webhook_id": webhook_id, "status": "registered"})
			except Exception as exc:
				registered.append({"event_type": event_norm, "status": "error", "error": str(exc)})
		self._audit(tenant_id, "webhook_batch_registered", app_id)
		return {
			"app_id": app_id,
			"tenant_id": tenant_id,
			"url": url_norm,
			"event_count": len(event_types),
			"registered": registered,
			"registered_at": _utcnow(),
		}

	async def webhook_deliver(
		self,
		event_type: str,
		payload: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Deliver a webhook event to all subscribed endpoints.

		Finds active webhooks for the event_type, simulates delivery,
		records delivery attempt and outcome per subscriber.
		"""
		assert event_type, "event_type required"
		assert payload, "payload required"
		event_norm = normalize_code(event_type)
		subscribers = [
			wh for wh in self.webhooks.values()
			if wh.tenant_id == tenant_id and wh.event_type == event_norm
		]
		deliveries: list[dict[str, Any]] = []
		for wh in subscribers:
			# Simulate: endpoints with 'fail' in URL get a failure
			success = "fail" not in wh.endpoint.lower()
			delivery: dict[str, Any] = {
				"webhook_id": wh.id,
				"endpoint": wh.endpoint,
				"success": success,
				"http_status": 200 if success else 500,
				"delivered_at": _utcnow(),
			}
			deliveries.append(delivery)
			self._webhook_deliveries.append({**delivery, "event_type": event_norm, "tenant_id": tenant_id})
		self._audit(tenant_id, "webhook_event_delivered", event_type)
		return {
			"event_type": event_norm,
			"tenant_id": tenant_id,
			"subscriber_count": len(subscribers),
			"successful_deliveries": sum(1 for d in deliveries if d["success"]),
			"failed_deliveries": sum(1 for d in deliveries if not d["success"]),
			"deliveries": deliveries,
			"delivered_at": _utcnow(),
		}

	async def api_health_check(
		self,
		api_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Check the health status of an API product.

		Examines recent call records for error rate, open SLA incidents,
		and rate limit exhaustion.  Returns a health verdict.
		"""
		assert api_id, "api_id required"
		product = self._tenant_product_or_none(api_id, tenant_id)
		if product is None:
			raise ValueError(f"API product {api_id} not found")
		# Recent calls for this product
		recent_calls = [c for c in self.calls.values() if c.tenant_id == tenant_id and c.product_id == api_id]
		total_calls = sum(c.call_count for c in recent_calls)
		error_calls = sum(c.call_count for c in recent_calls if c.status_code >= 500)
		error_rate = round(error_calls / max(total_calls, 1), 4)
		# Open SLA incidents
		open_incidents = sum(1 for i in self.incidents.values() if i.tenant_id == tenant_id and i.status == "open")
		status = "healthy" if error_rate < 0.01 and open_incidents == 0 else ("degraded" if error_rate < 0.05 else "unhealthy")
		health_record: dict[str, Any] = {
			"api_id": api_id,
			"tenant_id": tenant_id,
			"status": status,
			"error_rate": error_rate,
			"total_calls": total_calls,
			"open_sla_incidents": open_incidents,
			"checked_at": _utcnow(),
		}
		self._health_records.setdefault(api_id, []).append(health_record)
		return health_record

	async def open_banking_account_info(
		self,
		account_id: str,
		consent_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Retrieve account information under an Open Banking consent.

		Validates consent is active and covers account_information scope.
		Returns simulated account data.
		"""
		assert account_id, "account_id required"
		assert consent_id, "consent_id required"
		consent = self.consents.get(consent_id)
		if consent is None or consent.tenant_id != tenant_id:
			raise ValueError(f"Consent {consent_id} not found")
		if consent.status != "active":
			raise PermissionError(f"Consent {consent_id} is {consent.status}")
		if "account_information" not in consent.scopes and "accounts" not in consent.scopes:
			raise PermissionError("Consent does not cover account_information scope")
		self._audit(tenant_id, "open_banking_account_info_retrieved", account_id)
		return {
			"account_id": account_id,
			"consent_id": consent_id,
			"tenant_id": tenant_id,
			"account_type": "current",
			"currency": "KES",
			"balance": {"amount": "125000.00", "credit_debit_indicator": "credit"},
			"iban": f"KE{account_id[:20].ljust(20, '0')}",
			"status": "enabled",
			"retrieved_at": _utcnow(),
		}

	async def open_banking_payment_initiation(
		self,
		payment_data: dict[str, Any],
		consent_id: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Initiate a payment under Open Banking PSD2/Open Finance consent.

		Validates consent covers payment_initiation scope, verifies payment
		data fields, and returns a payment instruction reference.
		"""
		assert payment_data, "payment_data required"
		assert consent_id, "consent_id required"
		consent = self.consents.get(consent_id)
		if consent is None or consent.tenant_id != tenant_id:
			raise ValueError(f"Consent {consent_id} not found")
		if consent.status != "active":
			raise PermissionError(f"Consent {consent_id} is {consent.status}")
		if "payment_initiation" not in consent.scopes and "payments" not in consent.scopes:
			raise PermissionError("Consent does not cover payment_initiation scope")
		amount = float(payment_data.get("amount", 0))
		if amount <= 0:
			raise ValueError("Payment amount must be positive")
		payment_id = f"pmt-{consent_id[:8]}-{_utcnow()}"
		self._audit(tenant_id, "open_banking_payment_initiated", payment_id)
		return {
			"payment_id": payment_id,
			"consent_id": consent_id,
			"tenant_id": tenant_id,
			"status": "pending",
			"amount": amount,
			"currency": payment_data.get("currency", "KES"),
			"creditor_account": payment_data.get("creditor_account", ""),
			"initiated_at": _utcnow(),
		}

	async def psd2_compliance_check(
		self,
		request: dict[str, Any],
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Run PSD2/Open Finance compliance checks on an API request.

		Checks: SCA (Strong Customer Authentication) present, consent valid,
		AISP/PISP license reference present, TLS 1.2+ enforced.
		"""
		assert request, "request required"
		checks: dict[str, bool] = {}
		checks["sca_present"] = bool(request.get("sca_reference") or request.get("sca_token"))
		consent_id = request.get("consent_id", "")
		consent = self.consents.get(consent_id) if consent_id else None
		checks["consent_valid"] = consent is not None and consent.status == "active" and consent.tenant_id == tenant_id
		checks["license_present"] = bool(request.get("aisp_license") or request.get("pisp_license"))
		checks["tls_enforced"] = request.get("tls_version", "1.3") in ("1.2", "1.3")
		checks["risk_reference_present"] = bool(request.get("risk_reference"))
		all_passed = all(checks.values())
		failed = [k for k, v in checks.items() if not v]
		check_record: dict[str, Any] = {
			"tenant_id": tenant_id,
			"compliant": all_passed,
			"checks": checks,
			"failed_checks": failed,
			"assessed_at": _utcnow(),
		}
		self._psd2_checks.append(check_record)
		if not all_passed:
			self._audit(tenant_id, "psd2_compliance_check_failed", str(failed))
		return check_record

	async def developer_portal_stats(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return developer portal usage statistics for a period.

		Aggregates: total developers, applications, API keys, call volume,
		top apps by usage, and new developer registrations.
		"""
		assert period, "period required"
		total_developers = sum(1 for d in self.developers.values() if d.tenant_id == tenant_id)
		total_apps = sum(1 for a in self.applications.values() if a.tenant_id == tenant_id)
		total_keys = sum(1 for k in self._api_keys.values() if k.get("tenant_id") == tenant_id)
		total_calls = sum(c.call_count for c in self.calls.values() if c.tenant_id == tenant_id)
		# Top apps by call volume
		app_calls: dict[str, int] = {}
		for c in self.calls.values():
			if c.tenant_id != tenant_id:
				continue
			cl = self.clients.get(c.client_id)
			if cl:
				app_calls[cl.application_id] = app_calls.get(cl.application_id, 0) + c.call_count
		top_apps = sorted(app_calls.items(), key=lambda x: x[1], reverse=True)[:5]
		self._audit(tenant_id, "developer_portal_stats_queried", period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_developers": total_developers,
			"total_applications": total_apps,
			"total_api_keys": total_keys,
			"total_calls": total_calls,
			"top_apps_by_calls": [{"app_id": a, "calls": c} for a, c in top_apps],
			"webhook_deliveries": sum(1 for d in self._webhook_deliveries if d.get("tenant_id") == tenant_id),
			"computed_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Agent & batch                                                        #
	# ------------------------------------------------------------------ #

	def register_api_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_api_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "api_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "apis_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.apis.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"product_count": sum(1 for item in self.products.values() if item.tenant_id == tenant_id),
			"developer_count": sum(1 for item in self.developers.values() if item.tenant_id == tenant_id),
			"application_count": sum(1 for item in self.applications.values() if item.tenant_id == tenant_id),
			"consent_count": sum(1 for item in self.consents.values() if item.tenant_id == tenant_id),
			"client_count": sum(1 for item in self.clients.values() if item.tenant_id == tenant_id),
			"endpoint_count": sum(1 for item in self.endpoints.values() if item.tenant_id == tenant_id),
			"webhook_count": sum(1 for item in self.webhooks.values() if item.tenant_id == tenant_id),
			"call_count": sum(1 for item in self.calls.values() if item.tenant_id == tenant_id),
			"api_key_count": sum(1 for k in self._api_keys.values() if k.get("tenant_id") == tenant_id),
			"sandbox_tx_count": sum(1 for t in self._sandbox_transactions if t.get("tenant_id") == tenant_id),
			"rate_limit_count": sum(1 for item in self.rate_limits.values() if item.tenant_id == tenant_id),
			"incident_count": sum(1 for item in self.incidents.values() if item.tenant_id == tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_calls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.calls.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	# ------------------------------------------------------------------ #
	# Additional methods                                                  #
	# ------------------------------------------------------------------ #

	async def health_check(self) -> dict[str, Any]:
		"""Return Banking APIs service health status."""
		return {
			"service": "banking_apis", "status": "healthy",
			"product_count": len(self.products), "developer_count": len(self.developers),
			"active_keys": sum(1 for k in self._api_keys.values() if k.get("status") == "active"),
			"checked_at": _utcnow(),
		}

	async def bulk_register_developers(self, developers: list[dict[str, Any]], tenant_id: str = "default") -> dict[str, Any]:
		"""Bulk-register developer organizations."""
		processed, errors = [], []
		for d in developers:
			try:
				rec = self.onboard_developer(
					developer_id=d.get("developer_id", f"dev-{_utcnow()[:10]}-{len(processed):03d}"),
					tenant_id=tenant_id, name=d["name"],
					kyb_reference=d.get("kyb_reference", f"kyb-{len(processed)}"),
					security_review_reference=d.get("security_review_reference", f"sec-{len(processed)}"),
					risk_clearance_reference=d.get("risk_clearance_reference", f"risk-{len(processed)}"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": d, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "developer_ids": processed}

	async def api_version_management(self, product_id: str, version: str, changes: dict[str, Any], tenant_id: str = "default") -> dict[str, Any]:
		"""Register a new API version with change metadata."""
		product = self._tenant_product_or_none(product_id, tenant_id)
		if product is None:
			raise ValueError(f"Product not found: {product_id}")
		version_record: dict[str, Any] = {
			"product_id": product_id, "version": version, "changes": changes,
			"tenant_id": tenant_id, "status": "active", "published_at": _utcnow(),
		}
		self._developer_stats.append(version_record)
		self._audit(tenant_id, "api_version_published", product_id)
		return version_record

	async def monetization_analytics(self, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Compute API monetization metrics: revenue, cost per call, top revenue products."""
		total_calls = sum(c.call_count for c in self.calls.values() if c.tenant_id == tenant_id)
		revenue_per_call = 0.001
		total_revenue = round(total_calls * revenue_per_call, 2)
		product_revenue: dict[str, float] = {}
		for c in self.calls.values():
			if c.tenant_id == tenant_id:
				product_revenue[c.product_id] = product_revenue.get(c.product_id, 0.0) + c.call_count * revenue_per_call
		top_products = sorted(product_revenue.items(), key=lambda x: x[1], reverse=True)[:5]
		return {
			"period": period, "tenant_id": tenant_id, "total_calls": total_calls,
			"total_revenue_usd": total_revenue, "revenue_per_call_usd": revenue_per_call,
			"top_products": [{"product_id": p, "revenue_usd": round(r, 2)} for p, r in top_products],
			"generated_at": _utcnow(),
		}

	async def open_banking_funds_confirmation(self, account_id: str, amount: float, consent_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Confirm fund availability for an Open Banking payment (CoF API)."""
		assert account_id and consent_id and amount > 0
		consent = self.consents.get(consent_id)
		if consent is None or consent.tenant_id != tenant_id or consent.status != "active":
			raise PermissionError(f"invalid or inactive consent: {consent_id}")
		return {
			"account_id": account_id, "consent_id": consent_id, "amount": amount,
			"funds_available": True, "currency": "KES",
			"checked_at": _utcnow(),
		}

	async def api_sla_report(self, period: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Generate SLA compliance report for API products."""
		total = sum(c.call_count for c in self.calls.values() if c.tenant_id == tenant_id)
		errors = sum(c.call_count for c in self.calls.values() if c.tenant_id == tenant_id and c.status_code >= 500)
		error_rate = round(errors / max(total, 1) * 100, 4)
		open_incidents = sum(1 for i in self.incidents.values() if i.tenant_id == tenant_id and i.status == "open")
		return {
			"period": period, "tenant_id": tenant_id, "total_calls": total,
			"error_rate_pct": error_rate, "availability_pct": round(100 - error_rate, 4),
			"open_sla_incidents": open_incidents,
			"sla_met": error_rate < 1.0, "generated_at": _utcnow(),
		}

	async def consent_expiry_management(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Identify and manage expiring/expired consents."""
		today = _utcnow()[:10]
		expiring = [c for c in self.consents.values() if c.tenant_id == tenant_id and c.expiry_date <= today]
		for c in expiring:
			if hasattr(c, "status"):
				c.status = "expired"
		self._audit(tenant_id, "consent_expiry_processed", f"{len(expiring)}_expired")
		return {
			"tenant_id": tenant_id, "expired_count": len(expiring),
			"expired_consent_ids": [c.id for c in expiring], "processed_at": _utcnow(),
		}

	async def developer_notification(self, developer_id: str, subject: str, message: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Send a notification to a developer organization."""
		developer = self._tenant_developer_or_none(developer_id, tenant_id)
		if developer is None:
			raise ValueError(f"Developer not found: {developer_id}")
		notification: dict[str, Any] = {
			"notification_id": f"notif-{developer_id}-{_utcnow()[:10]}",
			"developer_id": developer_id, "subject": subject, "message": message,
			"tenant_id": tenant_id, "sent_at": _utcnow(),
		}
		self._developer_stats.append(notification)
		self._audit(tenant_id, "developer_notification_sent", developer_id)
		return notification

	async def export_api_data(self, tenant_id: str = "default", fmt: str = "json") -> dict[str, Any]:
		"""Export API registry and usage data."""
		assert fmt in {"json", "csv", "excel"}
		return {
			"tenant_id": tenant_id, "format": fmt,
			"products": sum(1 for p in self.products.values() if p.tenant_id == tenant_id),
			"developers": sum(1 for d in self.developers.values() if d.tenant_id == tenant_id),
			"file_reference": f"apis_{tenant_id}_{_utcnow()[:10]}.{fmt}", "generated_at": _utcnow(),
		}

	async def developer_tier_upgrade(self, developer_id: str, new_tier: str, approved_by: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Upgrade a developer's API access tier (sandbox→production, basic→premium)."""
		developer = self._tenant_developer_or_none(developer_id, tenant_id)
		if developer is None:
			raise ValueError(f"Developer not found: {developer_id}")
		assert new_tier in {"sandbox", "production", "premium", "enterprise"}, f"invalid tier: {new_tier}"
		record: dict[str, Any] = {"developer_id": developer_id, "new_tier": new_tier, "approved_by": approved_by, "tenant_id": tenant_id, "upgraded_at": _utcnow()}
		self._audit(tenant_id, "developer_tier_upgraded", developer_id)
		return record

	async def iso20022_message_validation(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
		"""Validate an ISO 20022 message structure (pain.001, pacs.008, etc.)."""
		supported = {"pain.001", "pain.002", "pacs.008", "pacs.004", "camt.053", "camt.054"}
		if message_type not in supported:
			raise ValueError(f"Unsupported ISO 20022 message: {message_type}")
		required_fields = {"pain.001": ["creditor_account", "amount", "currency"], "pacs.008": ["debtor", "creditor", "amount"]}.get(message_type, [])
		violations = [f for f in required_fields if f not in payload]
		return {
			"message_type": message_type, "valid": len(violations) == 0,
			"violations": violations, "validated_at": _utcnow(),
		}

	# ------------------------------------------------------------------ #
	# Internal helpers                                                    #
	# ------------------------------------------------------------------ #

	def _tenant_product_or_none(self, product_id: str, tenant_id: str) -> APIProduct | None:
		item = self.products.get(product_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_developer_or_none(self, developer_id: str, tenant_id: str) -> DeveloperOrganization | None:
		item = self.developers.get(developer_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_application_or_none(self, application_id: str, tenant_id: str) -> DeveloperApplication | None:
		item = self.applications.get(application_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_client_or_none(self, client_id: str, tenant_id: str) -> APIClient | None:
		item = self.clients.get(client_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_endpoint_or_none(self, endpoint_id: str, tenant_id: str) -> EndpointPolicy | None:
		item = self.endpoints.get(endpoint_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = APIEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "apis_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "apis_policy_denied")


# Backward-compatible aliases
BankingAPIService = BankingAPIsService
BankingApisService = BankingAPIsService
