"""Async service layer for APG Developer Portal (common/devp)."""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from .capability_contract import CAPABILITY_ID, evaluate_capability_rules, get_capability_contract
	from .models import (
		DpApiKey,
		DpApiProduct,
		DpDeveloperApp,
		DpEndpointStats,
		DpRateLimit,
		DpSubscription,
		DpUsageStats,
		DpWebhookEndpoint,
	)
except ImportError:  # pragma: no cover
	from capability_contract import CAPABILITY_ID, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import (  # type: ignore
		DpApiKey,
		DpApiProduct,
		DpDeveloperApp,
		DpEndpointStats,
		DpRateLimit,
		DpSubscription,
		DpUsageStats,
		DpWebhookEndpoint,
	)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _hash_secret(raw: str) -> str:
	"""SHA-256 hex digest of a raw secret string."""
	return hashlib.sha256(raw.encode()).hexdigest()


def _generate_api_key() -> tuple[str, str]:
	"""Generate a cryptographically-random API key.

	Returns (plaintext_key, sha256_hex_hash).
	The plaintext is only returned once at creation; only the hash is stored.
	"""
	raw = "apg_" + secrets.token_hex(32)
	return raw, _hash_secret(raw)


def _guard_tenant(tenant_id: str) -> None:
	assert tenant_id and tenant_id.strip(), "tenant_id must be non-empty"


def _guard_str(value: str, field: str) -> None:
	assert value and value.strip(), f"{field} must be non-empty"


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

class DeveloperPortalService:
	"""Tenant-scoped Developer Portal runtime for generated APG applications.

	In-memory store is suitable for testing and embedded operation.
	Production deployments should override _load / _persist methods with
	PostgreSQL-backed implementations.

	Keyword: developer_portal, api_self_service, api_monetization
	"""

	def __init__(self) -> None:
		# in-memory stores keyed by (tenant_id, id)
		self._api_products: dict[tuple[str, str], DpApiProduct] = {}
		self._api_keys: dict[tuple[str, str], DpApiKey] = {}
		# secondary index: key_hash → DpApiKey  (cross-tenant safe — hashes are global)
		self._key_hash_index: dict[str, DpApiKey] = {}
		self._developer_apps: dict[tuple[str, str], DpDeveloperApp] = {}
		self._subscriptions: dict[tuple[str, str], DpSubscription] = {}
		self._webhooks: dict[tuple[str, str], DpWebhookEndpoint] = {}
		# minimal usage ledger: (tenant_id, key_id) → list of call records
		# each record: {"endpoint": str, "status": int, "latency_ms": float, "ts": str}
		self._call_log: dict[tuple[str, str], list[dict[str, Any]]] = {}

	# -----------------------------------------------------------------------
	# API Products
	# -----------------------------------------------------------------------

	async def create_api_product(
		self,
		tenant_id: str,
		name: str,
		description: str,
		capabilities: list[str],
		plan: str,
		endpoints: list[dict[str, Any]] | None = None,
		monthly_call_quota: int | None = None,
		price_minor_units: int | None = None,
		currency: str = "KES",
	) -> DpApiProduct:
		"""Create a new API product.

		Keyword: create_api_product, publish_api
		"""
		_guard_tenant(tenant_id)
		_guard_str(name, "name")
		_guard_str(plan, "plan")

		from .models import DpEndpoint  # local import to avoid circular at module level
		endpoint_objs = [DpEndpoint(**e) for e in (endpoints or [])]

		product = DpApiProduct(
			tenant_id=tenant_id,
			name=name,
			description=description,
			capabilities=capabilities,
			plan=plan,
			endpoints=endpoint_objs,
			monthly_call_quota=monthly_call_quota,
			price_minor_units=price_minor_units,
			currency=currency,
		)
		self._api_products[(tenant_id, product.id)] = product
		return product

	async def list_api_products(self, tenant_id: str) -> list[DpApiProduct]:
		"""List all published API products for a tenant.

		Keyword: list_api_products, browse_catalog
		"""
		_guard_tenant(tenant_id)
		return [p for (tid, _), p in self._api_products.items() if tid == tenant_id]

	async def get_api_product(self, product_id: str, tenant_id: str) -> DpApiProduct | None:
		"""Fetch a single API product by ID.

		Keyword: get_api_product
		"""
		_guard_tenant(tenant_id)
		return self._api_products.get((tenant_id, product_id))

	# -----------------------------------------------------------------------
	# Developer Apps
	# -----------------------------------------------------------------------

	async def create_developer_app(
		self,
		tenant_id: str,
		name: str,
		owner_id: str,
		description: str = "",
		callback_urls: list[str] | None = None,
	) -> DpDeveloperApp:
		"""Register a new developer application.

		Keyword: create_developer_app, register_app
		"""
		_guard_tenant(tenant_id)
		_guard_str(name, "name")
		_guard_str(owner_id, "owner_id")

		app = DpDeveloperApp(
			tenant_id=tenant_id,
			name=name,
			description=description,
			owner_id=owner_id,
			callback_urls=callback_urls or [],
		)
		self._developer_apps[(tenant_id, app.id)] = app
		return app

	async def get_developer_app(self, app_id: str, tenant_id: str) -> DpDeveloperApp | None:
		"""Fetch a developer app by ID.

		Keyword: get_developer_app
		"""
		_guard_tenant(tenant_id)
		return self._developer_apps.get((tenant_id, app_id))

	async def list_developer_apps(self, tenant_id: str, owner_id: str | None = None) -> list[DpDeveloperApp]:
		"""List developer apps, optionally filtered by owner.

		Keyword: list_developer_apps
		"""
		_guard_tenant(tenant_id)
		apps = [a for (tid, _), a in self._developer_apps.items() if tid == tenant_id]
		if owner_id:
			apps = [a for a in apps if a.owner_id == owner_id]
		return apps

	# -----------------------------------------------------------------------
	# API Keys
	# -----------------------------------------------------------------------

	async def create_api_key(
		self,
		app_id: str,
		scopes: list[str],
		rate_limit: DpRateLimit | None,
		tenant_id: str,
		name: str = "",
		owner_id: str = "",
		expires_in_days: int | None = None,
	) -> tuple[DpApiKey, str]:
		"""Create an API key bound to a developer app.

		Returns (DpApiKey, plaintext_key).  The plaintext is returned **once**
		and must be delivered to the developer immediately — it cannot be
		recovered from the hash.

		Keyword: create_api_key, issue_api_key
		"""
		_guard_tenant(tenant_id)
		_guard_str(app_id, "app_id")

		# verify app exists
		app = self._developer_apps.get((tenant_id, app_id))
		assert app is not None, f"DeveloperApp {app_id!r} not found in tenant {tenant_id!r}"

		plaintext, key_hash = _generate_api_key()

		expires_at: str | None = None
		if expires_in_days is not None:
			expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()

		api_key = DpApiKey(
			tenant_id=tenant_id,
			key_hash=key_hash,
			name=name or f"key-{app.name}",
			owner_id=owner_id or app.owner_id,
			app_id=app_id,
			scopes=scopes,
			rate_limits=rate_limit or DpRateLimit(),
			status="active",
			expires_at=expires_at,
		)
		self._api_keys[(tenant_id, api_key.id)] = api_key
		self._key_hash_index[key_hash] = api_key
		return api_key, plaintext

	async def revoke_api_key(self, key_id: str, tenant_id: str, revoked_by: str = "system") -> None:
		"""Revoke an API key permanently.

		Keyword: revoke_api_key, invalidate_key
		"""
		_guard_tenant(tenant_id)
		_guard_str(key_id, "key_id")

		api_key = self._api_keys.get((tenant_id, key_id))
		assert api_key is not None, f"ApiKey {key_id!r} not found in tenant {tenant_id!r}"
		assert api_key.status != "revoked", "key is already revoked"

		api_key.status = "revoked"
		api_key.revoked_at = _now()
		api_key.revoked_by = revoked_by
		# keep hash index entry but mark revoked for fast validation
		if api_key.key_hash in self._key_hash_index:
			self._key_hash_index[api_key.key_hash] = api_key

	async def validate_api_key(self, key_hash: str) -> DpApiKey | None:
		"""Validate an API key by hash.  Used by the API gateway.

		Returns None if not found, revoked, suspended, or expired.

		Keyword: validate_api_key, authenticate_key
		"""
		api_key = self._key_hash_index.get(key_hash)
		if api_key is None:
			return None
		if api_key.status not in ("active",):
			return None
		# check expiry
		if api_key.expires_at:
			try:
				exp = datetime.fromisoformat(api_key.expires_at)
				if exp.tzinfo is None:
					exp = exp.replace(tzinfo=timezone.utc)
				if datetime.now(timezone.utc) > exp:
					api_key.status = "expired"
					return None
			except (ValueError, TypeError):
				pass
		# update last_used_at
		api_key.last_used_at = _now()
		return api_key

	async def list_api_keys(self, tenant_id: str, app_id: str | None = None) -> list[DpApiKey]:
		"""List API keys for a tenant, optionally filtered by app.

		Keyword: list_api_keys
		"""
		_guard_tenant(tenant_id)
		keys = [k for (tid, _), k in self._api_keys.items() if tid == tenant_id]
		if app_id:
			keys = [k for k in keys if k.app_id == app_id]
		return keys

	# -----------------------------------------------------------------------
	# Subscriptions
	# -----------------------------------------------------------------------

	async def subscribe_to_product(
		self,
		developer_app_id: str,
		product_id: str,
		tenant_id: str,
	) -> DpSubscription:
		"""Subscribe a developer app to an API product.

		Keyword: subscribe_to_product, activate_subscription
		"""
		_guard_tenant(tenant_id)
		_guard_str(developer_app_id, "developer_app_id")
		_guard_str(product_id, "product_id")

		app = self._developer_apps.get((tenant_id, developer_app_id))
		assert app is not None, f"DeveloperApp {developer_app_id!r} not found"

		product = self._api_products.get((tenant_id, product_id))
		assert product is not None, f"ApiProduct {product_id!r} not found"

		now = _now()
		subscription = DpSubscription(
			tenant_id=tenant_id,
			developer_app_id=developer_app_id,
			product_id=product_id,
			status="active",
			plan=product.plan,
			billing_cycle_start=now,
			activated_at=now,
		)
		self._subscriptions[(tenant_id, subscription.id)] = subscription

		# update app product list
		if product_id not in app.api_products:
			app.api_products.append(product_id)
			app.updated_at = now

		return subscription

	async def cancel_subscription(self, subscription_id: str, tenant_id: str) -> DpSubscription:
		"""Cancel an active subscription.

		Keyword: cancel_subscription
		"""
		_guard_tenant(tenant_id)
		sub = self._subscriptions.get((tenant_id, subscription_id))
		assert sub is not None, f"Subscription {subscription_id!r} not found"
		assert sub.status == "active", "only active subscriptions can be cancelled"

		sub.status = "cancelled"
		sub.cancelled_at = _now()
		return sub

	async def list_subscriptions(self, tenant_id: str, developer_app_id: str | None = None) -> list[DpSubscription]:
		"""List subscriptions for a tenant.

		Keyword: list_subscriptions
		"""
		_guard_tenant(tenant_id)
		subs = [s for (tid, _), s in self._subscriptions.items() if tid == tenant_id]
		if developer_app_id:
			subs = [s for s in subs if s.developer_app_id == developer_app_id]
		return subs

	# -----------------------------------------------------------------------
	# Webhooks
	# -----------------------------------------------------------------------

	async def register_webhook(
		self,
		app_id: str,
		url: str,
		events: list[str],
		secret: str,
		tenant_id: str,
	) -> DpWebhookEndpoint:
		"""Register a webhook endpoint for event delivery.

		The raw secret is hashed on registration; it cannot be retrieved later.

		Keyword: register_webhook, create_webhook
		"""
		_guard_tenant(tenant_id)
		_guard_str(app_id, "app_id")
		_guard_str(url, "url")
		_guard_str(secret, "secret")
		assert events, "events list must be non-empty"
		assert url.startswith("https://"), "webhook URL must use HTTPS"

		app = self._developer_apps.get((tenant_id, app_id))
		assert app is not None, f"DeveloperApp {app_id!r} not found"

		webhook = DpWebhookEndpoint(
			tenant_id=tenant_id,
			app_id=app_id,
			url=url,
			secret_hash=_hash_secret(secret),
			events=events,
		)
		self._webhooks[(tenant_id, webhook.id)] = webhook
		return webhook

	async def delete_webhook(self, webhook_id: str, tenant_id: str) -> None:
		"""Delete a registered webhook.

		Keyword: delete_webhook, remove_webhook
		"""
		_guard_tenant(tenant_id)
		assert (tenant_id, webhook_id) in self._webhooks, f"Webhook {webhook_id!r} not found"
		del self._webhooks[(tenant_id, webhook_id)]

	async def list_webhooks(self, tenant_id: str, app_id: str | None = None) -> list[DpWebhookEndpoint]:
		"""List webhooks for a tenant.

		Keyword: list_webhooks
		"""
		_guard_tenant(tenant_id)
		hooks = [h for (tid, _), h in self._webhooks.items() if tid == tenant_id]
		if app_id:
			hooks = [h for h in hooks if h.app_id == app_id]
		return hooks

	# -----------------------------------------------------------------------
	# Usage stats
	# -----------------------------------------------------------------------

	async def record_call(
		self,
		key_id: str,
		tenant_id: str,
		endpoint: str,
		status_code: int,
		latency_ms: float,
	) -> None:
		"""Append a call record to the usage ledger.

		Normally called by the API gateway after forwarding a request.

		Keyword: record_call, ingest_usage
		"""
		_guard_tenant(tenant_id)
		ledger_key = (tenant_id, key_id)
		if ledger_key not in self._call_log:
			self._call_log[ledger_key] = []
		self._call_log[ledger_key].append({
			"endpoint": endpoint,
			"status": status_code,
			"latency_ms": latency_ms,
			"ts": _now(),
		})

	async def get_usage_stats(
		self,
		key_id: str,
		period_days: int,
		tenant_id: str,
	) -> DpUsageStats:
		"""Compute usage statistics for a key over the last N days.

		Keyword: get_usage_stats, api_analytics
		"""
		_guard_tenant(tenant_id)
		_guard_str(key_id, "key_id")
		assert period_days >= 1, "period_days must be >= 1"

		cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
		period_start = cutoff.isoformat()
		period_end = _now()

		records = self._call_log.get((tenant_id, key_id), [])
		# filter to window
		window: list[dict[str, Any]] = []
		for r in records:
			try:
				ts = datetime.fromisoformat(r["ts"])
				if ts.tzinfo is None:
					ts = ts.replace(tzinfo=timezone.utc)
				if ts >= cutoff:
					window.append(r)
			except (ValueError, KeyError):
				continue

		total = len(window)
		errors = sum(1 for r in window if r.get("status", 200) >= 400)
		latencies = sorted([r["latency_ms"] for r in window if "latency_ms" in r])

		def _percentile(data: list[float], p: float) -> float:
			if not data:
				return 0.0
			idx = int(len(data) * p / 100)
			return data[min(idx, len(data) - 1)]

		# per-endpoint aggregation
		ep_map: dict[str, list[dict[str, Any]]] = {}
		for r in window:
			ep = r.get("endpoint", "unknown")
			ep_map.setdefault(ep, []).append(r)

		by_endpoint = []
		for ep, ep_records in ep_map.items():
			ep_latencies = sorted([r["latency_ms"] for r in ep_records if "latency_ms" in r])
			by_endpoint.append(DpEndpointStats(
				endpoint=ep,
				calls=len(ep_records),
				errors=sum(1 for r in ep_records if r.get("status", 200) >= 400),
				latency_p50_ms=_percentile(ep_latencies, 50),
				latency_p95_ms=_percentile(ep_latencies, 95),
				latency_p99_ms=_percentile(ep_latencies, 99),
			))

		# quota pct — look up via key
		api_key = self._api_keys.get((tenant_id, key_id))
		quota_used_pct: float | None = None
		if api_key and api_key.rate_limits.requests_per_day:
			quota_used_pct = min(100.0, total / api_key.rate_limits.requests_per_day * 100)

		return DpUsageStats(
			tenant_id=tenant_id,
			key_id=key_id,
			period_start=period_start,
			period_end=period_end,
			period_days=period_days,
			total_calls=total,
			total_errors=errors,
			error_rate=errors / total if total else 0.0,
			latency_p50_ms=_percentile(latencies, 50),
			latency_p95_ms=_percentile(latencies, 95),
			latency_p99_ms=_percentile(latencies, 99),
			by_endpoint=by_endpoint,
			quota_used_pct=quota_used_pct,
		)

	# -----------------------------------------------------------------------
	# OpenAPI browser proxy
	# -----------------------------------------------------------------------

	async def get_openapi_spec(self, capability_id: str) -> dict[str, Any]:
		"""Proxy the OpenAPI spec for a capability.

		In production this should delegate to the capability's own
		/openapi.json endpoint.  This stub returns a minimal valid spec.

		Keyword: get_openapi_spec, openapi_browser
		"""
		_guard_str(capability_id, "capability_id")
		return {
			"openapi": "3.1.0",
			"info": {
				"title": capability_id,
				"version": "1.0.0",
				"description": f"OpenAPI spec for APG capability: {capability_id}",
			},
			"paths": {},
			"components": {},
			"x-apg-capability": capability_id,
		}

	# -----------------------------------------------------------------------
	# Capability contract access
	# -----------------------------------------------------------------------

	def get_contract(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return the capability contract for this tenant.

		Keyword: get_contract, capability_contract
		"""
		return get_capability_contract(tenant_id)
