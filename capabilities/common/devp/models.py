"""Pydantic v2 models for APG Developer Portal (Dp prefix)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())

except ImportError:  # pragma: no cover
	import uuid

	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


def _non_empty(v: str) -> str:
	assert v.strip(), "must be non-empty"
	return v


def _non_empty_list(v: list[str]) -> list[str]:
	assert len(v) > 0, "must contain at least one item"
	return v


def _valid_url(v: str) -> str:
	assert v.startswith("https://"), "webhook URL must use HTTPS"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
NonEmptyStrList = Annotated[list[str], AfterValidator(_non_empty_list)]
HttpsUrl = Annotated[str, AfterValidator(_valid_url)]


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class DpRateLimit(BaseModel):
	"""Embedded rate-limit specification for an API key."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	requests_per_second: int | None = None
	requests_per_minute: int | None = Field(default=60, ge=1)
	requests_per_hour: int | None = None
	requests_per_day: int | None = Field(default=10000, ge=1)
	requests_per_month: int | None = None
	burst_limit: int | None = None


class DpEndpoint(BaseModel):
	"""A single endpoint entry in an API product."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	method: NonEmptyStr
	path: NonEmptyStr
	description: str = ""
	scopes_required: list[str] = Field(default_factory=list)


class DpApiProduct(BaseModel):
	"""Published API product grouping capabilities and their endpoints.

	Keyword: api_product, product_catalog
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	description: str = ""
	# IDs of APG capabilities exposed by this product
	capabilities: list[str] = Field(default_factory=list)
	# published endpoints
	endpoints: list[DpEndpoint] = Field(default_factory=list)
	plan: NonEmptyStr  # free | starter | professional | enterprise | custom
	# monthly quota; None = unlimited
	monthly_call_quota: int | None = None
	# base price per month in minor currency units (e.g. cents); None = free
	price_minor_units: int | None = None
	currency: str = "KES"
	is_public: bool = True
	is_deprecated: bool = False
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


class DpApiKey(BaseModel):
	"""Developer API key — plaintext is never stored, only the hash.

	Keyword: api_key, developer_key
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	# SHA-256 hex digest of the raw key
	key_hash: NonEmptyStr
	name: NonEmptyStr
	owner_id: NonEmptyStr
	# bound developer application
	app_id: NonEmptyStr
	# list of permission scopes (e.g. ["read:orders", "write:orders"])
	scopes: list[str] = Field(default_factory=list)
	rate_limits: DpRateLimit = Field(default_factory=DpRateLimit)
	# active | revoked | suspended | expired
	status: NonEmptyStr = "active"
	expires_at: str | None = None
	last_used_at: str | None = None
	created_at: str = Field(default_factory=_now_iso)
	revoked_at: str | None = None
	revoked_by: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


class DpDeveloperApp(BaseModel):
	"""Developer application that groups API keys and subscriptions.

	Keyword: developer_app, client_app
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	description: str = ""
	owner_id: NonEmptyStr
	# IDs of API products this app is subscribed to
	api_products: list[str] = Field(default_factory=list)
	callback_urls: list[str] = Field(default_factory=list)
	is_active: bool = True
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


class DpEndpointStats(BaseModel):
	"""Per-endpoint breakdown within a usage stats record."""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	endpoint: NonEmptyStr
	calls: int = 0
	errors: int = 0
	latency_p50_ms: float = 0.0
	latency_p95_ms: float = 0.0
	latency_p99_ms: float = 0.0


class DpUsageStats(BaseModel):
	"""Per-key usage statistics over a reporting period.

	Keyword: usage_stats, api_analytics
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	key_id: NonEmptyStr
	period_start: NonEmptyStr
	period_end: NonEmptyStr
	period_days: int = Field(ge=1)
	# aggregate totals
	total_calls: int = 0
	total_errors: int = 0
	error_rate: float = 0.0
	latency_p50_ms: float = 0.0
	latency_p95_ms: float = 0.0
	latency_p99_ms: float = 0.0
	# per-endpoint breakdown
	by_endpoint: list[DpEndpointStats] = Field(default_factory=list)
	quota_used_pct: float | None = None
	generated_at: str = Field(default_factory=_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


class DpSubscription(BaseModel):
	"""Developer subscription linking a DeveloperApp to an ApiProduct.

	Keyword: subscription, api_subscription
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	developer_app_id: NonEmptyStr
	product_id: NonEmptyStr
	# pending | active | suspended | cancelled
	status: NonEmptyStr = "pending"
	plan: NonEmptyStr
	# ISO date when billing cycle started
	billing_cycle_start: str | None = None
	# ISO date of next renewal
	next_renewal_at: str | None = None
	# cumulative calls this billing cycle
	calls_this_cycle: int = 0
	created_at: str = Field(default_factory=_now_iso)
	activated_at: str | None = None
	cancelled_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()


class DpWebhookEndpoint(BaseModel):
	"""Developer-registered webhook for receiving portal lifecycle events.

	Keyword: webhook, event_subscription
	"""

	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	app_id: NonEmptyStr
	# HTTPS endpoint URL
	url: HttpsUrl
	# HMAC-SHA256 signing secret (stored as hash, never returned)
	secret_hash: NonEmptyStr
	# list of event types subscribed to
	events: NonEmptyStrList
	is_active: bool = True
	# consecutive delivery failures
	failure_count: int = 0
	last_delivery_at: str | None = None
	last_delivery_status: int | None = None
	created_at: str = Field(default_factory=_now_iso)
	updated_at: str = Field(default_factory=_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return self.model_dump()
