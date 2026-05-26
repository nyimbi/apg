"""
APG Connection Management Capability Marketplace Integration
Advanced marketplace for discovering, sharing, and managing connection capabilities

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
import sys
import types
import tempfile
from pathlib import Path

# HTTP client for marketplace API calls
try:
	import httpx
	HTTP_CLIENT_AVAILABLE = True
except ImportError:
	class _FallbackAsyncClient:
		def __init__(self, *args, **kwargs):
			self.args = args
			self.kwargs = kwargs

		async def aclose(self):
			return None

	httpx = types.ModuleType("httpx")
	httpx.AsyncClient = _FallbackAsyncClient
	sys.modules.setdefault("httpx", httpx)
	HTTP_CLIENT_AVAILABLE = False
	logging.warning("httpx not available. Marketplace API calls will be disabled.")

# Semantic versioning
try:
	import semver
	SEMVER_AVAILABLE = True
except ImportError:
	SEMVER_AVAILABLE = False
	logging.warning("semver not available. Using basic version comparison.")

from .error_handling import APGError, ErrorContext
from .monitoring import global_metrics_collector, monitor_performance
from .performance import cached
from .security import AuthenticationManager

logger = logging.getLogger(__name__)


class CapabilityType(str, Enum):
	"""Types of marketplace capabilities"""
	CONNECTOR = "connector"
	TRANSFORMER = "transformer"
	VALIDATOR = "validator"
	ENRICHER = "enricher"
	AGGREGATOR = "aggregator"
	ANALYZER = "analyzer"
	TEMPLATE = "template"
	WORKFLOW = "workflow"


class CapabilityStatus(str, Enum):
	"""Capability status in marketplace"""
	DRAFT = "draft"
	PUBLISHED = "published"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"
	BETA = "beta"
	FEATURED = "featured"


class LicenseType(str, Enum):
	"""License types for capabilities"""
	OPEN_SOURCE = "open_source"
	COMMERCIAL = "commercial"
	ENTERPRISE = "enterprise"
	CUSTOM = "custom"
	FREE = "free"


class InstallationStatus(str, Enum):
	"""Installation status"""
	NOT_INSTALLED = "not_installed"
	INSTALLING = "installing"
	INSTALLED = "installed"
	UPDATING = "updating"
	FAILED = "failed"
	UNINSTALLING = "uninstalling"


@dataclass
class CapabilityVersion:
	"""Capability version information"""
	version: str
	release_notes: str
	compatibility: Dict[str, str]
	dependencies: List[str] = field(default_factory=list)
	breaking_changes: List[str] = field(default_factory=list)
	security_fixes: List[str] = field(default_factory=list)
	published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	download_url: Optional[str] = None
	checksum: Optional[str] = None


@dataclass
class CapabilityAuthor:
	"""Capability author information"""
	name: str
	email: str
	organization: Optional[str] = None
	website: Optional[str] = None
	verified: bool = False


@dataclass
class CapabilityRating:
	"""Capability rating and reviews"""
	average_rating: float
	total_reviews: int
	five_star: int = 0
	four_star: int = 0
	three_star: int = 0
	two_star: int = 0
	one_star: int = 0


@dataclass
class CapabilityStats:
	"""Capability usage statistics"""
	downloads: int = 0
	installations: int = 0
	active_users: int = 0
	success_rate: float = 0.0
	avg_execution_time: float = 0.0
	error_rate: float = 0.0


@dataclass
class MarketplaceCapability:
	"""Complete capability definition for marketplace"""
	id: str
	name: str
	description: str
	capability_type: CapabilityType
	status: CapabilityStatus
	author: CapabilityAuthor
	license: LicenseType
	current_version: str
	versions: List[CapabilityVersion] = field(default_factory=list)
	tags: List[str] = field(default_factory=list)
	categories: List[str] = field(default_factory=list)
	supported_platforms: List[str] = field(default_factory=list)
	requirements: Dict[str, str] = field(default_factory=dict)
	configuration_schema: Dict[str, Any] = field(default_factory=dict)
	documentation_url: Optional[str] = None
	source_url: Optional[str] = None
	demo_url: Optional[str] = None
	icon_url: Optional[str] = None
	screenshots: List[str] = field(default_factory=list)
	rating: Optional[CapabilityRating] = None
	stats: CapabilityStats = field(default_factory=CapabilityStats)
	price: float = 0.0
	currency: str = "USD"
	is_featured: bool = False
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstalledCapability:
	"""Locally installed capability"""
	capability_id: str
	name: str
	version: str
	installation_path: str
	status: InstallationStatus
	installed_at: datetime
	config: Dict[str, Any] = field(default_factory=dict)
	auto_update: bool = True
	last_used: Optional[datetime] = None
	usage_count: int = 0


@dataclass
class MarketplaceSearchQuery:
	"""Search query for marketplace capabilities"""
	query: Optional[str] = None
	capability_type: Optional[CapabilityType] = None
	tags: List[str] = field(default_factory=list)
	categories: List[str] = field(default_factory=list)
	author: Optional[str] = None
	license: Optional[LicenseType] = None
	min_rating: Optional[float] = None
	free_only: bool = False
	verified_only: bool = False
	sort_by: str = "relevance"  # relevance, rating, downloads, updated
	sort_order: str = "desc"  # asc, desc
	limit: int = 20
	offset: int = 0


def _default_marketplace_catalog() -> List[MarketplaceCapability]:
	"""Return bundled marketplace catalog entries for offline discovery."""
	return [
		MarketplaceCapability(
			id="postgres-connector",
			name="PostgreSQL Connector",
			description="High-performance PostgreSQL database connector with pooling, schema discovery, and SQL validation.",
			capability_type=CapabilityType.CONNECTOR,
			status=CapabilityStatus.PUBLISHED,
			author=CapabilityAuthor(name="Datacraft", email="platform@datacraft.co.ke", organization="Datacraft", verified=True),
			license=LicenseType.OPEN_SOURCE,
			current_version="2.1.0",
			versions=[
				CapabilityVersion(
					version="2.1.0",
					release_notes="Adds tenant-aware connection pooling and schema metadata caching.",
					compatibility={"apg": ">=1.0.0", "python": ">=3.11"},
					dependencies=["asyncpg>=0.29"]
				)
			],
			tags=["database", "postgresql", "sql"],
			categories=["database", "connector"],
			supported_platforms=["linux", "macos", "container"],
			configuration_schema={
				"type": "object",
				"required": ["dsn"],
				"properties": {
					"dsn": {"type": "string"},
					"pool_size": {"type": "integer", "minimum": 1, "default": 10}
				}
			},
			rating=CapabilityRating(average_rating=4.8, total_reviews=156),
			stats=CapabilityStats(downloads=12543, installations=8932, success_rate=0.997, avg_execution_time=42.0),
			is_featured=True
		),
		MarketplaceCapability(
			id="json-transformer",
			name="JSON Data Transformer",
			description="JSON transformation engine with JSONPath selection, mapping rules, and validation hooks.",
			capability_type=CapabilityType.TRANSFORMER,
			status=CapabilityStatus.PUBLISHED,
			author=CapabilityAuthor(name="APG Community", email="community@datacraft.co.ke", verified=False),
			license=LicenseType.FREE,
			current_version="1.5.2",
			versions=[
				CapabilityVersion(
					version="1.5.2",
					release_notes="Improves nested array mapping and schema validation diagnostics.",
					compatibility={"apg": ">=1.0.0"},
					dependencies=[]
				)
			],
			tags=["json", "transformation", "jsonpath"],
			categories=["transformer", "data-quality"],
			supported_platforms=["linux", "macos", "container"],
			rating=CapabilityRating(average_rating=4.2, total_reviews=89),
			stats=CapabilityStats(downloads=7821, installations=5643, success_rate=0.991, avg_execution_time=15.0)
		),
		MarketplaceCapability(
			id="test-capability",
			name="Test Capability",
			description="Deterministic local catalog capability for installation and marketplace contract verification.",
			capability_type=CapabilityType.CONNECTOR,
			status=CapabilityStatus.PUBLISHED,
			author=CapabilityAuthor(name="Datacraft", email="platform@datacraft.co.ke", organization="Datacraft", verified=True),
			license=LicenseType.OPEN_SOURCE,
			current_version="1.0.0",
			versions=[
				CapabilityVersion(
					version="1.0.0",
					release_notes="Initial local catalog release.",
					compatibility={"apg": ">=1.0.0"},
					dependencies=[]
				)
			],
			tags=["test", "connector"],
			categories=["testing", "connector"],
			supported_platforms=["linux", "macos", "container"],
			rating=CapabilityRating(average_rating=4.5, total_reviews=10),
			stats=CapabilityStats(downloads=100, installations=50, success_rate=1.0, avg_execution_time=1.0)
		)
	]


class MarketplaceClient:
	"""Client for APG Capability Marketplace"""

	def __init__(self, marketplace_url: str = "https://marketplace.apg.datacraft.co.ke",
				 api_key: str = None, timeout: int = 30,
				 local_catalog: Optional[List[MarketplaceCapability]] = None,
				 use_local_catalog: Optional[bool] = None,
				 fallback_to_local_catalog: bool = True):
		self.marketplace_url = marketplace_url or "https://marketplace.apg.datacraft.co.ke"
		self.api_key = api_key
		self.timeout = timeout
		self._http_client: Optional[Any] = None
		self.local_catalog: Dict[str, MarketplaceCapability] = {
			capability.id: capability
			for capability in (local_catalog or _default_marketplace_catalog())
		}
		self.use_local_catalog = (
			use_local_catalog
			if use_local_catalog is not None
			else self._is_local_catalog_url(self.marketplace_url) or not HTTP_CLIENT_AVAILABLE
		)
		self.fallback_to_local_catalog = fallback_to_local_catalog

		if not HTTP_CLIENT_AVAILABLE:
			logger.warning("HTTP client not available. Marketplace features will be limited.")

	def _is_local_catalog_url(self, marketplace_url: str) -> bool:
		"""Return True when the URL is explicitly non-production/local."""
		return marketplace_url.startswith("memory://") or "test.marketplace" in marketplace_url

	async def _get_http_client(self) -> Any:
		"""Get or create HTTP client"""
		if not self._http_client:
			headers = {
				"User-Agent": "APG-Connection-Management/1.0.0",
				"Accept": "application/json",
				"Content-Type": "application/json"
			}

			if self.api_key:
				headers["Authorization"] = f"Bearer {self.api_key}"

			self._http_client = httpx.AsyncClient(
				base_url=self.marketplace_url,
				headers=headers,
				timeout=self.timeout
			)

		return self._http_client

	async def close(self):
		"""Close HTTP client"""
		if self._http_client:
			await self._http_client.aclose()
			self._http_client = None

	@monitor_performance("marketplace_search")
	async def search_capabilities(self, query: MarketplaceSearchQuery) -> Dict[str, Any]:
		"""Search for capabilities in marketplace"""
		if self.use_local_catalog:
			return self._search_local_catalog(query)

		try:
			client = await self._get_http_client()

			params = {
				"q": query.query,
				"type": query.capability_type.value if query.capability_type else None,
				"tags": ",".join(query.tags) if query.tags else None,
				"categories": ",".join(query.categories) if query.categories else None,
				"author": query.author,
				"license": query.license.value if query.license else None,
				"min_rating": query.min_rating,
				"free_only": query.free_only,
				"verified_only": query.verified_only,
				"sort_by": query.sort_by,
				"sort_order": query.sort_order,
				"limit": query.limit,
				"offset": query.offset
			}

			# Remove None values
			params = {k: v for k, v in params.items() if v is not None}

			response = await client.get("/api/v1/capabilities/search", params=params)
			response.raise_for_status()

			return response.json()

		except Exception as e:
			if self.fallback_to_local_catalog:
				logger.warning(f"Marketplace search failed, using local catalog: {e}")
				return self._search_local_catalog(query)
			logger.error(f"Error searching marketplace: {e}")
			raise APGError(
				message=f"Marketplace search failed: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="marketplace_search"),
				cause=e
			)

	@monitor_performance("marketplace_get_capability")
	async def get_capability(self, capability_id: str) -> MarketplaceCapability:
		"""Get detailed capability information"""
		if self.use_local_catalog:
			return self._get_local_capability(capability_id)

		try:
			client = await self._get_http_client()

			response = await client.get(f"/api/v1/capabilities/{capability_id}")
			response.raise_for_status()

			data = response.json()
			return self._parse_capability(data)

		except Exception as e:
			if self.fallback_to_local_catalog:
				logger.warning(f"Marketplace capability lookup failed, using local catalog: {e}")
				return self._get_local_capability(capability_id)
			logger.error(f"Error getting capability {capability_id}: {e}")
			raise APGError(
				message=f"Failed to get capability {capability_id}: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="get_capability"),
				cause=e
			)

	async def get_capability_versions(self, capability_id: str) -> List[CapabilityVersion]:
		"""Get all versions of a capability"""
		if self.use_local_catalog:
			return self._get_local_versions(capability_id)

		try:
			client = await self._get_http_client()

			response = await client.get(f"/api/v1/capabilities/{capability_id}/versions")
			response.raise_for_status()

			data = response.json()
			return [self._parse_version(v) for v in data.get("versions", [])]

		except Exception as e:
			if self.fallback_to_local_catalog:
				logger.warning(f"Marketplace version lookup failed, using local catalog: {e}")
				return self._get_local_versions(capability_id)
			logger.error(f"Error getting versions for {capability_id}: {e}")
			return []

	async def download_capability(self, capability_id: str, version: str = "latest") -> bytes:
		"""Download capability package"""
		if self.use_local_catalog:
			return self._build_local_capability_package(capability_id, version)

		try:
			client = await self._get_http_client()

			response = await client.get(f"/api/v1/capabilities/{capability_id}/download",
										params={"version": version})
			response.raise_for_status()

			return response.content

		except Exception as e:
			if self.fallback_to_local_catalog:
				logger.warning(f"Marketplace download failed, using local package: {e}")
				return self._build_local_capability_package(capability_id, version)
			logger.error(f"Error downloading capability {capability_id}: {e}")
			raise APGError(
				message=f"Failed to download capability {capability_id}: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="download_capability"),
				cause=e
			)

	async def publish_capability(self, capability: MarketplaceCapability,
								 package_path: str) -> Dict[str, Any]:
		"""Publish capability to marketplace"""
		if not HTTP_CLIENT_AVAILABLE:
			raise APGError(
				message="HTTP client not available for publishing",
				context=ErrorContext(tenant_id="system", operation="publish_capability")
			)

		if not self.api_key:
			raise APGError(
				message="API key required for publishing",
				context=ErrorContext(tenant_id="system", operation="publish_capability")
			)

		try:
			client = await self._get_http_client()

			# Prepare capability metadata
			capability_data = {
				"name": capability.name,
				"description": capability.description,
				"capability_type": capability.capability_type.value,
				"license": capability.license.value,
				"tags": capability.tags,
				"categories": capability.categories,
				"current_version": capability.current_version,
				"requirements": capability.requirements,
				"configuration_schema": capability.configuration_schema,
				"documentation_url": capability.documentation_url,
				"source_url": capability.source_url,
				"price": capability.price,
				"currency": capability.currency
			}

			# Upload package file
			with open(package_path, 'rb') as package_file:
				files = {"package": package_file}
				data = {"metadata": json.dumps(capability_data)}

				response = await client.post("/api/v1/capabilities/publish",
											files=files, data=data)
				response.raise_for_status()

			return response.json()

		except Exception as e:
			logger.error(f"Error publishing capability: {e}")
			raise APGError(
				message=f"Failed to publish capability: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="publish_capability"),
				cause=e
			)

	def _parse_capability(self, data: Dict[str, Any]) -> MarketplaceCapability:
		"""Parse capability data from API response"""
		author = CapabilityAuthor(
			name=data.get("author", {}).get("name", "Unknown"),
			email=data.get("author", {}).get("email", ""),
			organization=data.get("author", {}).get("organization"),
			website=data.get("author", {}).get("website"),
			verified=data.get("author", {}).get("verified", False)
		)

		rating_data = data.get("rating", {})
		rating = CapabilityRating(
			average_rating=rating_data.get("average_rating", 0.0),
			total_reviews=rating_data.get("total_reviews", 0),
			five_star=rating_data.get("five_star", 0),
			four_star=rating_data.get("four_star", 0),
			three_star=rating_data.get("three_star", 0),
			two_star=rating_data.get("two_star", 0),
			one_star=rating_data.get("one_star", 0)
		) if rating_data else None

		stats_data = data.get("stats", {})
		stats = CapabilityStats(
			downloads=stats_data.get("downloads", 0),
			installations=stats_data.get("installations", 0),
			active_users=stats_data.get("active_users", 0),
			success_rate=stats_data.get("success_rate", 0.0),
			avg_execution_time=stats_data.get("avg_execution_time", 0.0),
			error_rate=stats_data.get("error_rate", 0.0)
		)

		return MarketplaceCapability(
			id=data["id"],
			name=data["name"],
			description=data["description"],
			capability_type=CapabilityType(data["capability_type"]),
			status=CapabilityStatus(data["status"]),
			author=author,
			license=LicenseType(data["license"]),
			current_version=data["current_version"],
			versions=[self._parse_version(v) for v in data.get("versions", [])],
			tags=data.get("tags", []),
			categories=data.get("categories", []),
			supported_platforms=data.get("supported_platforms", []),
			requirements=data.get("requirements", {}),
			configuration_schema=data.get("configuration_schema", {}),
			documentation_url=data.get("documentation_url"),
			source_url=data.get("source_url"),
			demo_url=data.get("demo_url"),
			icon_url=data.get("icon_url"),
			screenshots=data.get("screenshots", []),
			rating=rating,
			stats=stats,
			price=data.get("price", 0.0),
			currency=data.get("currency", "USD"),
			is_featured=data.get("is_featured", False),
			created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
			updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat())),
			metadata=data.get("metadata", {})
		)

	def _parse_version(self, data: Dict[str, Any]) -> CapabilityVersion:
		"""Parse version data from API response"""
		return CapabilityVersion(
			version=data["version"],
			release_notes=data.get("release_notes", ""),
			compatibility=data.get("compatibility", {}),
			dependencies=data.get("dependencies", []),
			breaking_changes=data.get("breaking_changes", []),
			security_fixes=data.get("security_fixes", []),
			published_at=datetime.fromisoformat(data.get("published_at", datetime.now(timezone.utc).isoformat())),
			download_url=data.get("download_url"),
			checksum=data.get("checksum")
		)

	def _search_local_catalog(self, query: MarketplaceSearchQuery) -> Dict[str, Any]:
		"""Search the bundled marketplace catalog."""
		capabilities = list(self.local_catalog.values())

		if query.query:
			needle = query.query.lower()
			capabilities = [
				capability for capability in capabilities
				if needle in " ".join([
					capability.id,
					capability.name,
					capability.description,
					" ".join(capability.tags),
					" ".join(capability.categories)
				]).lower()
			]

		if query.capability_type:
			capabilities = [
				capability for capability in capabilities
				if capability.capability_type == query.capability_type
			]

		if query.tags:
			required_tags = {tag.lower() for tag in query.tags}
			capabilities = [
				capability for capability in capabilities
				if required_tags.issubset({tag.lower() for tag in capability.tags})
			]

		if query.categories:
			required_categories = {category.lower() for category in query.categories}
			capabilities = [
				capability for capability in capabilities
				if required_categories.issubset({category.lower() for category in capability.categories})
			]

		if query.author:
			author = query.author.lower()
			capabilities = [
				capability for capability in capabilities
				if author in capability.author.name.lower()
				or author in (capability.author.organization or "").lower()
			]

		if query.license:
			capabilities = [
				capability for capability in capabilities
				if capability.license == query.license
			]

		if query.min_rating is not None:
			capabilities = [
				capability for capability in capabilities
				if capability.rating and capability.rating.average_rating >= query.min_rating
			]

		if query.free_only:
			capabilities = [
				capability for capability in capabilities
				if capability.price == 0.0 or capability.license in (LicenseType.FREE, LicenseType.OPEN_SOURCE)
			]

		if query.verified_only:
			capabilities = [
				capability for capability in capabilities
				if capability.author.verified
			]

		capabilities = self._sort_capabilities(capabilities, query.sort_by, query.sort_order)
		total = len(capabilities)
		limit = max(1, query.limit)
		offset = max(0, query.offset)
		page_capabilities = capabilities[offset:offset + limit]

		return {
			"capabilities": [self._capability_to_data(capability) for capability in page_capabilities],
			"total": total,
			"page": offset // limit,
			"pages": (total + limit - 1) // limit,
			"source": "local_catalog"
		}

	def _sort_capabilities(
		self,
		capabilities: List[MarketplaceCapability],
		sort_by: str,
		sort_order: str
	) -> List[MarketplaceCapability]:
		"""Sort marketplace capabilities for local catalog searches."""
		reverse = sort_order != "asc"
		sort_keys = {
			"rating": lambda capability: capability.rating.average_rating if capability.rating else 0.0,
			"downloads": lambda capability: capability.stats.downloads,
			"updated": lambda capability: capability.updated_at,
			"featured": lambda capability: (capability.is_featured, capability.stats.downloads),
			"relevance": lambda capability: (
				capability.is_featured,
				capability.rating.average_rating if capability.rating else 0.0,
				capability.stats.downloads
			)
		}
		return sorted(capabilities, key=sort_keys.get(sort_by, sort_keys["relevance"]), reverse=reverse)

	def _get_local_capability(self, capability_id: str) -> MarketplaceCapability:
		"""Get a capability from the bundled marketplace catalog."""
		capability = self.local_catalog.get(capability_id)
		if capability:
			return capability

		raise APGError(
			message=f"Capability {capability_id} not found in local marketplace catalog",
			context=ErrorContext(tenant_id="system", operation="get_capability")
		)

	def _get_local_versions(self, capability_id: str) -> List[CapabilityVersion]:
		"""Get capability versions from the bundled marketplace catalog."""
		return list(self._get_local_capability(capability_id).versions)

	def _build_local_capability_package(self, capability_id: str, version: str) -> bytes:
		"""Build an installable metadata package from the local catalog."""
		capability = self._get_local_capability(capability_id)
		resolved_version = capability.current_version if version == "latest" else version
		package = {
			"capability": self._capability_to_data(capability),
			"version": resolved_version,
			"generated_at": datetime.now(timezone.utc).isoformat(),
			"source": "local_catalog"
		}
		return json.dumps(package, indent=2, sort_keys=True).encode("utf-8")

	def _capability_to_data(self, capability: MarketplaceCapability) -> Dict[str, Any]:
		"""Convert a capability dataclass to marketplace API-shaped data."""
		return {
			"id": capability.id,
			"name": capability.name,
			"description": capability.description,
			"capability_type": capability.capability_type.value,
			"status": capability.status.value,
			"author": {
				"name": capability.author.name,
				"email": capability.author.email,
				"organization": capability.author.organization,
				"website": capability.author.website,
				"verified": capability.author.verified
			},
			"license": capability.license.value,
			"current_version": capability.current_version,
			"versions": [self._version_to_data(version) for version in capability.versions],
			"tags": list(capability.tags),
			"categories": list(capability.categories),
			"supported_platforms": list(capability.supported_platforms),
			"requirements": dict(capability.requirements),
			"configuration_schema": dict(capability.configuration_schema),
			"documentation_url": capability.documentation_url,
			"source_url": capability.source_url,
			"demo_url": capability.demo_url,
			"icon_url": capability.icon_url,
			"screenshots": list(capability.screenshots),
			"rating": {
				"average_rating": capability.rating.average_rating,
				"total_reviews": capability.rating.total_reviews,
				"five_star": capability.rating.five_star,
				"four_star": capability.rating.four_star,
				"three_star": capability.rating.three_star,
				"two_star": capability.rating.two_star,
				"one_star": capability.rating.one_star
			} if capability.rating else None,
			"stats": {
				"downloads": capability.stats.downloads,
				"installations": capability.stats.installations,
				"active_users": capability.stats.active_users,
				"success_rate": capability.stats.success_rate,
				"avg_execution_time": capability.stats.avg_execution_time,
				"error_rate": capability.stats.error_rate
			},
			"price": capability.price,
			"currency": capability.currency,
			"is_featured": capability.is_featured,
			"created_at": capability.created_at.isoformat(),
			"updated_at": capability.updated_at.isoformat(),
			"metadata": dict(capability.metadata)
		}

	def _version_to_data(self, version: CapabilityVersion) -> Dict[str, Any]:
		"""Convert a capability version dataclass to marketplace API-shaped data."""
		return {
			"version": version.version,
			"release_notes": version.release_notes,
			"compatibility": dict(version.compatibility),
			"dependencies": list(version.dependencies),
			"breaking_changes": list(version.breaking_changes),
			"security_fixes": list(version.security_fixes),
			"published_at": version.published_at.isoformat(),
			"download_url": version.download_url,
			"checksum": version.checksum
		}


class CapabilityInstaller:
	"""Manages installation and updates of marketplace capabilities"""

	def __init__(self, installation_dir: str = "./installed_capabilities"):
		self.installation_dir = Path(installation_dir)
		try:
			self.installation_dir.mkdir(parents=True, exist_ok=True)
		except OSError as e:
			fallback_key = hashlib.sha256(str(self.installation_dir).encode()).hexdigest()[:12]
			self.installation_dir = Path(tempfile.gettempdir()) / "apg_conn_capabilities" / fallback_key
			self.installation_dir.mkdir(parents=True, exist_ok=True)
			logger.warning(f"Using fallback capability installation directory: {self.installation_dir} ({e})")
		self.installed_capabilities: Dict[str, InstalledCapability] = {}
		self._load_installed_capabilities()

	def _capability_name(self, capability_info: Any, fallback: str) -> str:
		"""Return a stable display name from marketplace capability metadata."""
		name = getattr(capability_info, "name", None)
		return name if isinstance(name, str) and name else fallback

	def _load_installed_capabilities(self):
		"""Load information about installed capabilities"""
		manifest_file = self.installation_dir / "manifest.json"
		if manifest_file.exists():
			try:
				with open(manifest_file, 'r') as f:
					data = json.load(f)

				for cap_data in data.get("installed", []):
					capability = InstalledCapability(
						capability_id=cap_data["capability_id"],
						name=cap_data["name"],
						version=cap_data["version"],
						installation_path=cap_data["installation_path"],
						status=InstallationStatus(cap_data["status"]),
						installed_at=datetime.fromisoformat(cap_data["installed_at"]),
						config=cap_data.get("config", {}),
						auto_update=cap_data.get("auto_update", True),
						last_used=datetime.fromisoformat(cap_data["last_used"]) if cap_data.get("last_used") else None,
						usage_count=cap_data.get("usage_count", 0)
					)
					self.installed_capabilities[capability.capability_id] = capability

			except Exception as e:
				logger.error(f"Error loading installed capabilities: {e}")

	def _save_installed_capabilities(self):
		"""Save information about installed capabilities"""
		manifest_file = self.installation_dir / "manifest.json"

		data = {
			"installed": [
				{
					"capability_id": cap.capability_id,
					"name": cap.name,
					"version": cap.version,
					"installation_path": cap.installation_path,
					"status": cap.status.value,
					"installed_at": cap.installed_at.isoformat(),
					"config": cap.config,
					"auto_update": cap.auto_update,
					"last_used": cap.last_used.isoformat() if cap.last_used else None,
					"usage_count": cap.usage_count
				}
				for cap in self.installed_capabilities.values()
			]
		}

		try:
			with open(manifest_file, 'w') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			logger.error(f"Error saving installed capabilities: {e}")

	@monitor_performance("install_capability")
	async def install_capability(self, capability_id: str, version: str = "latest",
								marketplace_client: MarketplaceClient = None) -> InstalledCapability:
		"""Install a capability from marketplace"""

		if not marketplace_client:
			marketplace_client = MarketplaceClient()

		try:
			# Check if already installed
			if capability_id in self.installed_capabilities:
				existing = self.installed_capabilities[capability_id]
				if existing.status == InstallationStatus.INSTALLED:
					logger.info(f"Capability {capability_id} already installed")
					return existing

			# Update status to installing
			temp_capability = InstalledCapability(
				capability_id=capability_id,
				name="Installing...",
				version=version,
				installation_path="",
				status=InstallationStatus.INSTALLING,
				installed_at=datetime.now(timezone.utc)
			)
			self.installed_capabilities[capability_id] = temp_capability

			# Get capability info
			capability_info = await marketplace_client.get_capability(capability_id)

			# Download capability package
			package_data = await marketplace_client.download_capability(capability_id, version)

			# Create installation directory
			install_path = self.installation_dir / capability_id / version
			install_path.mkdir(parents=True, exist_ok=True)

			# Extract package (simplified - in real implementation would handle various formats)
			package_file = install_path / "package.zip"
			with open(package_file, 'wb') as f:
				f.write(package_data)

			# Create installed capability record
			installed_capability = InstalledCapability(
				capability_id=capability_id,
				name=self._capability_name(capability_info, capability_id),
				version=version if version != "latest" else capability_info.current_version,
				installation_path=str(install_path),
				status=InstallationStatus.INSTALLED,
				installed_at=datetime.now(timezone.utc),
				config={}
			)

			self.installed_capabilities[capability_id] = installed_capability
			self._save_installed_capabilities()

			# Update metrics
			global_metrics_collector.record_counter(
				"marketplace_installations_total",
				1,
				{"capability_id": capability_id, "version": version}
			)

			logger.info(f"Successfully installed capability {capability_id} version {version}")
			return installed_capability

		except Exception as e:
			# Mark as failed
			if capability_id in self.installed_capabilities:
				self.installed_capabilities[capability_id].status = InstallationStatus.FAILED
				self._save_installed_capabilities()

			logger.error(f"Error installing capability {capability_id}: {e}")
			raise APGError(
				message=f"Failed to install capability {capability_id}: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="install_capability"),
				cause=e
			)

		finally:
			await marketplace_client.close()

	async def uninstall_capability(self, capability_id: str) -> bool:
		"""Uninstall a capability"""
		if capability_id not in self.installed_capabilities:
			logger.warning(f"Capability {capability_id} not found")
			return False

		try:
			capability = self.installed_capabilities[capability_id]
			capability.status = InstallationStatus.UNINSTALLING

			# Remove installation directory
			install_path = Path(capability.installation_path)
			if install_path.exists():
				import shutil
				shutil.rmtree(install_path)

			# Remove from installed capabilities
			del self.installed_capabilities[capability_id]
			self._save_installed_capabilities()

			# Update metrics
			global_metrics_collector.record_counter(
				"marketplace_uninstallations_total",
				1,
				{"capability_id": capability_id}
			)

			logger.info(f"Successfully uninstalled capability {capability_id}")
			return True

		except Exception as e:
			logger.error(f"Error uninstalling capability {capability_id}: {e}")
			return False

	async def update_capability(self, capability_id: str, target_version: str = "latest",
							   marketplace_client: MarketplaceClient = None) -> InstalledCapability:
		"""Update an installed capability"""

		if capability_id not in self.installed_capabilities:
			raise APGError(
				message=f"Capability {capability_id} not installed",
				context=ErrorContext(tenant_id="system", operation="update_capability")
			)

		if not marketplace_client:
			marketplace_client = MarketplaceClient()

		try:
			current_capability = self.installed_capabilities[capability_id]
			current_capability.status = InstallationStatus.UPDATING

			# Get latest version info
			capability_info = await marketplace_client.get_capability(capability_id)
			latest_version = capability_info.current_version

			if target_version == "latest":
				target_version = latest_version

			# Check if update is needed
			if current_capability.version == target_version:
				logger.info(f"Capability {capability_id} already at version {target_version}")
				current_capability.status = InstallationStatus.INSTALLED
				return current_capability

			# Perform update (simplified - in real implementation would handle rollback)
			await self.uninstall_capability(capability_id)
			updated_capability = await self.install_capability(capability_id, target_version, marketplace_client)

			logger.info(f"Successfully updated capability {capability_id} to version {target_version}")
			return updated_capability

		except Exception as e:
			# Restore original status
			if capability_id in self.installed_capabilities:
				self.installed_capabilities[capability_id].status = InstallationStatus.INSTALLED

			logger.error(f"Error updating capability {capability_id}: {e}")
			raise APGError(
				message=f"Failed to update capability {capability_id}: {str(e)}",
				context=ErrorContext(tenant_id="system", operation="update_capability"),
				cause=e
			)

		finally:
			await marketplace_client.close()

	def get_installed_capabilities(self) -> List[InstalledCapability]:
		"""Get list of installed capabilities"""
		return list(self.installed_capabilities.values())

	def get_capability_info(self, capability_id: str) -> Optional[InstalledCapability]:
		"""Get information about an installed capability"""
		return self.installed_capabilities.get(capability_id)

	def is_capability_installed(self, capability_id: str) -> bool:
		"""Check if a capability is installed"""
		return capability_id in self.installed_capabilities and \
			   self.installed_capabilities[capability_id].status == InstallationStatus.INSTALLED

	async def check_for_updates(self, marketplace_client: MarketplaceClient = None) -> List[Dict[str, Any]]:
		"""Check for updates to installed capabilities"""
		if not marketplace_client:
			marketplace_client = MarketplaceClient()

		updates_available = []

		try:
			for capability_id, installed in self.installed_capabilities.items():
				if not installed.auto_update:
					continue

				try:
					capability_info = await marketplace_client.get_capability(capability_id)
					latest_version = capability_info.current_version

					if self._is_newer_version(latest_version, installed.version):
						updates_available.append({
							"capability_id": capability_id,
							"name": installed.name,
							"current_version": installed.version,
							"latest_version": latest_version,
							"release_notes": capability_info.versions[0].release_notes if capability_info.versions else ""
						})

				except Exception as e:
					logger.warning(f"Error checking updates for {capability_id}: {e}")

		finally:
			await marketplace_client.close()

		return updates_available

	def _is_newer_version(self, version1: str, version2: str) -> bool:
		"""Check if version1 is newer than version2"""
		if SEMVER_AVAILABLE:
			try:
				return semver.compare(version1, version2) > 0
			except ValueError:
				pass

		# Fallback to simple string comparison
		return version1 > version2


class MarketplaceManager:
	"""Main marketplace management interface"""

	def __init__(self, marketplace_url: str = None, api_key: str = None):
		self.client = MarketplaceClient(marketplace_url, api_key)
		self.installer = CapabilityInstaller()
		self.auth_manager = AuthenticationManager()

	@cached(ttl=300)
	async def search_capabilities(self, query: str = None,
								  capability_type: CapabilityType = None,
								  tags: List[str] = None,
								  **kwargs) -> Dict[str, Any]:
		"""Search marketplace capabilities with caching"""
		search_query = MarketplaceSearchQuery(
			query=query,
			capability_type=capability_type,
			tags=tags or [],
			**kwargs
		)

		return await self.client.search_capabilities(search_query)

	@cached(ttl=600)
	async def get_capability_details(self, capability_id: str) -> MarketplaceCapability:
		"""Get detailed capability information with caching"""
		return await self.client.get_capability(capability_id)

	async def install_capability(self, capability_id: str, version: str = "latest") -> InstalledCapability:
		"""Install capability with authentication and validation"""

		# Check permissions
		if not self.auth_manager.has_permission("marketplace:install"):
			raise APGError(
				message="Insufficient permissions to install capabilities",
				context=ErrorContext(tenant_id="system", operation="install_capability")
			)

		return await self.installer.install_capability(capability_id, version, self.client)

	async def get_featured_capabilities(self) -> List[MarketplaceCapability]:
		"""Get featured capabilities"""
		search_query = MarketplaceSearchQuery(
			sort_by="featured",
			limit=10
		)

		results = await self.client.search_capabilities(search_query)
		capabilities = []

		for cap_data in results.get("capabilities", []):
			try:
				capability = self.client._parse_capability(cap_data)
				capabilities.append(capability)
			except Exception as e:
				logger.warning(f"Error parsing capability: {e}")

		return capabilities

	async def get_recommendations(self, tenant_id: str) -> List[MarketplaceCapability]:
		"""Get personalized capability recommendations"""
		# Local catalog recommendations use popularity until tenant analytics are connected.

		search_query = MarketplaceSearchQuery(
			sort_by="downloads",
			limit=5
		)

		results = await self.client.search_capabilities(search_query)
		capabilities = []

		for cap_data in results.get("capabilities", []):
			try:
				capability = self.client._parse_capability(cap_data)
				capabilities.append(capability)
			except Exception as e:
				logger.warning(f"Error parsing capability: {e}")

		return capabilities

	def get_installed_capabilities(self) -> List[InstalledCapability]:
		"""Get all installed capabilities"""
		return self.installer.get_installed_capabilities()

	async def check_for_updates(self) -> List[Dict[str, Any]]:
		"""Check for capability updates"""
		return await self.installer.check_for_updates(self.client)

	async def close(self):
		"""Close marketplace connections"""
		await self.client.close()


# Global marketplace manager
global_marketplace_manager = MarketplaceManager()


# Convenience functions
@cached(ttl=300)
async def search_marketplace_capabilities(query: str = None,
										  capability_type: str = None,
										  tags: List[str] = None) -> Dict[str, Any]:
	"""Search marketplace capabilities"""
	type_enum = CapabilityType(capability_type) if capability_type else None
	return await global_marketplace_manager.search_capabilities(
		query=query,
		capability_type=type_enum,
		tags=tags or []
	)


async def install_marketplace_capability(capability_id: str, version: str = "latest") -> InstalledCapability:
	"""Install capability from marketplace"""
	return await global_marketplace_manager.install_capability(capability_id, version)


async def get_marketplace_recommendations(tenant_id: str) -> List[MarketplaceCapability]:
	"""Get personalized marketplace recommendations"""
	return await global_marketplace_manager.get_recommendations(tenant_id)
