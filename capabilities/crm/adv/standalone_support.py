"""
Standalone support shims for optional CRM integration modules.

These classes keep optional adapters importable in a lightweight APG checkout
without requiring network clients, Redis, or legacy UI packages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CRMError(Exception):
	"""CRM module error used when the legacy views module is unavailable."""


class CRMResponse(BaseModel):
	"""Minimal response envelope for standalone optional CRM modules."""

	success: bool = True
	data: Optional[Any] = None
	message: Optional[str] = None
	error: Optional[str] = None


class PaginationParams(BaseModel):
	"""Minimal pagination parameters for standalone optional CRM modules."""

	page: int = Field(default=1, ge=1)
	page_size: int = Field(default=50, ge=1)


class NoOpRedis:
	"""Async Redis-compatible no-op client for import and light smoke tests."""

	async def get(self, *_args, **_kwargs):
		return None

	async def set(self, *_args, **_kwargs):
		return True

	async def setex(self, *_args, **_kwargs):
		return True

	async def incr(self, *_args, **_kwargs):
		return 1

	async def expire(self, *_args, **_kwargs):
		return True

	async def delete(self, *_args, **_kwargs):
		return 0

	async def keys(self, *_args, **_kwargs):
		return []

	async def eval(self, *_args, **_kwargs):
		return 1

	async def ping(self):
		return True

	async def lpush(self, *_args, **_kwargs):
		return 1

	async def ltrim(self, *_args, **_kwargs):
		return True

	async def lrange(self, *_args, **_kwargs):
		return []

	async def hgetall(self, *_args, **_kwargs):
		return {}

	async def hset(self, *_args, **_kwargs):
		return 1

	async def publish(self, *_args, **_kwargs):
		return 0

	async def close(self):
		return None


class NoOpRedisModule:
	"""Redis module-shaped object with from_url and Redis attributes."""

	Redis = NoOpRedis

	@staticmethod
	def from_url(*_args, **_kwargs) -> NoOpRedis:
		return NoOpRedis()


class NoOpAsyncpgPool:
	"""Asyncpg Pool-compatible placeholder used for import-time annotations."""


class NoOpAsyncpgConnection:
	"""Asyncpg Connection-compatible placeholder used for annotations."""


class NoOpAsyncpgModule:
	"""Asyncpg module-shaped object for standalone imports."""

	Pool = NoOpAsyncpgPool
	Connection = NoOpAsyncpgConnection

	@staticmethod
	async def create_pool(*_args, **_kwargs) -> NoOpAsyncpgPool:
		return NoOpAsyncpgPool()


class NoOpClientTimeout:
	"""AIOHTTP ClientTimeout-compatible value object."""

	def __init__(self, **kwargs):
		self.kwargs = kwargs


class NoOpBasicAuth:
	"""AIOHTTP BasicAuth-compatible value object."""

	def __init__(self, login: str, password: str = ""):
		self.login = login
		self.password = password


class NoOpHTTPResponse:
	"""Async context manager response for no-op HTTP sessions."""

	status = 204

	async def __aenter__(self):
		return self

	async def __aexit__(self, *_exc_info):
		return False

	async def json(self) -> Dict[str, Any]:
		return {}

	async def text(self) -> str:
		return ""


class NoOpClientSession:
	"""AIOHTTP ClientSession-compatible no-op session."""

	def __init__(self, *_args, **_kwargs):
		self.closed = False

	async def close(self):
		self.closed = True

	def request(self, *_args, **_kwargs) -> NoOpHTTPResponse:
		return NoOpHTTPResponse()

	def get(self, *_args, **_kwargs) -> NoOpHTTPResponse:
		return NoOpHTTPResponse()

	def post(self, *_args, **_kwargs) -> NoOpHTTPResponse:
		return NoOpHTTPResponse()

	def put(self, *_args, **_kwargs) -> NoOpHTTPResponse:
		return NoOpHTTPResponse()

	def patch(self, *_args, **_kwargs) -> NoOpHTTPResponse:
		return NoOpHTTPResponse()


class NoOpAiohttpModule:
	"""AIOHTTP module-shaped object used when aiohttp is absent."""

	ClientSession = NoOpClientSession
	ClientTimeout = NoOpClientTimeout
	BasicAuth = NoOpBasicAuth

	class TCPConnector:
		def __init__(self, *_args, **_kwargs):
			pass


class StandaloneCoreObject:
	"""Permissive APG-core fallback object."""

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def __getattr__(self, name: str):
		async def _async_noop(*_args, **_kwargs):
			return None

		return _async_noop


class StandaloneEvent(StandaloneCoreObject):
	"""Minimal event fallback."""


class StandaloneEventBus(StandaloneCoreObject):
	"""Minimal event bus fallback."""

	async def publish(self, *_args, **_kwargs):
		return None

	async def subscribe(self, *_args, **_kwargs):
		return None


class StandaloneCapability(StandaloneCoreObject):
	"""Minimal APG capability fallback."""


class StandaloneCapabilityInfo(StandaloneCoreObject):
	"""Minimal APG capability info fallback."""


class StandaloneCapabilityStatus:
	INITIALIZING = "initializing"
	READY = "ready"
	STOPPING = "stopping"
	STOPPED = "stopped"
	ACTIVE = "active"
	INACTIVE = "inactive"
	HEALTHY = "healthy"
	UNHEALTHY = "unhealthy"
	ERROR = "error"
	FAILED = "failed"


class StandaloneRegistry(StandaloneCoreObject):
	async def register(self, *_args, **_kwargs):
		return True


capability_registry = StandaloneRegistry()


class StandaloneModel:
	"""Permissive placeholder for legacy UI model classes."""

	def __init__(self, **values):
		self.__dict__.update(values)


class StandaloneView:
	"""Permissive placeholder for legacy Flask-AppBuilder view classes."""


class StandaloneWidget:
	"""Permissive placeholder for legacy Flask-AppBuilder widgets."""


class StandaloneForm:
	"""Permissive placeholder for legacy WTForms forms."""


class StandaloneField:
	"""Permissive placeholder for legacy WTForms fields and validators."""

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs


def standalone_expose(*_args, **_kwargs):
	def _decorator(func):
		return func
	return _decorator


def standalone_has_access(func=None, *_args, **_kwargs):
	if func is None:
		return lambda wrapped: wrapped
	return func


standalone_protect = standalone_has_access


class StandaloneSQLAInterface:
	def __init__(self, model):
		self.model = model


class StandaloneSQLA:
	"""Minimal SQLA placeholder for legacy blueprint annotations."""


def standalone_lazy_gettext(value: str) -> str:
	return value
