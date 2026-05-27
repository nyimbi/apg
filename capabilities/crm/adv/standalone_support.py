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

	async def lpush(self, *_args, **_kwargs):
		return 1

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
