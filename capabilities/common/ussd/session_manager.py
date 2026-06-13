"""
USSD Session State Machine — top-level session manager for common/ussd.

Wraps the lower-level UssdEngService with a clean, contract-aligned API:
  - start_session / get_session / end_session
  - navigate(session_id, input_text) → UssdResponse
  - Strict 3-minute TTL with asyncio background reaper
  - NATS event emission for platform integration
  - 'Us' Pydantic model prefix throughout
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

_log = logging.getLogger(__name__)

SESSION_TTL_SECONDS: int = 180  # 3 minutes — hard USSD network limit
_REAPER_INTERVAL: int = 30      # background TTL reaper cadence


# ── Pydantic models ──────────────────────────────────────────────────────────

class UsSession(BaseModel):
	"""Live USSD session record."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	session_id: str = Field(default_factory=uuid7str)
	phone: str
	service_code: str
	tenant_id: str = "default"
	current_menu: str = "main"
	navigation_stack: list[str] = Field(default_factory=lambda: ["main"])
	data: dict[str, Any] = Field(default_factory=dict)
	input_history: list[str] = Field(default_factory=list)
	hop_count: int = 0
	language: str = "en"
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	expires_at: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)
	)
	ended_at: datetime | None = None
	state: str = "active"  # active | ended | expired | error


class UsResponse(BaseModel):
	"""Response returned from navigate() to the gateway."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	text: str
	continue_session: bool  # True → CON, False → END
	session_id: str
	hop_count: int = 0
	menu_id: str = ""


class UsSessionStartRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	phone: str
	service_code: str
	tenant_id: str = "default"
	language: str = "en"
	metadata: dict[str, Any] = Field(default_factory=dict)


class UsNavigateRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	session_id: str
	input_text: str = ""


# ── State machine ────────────────────────────────────────────────────────────

class UssdSessionManager:
	"""
	USSD session state machine.

	Maintains an in-process session store keyed by session_id with automatic
	TTL expiry.  Menu logic is delegated to a pluggable menu_resolver callable:

	    async def menu_resolver(
	        session: UsSession, input_text: str
	    ) -> UsResponse: ...

	If no resolver is provided the manager falls back to a pass-through that
	echoes the current menu ID — useful for testing gateway integration in
	isolation before wiring real menus.

	NATS events are emitted via an optional nats_client duck-typed to:
	    async def publish(subject: str, payload: bytes) -> None
	"""

	def __init__(
		self,
		menu_resolver: Any | None = None,
		nats_client: Any | None = None,
		session_ttl: int = SESSION_TTL_SECONDS,
	) -> None:
		self._sessions: dict[str, UsSession] = {}
		self._menu_resolver = menu_resolver
		self._nats = nats_client
		self._ttl = session_ttl
		self._reaper_task: asyncio.Task[None] | None = None  # type: ignore[type-arg]

	# ── Lifecycle ─────────────────────────────────────────────────────────────

	async def start(self) -> None:
		"""Start background TTL reaper.  Call once at app startup."""
		if self._reaper_task is None or self._reaper_task.done():
			self._reaper_task = asyncio.create_task(self._reaper_loop())
			_log.info("UssdSessionManager started — TTL=%ds reaper=%ds", self._ttl, _REAPER_INTERVAL)

	async def stop(self) -> None:
		"""Gracefully stop background reaper."""
		if self._reaper_task and not self._reaper_task.done():
			self._reaper_task.cancel()
			try:
				await self._reaper_task
			except asyncio.CancelledError:
				pass
		_log.info("UssdSessionManager stopped")

	# ── Public API ────────────────────────────────────────────────────────────

	async def start_session(
		self,
		phone: str,
		service_code: str,
		tenant_id: str = "default",
		language: str = "en",
		metadata: dict[str, Any] | None = None,
	) -> UsSession:
		"""
		Create and register a new USSD session.

		Emits: ussd.session_started
		"""
		assert phone, "phone must be non-empty"
		assert service_code, "service_code must be non-empty"
		now = datetime.now(timezone.utc)
		session = UsSession(
			session_id=uuid7str(),
			phone=phone,
			service_code=service_code,
			tenant_id=tenant_id,
			language=language,
			created_at=now,
			expires_at=now + timedelta(seconds=self._ttl),
			data=dict(metadata or {}),
		)
		self._sessions[session.session_id] = session
		_log.info(
			"session started: id=%s phone=%s service=%s tenant=%s",
			session.session_id, phone, service_code, tenant_id,
		)
		await self._publish("ussd.session_started", {
			"session_id": session.session_id,
			"phone": phone,
			"service_code": service_code,
			"tenant_id": tenant_id,
		})
		return session

	async def get_session(self, session_id: str) -> UsSession | None:
		"""Return session if active and not expired, else None."""
		session = self._sessions.get(session_id)
		if session is None:
			return None
		if self._is_expired(session):
			await self._expire(session)
			return None
		return session

	async def end_session(self, session_id: str, reason: str = "user_exit") -> None:
		"""
		Terminate a session.

		Emits: ussd.session_ended
		"""
		session = self._sessions.get(session_id)
		if session is None:
			_log.debug("end_session: session not found %s", session_id)
			return
		session.state = "ended"
		session.ended_at = datetime.now(timezone.utc)
		_log.info("session ended: id=%s reason=%s", session_id, reason)
		await self._publish("ussd.session_ended", {
			"session_id": session_id,
			"phone": session.phone,
			"service_code": session.service_code,
			"tenant_id": session.tenant_id,
			"reason": reason,
			"hop_count": session.hop_count,
		})
		# Retain ended sessions briefly for audit — reaper removes after TTL
		session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)

	async def navigate(self, session_id: str, input_text: str = "") -> UsResponse:
		"""
		Process subscriber input and advance the session state machine.

		Returns UsResponse with rendered screen text and continue/end signal.
		Automatically expires sessions that exceed TTL.
		"""
		session = await self.get_session(session_id)
		if session is None:
			return UsResponse(
				text="Your session has expired. Please dial again.",
				continue_session=False,
				session_id=session_id,
			)

		if session.state != "active":
			return UsResponse(
				text="Session is no longer active.",
				continue_session=False,
				session_id=session_id,
				menu_id=session.current_menu,
			)

		if session.hop_count >= 30:
			await self.end_session(session_id, reason="max_hops")
			return UsResponse(
				text="Session limit reached. Please try again.",
				continue_session=False,
				session_id=session_id,
			)

		# Record input
		if input_text:
			session.input_history.append(input_text)
		session.hop_count += 1
		# Bump TTL on every hop — active sessions stay alive
		session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)

		# Delegate to menu resolver
		if self._menu_resolver is not None:
			try:
				response: UsResponse = await self._menu_resolver(session, input_text)
				# Sync state back
				if not response.continue_session:
					await self.end_session(session_id, reason="end_screen")
				return response
			except Exception as exc:
				_log.error("menu_resolver error session=%s: %s", session_id, exc)
				session.state = "error"
				return UsResponse(
					text="Service error. Please try again later.",
					continue_session=False,
					session_id=session_id,
					menu_id=session.current_menu,
				)

		# Passthrough fallback — useful for gateway integration tests
		return UsResponse(
			text=f"[menu:{session.current_menu}] Hop {session.hop_count}. Input: {input_text or '(none)'}",
			continue_session=True,
			session_id=session_id,
			hop_count=session.hop_count,
			menu_id=session.current_menu,
		)

	# ── Navigation helpers ────────────────────────────────────────────────────

	def push_menu(self, session: UsSession, menu_id: str) -> None:
		"""Push a menu onto the navigation stack and set as current."""
		session.navigation_stack.append(menu_id)
		session.current_menu = menu_id

	def pop_menu(self, session: UsSession) -> str | None:
		"""Navigate back — pop current menu, return previous or None at root."""
		if len(session.navigation_stack) <= 1:
			return None
		session.navigation_stack.pop()
		session.current_menu = session.navigation_stack[-1]
		return session.current_menu

	def reset_navigation(self, session: UsSession) -> None:
		"""Reset navigation stack to root menu."""
		root = session.navigation_stack[0] if session.navigation_stack else "main"
		session.navigation_stack = [root]
		session.current_menu = root

	# ── Queries ───────────────────────────────────────────────────────────────

	def list_sessions(
		self,
		tenant_id: str | None = None,
		state: str | None = None,
	) -> list[UsSession]:
		"""Return sessions, optionally filtered by tenant and/or state."""
		results = list(self._sessions.values())
		if tenant_id:
			results = [s for s in results if s.tenant_id == tenant_id]
		if state:
			results = [s for s in results if s.state == state]
		return results

	def active_count(self, tenant_id: str | None = None) -> int:
		"""Count active non-expired sessions."""
		now = datetime.now(timezone.utc)
		return sum(
			1 for s in self._sessions.values()
			if s.state == "active"
			and s.expires_at > now
			and (tenant_id is None or s.tenant_id == tenant_id)
		)

	# ── Internal ──────────────────────────────────────────────────────────────

	def _is_expired(self, session: UsSession) -> bool:
		return datetime.now(timezone.utc) >= session.expires_at

	async def _expire(self, session: UsSession) -> None:
		if session.state == "active":
			session.state = "expired"
			session.ended_at = datetime.now(timezone.utc)
			_log.info("session expired: id=%s phone=%s", session.session_id, session.phone)
			await self._publish("ussd.session_ended", {
				"session_id": session.session_id,
				"phone": session.phone,
				"service_code": session.service_code,
				"tenant_id": session.tenant_id,
				"reason": "ttl_expired",
				"hop_count": session.hop_count,
			})

	async def _reaper_loop(self) -> None:
		"""Background task — sweep expired sessions every _REAPER_INTERVAL seconds."""
		while True:
			await asyncio.sleep(_REAPER_INTERVAL)
			now = datetime.now(timezone.utc)
			to_purge: list[str] = []
			for sid, session in list(self._sessions.items()):
				if session.expires_at <= now:
					await self._expire(session)
					to_purge.append(sid)
			for sid in to_purge:
				self._sessions.pop(sid, None)
			if to_purge:
				_log.debug("reaper purged %d expired sessions", len(to_purge))

	async def _publish(self, subject: str, payload: dict[str, Any]) -> None:
		"""Emit a NATS event if a client is wired in, else log at DEBUG."""
		if self._nats is None:
			_log.debug("NATS event %s: %s", subject, payload)
			return
		try:
			import json
			await self._nats.publish(subject, json.dumps(payload, default=str).encode())
		except Exception as exc:
			_log.warning("NATS publish failed subject=%s: %s", subject, exc)
