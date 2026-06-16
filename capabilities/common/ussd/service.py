"""
USSD Engine Service — session state machine, flow registry, and menu renderer.

Coordinates:
  - USSDSession lifecycle (create / navigate / end / TTL reap)
  - FlowDefinition registry (in-process dict; swap for Redis in production)
  - Menu rendering with i18n and 182-char truncation
  - Gateway-agnostic: callers pass USSDRequest, receive USSDResponse

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import (
	AT_MAX_CHARS,
	SESSION_TTL_SECONDS,
	FlowDefinition,
	SessionState,
	USSDMenu,
	USSDMenuItem,
	USSDRequest,
	USSDResponse,
	USSDSession,
	uuid7str,
)

_log = logging.getLogger(__name__)

_REAPER_INTERVAL: int = 30

_SYSTEM_PHRASES: dict[str, dict[str, str]] = {
	"back":    {"en": "0. Back",   "sw": "0. Rudi",   "am": "0. ተመለስ",   "fr": "0. Retour"},
	"exit":    {"en": "00. Exit",  "sw": "00. Toka",  "am": "00. ውጣ",    "fr": "00. Quitter"},
	"invalid": {
		"en": "Invalid option. Try again.",
		"sw": "Chaguo batili. Jaribu tena.",
		"am": "ልክ ያልሆነ አማራጭ። እንደገና ይሞክሩ።",
		"fr": "Option invalide. Réessayez.",
	},
}


class USSDEngineService:
	"""
	USSD state machine service.

	In-process stores for sessions and flows.  Production deployments should
	inject Redis-backed replacements via the constructor hooks.

	Session key: (msisdn, service_code) → canonical session_id lookup.
	All public methods are async.
	"""

	def __init__(
		self,
		session_ttl: int = SESSION_TTL_SECONDS,
		nats_client: Any | None = None,
	) -> None:
		# session_id → USSDSession
		self._sessions: dict[str, USSDSession] = {}
		# (msisdn, service_code) → session_id  (latest active only)
		self._session_keys: dict[tuple[str, str], str] = {}
		# flow_id → FlowDefinition
		self._flows: dict[str, FlowDefinition] = {}
		# service_code → flow_id  (active flow per service code)
		self._flow_index: dict[str, str] = {}

		self._ttl = session_ttl
		self._nats = nats_client
		self._reaper_task: asyncio.Task[None] | None = None  # type: ignore[type-arg]

	# ── Lifecycle ─────────────────────────────────────────────────────────────

	async def start(self) -> None:
		"""Start background TTL reaper.  Call once at app startup."""
		if self._reaper_task is None or self._reaper_task.done():
			self._reaper_task = asyncio.create_task(self._reaper_loop())
			_log.info("USSDEngineService started (ttl=%ds)", self._ttl)

	async def stop(self) -> None:
		"""Cancel background reaper gracefully."""
		if self._reaper_task and not self._reaper_task.done():
			self._reaper_task.cancel()
			try:
				await self._reaper_task
			except asyncio.CancelledError:
				pass
		_log.info("USSDEngineService stopped")

	# ── Primary gateway entry point ───────────────────────────────────────────

	async def handle_request(self, req: USSDRequest) -> USSDResponse:
		"""
		Main state machine entry point.

		1. Look up or create a session keyed by (msisdn, service_code).
		2. Advance the session by one hop.
		3. Resolve the active flow for service_code and navigate the menu tree.
		4. Return a USSDResponse (text <= 182 chars, continue_session flag).

		A missing flow falls back to a passthrough echo useful for gateway-only
		integration testing.
		"""
		assert req.msisdn, "msisdn required"
		assert req.service_code, "service_code required"

		# ── Resolve or create session ─────────────────────────────────────────
		key = (req.msisdn, req.service_code)
		sid = self._session_keys.get(key)
		session: USSDSession | None = None

		if sid:
			session = await self.get_session(sid)

		if session is None:
			session = await self._create_session(req)

		# ── Guard: ended / expired ────────────────────────────────────────────
		if session.state != SessionState.ACTIVE:
			return USSDResponse(
				text="Session is no longer active. Please dial again.",
				continue_session=False,
				session_id=session.session_id,
			)

		# ── Guard: hop limit ──────────────────────────────────────────────────
		if session.hop_count >= 30:
			await self.end_session(session.session_id, reason="max_hops")
			return USSDResponse(
				text="Session limit reached. Please try again.",
				continue_session=False,
				session_id=session.session_id,
			)

		# ── Advance session ───────────────────────────────────────────────────
		if req.text:
			session.input_history.append(req.text)
		session.hop_count += 1
		session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)

		# ── Navigate menu tree ────────────────────────────────────────────────
		flow = self._get_active_flow(req.service_code)
		if flow is None:
			# No flow registered — passthrough for integration testing
			return USSDResponse(
				text=f"[{req.service_code}] Hop {session.hop_count}. Input: {req.text or '(none)'}",
				continue_session=True,
				session_id=session.session_id,
				hop_count=session.hop_count,
			)

		try:
			response = await self._navigate_flow(session, flow, req.text)
		except Exception as exc:
			_log.error("flow navigation error session=%s: %s", session.session_id, exc, exc_info=True)
			session.state = SessionState.ERROR
			return USSDResponse(
				text="Service error. Please try again later.",
				continue_session=False,
				session_id=session.session_id,
			)

		if not response.continue_session:
			await self.end_session(session.session_id, reason="end_screen")

		return response

	# ── Flow management ───────────────────────────────────────────────────────

	async def create_flow(self, flow: FlowDefinition) -> str:
		"""
		Register a flow definition.

		If another flow is already active for the same service_code it is
		deactivated (active=False) before the new one takes over.

		Returns the flow_id.
		"""
		assert flow.service_code, "service_code required"
		assert flow.root_menu_id in flow.menus, f"root_menu_id '{flow.root_menu_id}' missing from menus"

		existing_fid = self._flow_index.get(flow.service_code)
		if existing_fid and existing_fid != flow.flow_id:
			old = self._flows.get(existing_fid)
			if old:
				old.active = False
				_log.info("deactivated flow %s for %s", existing_fid, flow.service_code)

		self._flows[flow.flow_id] = flow
		if flow.active:
			self._flow_index[flow.service_code] = flow.flow_id

		_log.info("flow registered: id=%s service=%s menus=%d", flow.flow_id, flow.service_code, len(flow.menus))
		await self._publish("ussd.flow_created", {"flow_id": flow.flow_id, "service_code": flow.service_code})
		return flow.flow_id

	def get_flow(self, flow_id: str) -> FlowDefinition | None:
		"""Return a flow definition by id."""
		return self._flows.get(flow_id)

	# ── Session management ────────────────────────────────────────────────────

	async def get_session(self, session_id: str) -> USSDSession | None:
		"""Return active session or None if not found / expired."""
		session = self._sessions.get(session_id)
		if session is None:
			return None
		if datetime.now(timezone.utc) >= session.expires_at and session.state == SessionState.ACTIVE:
			await self._expire(session)
			return None
		return session

	async def end_session(self, session_id: str, reason: str = "user_exit") -> None:
		"""Terminate a session.  No-op if already ended."""
		session = self._sessions.get(session_id)
		if session is None:
			return
		if session.state != SessionState.ACTIVE:
			return
		session.state = SessionState.ENDED
		session.ended_at = datetime.now(timezone.utc)
		# Keep in store for 60 s post-end for audit
		session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
		_log.info("session ended: id=%s reason=%s hops=%d", session_id, reason, session.hop_count)
		await self._publish("ussd.session_ended", {
			"session_id": session_id,
			"msisdn": session.msisdn,
			"service_code": session.service_code,
			"reason": reason,
			"hop_count": session.hop_count,
		})

	# ── Menu rendering ────────────────────────────────────────────────────────

	def render_menu(self, menu: USSDMenu, lang: str = "en") -> str:
		"""
		Render a USSDMenu to a USSD-safe string (<= AT_MAX_CHARS).

		Applies translation lookup (falls back to 'en' then base text),
		appends back/exit lines if enabled, and hard-truncates at 182 chars.
		"""
		title = self._t(menu.title, menu.titles, lang)
		body  = self._t(menu.body,  menu.bodies,  lang)

		lines: list[str] = []
		if title:
			lines.append(title)
		if body:
			lines.append(body)

		for item in menu.items:
			label = self._t(item.label, item.labels, lang)
			lines.append(f"{item.key}. {label}")

		if menu.show_back:
			lines.append(_SYSTEM_PHRASES["back"].get(lang) or _SYSTEM_PHRASES["back"]["en"])
		if menu.show_exit:
			lines.append(_SYSTEM_PHRASES["exit"].get(lang) or _SYSTEM_PHRASES["exit"]["en"])

		rendered = "\n".join(lines)
		if len(rendered) > AT_MAX_CHARS:
			rendered = rendered[:AT_MAX_CHARS]
		return rendered

	# ── Internal helpers ──────────────────────────────────────────────────────

	async def _create_session(self, req: USSDRequest) -> USSDSession:
		"""Create and register a new session, overwriting the key slot."""
		now = datetime.now(timezone.utc)
		session = USSDSession(
			session_id=req.session_id,
			msisdn=req.msisdn,
			service_code=req.service_code,
			tenant_id=req.tenant_id,
			gateway=req.gateway,
			language=req.language,
			created_at=now,
			expires_at=now + timedelta(seconds=self._ttl),
		)
		self._sessions[session.session_id] = session
		self._session_keys[(req.msisdn, req.service_code)] = session.session_id
		_log.info(
			"session created: id=%s msisdn=%s service=%s gw=%s",
			session.session_id, req.msisdn, req.service_code, req.gateway,
		)
		await self._publish("ussd.session_started", {
			"session_id": session.session_id,
			"msisdn": req.msisdn,
			"service_code": req.service_code,
			"gateway": req.gateway,
		})
		return session

	def _get_active_flow(self, service_code: str) -> FlowDefinition | None:
		fid = self._flow_index.get(service_code)
		if fid is None:
			return None
		return self._flows.get(fid)

	async def _navigate_flow(
		self,
		session: USSDSession,
		flow: FlowDefinition,
		input_text: str,
	) -> USSDResponse:
		"""Advance one hop through the flow's menu tree."""
		lang = session.language or flow.default_language

		# First hop: empty text → show root menu
		if not input_text:
			menu = flow.menus.get(flow.root_menu_id)
			if menu is None:
				raise ValueError(f"root menu '{flow.root_menu_id}' missing")
			session.current_menu_id = menu.menu_id
			session.navigation_stack = [menu.menu_id]
			return USSDResponse(
				text=self.render_menu(menu, lang),
				continue_session=not menu.is_terminal,
				session_id=session.session_id,
				menu_id=menu.menu_id,
				hop_count=session.hop_count,
			)

		# Subsequent hops: AT sends full concatenated chain "1*2*3"
		# We only care about the last segment for the current menu
		last_input = input_text.split("*")[-1].strip()

		current_menu = flow.menus.get(session.current_menu_id)
		if current_menu is None:
			raise ValueError(f"current menu '{session.current_menu_id}' missing from flow")

		# Back navigation
		if last_input == "0" and current_menu.show_back:
			if len(session.navigation_stack) > 1:
				session.navigation_stack.pop()
				session.current_menu_id = session.navigation_stack[-1]
			target_menu = flow.menus.get(session.current_menu_id)
			if target_menu is None:
				raise ValueError(f"back target menu '{session.current_menu_id}' missing")
			return USSDResponse(
				text=self.render_menu(target_menu, lang),
				continue_session=True,
				session_id=session.session_id,
				menu_id=target_menu.menu_id,
				hop_count=session.hop_count,
			)

		# Exit navigation
		if last_input == "00" and current_menu.show_exit:
			return USSDResponse(
				text="Thank you. Goodbye.",
				continue_session=False,
				session_id=session.session_id,
				menu_id=session.current_menu_id,
				hop_count=session.hop_count,
			)

		# Find matching item
		item: USSDMenuItem | None = next(
			(i for i in current_menu.items if i.key == last_input), None
		)

		if item is None:
			invalid_msg = _SYSTEM_PHRASES["invalid"].get(lang) or _SYSTEM_PHRASES["invalid"]["en"]
			return USSDResponse(
				text=f"{invalid_msg}\n{self.render_menu(current_menu, lang)}"[:AT_MAX_CHARS],
				continue_session=True,
				session_id=session.session_id,
				menu_id=current_menu.menu_id,
				hop_count=session.hop_count,
			)

		if item.action == "end":
			return USSDResponse(
				text=item.target or "Thank you.",
				continue_session=False,
				session_id=session.session_id,
				menu_id=current_menu.menu_id,
				hop_count=session.hop_count,
			)

		if item.action in ("navigate", "execute") and item.target:
			target_menu = flow.menus.get(item.target)
			if target_menu is None:
				raise ValueError(f"target menu '{item.target}' not found in flow")
			session.navigation_stack.append(target_menu.menu_id)
			session.current_menu_id = target_menu.menu_id
			return USSDResponse(
				text=self.render_menu(target_menu, lang),
				continue_session=not target_menu.is_terminal,
				session_id=session.session_id,
				menu_id=target_menu.menu_id,
				hop_count=session.hop_count,
			)

		# input action — store value and stay on current menu
		if item.target:
			session.data[item.target] = last_input
		return USSDResponse(
			text=self.render_menu(current_menu, lang),
			continue_session=True,
			session_id=session.session_id,
			menu_id=current_menu.menu_id,
			hop_count=session.hop_count,
		)

	async def _expire(self, session: USSDSession) -> None:
		if session.state == SessionState.ACTIVE:
			session.state = SessionState.EXPIRED
			session.ended_at = datetime.now(timezone.utc)
			_log.info("session expired: id=%s msisdn=%s", session.session_id, session.msisdn)
			await self._publish("ussd.session_ended", {
				"session_id": session.session_id,
				"msisdn": session.msisdn,
				"service_code": session.service_code,
				"reason": "ttl_expired",
				"hop_count": session.hop_count,
			})

	async def _reaper_loop(self) -> None:
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
				_log.debug("reaper purged %d sessions", len(to_purge))

	async def _publish(self, subject: str, payload: dict[str, Any]) -> None:
		if self._nats is None:
			_log.debug("NATS %s: %s", subject, payload)
			return
		try:
			import json
			await self._nats.publish(subject, json.dumps(payload, default=str).encode())
		except Exception as exc:
			_log.warning("NATS publish failed subject=%s: %s", subject, exc)

	@staticmethod
	def _t(base: str, translations: dict[str, str], lang: str) -> str:
		"""Resolve translation: lang → en → base."""
		if lang in translations:
			return translations[lang]
		if "en" in translations:
			return translations["en"]
		return base
