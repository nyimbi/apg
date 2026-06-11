"""USSD Engine service — session state machine, gateway integration, menu DSL."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

SUPPORTED_GATEWAYS = {"africastalking", "safaricom", "custom"}
SUPPORTED_ENVIRONMENTS = {"sandbox", "production"}
SUPPORTED_SESSION_STATES = {"active", "ended", "timeout", "error"}
SUPPORTED_MENU_ACTIONS = {"navigate", "execute", "end", "back", "input"}
USSD_MAX_RESPONSE_LENGTH = 182  # GSM 03.38 safe limit
USSD_MAX_HOPS = 30
USSD_DEFAULT_TIMEOUT = 180


class UssdEngService:
	"""In-memory USSD engine: session state machine, gateway config, menu DSL."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.sessions: dict[str, dict[str, Any]] = {}
		self.menus: dict[str, dict[str, Any]] = {}
		self.gateways: dict[str, dict[str, Any]] = {}
		self.service_codes: dict[str, str] = {}  # service_code -> root menu id
		self.session_variables: dict[str, dict[str, Any]] = {}
		self.handlers: dict[str, Any] = {}  # handler_name -> callable
		self._audit_events: list[dict[str, Any]] = []
		# I3: idempotency cache keyed by "session_id:hop:handler"
		self._idempotency_cache: dict[str, dict[str, Any]] = {}
		# I4: sliding-window rate-limit buckets keyed by "tenant:phone:service"
		self._rate_buckets: dict[str, list[float]] = {}
		# I6: menu version snapshots keyed by "composite_key:vN"
		self._menu_versions: dict[str, list[dict[str, Any]]] = {}
		# I11: dead-letter queue keyed by tenant_id
		self._dead_letters: list[dict[str, Any]] = []
		# I14: webhook registrations
		self._webhooks: list[dict[str, Any]] = []

	# ── Utility ─────────────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _emit(self, tenant_id: str, event_type: str, resource_id: str, resource_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"resource_id": resource_id,
			"resource_type": resource_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _truncate_ussd(self, text: str) -> str:
		"""Truncate to USSD safe length."""
		return text[:USSD_MAX_RESPONSE_LENGTH]

	def _render_menu(self, menu: dict[str, Any], session: dict[str, Any]) -> str:
		"""Render a menu dict to USSD string, substituting session variables."""
		body = menu.get("body", "")
		variables = session.get("variables", {})
		for key, val in variables.items():
			body = body.replace(f"{{{key}}}", str(val))
		lines = [menu.get("title", ""), body]
		items = menu.get("items", [])
		for i, item in enumerate(items, 1):
			cond = item.get("condition")
			if cond and not self._eval_condition(cond, session):
				continue
			lines.append(f"{i}. {item['label']}")
		return self._truncate_ussd("\n".join(filter(None, lines)))

	def _eval_condition(self, condition: str, session: dict[str, Any]) -> bool:
		"""Evaluate a simple condition expression against session variables."""
		variables = session.get("variables", {})
		try:
			# Only allow simple equality / inequality checks for safety
			match = re.match(r"(\w+)\s*(==|!=|>|<|>=|<=)\s*(.+)", condition.strip())
			if not match:
				return True
			key, op, rhs = match.group(1), match.group(2), match.group(3).strip().strip("'\"")
			lhs = str(variables.get(key, ""))
			ops = {"==": lhs == rhs, "!=": lhs != rhs, ">": lhs > rhs, "<": lhs < rhs, ">=": lhs >= rhs, "<=": lhs <= rhs}
			return ops.get(op, True)
		except Exception as exc:
			_log.debug("condition eval error '%s': %s", condition, exc)
			return True

	def _resolve_input(self, text: str) -> list[str]:
		"""Split USSD multi-hop input text into individual responses."""
		if not text:
			return []
		return [part for part in text.split("*") if part is not None]

	# ── Health & describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		active_sessions = sum(1 for s in self.sessions.values() if s["session_state"] == "active")
		return {
			"service": "ussd_eng",
			"status": "healthy",
			"active_sessions": active_sessions,
			"total_sessions": len(self.sessions),
			"registered_menus": len(self.menus),
			"registered_gateways": len(self.gateways),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		"""Return capability contract descriptor."""
		return {
			"capability_id": "ussd_eng",
			"domain": "common",
			"version": "1.0.0",
			"description": "USSD session state machine, gateway integration, menu DSL, session persistence",
			"supported_gateways": list(SUPPORTED_GATEWAYS),
			"max_response_length": USSD_MAX_RESPONSE_LENGTH,
			"max_hops": USSD_MAX_HOPS,
			"default_session_timeout": USSD_DEFAULT_TIMEOUT,
		}

	# ── Gateway management ───────────────────────────────────────────────────

	async def create_gateway(
		self,
		name: str,
		gateway_type: str,
		service_code: str,
		tenant_id: str | None = None,
		api_key: str | None = None,
		api_secret: str | None = None,
		username: str | None = None,
		webhook_url: str | None = None,
		environment: str = "sandbox",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(service_code, "service_code")
		if gateway_type not in SUPPORTED_GATEWAYS:
			raise ValueError(f"gateway_type must be one of {SUPPORTED_GATEWAYS}")
		if environment not in SUPPORTED_ENVIRONMENTS:
			raise ValueError(f"environment must be one of {SUPPORTED_ENVIRONMENTS}")
		record = {
			"id": self._record_id("gw"),
			"tenant_id": tenant,
			"name": name,
			"gateway_type": gateway_type,
			"service_code": service_code,
			"api_key": api_key,
			"api_secret": api_secret,
			"username": username,
			"webhook_url": webhook_url,
			"environment": environment,
			"status": "active",
			"session_count": 0,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.gateways[record["id"]] = record
		self._emit(tenant, "gateway_created", record["id"], "ussd_gateway", {"name": name, "gateway_type": gateway_type})
		_log.info("gateway created: %s (%s) for tenant %s", name, gateway_type, tenant)
		return deepcopy(record)

	async def get_gateway(self, gateway_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.gateways.get(gateway_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"gateway_not_found: {gateway_id}")
		return deepcopy(record)

	async def list_gateways(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.gateways.values() if r["tenant_id"] == tenant]

	async def update_gateway(
		self,
		gateway_id: str,
		tenant_id: str | None = None,
		webhook_url: str | None = None,
		environment: str | None = None,
		status: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.gateways.get(gateway_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"gateway_not_found: {gateway_id}")
		if webhook_url is not None:
			record["webhook_url"] = webhook_url
		if environment is not None:
			if environment not in SUPPORTED_ENVIRONMENTS:
				raise ValueError(f"environment must be one of {SUPPORTED_ENVIRONMENTS}")
			record["environment"] = environment
		if status is not None:
			record["status"] = status
		if metadata is not None:
			record["metadata"].update(metadata)
		record["updated_at"] = self._now()
		self._emit(tenant, "gateway_updated", gateway_id, "ussd_gateway")
		return deepcopy(record)

	async def delete_gateway(self, gateway_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.gateways.get(gateway_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"gateway_not_found: {gateway_id}")
		record["status"] = "deleted"
		record["updated_at"] = self._now()
		del self.gateways[gateway_id]
		self._emit(tenant, "gateway_deleted", gateway_id, "ussd_gateway")
		return deepcopy(record)

	# ── Menu management (DSL) ────────────────────────────────────────────────

	async def create_menu(
		self,
		menu_id: str,
		title: str,
		body: str,
		service_code: str,
		tenant_id: str | None = None,
		items: list[dict[str, Any]] | None = None,
		language: str = "en",
		is_end_screen: bool = False,
		timeout_seconds: int = USSD_DEFAULT_TIMEOUT,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(menu_id, "menu_id")
		guard_non_empty_string(title, "title")
		guard_non_empty_string(service_code, "service_code")
		record = {
			"id": self._record_id("menu"),
			"tenant_id": tenant,
			"menu_id": menu_id,
			"title": title,
			"body": body,
			"items": deepcopy(items or []),
			"service_code": service_code,
			"language": language,
			"is_end_screen": is_end_screen,
			"timeout_seconds": timeout_seconds,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		self.menus[composite_key] = record
		self._emit(tenant, "menu_created", record["id"], "ussd_menu", {"menu_id": menu_id, "service_code": service_code})
		_log.info("menu created: %s for service %s tenant %s", menu_id, service_code, tenant)
		return deepcopy(record)

	async def get_menu(self, menu_id: str, service_code: str, tenant_id: str | None = None, language: str = "en") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		record = self.menus.get(composite_key)
		if not record:
			# Fall back to default language
			composite_key = f"{tenant}:{service_code}:en:{menu_id}"
			record = self.menus.get(composite_key)
		if not record:
			raise KeyError(f"menu_not_found: {menu_id}")
		return deepcopy(record)

	async def list_menus(self, service_code: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		results = [deepcopy(r) for r in self.menus.values() if r["tenant_id"] == tenant]
		if service_code:
			results = [r for r in results if r["service_code"] == service_code]
		return results

	async def update_menu(
		self,
		menu_id: str,
		service_code: str,
		tenant_id: str | None = None,
		language: str = "en",
		title: str | None = None,
		body: str | None = None,
		items: list[dict[str, Any]] | None = None,
		is_end_screen: bool | None = None,
		timeout_seconds: int | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		record = self.menus.get(composite_key)
		if not record:
			raise KeyError(f"menu_not_found: {menu_id}")
		if title is not None:
			record["title"] = title
		if body is not None:
			record["body"] = body
		if items is not None:
			record["items"] = deepcopy(items)
		if is_end_screen is not None:
			record["is_end_screen"] = is_end_screen
		if timeout_seconds is not None:
			record["timeout_seconds"] = timeout_seconds
		record["updated_at"] = self._now()
		self._emit(tenant, "menu_updated", record["id"], "ussd_menu", {"menu_id": menu_id})
		return deepcopy(record)

	async def delete_menu(self, menu_id: str, service_code: str, tenant_id: str | None = None, language: str = "en") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		record = self.menus.get(composite_key)
		if not record:
			raise KeyError(f"menu_not_found: {menu_id}")
		del self.menus[composite_key]
		self._emit(tenant, "menu_deleted", record["id"], "ussd_menu", {"menu_id": menu_id})
		return deepcopy(record)

	async def set_root_menu(self, service_code: str, menu_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Designate a menu as the root (entry point) for a service code."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(service_code, "service_code")
		key = f"{tenant}:{service_code}"
		self.service_codes[key] = menu_id
		self._emit(tenant, "root_menu_set", service_code, "service_code", {"menu_id": menu_id})
		return {"tenant_id": tenant, "service_code": service_code, "root_menu": menu_id, "updated_at": self._now()}

	# ── Session lifecycle ────────────────────────────────────────────────────

	async def create_session(
		self,
		phone_number: str,
		service_code: str,
		gateway_session_id: str | None = None,
		tenant_id: str | None = None,
		gateway: str = "africastalking",
		language: str = "en",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(phone_number, "phone_number")
		guard_non_empty_string(service_code, "service_code")
		root_key = f"{tenant}:{service_code}"
		root_menu_id = self.service_codes.get(root_key, "main")
		session_id = self._record_id("sess", gateway_session_id)
		record = {
			"id": session_id,
			"tenant_id": tenant,
			"phone_number": phone_number,
			"service_code": service_code,
			"gateway": gateway,
			"language": language,
			"current_menu": root_menu_id,
			"session_state": "active",
			"variables": {},
			"input_history": [],
			"menu_history": [root_menu_id],
			"hop_count": 0,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
			"ended_at": None,
		}
		self.sessions[session_id] = record
		self.session_variables[session_id] = {}
		# Increment gateway session count
		for gw in self.gateways.values():
			if gw["tenant_id"] == tenant and gw["service_code"] == service_code:
				gw["session_count"] += 1
				break
		self._emit(tenant, "session_created", session_id, "ussd_session", {"phone": phone_number, "service_code": service_code})
		_log.info("session created: %s phone=%s service=%s", session_id, phone_number, service_code)
		return deepcopy(record)

	async def get_session(self, session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		return deepcopy(record)

	async def list_sessions(
		self,
		tenant_id: str | None = None,
		phone_number: str | None = None,
		service_code: str | None = None,
		session_state: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		results = [deepcopy(r) for r in self.sessions.values() if r["tenant_id"] == tenant]
		if phone_number:
			results = [r for r in results if r["phone_number"] == phone_number]
		if service_code:
			results = [r for r in results if r["service_code"] == service_code]
		if session_state:
			results = [r for r in results if r["session_state"] == session_state]
		return results

	async def update_session(
		self,
		session_id: str,
		tenant_id: str | None = None,
		current_menu: str | None = None,
		language: str | None = None,
		variables: dict[str, Any] | None = None,
		status: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		if current_menu is not None:
			record["current_menu"] = current_menu
		if language is not None:
			record["language"] = language
		if variables is not None:
			record["variables"].update(variables)
		if status is not None:
			if status not in SUPPORTED_SESSION_STATES:
				raise ValueError(f"status must be one of {SUPPORTED_SESSION_STATES}")
			record["session_state"] = status
		record["updated_at"] = self._now()
		self._emit(tenant, "session_updated", session_id, "ussd_session")
		return deepcopy(record)

	async def end_session(self, session_id: str, tenant_id: str | None = None, reason: str = "user_exit") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		record["session_state"] = "ended"
		record["ended_at"] = self._now()
		record["updated_at"] = self._now()
		record["end_reason"] = reason
		self._emit(tenant, "session_ended", session_id, "ussd_session", {"reason": reason})
		_log.info("session ended: %s reason=%s", session_id, reason)
		return deepcopy(record)

	async def delete_session(self, session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		del self.sessions[session_id]
		self.session_variables.pop(session_id, None)
		self._emit(tenant, "session_deleted", session_id, "ussd_session")
		return deepcopy(record)

	# ── Core USSD request handler ────────────────────────────────────────────

	async def handle_ussd_request(
		self,
		session_id: str,
		service_code: str,
		phone_number: str,
		text: str = "",
		tenant_id: str | None = None,
		gateway: str = "africastalking",
		language: str = "en",
	) -> dict[str, Any]:
		"""
		Handle an incoming USSD request (Africa's Talking / Safaricom callback).
		Returns a dict with response_type ('CON' or 'END') and body text.
		"""
		tenant = self._tenant(tenant_id)
		# Retrieve or create session
		session = self.sessions.get(session_id)
		if session is None or session["session_state"] == "ended":
			session_record = await self.create_session(
				phone_number=phone_number,
				service_code=service_code,
				gateway_session_id=session_id,
				tenant_id=tenant,
				gateway=gateway,
				language=language,
			)
			session = self.sessions[session_record["id"]]

		if session["hop_count"] >= USSD_MAX_HOPS:
			await self.end_session(session["id"], tenant, reason="max_hops_exceeded")
			return {"response_type": "END", "body": "Session limit reached. Please try again.", "session_id": session["id"]}

		# Parse input chain
		inputs = self._resolve_input(text)
		current_menu_id = session["current_menu"]

		try:
			menu = await self.get_menu(current_menu_id, service_code, tenant, session.get("language", language))
		except KeyError:
			await self.end_session(session["id"], tenant, reason="menu_not_found")
			return {"response_type": "END", "body": "Service unavailable. Please try again.", "session_id": session["id"]}

		# Process latest input (last element of multi-hop chain)
		latest_input = inputs[-1] if inputs else ""

		if latest_input:
			session["input_history"].append(latest_input)
			session["hop_count"] += 1
			# Resolve which item was selected
			try:
				item_index = int(latest_input) - 1
				items = [
					item for item in menu.get("items", [])
					if not item.get("condition") or self._eval_condition(item["condition"], session)
				]
				if 0 <= item_index < len(items):
					selected_item = items[item_index]
					action = selected_item.get("action", "navigate")

					if action == "navigate" and selected_item.get("target"):
						session["menu_history"].append(selected_item["target"])
						session["current_menu"] = selected_item["target"]
						current_menu_id = selected_item["target"]
						try:
							menu = await self.get_menu(current_menu_id, service_code, tenant, session.get("language", language))
						except KeyError:
							await self.end_session(session["id"], tenant, reason="target_menu_not_found")
							return {"response_type": "END", "body": "Navigation error. Please try again.", "session_id": session["id"]}

					elif action == "back":
						if len(session["menu_history"]) > 1:
							session["menu_history"].pop()
							session["current_menu"] = session["menu_history"][-1]
							current_menu_id = session["current_menu"]
							try:
								menu = await self.get_menu(current_menu_id, service_code, tenant, session.get("language", language))
							except KeyError:
								await self.end_session(session["id"], tenant, reason="back_menu_not_found")
								return {"response_type": "END", "body": "Navigation error.", "session_id": session["id"]}

					elif action == "execute" and selected_item.get("handler"):
						handler = self.handlers.get(selected_item["handler"])
						if handler:
							try:
								result = await handler(session, selected_item)
								session["variables"].update(result.get("variables", {}))
								if result.get("end_session"):
									await self.end_session(session["id"], tenant, reason="handler_end")
									return {"response_type": "END", "body": self._truncate_ussd(result.get("body", "Done.")), "session_id": session["id"]}
								if result.get("next_menu"):
									session["current_menu"] = result["next_menu"]
									current_menu_id = result["next_menu"]
									try:
										menu = await self.get_menu(current_menu_id, service_code, tenant, session.get("language", language))
									except KeyError as _exc:
										_log.debug("Handled exception: %s", _exc)
							except Exception as exc:
								_log.error("handler %s failed: %s", selected_item["handler"], exc)

					elif action == "end":
						await self.end_session(session["id"], tenant, reason="menu_end")
						end_body = self._render_menu(menu, session) if menu.get("is_end_screen") else selected_item.get("label", "Thank you.")
						return {"response_type": "END", "body": self._truncate_ussd(end_body), "session_id": session["id"]}

			except (ValueError, IndexError) as exc:
				_log.debug("input parse error session %s: %s", session["id"], exc)
				# Free-text input — store as variable
				current_items = menu.get("items", [])
				input_items = [i for i in current_items if i.get("action") == "input"]
				if input_items:
					var_name = input_items[0].get("target", "user_input")
					session["variables"][var_name] = latest_input
					if input_items[0].get("handler"):
						handler = self.handlers.get(input_items[0]["handler"])
						if handler:
							try:
								result = await handler(session, input_items[0])
								session["variables"].update(result.get("variables", {}))
								if result.get("end_session"):
									await self.end_session(session["id"], tenant, reason="input_handler_end")
									return {"response_type": "END", "body": self._truncate_ussd(result.get("body", "Done.")), "session_id": session["id"]}
								if result.get("next_menu"):
									session["current_menu"] = result["next_menu"]
									current_menu_id = result["next_menu"]
									try:
										menu = await self.get_menu(current_menu_id, service_code, tenant, session.get("language", language))
									except KeyError as _exc:
										_log.debug("Handled exception: %s", _exc)
							except Exception as exc:
								_log.error("input handler error: %s", exc)

		session["updated_at"] = self._now()
		response_type = "END" if menu.get("is_end_screen") else "CON"
		if response_type == "END":
			await self.end_session(session["id"], tenant, reason="end_screen")

		body = self._render_menu(menu, session)
		self._emit(tenant, "ussd_request_handled", session["id"], "ussd_session", {
			"phone": phone_number, "hop": session["hop_count"], "menu": current_menu_id,
		})
		return {"response_type": response_type, "body": body, "session_id": session["id"]}

	# ── Handler registry ────────────────────────────────────────────────────

	async def register_handler(self, name: str, handler: Any, tenant_id: str | None = None) -> dict[str, Any]:
		"""Register a callable handler for menu execute actions."""
		guard_non_empty_string(name, "name")
		if not callable(handler):
			raise TypeError("handler must be callable")
		self.handlers[name] = handler
		_log.info("handler registered: %s", name)
		return {"name": name, "registered_at": self._now()}

	async def list_handlers(self) -> list[str]:
		"""List all registered handler names."""
		return list(self.handlers.keys())

	async def unregister_handler(self, name: str) -> dict[str, Any]:
		"""Remove a registered handler."""
		if name not in self.handlers:
			raise KeyError(f"handler_not_found: {name}")
		del self.handlers[name]
		return {"name": name, "removed_at": self._now()}

	# ── Session variable management ──────────────────────────────────────────

	async def set_session_variable(self, session_id: str, key: str, value: Any, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		record["variables"][key] = value
		record["updated_at"] = self._now()
		return {"session_id": session_id, "key": key, "value": value, "updated_at": record["updated_at"]}

	async def get_session_variables(self, session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		return deepcopy(record["variables"])

	async def clear_session_variables(self, session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.sessions.get(session_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		record["variables"] = {}
		record["updated_at"] = self._now()
		return {"session_id": session_id, "cleared_at": record["updated_at"]}

	# ── Session timeout management ───────────────────────────────────────────

	async def expire_timed_out_sessions(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Mark sessions that have exceeded their menu timeout as timed out."""
		tenant = self._tenant(tenant_id)
		now_ts = datetime.utcnow()
		expired = []
		for sid, record in list(self.sessions.items()):
			if record["tenant_id"] != tenant or record["session_state"] != "active":
				continue
			updated = datetime.fromisoformat(record["updated_at"].rstrip("Z"))
			elapsed = (now_ts - updated).total_seconds()
			# Look up current menu timeout
			service_code = record["service_code"]
			menu_id = record["current_menu"]
			try:
				menu = await self.get_menu(menu_id, service_code, tenant, record.get("language", "en"))
				timeout = menu.get("timeout_seconds", USSD_DEFAULT_TIMEOUT)
			except KeyError:
				timeout = USSD_DEFAULT_TIMEOUT
			if elapsed > timeout:
				record["session_state"] = "timeout"
				record["ended_at"] = self._now()
				record["updated_at"] = self._now()
				expired.append(sid)
				self._emit(tenant, "session_timeout", sid, "ussd_session", {"elapsed_seconds": elapsed})
		return {"expired_count": len(expired), "expired_session_ids": expired, "checked_at": self._now()}

	# ── Gateway-specific integration helpers ─────────────────────────────────

	async def format_africastalking_response(self, response_type: str, body: str) -> str:
		"""Format response as Africa's Talking expects (CON/END prefix)."""
		prefix = "CON " if response_type == "CON" else "END "
		return prefix + self._truncate_ussd(body)

	async def format_safaricom_response(self, response_type: str, body: str, session_id: str) -> dict[str, Any]:
		"""Format response for Safaricom USSD (XML/JSON response structure)."""
		return {
			"sessionID": session_id,
			"responseType": response_type,
			"responseMsg": self._truncate_ussd(body),
		}

	async def validate_africastalking_callback(self, payload: dict[str, Any]) -> bool:
		"""Validate required fields in an Africa's Talking USSD callback."""
		required = {"sessionId", "serviceCode", "phoneNumber", "text"}
		missing = required - set(payload.keys())
		if missing:
			_log.warning("AT callback missing fields: %s", missing)
			return False
		return True

	async def validate_safaricom_callback(self, payload: dict[str, Any]) -> bool:
		"""Validate required fields in a Safaricom USSD callback."""
		required = {"msisdn", "input", "serviceCode", "sessionId"}
		missing = required - set(payload.keys())
		if missing:
			_log.warning("Safaricom callback missing fields: %s", missing)
			return False
		return True

	async def parse_africastalking_callback(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Normalize Africa's Talking USSD callback to internal format."""
		return {
			"session_id": payload.get("sessionId", ""),
			"service_code": payload.get("serviceCode", ""),
			"phone_number": payload.get("phoneNumber", ""),
			"text": payload.get("text", ""),
			"network_code": payload.get("networkCode"),
			"gateway": "africastalking",
		}

	async def parse_safaricom_callback(self, payload: dict[str, Any]) -> dict[str, Any]:
		"""Normalize Safaricom USSD callback to internal format."""
		return {
			"session_id": payload.get("sessionId", ""),
			"service_code": payload.get("serviceCode", ""),
			"phone_number": payload.get("msisdn", ""),
			"text": payload.get("input", ""),
			"network_code": payload.get("networkCode"),
			"gateway": "safaricom",
		}

	# ── Analytics & reporting ─────────────────────────────────────────────────

	async def get_session_analytics(self, tenant_id: str | None = None, service_code: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sessions = [r for r in self.sessions.values() if r["tenant_id"] == tenant]
		if service_code:
			sessions = [s for s in sessions if s["service_code"] == service_code]
		state_counts: dict[str, int] = {}
		for s in sessions:
			st = s["session_state"]
			state_counts[st] = state_counts.get(st, 0) + 1
		avg_hops = sum(s["hop_count"] for s in sessions) / len(sessions) if sessions else 0.0
		completion_rate = (state_counts.get("ended", 0) / len(sessions) * 100) if sessions else 0.0
		return {
			"tenant_id": tenant,
			"service_code": service_code,
			"total_sessions": len(sessions),
			"by_state": state_counts,
			"avg_hops": round(avg_hops, 2),
			"completion_rate_pct": round(completion_rate, 2),
			"generated_at": self._now(),
		}

	async def get_menu_analytics(self, service_code: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sessions = [r for r in self.sessions.values() if r["tenant_id"] == tenant and r["service_code"] == service_code]
		menu_visits: dict[str, int] = {}
		for s in sessions:
			for m in s.get("menu_history", []):
				menu_visits[m] = menu_visits.get(m, 0) + 1
		return {
			"tenant_id": tenant,
			"service_code": service_code,
			"total_sessions": len(sessions),
			"menu_visit_counts": menu_visits,
			"generated_at": self._now(),
		}

	async def get_drop_off_analysis(self, service_code: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Identify menus where sessions drop off most frequently."""
		tenant = self._tenant(tenant_id)
		sessions = [r for r in self.sessions.values() if r["tenant_id"] == tenant and r["service_code"] == service_code and r["session_state"] in {"ended", "timeout", "error"}]
		drop_off_menus: dict[str, int] = {}
		for s in sessions:
			last_menu = s.get("current_menu", "unknown")
			drop_off_menus[last_menu] = drop_off_menus.get(last_menu, 0) + 1
		sorted_drops = sorted(drop_off_menus.items(), key=lambda x: x[1], reverse=True)
		return {
			"tenant_id": tenant,
			"service_code": service_code,
			"drop_off_by_menu": dict(sorted_drops),
			"total_dropped": len(sessions),
			"generated_at": self._now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sessions = [r for r in self.sessions.values() if r["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"total_sessions": len(sessions),
			"active_sessions": sum(1 for s in sessions if s["session_state"] == "active"),
			"ended_sessions": sum(1 for s in sessions if s["session_state"] == "ended"),
			"timeout_sessions": sum(1 for s in sessions if s["session_state"] == "timeout"),
			"total_menus": len([m for m in self.menus.values() if m["tenant_id"] == tenant]),
			"total_gateways": len([g for g in self.gateways.values() if g["tenant_id"] == tenant]),
			"total_handlers": len(self.handlers),
			"audit_event_count": len([e for e in self._audit_events if e["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	# ── Audit events ──────────────────────────────────────────────────────────

	async def get_audit_events(self, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	# ── Batch operations ──────────────────────────────────────────────────────

	async def bulk_create_menus(self, menus: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
		"""Create multiple menus in one call."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		tasks = [
			self.create_menu(
				menu_id=m["menu_id"], title=m["title"], body=m.get("body", ""),
				service_code=m["service_code"], tenant_id=tenant,
				items=m.get("items", []), language=m.get("language", "en"),
				is_end_screen=m.get("is_end_screen", False),
				timeout_seconds=m.get("timeout_seconds", USSD_DEFAULT_TIMEOUT),
			)
			for m in menus
		]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		for m, r in zip(menus, raw):
			if isinstance(r, Exception):
				errors.append({"input": m, "error": str(r)})
			else:
				results.append(r)
		return {"created": len(results), "failed": len(errors), "menus": results, "errors": errors}

	async def export_session_data(self, tenant_id: str | None = None, fmt: str = "json") -> dict[str, Any]:
		"""Export all session data for a tenant."""
		tenant = self._tenant(tenant_id)
		assert fmt in {"json", "csv"}, "fmt must be json or csv"
		sessions = [r for r in self.sessions.values() if r["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"format": fmt,
			"record_count": len(sessions),
			"export_reference": f"ussd-sessions-{tenant}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}

	async def purge_ended_sessions(self, tenant_id: str | None = None, older_than_days: int = 30) -> dict[str, Any]:
		"""Remove ended/timeout sessions older than the given number of days."""
		tenant = self._tenant(tenant_id)
		now_ts = datetime.utcnow()
		removed = []
		for sid in list(self.sessions.keys()):
			record = self.sessions[sid]
			if record["tenant_id"] != tenant:
				continue
			if record["session_state"] not in {"ended", "timeout", "error"}:
				continue
			ended_str = record.get("ended_at") or record.get("updated_at", "")
			try:
				ended_ts = datetime.fromisoformat(ended_str.rstrip("Z"))
				if (now_ts - ended_ts).days >= older_than_days:
					del self.sessions[sid]
					self.session_variables.pop(sid, None)
					removed.append(sid)
			except Exception as exc:
				_log.debug("purge parse error for session %s: %s", sid, exc)
		self._emit(tenant, "sessions_purged", tenant, "ussd_session", {"removed_count": len(removed)})
		return {"removed_count": len(removed), "removed_session_ids": removed, "purged_at": self._now()}

	# ── I2: Session resumption after timeout ─────────────────────────────────

	async def resume_session(
		self,
		phone_number: str,
		service_code: str,
		tenant_id: str | None = None,
		grace_seconds: int = 90,
	) -> dict[str, Any]:
		"""
		Re-activate the most recent timed-out session for a phone number if it
		fell within the grace window.  Preserves hop_count, menu position and
		all session variables so the user continues mid-flow without re-entering
		data.  Returns the re-activated session or raises KeyError when none
		qualifies.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		guard_non_empty_string(phone_number, "phone_number")
		guard_non_empty_string(service_code, "service_code")
		now_ts = datetime.utcnow()
		candidate: dict[str, Any] | None = None
		candidate_ts: datetime | None = None
		for record in self.sessions.values():
			if record["tenant_id"] != tenant:
				continue
			if record["phone_number"] != phone_number:
				continue
			if record["service_code"] != service_code:
				continue
			if record["session_state"] != "timeout":
				continue
			ended_str = record.get("ended_at") or record.get("updated_at", "")
			try:
				ended_ts = datetime.fromisoformat(ended_str.rstrip("Z"))
			except Exception:
				continue
			elapsed = (now_ts - ended_ts).total_seconds()
			if elapsed <= grace_seconds:
				if candidate_ts is None or ended_ts > candidate_ts:
					candidate = record
					candidate_ts = ended_ts
		if candidate is None:
			raise KeyError(f"no_resumable_session: {phone_number}:{service_code}")
		candidate["session_state"] = "active"
		candidate["ended_at"] = None
		candidate["updated_at"] = self._now()
		candidate["metadata"]["resumed_at"] = self._now()
		candidate["metadata"]["resume_count"] = candidate["metadata"].get("resume_count", 0) + 1
		self._emit(tenant, "session_resumed", candidate["id"], "ussd_session", {
			"phone": phone_number,
			"grace_seconds": grace_seconds,
			"hop_count_at_resume": candidate["hop_count"],
		})
		_log.info(
			"session resumed: %s phone=%s service=%s hop=%d",
			candidate["id"], phone_number, service_code, candidate["hop_count"],
		)
		return deepcopy(candidate)

	# ── I3: Idempotent transaction execution ─────────────────────────────────

	async def execute_idempotent(
		self,
		session_id: str,
		hop_count: int,
		handler_name: str,
		payload: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Execute a handler exactly once per (session, hop, handler) triple.
		Duplicate USSD callbacks from GSM DTAP retransmission will receive the
		cached result rather than re-triggering the handler.  All monetary values
		in handler results must use Decimal — this method enforces that by
		coercing any ``amount`` / ``balance`` / ``total`` keys.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		guard_non_empty_string(handler_name, "handler_name")
		idem_key = hashlib.sha256(
			f"{session_id}:{hop_count}:{handler_name}".encode()
		).hexdigest()
		if idem_key in self._idempotency_cache:
			cached = self._idempotency_cache[idem_key]
			_log.info(
				"idempotent hit: key=%s session=%s hop=%d handler=%s",
				idem_key[:12], session_id, hop_count, handler_name,
			)
			return {"idempotent": True, "cached_at": cached["executed_at"], "result": deepcopy(cached["result"])}
		handler = self.handlers.get(handler_name)
		if handler is None:
			raise KeyError(f"handler_not_found: {handler_name}")
		session_record = self.sessions.get(session_id)
		if not session_record or session_record["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		result: dict[str, Any] = await handler(session_record, payload)
		# Coerce monetary fields to Decimal for downstream financial integrity
		for money_key in ("amount", "balance", "total", "fee", "charge"):
			if money_key in result:
				try:
					result[money_key] = Decimal(str(result[money_key]))
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		executed_at = self._now()
		self._idempotency_cache[idem_key] = {"result": deepcopy(result), "executed_at": executed_at}
		self._emit(tenant, "idempotent_execution", session_id, "ussd_session", {
			"handler": handler_name, "hop": hop_count, "idem_key": idem_key[:12],
		})
		_log.info(
			"idempotent execute: key=%s session=%s hop=%d handler=%s",
			idem_key[:12], session_id, hop_count, handler_name,
		)
		return {"idempotent": False, "executed_at": executed_at, "result": result}

	# ── I4: Rate limiting per phone number ───────────────────────────────────

	async def check_rate_limit(
		self,
		phone_number: str,
		service_code: str,
		tenant_id: str | None = None,
		window_seconds: int = 3600,
		max_sessions: int = 10,
	) -> dict[str, Any]:
		"""
		Sliding-window session rate limit per phone+service per tenant.
		Returns ``{"allowed": bool, "remaining": int, "reset_at": str}``.
		Callers should invoke this before ``create_session`` and refuse when
		``allowed`` is False to prevent bot scraping and credential stuffing.
		"""
		import time
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		guard_non_empty_string(phone_number, "phone_number")
		guard_non_empty_string(service_code, "service_code")
		bucket_key = f"{tenant}:{phone_number}:{service_code}"
		now = time.monotonic()
		window_start = now - window_seconds
		timestamps = self._rate_buckets.get(bucket_key, [])
		# Evict expired entries from sliding window
		timestamps = [t for t in timestamps if t > window_start]
		self._rate_buckets[bucket_key] = timestamps
		count = len(timestamps)
		allowed = count < max_sessions
		remaining = max(0, max_sessions - count)
		reset_at_epoch = (timestamps[0] + window_seconds) if timestamps else (now + window_seconds)
		reset_dt = datetime.utcfromtimestamp(
			datetime.utcnow().timestamp() + (reset_at_epoch - now)
		).isoformat(timespec="seconds") + "Z"
		if allowed:
			timestamps.append(now)
			self._rate_buckets[bucket_key] = timestamps
		else:
			_log.info(
				"rate limit hit: phone=%s service=%s tenant=%s count=%d/%d",
				phone_number, service_code, tenant, count, max_sessions,
			)
			self._emit(tenant, "rate_limit_exceeded", phone_number, "phone", {
				"service_code": service_code, "count": count, "max": max_sessions,
			})
		return {
			"allowed": allowed,
			"remaining": remaining if allowed else 0,
			"count": count + (1 if allowed else 0),
			"max_sessions": max_sessions,
			"window_seconds": window_seconds,
			"reset_at": reset_dt,
		}

	# ── I5: Input validation schema ──────────────────────────────────────────

	async def validate_input_against_schema(
		self,
		value: str,
		schema: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Validate a free-text USSD input value against a field schema dict.
		Schema keys: ``type`` (str|int|decimal|phone|pin), ``pattern`` (regex),
		``min_value`` (Decimal), ``max_value`` (Decimal), ``max_length`` (int),
		``min_length`` (int), ``required`` (bool).
		Returns ``{"valid": bool, "error_message": str | None, "coerced": Any}``.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		errors: list[str] = []
		coerced: Any = value
		field_type = schema.get("type", "str")
		required = schema.get("required", True)
		if not value and required:
			return {"valid": False, "error_message": "This field is required.", "coerced": None}
		if not value and not required:
			return {"valid": True, "error_message": None, "coerced": None}
		max_length = schema.get("max_length")
		min_length = schema.get("min_length")
		if max_length and len(value) > max_length:
			errors.append(f"Input too long (max {max_length} chars).")
		if min_length and len(value) < min_length:
			errors.append(f"Input too short (min {min_length} chars).")
		pattern = schema.get("pattern")
		if pattern and not re.fullmatch(pattern, value):
			errors.append(f"Input does not match expected format.")
		if field_type in ("int", "decimal", "amount"):
			try:
				coerced = Decimal(value.replace(",", ""))
				min_val = schema.get("min_value")
				max_val = schema.get("max_value")
				if min_val is not None and coerced < Decimal(str(min_val)):
					errors.append(f"Value must be at least {min_val}.")
				if max_val is not None and coerced > Decimal(str(max_val)):
					errors.append(f"Value must be at most {max_val}.")
			except Exception:
				errors.append("Please enter a valid number.")
				coerced = None
		elif field_type == "phone":
			phone_re = r"^\+?[0-9]{9,15}$"
			if not re.fullmatch(phone_re, value.replace(" ", "")):
				errors.append("Please enter a valid phone number.")
		elif field_type == "pin":
			pin_len = schema.get("pin_length", 4)
			if not re.fullmatch(r"\d+", value) or len(value) != pin_len:
				errors.append(f"PIN must be {pin_len} digits.")
		valid = len(errors) == 0
		_log.info("input_validation: type=%s valid=%s tenant=%s", field_type, valid, tenant)
		return {"valid": valid, "error_message": errors[0] if errors else None, "coerced": coerced}

	# ── I6: Menu versioning and rollback ─────────────────────────────────────

	async def create_menu_version(
		self,
		menu_id: str,
		service_code: str,
		tenant_id: str | None = None,
		language: str = "en",
	) -> dict[str, Any]:
		"""
		Snapshot the current state of a menu as a named version.
		Returns ``{"menu_id": str, "version": int, "snapshotted_at": str}``.
		Keeps the last 20 versions per menu to bound memory usage.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		record = self.menus.get(composite_key)
		if not record:
			raise KeyError(f"menu_not_found: {menu_id}")
		versions = self._menu_versions.setdefault(composite_key, [])
		version_num = len(versions) + 1
		snapshot = deepcopy(record)
		snapshot["_version"] = version_num
		snapshot["_snapshotted_at"] = self._now()
		versions.append(snapshot)
		if len(versions) > 20:
			versions.pop(0)
		self._emit(tenant, "menu_versioned", record["id"], "ussd_menu", {
			"menu_id": menu_id, "version": version_num,
		})
		_log.info(
			"menu version created: %s v%d service=%s tenant=%s",
			menu_id, version_num, service_code, tenant,
		)
		return {"menu_id": menu_id, "service_code": service_code, "version": version_num, "snapshotted_at": snapshot["_snapshotted_at"]}

	async def rollback_menu(
		self,
		menu_id: str,
		service_code: str,
		version: int,
		tenant_id: str | None = None,
		language: str = "en",
	) -> dict[str, Any]:
		"""
		Restore a menu to a previously snapshotted version.
		Atomically replaces the live menu entry so all new sessions
		immediately use the rolled-back definition.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		composite_key = f"{tenant}:{service_code}:{language}:{menu_id}"
		versions = self._menu_versions.get(composite_key, [])
		snapshot = next((v for v in versions if v.get("_version") == version), None)
		if snapshot is None:
			raise KeyError(f"menu_version_not_found: {menu_id}:v{version}")
		restored = deepcopy(snapshot)
		restored.pop("_version", None)
		restored.pop("_snapshotted_at", None)
		restored["updated_at"] = self._now()
		self.menus[composite_key] = restored
		self._emit(tenant, "menu_rolled_back", restored["id"], "ussd_menu", {
			"menu_id": menu_id, "version": version,
		})
		_log.info(
			"menu rolled back: %s to v%d service=%s tenant=%s",
			menu_id, version, service_code, tenant,
		)
		return deepcopy(restored)

	# ── I11: Dead-letter queue for failed handler executions ─────────────────

	async def queue_dead_letter(
		self,
		session_id: str,
		handler_name: str,
		payload: dict[str, Any],
		error: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Append a failed handler execution to the dead-letter store.
		Captures full execution context (session snapshot, menu position,
		input payload, exception message) for ops replay or alerting.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		guard_non_empty_string(handler_name, "handler_name")
		guard_non_empty_string(error, "error")
		session_record = self.sessions.get(session_id, {})
		entry = {
			"id": self._record_id("dlq"),
			"tenant_id": tenant,
			"session_id": session_id,
			"handler_name": handler_name,
			"payload": deepcopy(payload),
			"error": error,
			"session_snapshot": {
				"current_menu": session_record.get("current_menu"),
				"hop_count": session_record.get("hop_count"),
				"phone_number": session_record.get("phone_number"),
				"service_code": session_record.get("service_code"),
				"variables": deepcopy(session_record.get("variables", {})),
			},
			"retry_count": 0,
			"status": "pending",
			"queued_at": self._now(),
			"last_retried_at": None,
		}
		self._dead_letters.append(entry)
		self._emit(tenant, "dead_letter_queued", session_id, "ussd_session", {
			"handler": handler_name, "error": error[:200],
		})
		_log.info(
			"dead letter queued: handler=%s session=%s error=%s",
			handler_name, session_id, error[:80],
		)
		return deepcopy(entry)

	async def get_dead_letters(
		self,
		tenant_id: str | None = None,
		handler_name: str | None = None,
		status: str | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]:
		"""Return dead-letter entries for ops dashboards and automated replay."""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		results = [deepcopy(e) for e in self._dead_letters if e["tenant_id"] == tenant]
		if handler_name:
			results = [e for e in results if e["handler_name"] == handler_name]
		if status:
			results = [e for e in results if e["status"] == status]
		return results[-limit:]

	# ── I12: Paginated session queries ───────────────────────────────────────

	async def list_sessions_paginated(
		self,
		tenant_id: str | None = None,
		page: int = 1,
		page_size: int = 50,
		phone_number: str | None = None,
		service_code: str | None = None,
		session_state: str | None = None,
		sort_by: str = "created_at",
		sort_dir: str = "desc",
	) -> dict[str, Any]:
		"""
		Return a paginated, filterable, sortable view of sessions.
		At production scale (10M+ sessions) this avoids the O(n) full-copy
		of ``list_sessions()``.  Returns
		``{"items": [...], "total": int, "page": int, "pages": int, "page_size": int}``.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		assert page >= 1, "page must be >= 1"
		assert 1 <= page_size <= 1000, "page_size must be 1-1000"
		assert sort_dir in ("asc", "desc"), "sort_dir must be asc or desc"
		results = [r for r in self.sessions.values() if r["tenant_id"] == tenant]
		if phone_number:
			results = [r for r in results if r["phone_number"] == phone_number]
		if service_code:
			results = [r for r in results if r["service_code"] == service_code]
		if session_state:
			results = [r for r in results if r["session_state"] == session_state]
		reverse = sort_dir == "desc"
		results.sort(key=lambda r: r.get(sort_by, ""), reverse=reverse)
		total = len(results)
		pages = max(1, (total + page_size - 1) // page_size)
		start = (page - 1) * page_size
		end = start + page_size
		items = [deepcopy(r) for r in results[start:end]]
		_log.info(
			"list_sessions_paginated: tenant=%s page=%d/%d total=%d",
			tenant, page, pages, total,
		)
		return {
			"items": items,
			"total": total,
			"page": page,
			"pages": pages,
			"page_size": page_size,
		}

	# ── I15: Session replay for debugging ────────────────────────────────────

	async def replay_session(
		self,
		session_id: str,
		tenant_id: str | None = None,
		stop_at_hop: int | None = None,
	) -> list[dict[str, Any]]:
		"""
		Re-execute a completed session's input history against the current menu
		tree and return a step-by-step trace.  Useful for diagnosing failed
		transactions without needing to reconstruct the flow manually from audit
		logs.  Uses a shadow session so the live session store is not mutated.
		Returns a list of hop dicts:
		``[{"hop": int, "input": str, "menu": str, "response_type": str, "body": str}]``.
		"""
		tenant = guard_tenant_id(tenant_id or self.tenant_id)
		original = self.sessions.get(session_id)
		if not original or original["tenant_id"] != tenant:
			raise KeyError(f"session_not_found: {session_id}")
		input_history: list[str] = list(original.get("input_history", []))
		service_code = original["service_code"]
		phone_number = original["phone_number"]
		language = original.get("language", "en")
		gateway = original.get("gateway", "africastalking")
		if stop_at_hop is not None:
			input_history = input_history[:stop_at_hop]
		# Build a shadow session id so replay never collides with live data
		shadow_id = f"replay-{session_id[:12]}-{uuid4().hex[:8]}"
		trace: list[dict[str, Any]] = []
		# Prime the shadow session (hop 0 — initial menu render)
		shadow_resp = await self.handle_ussd_request(
			session_id=shadow_id,
			service_code=service_code,
			phone_number=phone_number,
			text="",
			tenant_id=tenant,
			gateway=gateway,
			language=language,
		)
		shadow_session_id = shadow_resp["session_id"]
		shadow = self.sessions.get(shadow_session_id, {})
		trace.append({
			"hop": 0,
			"input": "",
			"menu": shadow.get("current_menu", ""),
			"response_type": shadow_resp["response_type"],
			"body": shadow_resp["body"],
		})
		# Replay each recorded input
		cumulative_text = ""
		for hop_idx, user_input in enumerate(input_history, start=1):
			cumulative_text = cumulative_text + ("*" if cumulative_text else "") + user_input
			shadow_resp = await self.handle_ussd_request(
				session_id=shadow_session_id,
				service_code=service_code,
				phone_number=phone_number,
				text=cumulative_text,
				tenant_id=tenant,
				gateway=gateway,
				language=language,
			)
			shadow = self.sessions.get(shadow_session_id, {})
			trace.append({
				"hop": hop_idx,
				"input": user_input,
				"menu": shadow.get("current_menu", ""),
				"response_type": shadow_resp["response_type"],
				"body": shadow_resp["body"],
			})
			if shadow_resp["response_type"] == "END":
				break
		# Clean up shadow session from the live store
		self.sessions.pop(shadow_session_id, None)
		self.session_variables.pop(shadow_session_id, None)
		_log.info(
			"session replay complete: original=%s hops_replayed=%d tenant=%s",
			session_id, len(trace) - 1, tenant,
		)
		self._emit(tenant, "session_replayed", session_id, "ussd_session", {
			"hops_replayed": len(trace) - 1, "shadow_id": shadow_id,
		})
		return trace
