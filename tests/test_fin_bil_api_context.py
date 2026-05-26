"""Context and async decorator regressions for Billing API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "fin" / "bil" / "api.py"


class BillingError(Exception):
	def __init__(self, message: str, error_code: str = "billing_error"):
		super().__init__(message)
		self.error_code = error_code


class SubscriptionError(BillingError):
	pass


class UsageError(BillingError):
	pass


class InvoiceError(BillingError):
	pass


class PaymentError(BillingError):
	pass


def _helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Customer API endpoints")
	namespace: dict[str, Any] = {
		"Any": Any,
		"BillingError": BillingError,
		"InvoiceError": InvoiceError,
		"List": List,
		"Optional": Optional,
		"PaymentError": PaymentError,
		"SubscriptionError": SubscriptionError,
		"UsageError": UsageError,
		"current_app": type("CurrentApp", (), {"logger": type("Logger", (), {"error": lambda self, message: None})()})(),
		"g": g,
		"inspect": __import__("inspect"),
		"os": __import__("os"),
		"request": request,
		"session": session,
		"wraps": __import__("functools").wraps,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_billing_api_no_longer_uses_fixed_api_user_context():
	source = API_PATH.read_text(encoding="utf-8")
	assert "'api-user'" not in source
	assert "request.headers.get(\"X-APG-User-ID\")" in source
	assert "inspect.iscoroutinefunction(func)" in source
	assert "return await func(*args, **kwargs)" in source


def test_billing_api_user_context_resolves_flask_context_headers_and_env(monkeypatch):
	resolve_user = _helpers()["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	with app.test_request_context("/billing"):
		assert resolve_user() == "env-user"

	with app.test_request_context("/billing?user_id=query-user", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		request.current_user = {"user_id": "request-user"}
		assert resolve_user() == "request-user"

	with app.test_request_context("/billing", headers={"X-User-ID": "header-user"}):
		assert resolve_user() == "header-user"


def test_billing_error_decorator_awaits_async_handlers():
	handle = _helpers()["handle_billing_error"]

	@handle
	async def ok_handler():
		return {"ok": True}

	@handle
	async def failing_handler():
		raise BillingError("bad billing", "bad_billing")

	assert asyncio.run(ok_handler()) == {"ok": True}
	assert asyncio.run(failing_handler()) == ({"error": "bad billing", "error_code": "bad_billing"}, 400)
