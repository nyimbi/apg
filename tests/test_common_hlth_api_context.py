"""Context resolution regressions for System Health API resources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "common" / "hlth" / "api.py"


def _helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Marshmallow Schemas")
	namespace: dict[str, Any] = {
		"Any": Any,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_health_api_no_longer_uses_fixed_actor_fallback():
	source = API_PATH.read_text(encoding="utf-8")
	assert "'api_" + "user'" not in source
	assert "request.headers.get('X-" + "User-ID'" not in source
	assert source.count("return resolve_current_user_id()") == 2


def test_health_api_actor_context_resolves_flask_context_headers_and_env(monkeypatch):
	resolve_user = _helpers()["resolve_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	with app.test_request_context("/health"):
		assert resolve_user() == "env-user"

	with app.test_request_context("/health?user_id=query-user", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		request.current_user = {"user_id": "request-user"}
		assert resolve_user() == "request-user"

	with app.test_request_context("/health", headers={"X-User-ID": "header-user"}):
		assert resolve_user() == "header-user"

	with app.test_request_context("/health"):
		g.current_user = {"id": "g-user"}
		assert resolve_user() == "g-user"
