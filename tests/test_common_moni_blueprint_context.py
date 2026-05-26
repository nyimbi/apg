"""Context resolution regressions for Monitoring blueprint action actors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "common" / "moni" / "blueprint.py"


def _helpers() -> dict[str, Any]:
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# APG Capability Metadata")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(BLUEPRINT_PATH), "exec"), namespace)
	return namespace


def test_monitoring_blueprint_no_longer_uses_fixed_action_actor():
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")
	assert "'api_" + "user'" not in source
	assert "request.json.get('acknowledged_" + "by'" not in source
	assert "request.json.get('resolved_" + "by'" not in source
	assert "acknowledged_by = resolve_current_user_id(payload)" in source
	assert "resolved_by = resolve_current_user_id(payload)" in source


def test_monitoring_actor_context_resolves_payload_flask_headers_and_env(monkeypatch):
	resolve_user = _helpers()["resolve_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	with app.test_request_context("/monitoring"):
		assert resolve_user({}) == "env-user"

	with app.test_request_context("/monitoring", json={"acknowledged_by": "payload-actor"}):
		assert resolve_user({"acknowledged_by": "payload-actor"}) == "payload-actor"

	with app.test_request_context("/monitoring?user_id=query-user", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		request.current_user = {"user_id": "request-user"}
		assert resolve_user({}) == "request-user"

	with app.test_request_context("/monitoring", headers={"X-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/monitoring"):
		g.current_user = {"id": "g-user"}
		assert resolve_user({}) == "g-user"
