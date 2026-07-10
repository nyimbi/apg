"""Generated authentication UI regressions."""

from __future__ import annotations

import re

from compiler.compiler import APGCompiler


AUTH_SOURCE = """
module secure_customer_app version 1.0.0 {
	description: "Secure generated UI";
}

table Customer {
	name: str;
	email: str;
}

security {
	authentication: required;
}
"""


def _generated_namespace(source: str = AUTH_SOURCE) -> dict[str, object]:
	result = APGCompiler().compile_string(source, "secure_customer_app")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	return namespace


def _assert_security_headers(response) -> None:
	headers = response.headers
	assert headers["X-Content-Type-Options"] == "nosniff"
	assert headers["X-Frame-Options"] == "DENY"
	assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
	assert "camera=()" in headers["Permissions-Policy"]
	assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
	assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
	csp = headers["Content-Security-Policy"]
	assert "default-src 'self'" in csp
	assert "script-src 'self' 'unsafe-inline'" in csp
	assert "style-src 'self' 'unsafe-inline'" in csp
	assert "object-src 'none'" in csp
	assert "frame-ancestors 'none'" in csp
	assert "form-action 'self'" in csp


def _csrf_token_from_html(payload: bytes) -> str:
	match = re.search(
		r'name="apg_csrf_token"\s+value="([^"]+)"',
		payload.decode("utf-8"),
	)
	assert match is not None
	return match.group(1)


def _session_cookie_header(response) -> str:
	for header in response.headers.getlist("Set-Cookie"):
		if header.startswith("apg_session="):
			return header
	raise AssertionError("missing generated session cookie")


def test_generated_flask_responses_apply_security_headers():
	source = """
module open_customer_app version 1.0.0 {}
entity Customer { name: str; }
"""
	namespace = _generated_namespace(source)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		for response in (
			client.get("/ui"),
			client.get("/openapi.json"),
			client.get("/theme.css"),
			client.post("/locale", data={"lang": "en", "next": "/ui"}),
		):
			_assert_security_headers(response)

		plain = client.get("/ui")
		assert "Strict-Transport-Security" not in plain.headers

		secure = client.get("/ui", base_url="https://localhost")
		_assert_security_headers(secure)
		assert (
			secure.headers["Strict-Transport-Security"]
			== "max-age=63072000; includeSubDomains"
		)


def test_generated_session_secret_uses_config_or_ephemeral_secret(monkeypatch):
	monkeypatch.delenv("APG_SESSION_SECRET", raising=False)
	monkeypatch.delenv("APG_JWT_SECRET", raising=False)
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	assert app.secret_key
	assert app.secret_key != "apg-generated-session-secret"
	assert len(app.secret_key) >= 32

	monkeypatch.setenv("APG_SESSION_SECRET", "configured-session-secret-value")
	configured_namespace = _generated_namespace()
	configured_app = configured_namespace["_flask_app"]
	assert configured_app.secret_key == "configured-session-secret-value"


def test_generated_session_cookie_is_hardened(monkeypatch):
	monkeypatch.setenv(
		"APG_AUTH_USERS",
		'{"operator": {"password": "secret", "name": "Ops User", "roles": ["admin"], "permissions": ["*"]}}',
	)
	monkeypatch.setenv("APG_SESSION_SECRET", "configured-session-secret-value")
	monkeypatch.setenv("APG_SESSION_COOKIE_SECURE", "true")
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	assert app.config["SESSION_COOKIE_NAME"] == "apg_session"
	assert app.config["SESSION_COOKIE_HTTPONLY"] is True
	assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
	assert app.config["SESSION_COOKIE_SECURE"] is True

	with app.test_client() as client:
		login_page = client.get("/login", base_url="https://localhost")
		login_token = _csrf_token_from_html(login_page.data)
		accepted_login = client.post(
			"/login",
			base_url="https://localhost",
			data={
				"username": "operator",
				"password": "secret",
				"next": "/ui",
				"apg_csrf_token": login_token,
			},
		)

	session_cookie = _session_cookie_header(accepted_login)
	assert "HttpOnly" in session_cookie
	assert "SameSite=Lax" in session_cookie
	assert "Secure" in session_cookie


def test_generated_session_forms_require_csrf_token(monkeypatch):
	monkeypatch.setenv(
		"APG_AUTH_USERS",
		'{"operator": {"password": "secret", "name": "Ops User", "roles": ["admin"], "permissions": ["*"]}}',
	)
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		login_page = client.get("/login")
		login_token = _csrf_token_from_html(login_page.data)

		rejected_login = client.post(
			"/login",
			data={"username": "operator", "password": "secret", "next": "/ui"},
		)
		assert rejected_login.status_code == 400
		assert rejected_login.get_json()["error"] == "csrf_failed"

		accepted_login = client.post(
			"/login",
			data={
				"username": "operator",
				"password": "secret",
				"next": "/ui",
				"apg_csrf_token": login_token,
			},
		)
		assert accepted_login.status_code == 302

		ui = client.get("/ui/entities/Customer")
		ui_token = _csrf_token_from_html(ui.data)

		missing_token = client.post(
			"/ui/entities/Customer/records",
			data={"name": "Forged", "email": "forged@example.com"},
		)
		assert missing_token.status_code == 400
		assert missing_token.get_json()["error"] == "csrf_failed"
		assert namespace["list_records"]("Customer") == []

		created = client.post(
			"/ui/entities/Customer/records",
			data={
				"name": "Asha",
				"email": "asha@example.com",
				"apg_csrf_token": ui_token,
			},
		)
		assert created.status_code == 303
		assert namespace["list_records"]("Customer")[0]["name"] == "Asha"

		logout_without_token = client.post("/logout")
		assert logout_without_token.status_code == 400
		assert client.get("/ui").status_code == 200

		logout = client.post("/logout", data={"apg_csrf_token": ui_token})
		assert logout.status_code == 302
		assert client.get("/ui").status_code == 302


def test_generated_api_key_mutations_do_not_require_csrf(monkeypatch):
	monkeypatch.setenv("APG_API_KEY", "test-key")
	source = """
module open_customer_app version 1.0.0 {}
entity Customer { name: str; email: str; }
"""
	namespace = _generated_namespace(source)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		forbidden = client.post(
			"/entities/Customer/records",
			json={"record": {"name": "No Key", "email": "nokey@example.com"}},
		)
		assert forbidden.status_code == 401

		created = client.post(
			"/entities/Customer/records",
			json={"record": {"name": "Header Key", "email": "key@example.com"}},
			headers={"X-APG-API-Key": "test-key"},
		)
		assert created.status_code == 201
		assert namespace["list_records"]("Customer")[0]["name"] == "Header Key"


def test_auth_declared_generated_ui_login_logout_flow(monkeypatch):
	monkeypatch.setenv(
		"APG_AUTH_USERS",
		'{"operator": {"password": "secret", "name": "Ops User", "roles": ["admin"], "permissions": ["*"]}}',
	)
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		response = client.get("/ui")
		assert response.status_code == 302
		assert response.headers["Location"].startswith("/login?next=/ui")

		login_page = client.get("/login")
		assert login_page.status_code == 200
		assert b"secure_customer_app" in login_page.data
		assert b"Secure workspace sign-in" in login_page.data
		assert b"Continue to" in login_page.data
		assert b'apg-login-username' in login_page.data
		assert b'apg-login-password' in login_page.data
		assert b'id="apg-sidebar"' not in login_page.data
		login_token = _csrf_token_from_html(login_page.data)

		rejected = client.post(
			"/login",
			data={
				"username": "operator",
				"password": "wrong",
				"next": "/ui",
				"apg_csrf_token": login_token,
			},
		)
		assert rejected.status_code == 401
		assert b"We could not sign you in with those credentials." in rejected.data
		assert b'value="operator"' in rejected.data
		assert b"Invalid username or password" not in rejected.data

		accepted = client.post(
			"/login",
			data={
				"username": "operator",
				"password": "secret",
				"next": "/ui",
				"apg_csrf_token": login_token,
			},
		)
		assert accepted.status_code == 302
		assert accepted.headers["Location"] == "/ui"

		ui = client.get("/ui")
		assert ui.status_code == 200
		assert b"Ops User" in ui.data
		ui_token = _csrf_token_from_html(ui.data)

		logout = client.post("/logout", data={"apg_csrf_token": ui_token})
		assert logout.status_code == 302
		assert logout.headers["Location"] == "/login"
		assert client.get("/ui").status_code == 302


def test_authless_generated_ui_does_not_enable_login_route():
	source = """
module open_customer_app version 1.0.0 {}
entity Customer { name: str; }
"""
	namespace = _generated_namespace(source)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		assert namespace["auth_status"]()["mode"] == "open"
		assert client.get("/ui").status_code == 200
		assert client.get("/login").status_code == 404
