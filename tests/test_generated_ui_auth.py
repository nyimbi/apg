"""Generated authentication UI regressions."""

from __future__ import annotations

from compiler.compiler import APGCompiler


AUTH_SOURCE = """
module secure_customer_app version 1.0.0 {
	description: "Secure generated UI";
}

entity Customer {
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

		rejected = client.post(
			"/login",
			data={"username": "operator", "password": "wrong", "next": "/ui"},
		)
		assert rejected.status_code == 401
		assert b"We could not sign you in with those credentials." in rejected.data
		assert b'value="operator"' in rejected.data
		assert b"Invalid username or password" not in rejected.data

		accepted = client.post(
			"/login",
			data={"username": "operator", "password": "secret", "next": "/ui"},
		)
		assert accepted.status_code == 302
		assert accepted.headers["Location"] == "/ui"

		ui = client.get("/ui")
		assert ui.status_code == 200
		assert b"Ops User" in ui.data

		logout = client.post("/logout")
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
