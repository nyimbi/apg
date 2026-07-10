"""Generated app hardening regressions: password hashing, login throttling,
session fixation, request limits, and graceful error pages."""

from __future__ import annotations

import hashlib
import json
import re

from compiler.compiler import APGCompiler


AUTH_SOURCE = """
module hardened_customer_app version 1.0.0 {
	description: "Hardened generated UI";
}

table Customer {
	name: str;
	email: str;
}

security {
	authentication: required;
}
"""

OPEN_SOURCE = """
module open_records_app version 1.0.0 {}

entity Customer { name: str; }
"""


def _generated_namespace(source: str = AUTH_SOURCE) -> dict[str, object]:
	result = APGCompiler().compile_string(source, "hardened_customer_app")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	return namespace


def _csrf_token_from_html(payload: bytes) -> str:
	match = re.search(r'name="apg_csrf_token"\s+value="([^"]+)"', payload.decode("utf-8"))
	assert match is not None
	return match.group(1)


def _login(client, username: str, password: str, token: str):
	return client.post(
		"/login",
		data={"username": username, "password": password, "next": "/ui", "apg_csrf_token": token},
	)


def test_generated_app_verifies_scrypt_password_hashes(monkeypatch):
	namespace = _generated_namespace()
	hash_password = namespace["hash_password"]
	stored = hash_password("correct horse battery staple", n=2**14)
	assert stored.startswith("scrypt$16384$8$1$")

	monkeypatch.setenv(
		"APG_AUTH_USERS",
		json.dumps({"operator": {"password_hash": stored, "roles": ["admin"], "permissions": ["*"]}}),
	)
	authenticate = namespace["_authenticate_user"]
	assert authenticate("operator", "correct horse battery staple") is not None
	assert authenticate("operator", "wrong password") is None


def test_generated_hash_password_defaults_meet_owasp_parameters():
	namespace = _generated_namespace()
	scrypt_default = namespace["hash_password"]("s3cret", n=2**14)
	_, n, r, p, salt_hex, digest_hex = scrypt_default.split("$")
	expected = hashlib.scrypt(
		b"s3cret", salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p),
		maxmem=256 * 1024 * 1024, dklen=32,
	)
	assert digest_hex == expected.hex()
	assert (int(n), int(r), int(p)) == (16384, 8, 1)
	# OWASP default when no override is supplied: N=2^17, r=8, p=1.
	assert namespace["_APG_SCRYPT_N"] == 2**17

	pbkdf2 = namespace["hash_password"]("s3cret", scheme="pbkdf2_sha256")
	algorithm, iterations, salt_hex, digest_hex = pbkdf2.split("$")
	assert algorithm == "pbkdf2_sha256"
	assert int(iterations) >= 600_000
	expected = hashlib.pbkdf2_hmac("sha256", b"s3cret", bytes.fromhex(salt_hex), int(iterations))
	assert digest_hex == expected.hex()


def test_generated_app_verifies_pbkdf2_password_hashes(monkeypatch):
	namespace = _generated_namespace()
	stored = namespace["hash_password"]("legacy-import", scheme="pbkdf2_sha256", iterations=1000)
	assert stored.startswith("pbkdf2_sha256$1000$")
	monkeypatch.setenv("APG_AUTH_USERS", json.dumps({"operator": {"password_hash": stored}}))
	authenticate = namespace["_authenticate_user"]
	assert authenticate("operator", "legacy-import") is not None
	assert authenticate("operator", "not-it") is None


def test_generated_password_hash_takes_precedence_over_plaintext(monkeypatch):
	namespace = _generated_namespace()
	stored = namespace["hash_password"]("only-the-hash-counts", n=2**14)
	monkeypatch.setenv(
		"APG_AUTH_USERS",
		json.dumps({"operator": {"password": "plaintext-decoy", "password_hash": stored}}),
	)
	authenticate = namespace["_authenticate_user"]
	assert authenticate("operator", "only-the-hash-counts") is not None
	assert authenticate("operator", "plaintext-decoy") is None


def test_generated_authentication_rejects_oversized_passwords(monkeypatch):
	# KDF DoS guard: never feed multi-kilobyte passwords into scrypt/pbkdf2.
	monkeypatch.setenv("APG_AUTH_USERS", '{"operator": {"password": "secret"}}')
	namespace = _generated_namespace()
	authenticate = namespace["_authenticate_user"]
	assert authenticate("operator", "x" * 2048) is None
	assert authenticate("operator", "secret") is not None


def test_generated_login_throttles_repeated_failures(monkeypatch):
	monkeypatch.setenv("APG_AUTH_USERS", '{"operator": {"password": "secret"}}')
	monkeypatch.setenv("APG_LOGIN_MAX_ATTEMPTS", "3")
	monkeypatch.setenv("APG_LOGIN_WINDOW_SECONDS", "300")
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		token = _csrf_token_from_html(client.get("/login").data)
		for _ in range(3):
			response = _login(client, "operator", "wrong", token)
			assert response.status_code == 401
		blocked = _login(client, "operator", "wrong", token)
		assert blocked.status_code == 429
		assert blocked.headers.get("Retry-After")
		# Correct credentials are also refused while the lockout is active.
		still_blocked = _login(client, "operator", "secret", token)
		assert still_blocked.status_code == 429


def test_generated_login_success_resets_throttle_counter(monkeypatch):
	monkeypatch.setenv("APG_AUTH_USERS", '{"operator": {"password": "secret"}}')
	monkeypatch.setenv("APG_LOGIN_MAX_ATTEMPTS", "3")
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		token = _csrf_token_from_html(client.get("/login").data)
		assert _login(client, "operator", "wrong", token).status_code == 401
		assert _login(client, "operator", "secret", token).status_code == 302

	with app.test_client() as client:
		token = _csrf_token_from_html(client.get("/login").data)
		for _ in range(2):
			assert _login(client, "operator", "wrong", token).status_code == 401
		assert _login(client, "operator", "secret", token).status_code == 302


def test_generated_login_rotates_session_against_fixation(monkeypatch):
	monkeypatch.setenv("APG_AUTH_USERS", '{"operator": {"password": "secret"}}')
	namespace = _generated_namespace()
	app = namespace["_flask_app"]

	with app.test_client() as client:
		pre_login_token = _csrf_token_from_html(client.get("/login").data)
		response = _login(client, "operator", "secret", pre_login_token)
		assert response.status_code == 302
		post_login_token = _csrf_token_from_html(client.get("/ui").data)
	assert post_login_token != pre_login_token


def test_generated_app_limits_request_body_size(monkeypatch):
	monkeypatch.setenv("APG_MAX_BODY_BYTES", "1024")
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]
	assert app.config["MAX_CONTENT_LENGTH"] == 1024

	with app.test_client() as client:
		response = client.post(
			"/records/Customer",
			data=b"x" * 4096,
			content_type="application/json",
		)
		assert response.status_code == 413


def test_generated_app_default_body_limit_is_sane():
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]
	assert app.config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024


def test_generated_app_serves_branded_not_found_page():
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		html_response = client.get(
			"/definitely/not/a/route",
			headers={"Accept": "text/html,application/xhtml+xml"},
		)
		assert html_response.status_code == 404
		assert b"<html" in html_response.data.lower()
		assert b"werkzeug" not in html_response.data.lower()

		api_response = client.get("/api/definitely/not/a/route", headers={"Accept": "application/json"})
		assert api_response.status_code == 404
		payload = api_response.get_json()
		assert payload["error"] == "not_found"


def test_generated_production_mode_denies_unconfigured_api_mutations(monkeypatch):
	monkeypatch.setenv("APG_PRODUCTION", "1")
	monkeypatch.delenv("APG_API_KEY", raising=False)
	monkeypatch.delenv("APG_JWT_SECRET", raising=False)
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		denied = client.post("/records/Customer", json={"name": "intruder"})
		assert denied.status_code == 401

	monkeypatch.setenv("APG_API_KEY", "ops-key")
	with app.test_client() as client:
		allowed = client.post(
			"/records/Customer",
			json={"name": "operator"},
			headers={"X-APG-API-Key": "ops-key"},
		)
		assert allowed.status_code in {200, 201}


def test_generated_dev_mode_keeps_zero_config_api_mutations(monkeypatch):
	monkeypatch.delenv("APG_PRODUCTION", raising=False)
	monkeypatch.delenv("APG_ENV", raising=False)
	monkeypatch.delenv("APG_API_KEY", raising=False)
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]

	with app.test_client() as client:
		response = client.post("/records/Customer", json={"name": "dev-user"})
		assert response.status_code in {200, 201}


def test_generated_app_serves_branded_error_page_without_traceback():
	namespace = _generated_namespace(OPEN_SOURCE)
	app = namespace["_flask_app"]

	@app.route("/__boom")
	def _boom():  # pragma: no cover - exercised via test client
		raise RuntimeError("sensitive internal details")

	app.testing = False
	with app.test_client() as client:
		response = client.get("/__boom")
		assert response.status_code == 500
		assert b"sensitive internal details" not in response.data
		assert b"Traceback" not in response.data

		api_response = client.get("/__boom", headers={"Accept": "application/json"})
		assert api_response.status_code == 500
		payload = api_response.get_json()
		assert payload["error"] == "internal_error"
		assert "sensitive" not in json.dumps(payload)
