"""Generated app SMTP notification regressions."""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock

import pytest

from compiler.compiler import APGCompiler


EMAIL_SOURCE = """
module email_probe version 1.0.0 {}

table Customer {
	name: str;
	email: str;
}

security {
	authentication: required;
}
"""

_EMAIL_ENV_KEYS = (
	"APG_ALERT_EMAIL",
	"APG_API_KEY",
	"APG_AUTH_EMAIL",
	"APG_AUTH_PASSWORD",
	"APG_AUTH_PASSWORD_HASH",
	"APG_AUTH_USERNAME",
	"APG_AUTH_USERS",
	"APG_EMAIL_ON_LOGIN",
	"APG_ENV",
	"APG_JWT_SECRET",
	"APG_NOTIFY_EMAIL",
	"APG_PRODUCTION",
	"APG_SESSION_SECRET",
	"APG_SMTP_FROM",
	"APG_SMTP_HOST",
	"APG_SMTP_PASSWORD",
	"APG_SMTP_PORT",
	"APG_SMTP_USER",
)


@pytest.fixture()
def generated_email_app(monkeypatch, tmp_path):
	for key in _EMAIL_ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	result = APGCompiler().compile_string(EMAIL_SOURCE, "email_probe")
	assert result.success, result.errors
	namespace: dict[str, object] = {"__file__": str(tmp_path / "generated_email_app.py")}
	exec(compile(result.generated_files["app.py"], "generated_email_app.py", "exec"), namespace)
	namespace["_flask_app"].config["TESTING"] = True
	return namespace


def _csrf_token_from_html(payload: bytes) -> str:
	match = re.search(r'name="apg_csrf_token"\s+value="([^"]+)"', payload.decode("utf-8"))
	assert match is not None
	return match.group(1)


def test_email_sent_on_login(monkeypatch, generated_email_app):
	monkeypatch.setenv(
		"APG_AUTH_USERS",
		json.dumps({
			"operator": {
				"password": "secret",
				"email": "operator@example.com",
			}
		}),
	)
	monkeypatch.setenv("APG_EMAIL_ON_LOGIN", "1")
	monkeypatch.setenv("APG_SMTP_HOST", "smtp.example.test")
	monkeypatch.setenv("APG_SMTP_PORT", "587")
	monkeypatch.setenv("APG_SMTP_USER", "smtp-user")
	monkeypatch.setenv("APG_SMTP_PASSWORD", "smtp-pass")
	monkeypatch.setenv("APG_SMTP_FROM", "noreply@example.com")
	smtp_class = MagicMock()
	smtp = smtp_class.return_value.__enter__.return_value
	monkeypatch.setattr(generated_email_app["_smtplib"], "SMTP", smtp_class)

	with generated_email_app["_flask_app"].test_client() as client:
		token = _csrf_token_from_html(client.get("/login").data)
		response = client.post(
			"/login",
			data={
				"username": "operator",
				"password": "secret",
				"next": "/ui",
				"apg_csrf_token": token,
			},
		)

	for thread in list(generated_email_app["_APG_EMAIL_THREADS"]):
		thread.join(timeout=2)

	assert response.status_code == 302
	smtp_class.assert_called_once_with("smtp.example.test", 587, timeout=10)
	smtp.starttls.assert_called_once()
	smtp.login.assert_called_once_with("smtp-user", "smtp-pass")
	from_addr, recipients, message = smtp.sendmail.call_args.args
	assert from_addr == "noreply@example.com"
	assert recipients == ["operator@example.com"]
	assert "Subject: New login to email_probe" in message
