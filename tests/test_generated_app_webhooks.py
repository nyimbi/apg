"""Outbound webhook coverage for the generated Flask app template."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from compiler.compiler import compile_apg_string


WEBHOOK_APP_SOURCE = """
module webhook_probe version 1.0.0 {}

table Customer {
    name: str;
    email: str;
}
"""


_WEBHOOK_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DATABASE_URL",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "APG_WEBHOOK_SECRET",
    "APG_WEBHOOK_URL",
    "DATABASE_URL",
)


@pytest.fixture()
def generated_webhook_app(monkeypatch, tmp_path):
    for key in _WEBHOOK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(WEBHOOK_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": str(tmp_path / "generated_webhook_app.py")}
    exec(compile(result.generated_files["app.py"], "generated_webhook_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


@pytest.fixture()
def client(generated_webhook_app):
    return generated_webhook_app["_flask_app"].test_client()


def _webhook_handler(httpserver):
    handler = httpserver.expect_oneshot_request("/webhook", method="POST")
    handler.respond_with_data("ok")
    return handler


def _received_request(httpserver, handler):
    httpserver.check_assertions()
    requests = list(httpserver.iter_matching_requests(handler.matcher))
    assert len(requests) == 1
    return requests[0][0]


def test_webhook_fired_on_create(monkeypatch, httpserver, client):
    handler = _webhook_handler(httpserver)
    monkeypatch.setenv("APG_WEBHOOK_URL", httpserver.url_for("/webhook"))

    with httpserver.wait(timeout=2):
        response = client.post("/records/Customer", json={"name": "Asha", "email": "asha@example.com"})

    assert response.status_code == 201, response.get_json()
    request = _received_request(httpserver, handler)
    assert request.method == "POST"


def test_webhook_payload_correct(monkeypatch, httpserver, client):
    handler = _webhook_handler(httpserver)
    monkeypatch.setenv("APG_WEBHOOK_URL", httpserver.url_for("/webhook"))
    posted = {"name": "Asha", "email": "asha@example.com"}

    with httpserver.wait(timeout=2):
        response = client.post("/records/Customer", json=posted)

    assert response.status_code == 201, response.get_json()
    request = _received_request(httpserver, handler)
    payload = json.loads(request.get_data(as_text=True))
    assert payload["event"] == "entity.created"
    assert payload["entity"] == "Customer"
    assert payload["data"] == posted
    assert payload["id"] == str(response.get_json()["record"]["id"])
    assert payload["req_id"]


def test_webhook_hmac_signature(monkeypatch, httpserver, client):
    handler = _webhook_handler(httpserver)
    secret = "signed-secret"
    monkeypatch.setenv("APG_WEBHOOK_URL", httpserver.url_for("/webhook"))
    monkeypatch.setenv("APG_WEBHOOK_SECRET", secret)

    with httpserver.wait(timeout=2):
        response = client.post("/records/Customer", json={"name": "Asha", "email": "asha@example.com"})

    assert response.status_code == 201, response.get_json()
    request = _received_request(httpserver, handler)
    body = request.get_data()
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert request.headers["X-APG-Signature"] == expected


def test_webhook_failure_does_not_break_response(monkeypatch, client):
    monkeypatch.setenv("APG_WEBHOOK_URL", "http://127.0.0.1:1/webhook")

    response = client.post("/records/Customer", json={"name": "Asha", "email": "asha@example.com"})

    assert response.status_code == 201, response.get_json()
