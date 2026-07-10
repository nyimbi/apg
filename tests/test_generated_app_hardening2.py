"""Wave F generated app production hardening coverage."""

from __future__ import annotations

import json
import logging
import re

import pytest

from compiler.compiler import APGCompiler


OPEN_SOURCE = """
module wave_f_open_app version 1.0.0 {}

table Customer {
    name: str;
}
"""

AUTH_SOURCE = """
module wave_f_auth_app version 1.0.0 {}

table Customer {
    name: str;
}

security {
    authentication: required;
}
"""

ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUDIT_LOG_FILE",
    "APG_AUTH_USERS",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PRODUCTION",
    "APG_RATE_LIMIT_ANON",
    "APG_RATE_LIMIT_AUTH",
    "APG_SECRET_KEY",
    "APG_SESSION_COOKIE_SECURE",
    "APG_SESSION_SECRET",
)


def _generated_app_source(source: str) -> str:
    result = APGCompiler().compile_string(source, "wave_f_test")
    assert result.success, result.errors
    return result.generated_files["app.py"]


def _generated_namespace(monkeypatch, source: str = OPEN_SOURCE, **env: str) -> dict[str, object]:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    namespace: dict[str, object] = {"__file__": "generated_wave_f_app.py"}
    exec(compile(_generated_app_source(source), "generated_wave_f_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


def _csrf_token_from_html(payload: bytes) -> str:
    match = re.search(r'name="apg_csrf_token"\s+value="([^"]+)"', payload.decode("utf-8"))
    assert match is not None
    return match.group(1)


def _audit_records(caplog) -> list[dict[str, object]]:
    return [json.loads(record.getMessage()) for record in caplog.records if record.name == "apg.audit"]


def test_startup_raises_in_production_with_default_key(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APG_PRODUCTION", "1")
    monkeypatch.delenv("APG_SECRET_KEY", raising=False)
    namespace: dict[str, object] = {"__file__": "generated_wave_f_app.py"}

    with pytest.raises(RuntimeError, match="Set APG_SECRET_KEY in production"):
        exec(compile(_generated_app_source(OPEN_SOURCE), "generated_wave_f_app.py", "exec"), namespace)


def test_audit_logger_emits_on_entity_create(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="apg.audit")
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    response = client.post("/records/Customer", json={"name": "Ada"})

    assert response.status_code == 201
    events = _audit_records(caplog)
    assert events[-1]["audit"] is True
    assert events[-1]["action"] == "create"
    assert events[-1]["entity"] == "Customer"


def test_audit_logger_emits_on_login(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="apg.audit")
    namespace = _generated_namespace(
        monkeypatch,
        AUTH_SOURCE,
        APG_AUTH_USERS='{"operator": {"password": "secret"}}',
    )
    client = namespace["_flask_app"].test_client()
    token = _csrf_token_from_html(client.get("/login").data)

    response = client.post(
        "/login",
        data={"username": "operator", "password": "secret", "next": "/ui", "apg_csrf_token": token},
    )

    assert response.status_code == 302
    assert any(event["action"] == "login" for event in _audit_records(caplog))


def test_rate_limiter_returns_429(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_RATE_LIMIT_ANON="2")
    client = namespace["_flask_app"].test_client()

    first = client.post("/records/Customer", json={"name": "Ada"})
    second = client.post("/records/Customer", json={"name": "Grace"})
    third = client.post("/records/Customer", json={"name": "Katherine"})

    assert [first.status_code, second.status_code, third.status_code] == [201, 201, 429]
    assert third.headers["Retry-After"] == "60"


def test_content_type_guard_returns_415(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    response = client.post("/records/Customer", data='{"name": "Ada"}', content_type="text/plain")

    assert response.status_code == 415
