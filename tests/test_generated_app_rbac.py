"""Wave L RBAC regressions for the generated Flask app template."""

from __future__ import annotations

import json
import logging

from compiler.compiler import compile_apg_string


RBAC_APP_SOURCE = """
module wave_l_rbac_app version 1.0.0 {}

table Customer {
    name: str;
    secret: str;
}
"""


ENV_KEYS = (
    "APG_API_KEY",
    "APG_API_KEY_OWNER",
    "APG_AUDIT_LOG_FILE",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_FIELD_ACL",
    "APG_JWT_SECRET",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_ROW_OWNERSHIP",
    "APG_SECRET_KEY",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)


def _generated_namespace(monkeypatch, **env: str) -> dict[str, object]:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = compile_apg_string(RBAC_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_wave_l_rbac_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_wave_l_rbac_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


def _client(namespace: dict[str, object]):
    return namespace["_flask_app"].test_client()


def _set_user(client, username: str, role: str, permissions: list[str] | None = None) -> None:
    with client.session_transaction() as session:
        session["apg_user"] = {
            "username": username,
            "name": username,
            "roles": [role],
            "permissions": permissions or [],
        }


def _create_customer(client, name: str = "Ada", secret: str = "classified") -> dict[str, object]:
    response = client.post("/records/Customer", json={"name": name, "secret": secret})
    assert response.status_code == 201, response.get_json()
    return response.get_json()["record"]


def _audit_records(caplog) -> list[dict[str, object]]:
    return [json.loads(record.getMessage()) for record in caplog.records if record.name == "apg.audit"]


def test_field_acl_hides_field_from_non_admin(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_FIELD_ACL='{"Customer":{"secret":["admin"]}}')
    client = _client(namespace)
    _create_customer(client)
    _set_user(client, "operator", "user")

    response = client.get("/records/Customer")

    assert response.status_code == 200
    record = response.get_json()["data"][0]
    assert "secret" not in record


def test_field_acl_shows_field_to_admin(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_FIELD_ACL='{"Customer":{"secret":["admin"]}}')
    client = _client(namespace)
    _create_customer(client, secret="visible")
    _set_user(client, "admin", "admin")

    response = client.get("/records/Customer")

    assert response.status_code == 200
    record = response.get_json()["data"][0]
    assert record["secret"] == "visible"


def test_row_ownership_hides_other_users_rows(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_ROW_OWNERSHIP="1")
    client = _client(namespace)
    _set_user(client, "user-a", "user")
    created = _create_customer(client)
    _set_user(client, "user-b", "user")

    response = client.get(f"/records/Customer/{created['id']}")

    assert response.status_code == 404


def test_row_ownership_admin_sees_all(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_ROW_OWNERSHIP="1")
    client = _client(namespace)
    _set_user(client, "user-a", "user")
    created = _create_customer(client, name="Owned")
    _set_user(client, "admin", "admin")

    response = client.get(f"/records/Customer/{created['id']}")

    assert response.status_code == 200
    assert response.get_json()["record"]["name"] == "Owned"


def test_field_diff_in_audit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="apg.audit")
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    created = _create_customer(client, name="Ada")

    response = client.put(f"/records/Customer/{created['id']}", json={"name": "Grace"})

    assert response.status_code == 200
    assert response.get_json()["event"]["changed_fields"] == ["name"]
    updates = [event for event in _audit_records(caplog) if event["action"] == "update"]
    assert updates[-1]["changed_fields"] == ["name"]
