"""Generated app multi-tenant isolation coverage."""

from __future__ import annotations

import pytest

from compiler.compiler import compile_apg_string


MULTITENANT_APP_SOURCE = """
module multitenant_probe version 1.0.0 {}

entity Customer {
    name: str;
}
"""


_TENANT_ENV_KEYS = (
    "APG_ADMIN_KEY",
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_LOCALE",
    "APG_LOCALE_FILE",
    "APG_MULTI_TENANT",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SECRET_KEY",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "APG_TENANT_HEADER",
    "DATABASE_URL",
)


def _generated_app(monkeypatch: pytest.MonkeyPatch, **env: str) -> tuple[dict[str, object], object]:
    for key in _TENANT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = compile_apg_string(MULTITENANT_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_multitenant_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_multitenant_app.py", "exec"), namespace)
    app = namespace["_flask_app"]
    app.config["TESTING"] = True
    return namespace, app.test_client()


def test_multitenant_schema_and_default_tenant(monkeypatch: pytest.MonkeyPatch):
    namespace, client = _generated_app(monkeypatch, APG_MULTI_TENANT="1")
    conn = namespace["_sqlite_connection"]()

    columns = {row["name"]: dict(row) for row in conn.execute('PRAGMA table_info("Customer")').fetchall()}
    created = client.post("/records/Customer", json={"name": "Default"})

    assert columns["tenant_id"]["type"] == "TEXT"
    assert created.status_code == 201, created.get_json()
    assert created.get_json()["record"]["tenant_id"] == "default"


def test_tenant_header_is_stored_and_selects_are_scoped(monkeypatch: pytest.MonkeyPatch):
    _namespace, client = _generated_app(monkeypatch, APG_MULTI_TENANT="1")

    acme = client.post("/records/Customer", json={"name": "Asha"}, headers={"X-APG-Tenant": "acme"})
    globex = client.post("/records/Customer", json={"name": "Bayo"}, headers={"X-APG-Tenant": "globex"})
    acme_list = client.get("/records/Customer", headers={"X-APG-Tenant": "acme"})
    globex_list = client.get("/records/Customer", headers={"X-APG-Tenant": "globex"})

    assert acme.status_code == 201, acme.get_json()
    assert globex.status_code == 201, globex.get_json()
    assert acme.get_json()["record"]["tenant_id"] == "acme"
    assert [record["name"] for record in acme_list.get_json()["data"]] == ["Asha"]
    assert [record["name"] for record in globex_list.get_json()["data"]] == ["Bayo"]


def test_production_requires_tenant_header(monkeypatch: pytest.MonkeyPatch):
    _namespace, client = _generated_app(
        monkeypatch,
        APG_MULTI_TENANT="1",
        APG_PRODUCTION="1",
        APG_SECRET_KEY="test-secret",
    )

    response = client.post("/records/Customer", json={"name": "Missing tenant"})

    assert response.status_code == 400
    assert response.get_json() == {"error": "tenant_required"}


def test_admin_key_bypasses_tenant_filter(monkeypatch: pytest.MonkeyPatch):
    _namespace, client = _generated_app(
        monkeypatch,
        APG_MULTI_TENANT="1",
        APG_API_KEY="user-key",
        APG_ADMIN_KEY="admin-key",
    )
    user_headers = {"X-APG-API-Key": "user-key"}

    client.post("/records/Customer", json={"name": "Asha"}, headers={**user_headers, "X-APG-Tenant": "acme"})
    client.post("/records/Customer", json={"name": "Bayo"}, headers={**user_headers, "X-APG-Tenant": "globex"})
    admin_response = client.get(
        "/records/Customer?sort=name",
        headers={"X-APG-API-Key": "admin-key", "X-APG-Admin": "1"},
    )

    assert admin_response.status_code == 200, admin_response.get_json()
    assert [record["name"] for record in admin_response.get_json()["data"]] == ["Asha", "Bayo"]
