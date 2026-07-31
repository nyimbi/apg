"""Wave I database-pattern coverage for the generated Flask app template."""

from __future__ import annotations

import pytest

from compiler.compiler import compile_apg_string


DB_APP_SOURCE = """
module db_patterns_probe version 1.0.0 {}

table Customer {
    name: str;
}
"""


_DB_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)


@pytest.fixture()
def generated_db_app(monkeypatch):
    for key in _DB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(DB_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_db_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_db_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


@pytest.fixture()
def client(generated_db_app):
    return generated_db_app["_flask_app"].test_client()


def _create_customer(client, name: str = "a") -> dict[str, object]:
    response = client.post("/records/Customer", json={"name": name})
    assert response.status_code == 201, response.get_json()
    return response.get_json()["record"]


def test_timestamps_in_response(generated_db_app, client):
    # APG_EXPOSE_TIMESTAMPS defaults to off; patch the module-level bool to opt in.
    generated_db_app["APG_EXPOSE_TIMESTAMPS"] = True
    created = _create_customer(client)
    response = client.get(f"/records/Customer/{created['id']}")
    record = response.get_json()["record"]

    assert response.status_code == 200
    assert "created_at" in record
    assert "updated_at" in record
    assert "deleted_at" not in record


def test_soft_delete_hides_row(client):
    created = _create_customer(client)

    delete_response = client.delete(f"/records/Customer/{created['id']}")
    normal_response = client.get("/records/Customer")
    deleted_response = client.get("/records/Customer?include_deleted=1")

    assert delete_response.status_code == 200
    assert created["id"] not in [record["id"] for record in normal_response.get_json()["data"]]
    assert created["id"] in [record["id"] for record in deleted_response.get_json()["data"]]


def test_restore_endpoint(client):
    created = _create_customer(client)
    delete_response = client.delete(f"/records/Customer/{created['id']}")
    restore_response = client.delete(f"/records/Customer/{created['id']}/restore")
    list_response = client.get("/records/Customer")

    assert delete_response.status_code == 200
    assert restore_response.status_code == 200
    assert created["id"] in [record["id"] for record in list_response.get_json()["data"]]


def test_bulk_create(client):
    response = client.post("/records/Customer/bulk", json={"create": [{"name": "a"}, {"name": "b"}]})

    assert response.status_code == 200
    assert response.get_json()["created"] == 2


def test_bulk_limit(client):
    response = client.post(
        "/records/Customer/bulk",
        json={"create": [{"name": str(index)} for index in range(1001)]},
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": "bulk_limit_exceeded"}
