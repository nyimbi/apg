"""Relationship support coverage for generated APG apps."""

from __future__ import annotations

import json

from compiler.compiler import APGCompiler


RELATIONSHIP_SOURCE = """
module relationship_probe version 1.0.0 {}

entity Customer {
    name: str;
    has_many Order;
}

entity Order {
    amount: float;
    belongs_to Customer;
}
"""


ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_DATABASE_URL",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PRODUCTION",
    "APG_SECRET_KEY",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
)


def _generated_namespace(monkeypatch, tmp_path):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APG_SQLITE_PATH", str(tmp_path / "relationships.sqlite"))
    result = APGCompiler().compile_string(RELATIONSHIP_SOURCE, "relationships.apg")
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": str(tmp_path / "generated_relationship_app.py")}
    exec(compile(result.generated_files["app.py"], "generated_relationship_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


def test_fk_column_created(monkeypatch, tmp_path):
    namespace = _generated_namespace(monkeypatch, tmp_path)
    conn = namespace["_sqlite_connection"]()

    columns = {
        row["name"]
        for row in conn.execute('PRAGMA table_info("Order")').fetchall()
    }

    assert "customer_id" in columns


def test_nested_list_endpoint(monkeypatch, tmp_path):
    namespace = _generated_namespace(monkeypatch, tmp_path)
    namespace["create_record"]("Customer", {"id": 1, "name": "Asha"})
    namespace["create_record"]("Customer", {"id": 2, "name": "Noor"})
    namespace["create_record"]("Order", {"id": 10, "amount": 11.5, "customer_id": 1})
    namespace["create_record"]("Order", {"id": 20, "amount": 99.0, "customer_id": 2})
    client = namespace["_flask_app"].test_client()

    response = client.get("/records/Customer/1/orders")
    payload = json.loads(response.data)

    assert response.status_code == 200
    assert [record["id"] for record in payload["records"]] == [10]


def test_fk_set_on_create(monkeypatch, tmp_path):
    namespace = _generated_namespace(monkeypatch, tmp_path)
    namespace["create_record"]("Customer", {"id": 1, "name": "Asha"})
    client = namespace["_flask_app"].test_client()

    create_response = client.post("/records/Order", json={"amount": 27.5, "customer_id": 1})
    nested_response = client.get("/records/Customer/1/orders")
    payload = json.loads(nested_response.data)

    assert create_response.status_code == 201
    assert nested_response.status_code == 200
    assert payload["records"][0]["customer_id"] == 1
    assert payload["records"][0]["amount"] == 27.5


def test_openapi_includes_nested_paths(monkeypatch, tmp_path):
    namespace = _generated_namespace(monkeypatch, tmp_path)
    client = namespace["_flask_app"].test_client()

    response = client.get("/openapi.json")
    payload = json.loads(response.data)

    assert response.status_code == 200
    assert "/records/Customer/{id}/orders" in payload["paths"]
