"""Computed field regressions for the generated Flask app template."""

from __future__ import annotations

from compiler.ast_builder import ComputedFieldNode
from compiler.compiler import compile_apg_string


COMPUTED_APP_SOURCE = """
module wave_r_computed_app version 1.0.0 {}

table Product {
    name: str;
    price: float;
    tax_rate: float;
    total: float = price * (1 + tax_rate);
    display_name: str = 'Product: ' + name;
}
"""


ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_PRODUCTION",
    "APG_SECRET_KEY",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)


def _generated_namespace(monkeypatch) -> dict[str, object]:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(COMPUTED_APP_SOURCE)
    assert result.success, result.errors
    product = next(entity for entity in result.module.entities if entity.name == "Product")
    computed_fields = {
        field.name: field
        for field in product.properties
        if isinstance(field, ComputedFieldNode)
    }
    assert computed_fields["total"].expression == "price * (1 + tax_rate)"
    assert computed_fields["display_name"].expression == "'Product: ' + name"
    namespace: dict[str, object] = {"__file__": "generated_wave_r_computed_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_wave_r_computed_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


def _client(namespace: dict[str, object]):
    return namespace["_flask_app"].test_client()


def _create_product(client, name: str = "Widget") -> dict[str, object]:
    response = client.post(
        "/records/Product",
        json={"name": name, "price": 10, "tax_rate": 0.1},
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["record"]


def test_computed_field_in_response(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    created = _create_product(client)

    response = client.get(f"/records/Product/{created['id']}")
    openapi = namespace["openapi_document"]()

    assert response.status_code == 200
    assert response.get_json()["record"]["total"] == 11.0
    assert openapi["components"]["schemas"]["ProductRecord"]["properties"]["total"]["readOnly"] is True


def test_computed_field_not_in_db(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    conn = namespace["_sqlite_connection"]()

    columns = {row["name"] for row in conn.execute('PRAGMA table_info("Product")').fetchall()}

    assert "total" not in columns
    assert "display_name" not in columns


def test_computed_string_field(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    created = _create_product(client, name="Anvil")

    response = client.get(f"/records/Product/{created['id']}")

    assert response.status_code == 200
    assert response.get_json()["record"]["display_name"] == "Product: Anvil"
