"""Wave E API completeness coverage for the generated Flask app template."""

from __future__ import annotations

import csv
import io

import pytest

from compiler.compiler import compile_apg_string


API_APP_SOURCE = """
module api_probe version 1.2.3 {
	description: "API probe app";
}

table Customer {
	name: str;
	email: str;
	notes: str;
	age: int;
}
"""


_API_ENV_KEYS = (
	"APG_API_KEY",
	"APG_AUTH_USERS",
	"APG_DATABASE_URL",
	"APG_DATA_FILE",
	"APG_DATA_PATH",
	"APG_ENV",
	"APG_JWT_SECRET",
	"APG_PG_URL",
	"APG_PRODUCTION",
	"APG_SESSION_SECRET",
	"APG_SWAGGER_UI",
	"DATABASE_URL",
)


@pytest.fixture()
def generated_api_app(monkeypatch):
	for key in _API_ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	result = compile_apg_string(API_APP_SOURCE)
	assert result.success, result.errors
	namespace: dict[str, object] = {"__file__": "generated_api_app.py"}
	exec(compile(result.generated_files["app.py"], "generated_api_app.py", "exec"), namespace)
	return namespace


@pytest.fixture()
def client(generated_api_app):
	app = generated_api_app["_flask_app"]
	app.config["TESTING"] = True
	return app.test_client()


def _seed_customers(generated_api_app, records: list[dict[str, object]]) -> list[dict[str, object]]:
	created: list[dict[str, object]] = []
	for record in records:
		status, payload = generated_api_app["create_record"]("Customer", record)
		assert status == 201, payload
		created.append(payload["record"])
	return created


def test_openapi_json_returns_entity_record_paths_and_api_docs(monkeypatch, generated_api_app, client):
	response = client.get("/openapi.json")

	spec = response.get_json()
	assert response.status_code == 200
	assert spec["openapi"] == "3.1.0"
	assert spec["info"] == {
		"title": generated_api_app["APG_APP_NAME"],
		"version": generated_api_app["APG_APP_VERSION"],
		"description": generated_api_app["APG_APP_DESCRIPTION"],
	}
	assert "/records/Customer" in spec["paths"]
	assert "/records/Customer/{id}" in spec["paths"]
	assert set(spec["paths"]["/records/Customer"]) == {"get", "post"}
	assert set(spec["paths"]["/records/Customer/{id}"]) == {"get", "put", "delete"}

	monkeypatch.setenv("APG_SWAGGER_UI", "1")
	docs_response = client.get("/api-docs")
	assert docs_response.status_code == 200
	assert docs_response.content_type == "text/html; charset=utf-8"
	assert "/openapi.json" in docs_response.data.decode("utf-8")


def test_records_pagination_limit_and_next_cursor(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[
			{"name": "Alpha", "email": "alpha@example.com", "notes": "first", "age": 10},
			{"name": "Bravo", "email": "bravo@example.com", "notes": "second", "age": 20},
			{"name": "Charlie", "email": "charlie@example.com", "notes": "third", "age": 30},
		],
	)

	response = client.get("/records/Customer?limit=2")
	payload = response.get_json()

	assert response.status_code == 200
	assert [record["name"] for record in payload["data"]] == ["Alpha", "Bravo"]
	assert payload["total"] == 3
	assert payload["next_cursor"] == payload["data"][-1]["id"]
	assert 'rel="next"' in response.headers["Link"]
	assert "after=2" in response.headers["Link"]

	compat_response = client.get("/records/Customer?limit=2", headers={"X-APG-Compat": "v1"})
	compat_payload = compat_response.get_json()
	assert compat_response.status_code == 200
	assert isinstance(compat_payload, list)
	assert [record["name"] for record in compat_payload] == ["Alpha", "Bravo"]


def test_records_filter_exact_match(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[
			{"name": "foo", "email": "foo@example.com", "notes": "target", "age": 10},
			{"name": "bar", "email": "bar@example.com", "notes": "other", "age": 20},
		],
	)

	response = client.get("/records/Customer", query_string={"filter[name]": "foo"})
	payload = response.get_json()

	assert response.status_code == 200
	assert [record["name"] for record in payload["data"]] == ["foo"]
	assert payload["total"] == 1


def test_records_q_searches_string_fields(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[
			{"name": "Alpha", "email": "alpha@example.com", "notes": "needle match", "age": 10},
			{"name": "Bravo", "email": "bravo@example.com", "notes": "other", "age": 20},
		],
	)

	response = client.get("/records/Customer?q=needle")
	payload = response.get_json()

	assert response.status_code == 200
	assert [record["name"] for record in payload["data"]] == ["Alpha"]


def test_records_sort_descending(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[
			{"name": "Alpha", "email": "alpha@example.com", "notes": "one", "age": 10},
			{"name": "Charlie", "email": "charlie@example.com", "notes": "two", "age": 30},
			{"name": "Bravo", "email": "bravo@example.com", "notes": "three", "age": 20},
		],
	)

	response = client.get("/records/Customer?sort=name&sort_dir=desc")
	payload = response.get_json()

	assert response.status_code == 200
	assert [record["name"] for record in payload["data"]] == ["Charlie", "Bravo", "Alpha"]


def test_records_unknown_filter_field_returns_400(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[{"name": "Alpha", "email": "alpha@example.com", "notes": "one", "age": 10}],
	)

	response = client.get("/records/Customer", query_string={"filter[unknown]": "x"})

	assert response.status_code == 400
	assert response.get_json() == {"error": "invalid_field"}


def test_records_csv_export_returns_text_csv_with_header(generated_api_app, client):
	_seed_customers(
		generated_api_app,
		[
			{"name": "Bravo", "email": "bravo@example.com", "notes": "two", "age": 20},
			{"name": "Alpha", "email": "alpha@example.com", "notes": "one", "age": 10},
		],
	)

	response = client.get("/records/Customer?format=csv&sort=name&sort_dir=asc")
	rows = list(csv.reader(io.StringIO(response.data.decode("utf-8"))))

	assert response.status_code == 200
	assert response.content_type == "text/csv; charset=utf-8"
	assert response.headers["Content-Disposition"].startswith("attachment; filename=\"Customer_")
	assert rows[0] == ["id", "name", "email", "notes", "age"]
	assert [row[1] for row in rows[1:]] == ["Alpha", "Bravo"]
