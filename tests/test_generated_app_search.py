"""Wave K FTS5 search coverage for generated Flask apps."""

from __future__ import annotations

import pytest

from compiler.compiler import compile_apg_string


SEARCH_APP_SOURCE = """
module search_probe version 1.0.0 {}

table Article {
    title: str;
    body: text;
    views: int;
}
"""


_SEARCH_ENV_KEYS = (
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
def client(monkeypatch):
    for key in _SEARCH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    result = compile_apg_string(SEARCH_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_search_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_search_app.py", "exec"), namespace)
    app = namespace["_flask_app"]
    app.config["TESTING"] = True
    return app.test_client()


def _create_article(client, title: str, body: str = "body") -> dict[str, object]:
    response = client.post(
        "/records/Article",
        json={"record": {"title": title, "body": body, "views": 1}},
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["record"]


def test_search_finds_match(client):
    created = _create_article(client, "Needle Alpha", "exact phrase lives here")

    response = client.get("/records/Article/search", query_string={"q": "Needle"})

    assert response.status_code == 200, response.get_json()
    assert [record["id"] for record in response.get_json()] == [created["id"]]


def test_search_no_match_returns_empty(client):
    _create_article(client, "Needle Alpha")

    response = client.get("/records/Article/search", query_string={"q": "Absent"})

    assert response.status_code == 200, response.get_json()
    assert response.get_json() == []


def test_search_respects_limit(client):
    _create_article(client, "Needle Alpha")
    _create_article(client, "Needle Bravo")
    _create_article(client, "Needle Charlie")

    response = client.get("/records/Article/search", query_string={"q": "Needle", "limit": "2"})

    assert response.status_code == 200, response.get_json()
    assert len(response.get_json()) == 2
