"""Generated app enum and field validation coverage."""

from __future__ import annotations

import os

import pytest

from compiler.compiler import compile_apg_string


VALIDATION_APP_SOURCE = """
module validation_probe version 1.0.0 {}

enum Status { Draft; Published; Archived; }

entity Article {
    title: str @min_length(3) @max_length(200);
    status: Status;
    views: int @min(0);
    price: float @min(0.0) @max(999999.99);
    email: str @email;
    slug: str @pattern('[a-z0-9-]+');
}
"""


_GENERATED_ENV_KEYS = (
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


@pytest.fixture(scope="module")
def generated_app():
    original_env = {key: os.environ.get(key) for key in _GENERATED_ENV_KEYS}
    for key in _GENERATED_ENV_KEYS:
        os.environ.pop(key, None)
    result = compile_apg_string(VALIDATION_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_validation_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_validation_app.py", "exec"), namespace)
    app = namespace["_flask_app"]
    app.config["TESTING"] = True
    try:
        yield app
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture()
def client(generated_app):
    return generated_app.test_client()


def _valid_article(**overrides):
    record = {
        "title": "Draft article",
        "status": "Draft",
        "views": 0,
        "price": 10.5,
        "email": "editor@example.com",
        "slug": "draft-article",
    }
    record.update(overrides)
    return record


def _post_article(client, **overrides):
    return client.post("/records/Article", json={"record": _valid_article(**overrides)})


def test_enum_constraint_rejects_invalid(client):
    response = _post_article(client, status="Invalid")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "invalid_enum_value",
        "field": "status",
        "allowed": ["Draft", "Published", "Archived"],
    }


def test_enum_constraint_accepts_valid(client):
    response = _post_article(client, status="Draft")

    assert response.status_code == 201, response.get_json()


def test_min_length_validation(client):
    response = _post_article(client, title="Hi")

    assert response.status_code == 400
    assert response.get_json()["rule"] == "min_length"


def test_max_length_validation(client):
    response = _post_article(client, title="x" * 201)

    assert response.status_code == 400
    assert response.get_json()["rule"] == "max_length"


def test_email_validation(client):
    response = _post_article(client, email="notanemail")

    assert response.status_code == 400
    assert response.get_json()["rule"] == "email"


def test_pattern_validation(client):
    response = _post_article(client, slug="Has Spaces")

    assert response.status_code == 400
    assert response.get_json()["rule"] == "pattern"
