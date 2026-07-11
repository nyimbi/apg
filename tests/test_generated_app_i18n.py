"""Generated app runtime i18n scaffold coverage."""

from __future__ import annotations

import json

import pytest

from compiler.compiler import compile_apg_string


I18N_APP_SOURCE = """
module i18n_probe version 1.0.0 {}

entity Customer {
    name: str;
}
"""


_I18N_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_AUTO_MIGRATE",
    "APG_DATABASE_URL",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_EXPORT_LOCALE",
    "APG_JWT_SECRET",
    "APG_LOCALE",
    "APG_LOCALE_DIR",
    "APG_LOCALE_FILE",
    "APG_MULTI_TENANT",
    "APG_PG_URL",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "DATABASE_URL",
)

_REQUIRED_KEYS = {
    "save",
    "cancel",
    "delete",
    "confirm",
    "error",
    "success",
    "loading",
    "no_records",
    "search",
    "login",
    "logout",
}


def _generated_app(monkeypatch: pytest.MonkeyPatch, **env: str) -> tuple[dict[str, object], object, str]:
    for key in _I18N_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = compile_apg_string(I18N_APP_SOURCE)
    assert result.success, result.errors
    source = result.generated_files["app.py"]
    namespace: dict[str, object] = {"__file__": "generated_i18n_app.py"}
    exec(compile(source, "generated_i18n_app.py", "exec"), namespace)
    app = namespace["_flask_app"]
    app.config["TESTING"] = True
    return namespace, app.test_client(), source


def test_builtin_locale_endpoint_exposes_required_keys_and_placeholders(monkeypatch: pytest.MonkeyPatch):
    _namespace, client, source = _generated_app(monkeypatch)

    english = client.get("/locales/en.json").get_json()
    swahili = client.get("/locales/sw.json").get_json()

    assert "def _apg_t" in source
    assert _REQUIRED_KEYS <= set(english)
    assert {key: english[key] for key in _REQUIRED_KEYS} == {
        "save": "Save",
        "cancel": "Cancel",
        "delete": "Delete",
        "confirm": "Confirm",
        "error": "Error",
        "success": "Success",
        "loading": "Loading",
        "no_records": "No records",
        "search": "Search",
        "login": "Login",
        "logout": "Logout",
    }
    assert _REQUIRED_KEYS <= set(swahili)
    assert all(swahili[key] == "" for key in _REQUIRED_KEYS)


def test_accept_language_selects_locale_when_apg_locale_unset(monkeypatch: pytest.MonkeyPatch):
    namespace, client, _source = _generated_app(monkeypatch)

    response = client.get("/ui", headers={"Accept-Language": "fr"})

    assert response.status_code == 200
    assert '<html lang="fr"' in response.data.decode("utf-8")
    with namespace["_flask_app"].test_request_context("/", headers={"Accept-Language": "fr"}):
        assert namespace["_apg_t"]("save") == "Save"


def test_custom_locale_file_loads_for_configured_locale(monkeypatch: pytest.MonkeyPatch, tmp_path):
    locale_path = tmp_path / "test_locale.json"
    expected = {
        "save": "Hifadhi",
        "cancel": "Ghairi",
        "delete": "Futa",
        "confirm": "Thibitisha",
        "error": "Hitilafu",
        "success": "Imefanikiwa",
        "loading": "Inapakia",
        "no_records": "Hakuna rekodi",
        "search": "Tafuta",
        "login": "Ingia",
        "logout": "Toka",
    }
    locale_path.write_text(json.dumps(expected), encoding="utf-8")
    namespace, client, _source = _generated_app(
        monkeypatch,
        APG_LOCALE="sw",
        APG_LOCALE_FILE=str(locale_path),
    )

    response = client.get("/locales/sw.json")

    assert response.status_code == 200, response.get_json()
    assert response.get_json() == expected
    with namespace["_flask_app"].test_request_context("/"):
        assert namespace["_apg_t"]("save") == "Hifadhi"
