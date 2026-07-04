"""Generated UI i18n regressions."""

from __future__ import annotations

from compiler.compiler import compile_apg_file


def _example_10_namespace() -> dict[str, object]:
	result = compile_apg_file("examples/10_themed_i18n_streaming_capability/main.apg")
	assert result.success, result.errors
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	return namespace


def test_example_10_generates_language_switcher_and_locale_cookie():
	namespace = _example_10_namespace()
	app = namespace["_flask_app"]

	assert "sw" in namespace["APG_SUPPORTED_LANGUAGES"]
	assert "ar" in namespace["APG_SUPPORTED_LANGUAGES"]

	with app.test_client() as client:
		default_ui = client.get("/ui")
		assert default_ui.status_code == 200
		assert b'lang="en"' in default_ui.data
		assert b"apg-locale-select" in default_ui.data

		switched = client.post("/locale", data={"lang": "sw", "next": "/ui"})
		assert switched.status_code == 302
		assert switched.headers["Location"] == "/ui"

		sw_ui = client.get("/ui")
		assert b'lang="sw"' in sw_ui.data
		assert "Nyumbani".encode("utf-8") in sw_ui.data

		client.post("/locale", data={"lang": "ar", "next": "/ui"})
		ar_ui = client.get("/ui")
		assert b'lang="ar"' in ar_ui.data
		assert b'dir="rtl"' in ar_ui.data


def test_locale_format_helpers_are_available_to_templates():
	namespace = _example_10_namespace()

	assert namespace["format_currency"](1234.5) == "$1,234.50"
	assert namespace["format_date"]("2026-07-04") == "2026-07-04"
