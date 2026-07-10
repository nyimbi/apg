"""Generated app accessibility and CSP hardening regressions."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from compiler.compiler import APGCompiler


A11Y_SOURCE = """
module a11y_customer_app version 1.0.0 {
    description: "Accessible generated UI";
}

entity Customer {
    name: str;
    email: str;
}
"""


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.captions: list[str] = []
        self._caption_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag == "caption":
            self._caption_parts = []

    def handle_data(self, data: str) -> None:
        if self._caption_parts is not None:
            self._caption_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "caption" and self._caption_parts is not None:
            self.captions.append(" ".join("".join(self._caption_parts).split()))
            self._caption_parts = None


def _generated_namespace() -> dict[str, object]:
    result = APGCompiler().compile_string(A11Y_SOURCE, "a11y_customer_app")
    assert result.success, result.errors
    namespace: dict[str, object] = {}
    exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
    return namespace


def _ui_response_with_customer_record():
    namespace = _generated_namespace()
    status, payload = namespace["create_record"](
        "Customer",
        {"name": "Ada Lovelace", "email": "ada@example.com"},
    )
    assert status == 201, payload
    app = namespace["_flask_app"]
    with app.test_client() as client:
        response = client.get("/ui/entities/Customer")
    assert response.status_code == 200
    return response


def _parsed_ui() -> tuple[str, _TagCollector, object]:
    response = _ui_response_with_customer_record()
    html = response.get_data(as_text=True)
    parser = _TagCollector()
    parser.feed(html)
    return html, parser, response


def test_generated_table_caption_contains_entity_name():
    _html, parser, _response = _parsed_ui()

    assert any("Customer records" in caption for caption in parser.captions)


def test_generated_table_header_cells_have_col_scope():
    _html, parser, _response = _parsed_ui()

    header_cells = [attrs for tag, attrs in parser.tags if tag == "th"]
    assert header_cells
    assert all(attrs.get("scope") == "col" for attrs in header_cells)


def test_generated_ui_has_density_toggle_button():
    html, _parser, _response = _parsed_ui()

    assert 'id="apg-density-toggle"' in html
    assert 'data-apg-density-toggle' in html


def test_generated_csp_header_uses_nonce_without_unsafe_inline():
    _html, _parser, response = _parsed_ui()

    csp = response.headers["Content-Security-Policy"]
    assert "nonce-" in csp
    assert "unsafe-inline" not in csp


def test_generated_inline_script_tags_match_csp_nonce():
    _html, parser, response = _parsed_ui()

    csp = response.headers["Content-Security-Policy"]
    match = re.search(r"script-src[^;]*'nonce-([^']+)'", csp)
    assert match is not None
    nonce = match.group(1)
    inline_scripts = [
        attrs
        for tag, attrs in parser.tags
        if tag == "script" and not attrs.get("src")
    ]
    assert inline_scripts
    assert all(attrs.get("nonce") == nonce for attrs in inline_scripts)
