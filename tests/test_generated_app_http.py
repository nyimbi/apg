"""HTTP efficiency coverage for the generated Flask app template."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

from compiler.ast_builder import EntityDeclaration, EntityType, ModuleDeclaration, PropertyDeclaration, TypeAnnotation
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator
from compiler.compiler import APGCompiler


_HTTP_ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_DISABLE_GZIP",
    "APG_ENV",
    "APG_JWT_SECRET",
    "APG_PRODUCTION",
    "APG_SESSION_SECRET",
)


def _clear_http_env(monkeypatch) -> None:
    for key in _HTTP_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _generated_namespace(monkeypatch, *, root_path=None, **env_overrides):
    _clear_http_env(monkeypatch)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    app_file = Path(root_path or ".").resolve() / "generated_http_app.py"
    module = ModuleDeclaration(
        name="http_probe",
        entities=[
            EntityDeclaration(
                entity_type=EntityType.FORM,
                name="Customer",
                properties=[
                    PropertyDeclaration("name", TypeAnnotation("str")),
                    PropertyDeclaration("description", TypeAnnotation("str")),
                ],
            )
        ],
    )
    files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(module)
    namespace = {"__file__": str(app_file)}
    exec(compile(files["app.py"], "generated_http_app.py", "exec"), namespace)
    return namespace


def _auth_namespace(monkeypatch):
    _clear_http_env(monkeypatch)
    source = """
module secure_http_probe version 1.0.0 {}

entity Customer {
    name: str;
}

security {
    authentication: required;
}
"""
    result = APGCompiler().compile_string(source, "secure_http_probe")
    assert result.success, result.errors
    namespace = {"__file__": "generated_secure_http_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_secure_http_app.py", "exec"), namespace)
    return namespace


def _seed_large_records(namespace, count: int = 20) -> None:
    for index in range(count):
        status, payload = namespace["create_record"](
            "Customer",
            {
                "name": f"Customer {index:02d}",
                "description": "generated payload " * 12,
            },
        )
        assert status == 201, payload


def test_gzip_compresses_large_json_response(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    _seed_large_records(namespace)
    client = namespace["_flask_app"].test_client()

    response = client.get("/records/Customer", headers={"Accept-Encoding": "br, gzip"})
    decompressed = gzip.decompress(response.data)

    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert int(response.headers["Content-Length"]) == len(response.data)
    assert len(decompressed) > 860
    assert b'"entity": "Customer"' in decompressed


def test_gzip_can_be_disabled_with_environment_flag(monkeypatch):
    namespace = _generated_namespace(monkeypatch, APG_DISABLE_GZIP="1")
    _seed_large_records(namespace)
    client = namespace["_flask_app"].test_client()

    response = client.get("/records/Customer", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert "Content-Encoding" not in response.headers
    assert len(response.data) > 860


def test_records_list_response_sets_etag(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    status, payload = namespace["create_record"]("Customer", {"name": "Asha", "description": "etag probe"})
    assert status == 201, payload
    client = namespace["_flask_app"].test_client()

    record_id = payload["record"]["id"]
    response = client.get(f"/records/Customer/{record_id}")
    expected = '"' + hashlib.sha256(response.data).hexdigest()[:16] + '"'

    assert response.status_code == 200
    assert response.headers["ETag"] == expected


def test_records_list_response_returns_304_for_matching_etag(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    namespace["create_record"]("Customer", {"name": "Asha", "description": "etag probe"})
    client = namespace["_flask_app"].test_client()

    first = client.get("/records/Customer")
    second = client.get("/records/Customer", headers={"If-None-Match": first.headers["ETag"]})

    assert second.status_code == 304
    assert second.headers["ETag"] == first.headers["ETag"]
    assert second.data == b""


def test_cache_control_headers_for_static_records_and_private_pages(monkeypatch, tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "probe.txt").write_text("static asset", encoding="utf-8")
    namespace = _generated_namespace(monkeypatch, root_path=tmp_path)
    client = namespace["_flask_app"].test_client()

    static_response = client.get("/static/probe.txt")
    records_response = client.get("/records/Customer")
    ui_response = client.get("/ui")

    auth_namespace = _auth_namespace(monkeypatch)
    auth_client = auth_namespace["_flask_app"].test_client()
    login_response = auth_client.get("/login")

    assert static_response.status_code == 200
    assert static_response.headers["Cache-Control"] == "public,max-age=31536000,immutable"
    assert records_response.status_code == 200
    assert records_response.headers["Cache-Control"] == "no-cache"
    assert ui_response.status_code == 200
    assert ui_response.headers["Cache-Control"] == "no-store,private"
    assert login_response.status_code == 200
    assert login_response.headers["Cache-Control"] == "no-store,private"


def test_openui_library_and_renderer_are_shipped(monkeypatch, tmp_path):
	"""A generated app serves the OpenUI contract and the browser renderer.

	The renderer is only useful if both halves ship: the asset that parses and
	paints, and the library document it parses against. A page that loads the
	script without the document renders nothing, which no unit test would catch.
	"""
	_clear_http_env(monkeypatch)
	module = ModuleDeclaration(
		name="openui_probe",
		entities=[
			EntityDeclaration(
				entity_type=EntityType.FORM,
				name="Customer",
				properties=[
					PropertyDeclaration("name", TypeAnnotation("str")),
					PropertyDeclaration("description", TypeAnnotation("str")),
				],
			)
		],
	)
	generator = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False))
	files = generator.generate(module)
	files.update(generator.generate_deployment_bundle(module))

	# Write the static half out, so the assertions read the files the app serves.
	static_dir = tmp_path / "static"
	static_dir.mkdir()
	for name, body in files.items():
		if name.startswith("static/"):
			(static_dir / name.split("/", 1)[1]).write_text(body, encoding="utf-8")

	namespace = {"__file__": str(tmp_path / "generated_openui_app.py")}
	exec(compile(files["app.py"], "generated_openui_app.py", "exec"), namespace)
	client = namespace["_flask_app"].test_client()

	library_response = client.get("/openui-library.json")
	assert library_response.status_code == 200
	document = library_response.get_json()
	assert document["root"] == "Page"
	assert "Page" in document["schema"]["properties"]
	assert "Stack" in document["schema"]["$defs"]

	assert "openui_library.json" in files
	assert "openui_prompt.txt" in files
	assert "Page(" in files["openui_prompt.txt"]

	renderer = (static_dir / "apg-openui.js").read_text(encoding="utf-8")
	assert "APGOpenUI" in renderer

	# Every class the renderer emits has to be in the stylesheet it ships with.
	stylesheet = (static_dir / "apg.css").read_text(encoding="utf-8")
	for css_class in ("apg-openui-page", "apg-openui-stack", "apg-openui-stream", "apg-openui-text"):
		assert f".{css_class}" in stylesheet, f"{css_class} missing from the shipped stylesheet"

	# The shell loads the renderer, or nothing above is reachable.
	index = client.get("/ui")
	assert index.status_code == 200
	assert "/static/apg-openui.js" in index.get_data(as_text=True)
