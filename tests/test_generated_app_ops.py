"""Ops-hardening coverage for the generated Flask app template."""

import json
import uuid

from compiler.ast_builder import EntityDeclaration, EntityType, ModuleDeclaration, PropertyDeclaration, TypeAnnotation
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator


_OPS_ENV_KEYS = (
    "APG_ENV",
    "APG_JSON_LOGS",
    "APG_PRODUCTION",
    "APG_METRICS",
    "APG_METRICS_TOKEN",
)


def _generated_namespace(monkeypatch, **env_overrides):
    for key in _OPS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    module = ModuleDeclaration(
        name="ops_probe",
        entities=[
            EntityDeclaration(
                entity_type=EntityType.FORM,
                name="Customer",
                properties=[
                    PropertyDeclaration("name", TypeAnnotation("str")),
                ],
            )
        ],
    )
    files = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False)).generate(module)
    namespace = {"__file__": "generated_ops_app.py"}
    exec(files["app.py"], namespace)
    return namespace


def test_x_request_id_is_echoed_back(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    response = client.get("/health", headers={"X-Request-ID": "req-from-client"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-from-client"


def test_x_request_id_is_generated_when_absent(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    response = client.get("/health")
    request_id = response.headers.get("X-Request-ID")

    assert response.status_code == 200
    assert request_id
    assert str(uuid.UUID(request_id)) == request_id


def test_livez_returns_ok_json(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    response = client.get("/livez")

    assert response.status_code == 200
    assert response.content_type == "application/json"
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert isinstance(payload["uptime_s"], int)


def test_readyz_transitions_after_first_successful_request(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = namespace["_flask_app"].test_client()

    starting = client.get("/readyz")
    client.get("/livez")
    ready = client.get("/readyz")

    assert starting.status_code == 503
    assert starting.get_json() == {"status": "starting"}
    assert ready.status_code == 200
    assert ready.get_json() == {"status": "ready"}


def test_metrics_token_and_text_exposition(monkeypatch):
    namespace = _generated_namespace(
        monkeypatch,
        APG_METRICS="1",
        APG_METRICS_TOKEN="secret-token",
    )
    client = namespace["_flask_app"].test_client()
    client.get("/records/Customer/123")

    unauthorized = client.get("/metrics", headers={"X-Metrics-Token": "wrong"})
    ok = client.get("/metrics", headers={"X-Metrics-Token": "secret-token"})
    body = ok.get_data(as_text=True)

    assert unauthorized.status_code == 401
    assert ok.status_code == 200
    assert ok.content_type.startswith("text/plain; version=0.0.4")
    assert "# TYPE apg_http_requests_total counter" in body
    assert 'path_template="/records/Customer/:id"' in body
    assert "# TYPE apg_http_request_duration_seconds histogram" in body
    assert "# TYPE apg_active_requests gauge" in body


def test_json_log_formatter_emits_parseable_json(monkeypatch, capsys):
    namespace = _generated_namespace(monkeypatch, APG_JSON_LOGS="1")
    client = namespace["_flask_app"].test_client()

    response = client.get("/health", headers={"X-Request-ID": "json-log-id"})
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    finish = next(record for record in records if record["msg"] == "request_finish")

    assert response.status_code == 200
    assert finish["req_id"] == "json-log-id"
    assert finish["method"] == "GET"
    assert finish["path"] == "/health"
    assert finish["status"] == 200
    assert isinstance(finish["ms"], int)
