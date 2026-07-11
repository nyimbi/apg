"""Background job regressions for the generated Flask app template."""

from __future__ import annotations

import time

from compiler.compiler import compile_apg_string


JOB_APP_SOURCE = """
module wave_r_jobs_app version 1.0.0 {}

table Task {
    name: str;
}
"""


ENV_KEYS = (
    "APG_API_KEY",
    "APG_AUTH_USERS",
    "APG_DATA_FILE",
    "APG_DATA_PATH",
    "APG_DB_PATH",
    "APG_ENV",
    "APG_JOB_MAX_RETRIES",
    "APG_PRODUCTION",
    "APG_SECRET_KEY",
    "APG_SESSION_SECRET",
    "APG_SQLITE_PATH",
    "APG_WORKER_THREADS",
    "DATABASE_URL",
)


def _generated_namespace(monkeypatch) -> dict[str, object]:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APG_WORKER_THREADS", "1")
    monkeypatch.setenv("APG_JOB_MAX_RETRIES", "1")
    result = compile_apg_string(JOB_APP_SOURCE)
    assert result.success, result.errors
    namespace: dict[str, object] = {"__file__": "generated_wave_r_jobs_app.py"}
    exec(compile(result.generated_files["app.py"], "generated_wave_r_jobs_app.py", "exec"), namespace)
    namespace["_flask_app"].config["TESTING"] = True
    return namespace


def _client(namespace: dict[str, object]):
    return namespace["_flask_app"].test_client()


def _enqueue_echo(client, payload: dict[str, object] | None = None) -> str:
    response = client.post("/jobs", json={"type": "apg.echo", "payload": payload or {"ok": True}})
    assert response.status_code == 201, response.get_json()
    return str(response.get_json()["job_id"])


def test_enqueue_job_returns_id(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)

    response = client.post("/jobs", json={"type": "apg.echo", "payload": {"message": "hello"}})

    assert response.status_code == 201
    assert response.get_json()["job_id"]


def test_job_status_endpoint(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    job_id = _enqueue_echo(client)

    response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert "status" in response.get_json()


def test_job_list_endpoint(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    _enqueue_echo(client)

    response = client.get("/jobs")

    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


def test_job_worker_processes_job(monkeypatch):
    namespace = _generated_namespace(monkeypatch)
    client = _client(namespace)
    job_id = _enqueue_echo(client, {"message": "done"})

    status = None
    for _attempt in range(40):
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        status = response.get_json()["status"]
        if status == "done":
            break
        time.sleep(0.05)

    assert status == "done"
