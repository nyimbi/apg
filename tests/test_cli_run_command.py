"""Focused tests for the APG application runner CLI."""

from __future__ import annotations

import subprocess
import socket
import urllib.request
from pathlib import Path

from click.testing import CliRunner

from cli import run_command


def test_detect_application_kind_accepts_generated_runtime_styles():
	assert run_command._detect_application_kind("app = Flask(__name__)\napp.run()") == "flask"
	assert run_command._detect_application_kind("app = FastAPI()\nuvicorn.run(app)") == "fastapi"
	assert run_command._detect_application_kind("def main():\n\treturn 0\nif __name__ == '__main__':\n\tmain()") == "python"
	assert run_command._detect_application_kind("VALUE = 1") is None


def test_run_single_runs_fastapi_generated_app(tmp_path, monkeypatch):
	app = tmp_path / "app.py"
	app.write_text("from fastapi import FastAPI\nimport uvicorn\napp = FastAPI()\nuvicorn.run(app)\n")
	calls: list[dict[str, object]] = []

	def fake_run(command, cwd, env, check):
		calls.append({"command": command, "cwd": cwd, "env": env, "check": check})
		return subprocess.CompletedProcess(command, 0)

	monkeypatch.setattr(run_command.subprocess, "run", fake_run)

	run_command._run_single(app, "127.0.0.1", 8123, debug=True)

	assert len(calls) == 1
	assert calls[0]["cwd"] == tmp_path
	assert calls[0]["env"]["HOST"] == "127.0.0.1"
	assert calls[0]["env"]["PORT"] == "8123"
	assert calls[0]["env"]["FLASK_PORT"] == "8123"
	assert calls[0]["env"]["FLASK_DEBUG"] == "1"


def test_run_single_rejects_non_executable_python(tmp_path, monkeypatch):
	app = tmp_path / "constants.py"
	app.write_text("VALUE = 1\n")
	calls = []

	def fake_run(*args, **kwargs):
		calls.append((args, kwargs))

	monkeypatch.setattr(run_command.subprocess, "run", fake_run)

	run_command._run_single(app, "127.0.0.1", 8123, debug=False)

	assert calls == []


def test_check_command_prefers_health_endpoint(monkeypatch):
	opened_urls: list[str] = []

	class FakeSocket:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, traceback):
			return False

		def getpeername(self):
			return ("127.0.0.1", 8080)

	class FakeResponse:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, traceback):
			return False

		def getcode(self):
			return 200

	def fake_create_connection(address, timeout):
		return FakeSocket()

	def fake_urlopen(url, timeout):
		opened_urls.append(url)
		return FakeResponse()

	monkeypatch.setattr(socket, "create_connection", fake_create_connection)
	monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

	result = CliRunner().invoke(run_command.check, ["--port", "8080"])

	assert result.exit_code == 0
	assert opened_urls == ["http://127.0.0.1:8080/health"]
	assert "Application is running" in result.output
