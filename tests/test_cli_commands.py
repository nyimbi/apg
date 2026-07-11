from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import cli


APG_SOURCE = """\
module cli_sample version 1.0.0 {}

entity Customer {
    name: str;
    email: str;
}
"""


def test_init_creates_app_apg():
	runner = CliRunner()
	with runner.isolated_filesystem():
		result = runner.invoke(cli, ["init", "myapp"])

		assert result.exit_code == 0, result.output
		assert Path("myapp/app.apg").exists()


def test_doctor_ok_in_dev_mode(monkeypatch):
	for name in ("APG_PRODUCTION", "APG_SECRET_KEY", "APG_AUTH_USERS", "APG_SMTP_HOST"):
		monkeypatch.delenv(name, raising=False)

	result = CliRunner().invoke(cli, ["doctor"])

	assert result.exit_code == 0, result.output
	assert "[OK]" in result.output


def test_doctor_fails_in_production_without_key(monkeypatch):
	for name in ("APG_SECRET_KEY", "APG_AUTH_USERS", "APG_SMTP_HOST"):
		monkeypatch.delenv(name, raising=False)
	monkeypatch.setenv("APG_PRODUCTION", "1")

	result = CliRunner().invoke(cli, ["doctor"])

	assert result.exit_code != 0, result.output
	assert "APG_SECRET_KEY is required" in result.output


def test_compile_outputs_app_py():
	runner = CliRunner()
	with runner.isolated_filesystem():
		Path("app.apg").write_text(APG_SOURCE, encoding="utf-8")

		result = runner.invoke(cli, ["compile", "app.apg", "-o", "out"])

		assert result.exit_code == 0, result.output
		assert Path("out/app.py").exists()


def test_export_docker_creates_dockerfile():
	runner = CliRunner()
	with runner.isolated_filesystem():
		Path("app.apg").write_text(APG_SOURCE, encoding="utf-8")

		result = runner.invoke(cli, ["export", "app.apg", "--format", "docker", "-o", "out"])

		assert result.exit_code == 0, result.output
		assert Path("out/Dockerfile").exists()
