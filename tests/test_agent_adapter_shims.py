"""
APG AI agent adapter shim tests.
"""

from __future__ import annotations

import io
import json
import shlex
import sys

from cli.agent_adapter import provider_environment_keys, run_adapter


ENVELOPE = {
	"agent": {"name": "Planner"},
	"runtime": "codex",
	"input": {"ticket": "late order"},
	"payload": {"input": {"ticket": "late order"}},
}


def _run(runtime: str, envelope: dict[str, object], environ: dict[str, str]) -> tuple[int, dict[str, object]]:
	stdout = io.StringIO()
	exit_code = run_adapter(
		runtime,
		stdin=io.StringIO(json.dumps(envelope)),
		stdout=stdout,
		environ=environ,
	)
	return exit_code, json.loads(stdout.getvalue())


def test_agent_adapter_shim_reports_missing_provider_command():
	exit_code, payload = _run("codex", ENVELOPE, {})

	assert exit_code == 0
	assert payload["status"] == "adapter_required"
	assert payload["mode"] == "adapter_shim"
	assert payload["runtime"] == "codex"
	assert payload["agent"] == "Planner"
	assert payload["provider_environment_keys"] == provider_environment_keys("codex")
	assert "APG_AGENT_CODEX_PROVIDER_COMMAND" in payload["provider_environment_keys"]
	assert payload["input"] == {"ticket": "late order"}


def test_agent_adapter_shim_executes_configured_provider_command():
	code = (
		"import json, sys; "
		"envelope=json.load(sys.stdin); "
		"print(json.dumps({"
		"'agent': envelope['agent']['name'], "
		"'runtime': envelope['runtime'], "
		"'input': envelope['input']"
		"}))"
	)
	command = shlex.join([sys.executable, "-c", code])

	exit_code, payload = _run("codex", ENVELOPE, {"APG_AGENT_CODEX_PROVIDER_COMMAND": command})

	assert exit_code == 0
	assert payload["status"] == "completed"
	assert payload["mode"] == "adapter_shim"
	assert payload["provider_source"] == "APG_AGENT_CODEX_PROVIDER_COMMAND"
	assert payload["returncode"] == 0
	assert payload["parsed"] == {
		"agent": "Planner",
		"runtime": "codex",
		"input": {"ticket": "late order"},
	}


def test_agent_adapter_shim_rejects_invalid_json_envelope():
	stdout = io.StringIO()
	exit_code = run_adapter("codex", stdin=io.StringIO("{"), stdout=stdout, environ={})
	payload = json.loads(stdout.getvalue())

	assert exit_code == 2
	assert payload["status"] == "failed"
	assert payload["mode"] == "adapter_shim"
	assert "Invalid APG agent adapter envelope" in payload["message"]


def test_setup_exposes_default_agent_adapter_console_scripts():
	setup_py = open("setup.py", encoding="utf-8").read()

	for command in [
		"apg-agent-codex=cli.agent_adapter:codex",
		"apg-agent-claude-code=cli.agent_adapter:claude_code",
		"apg-agent-claude=cli.agent_adapter:claude_code",
		"apg-agent-opencode=cli.agent_adapter:opencode",
		"apg-agent-openai=cli.agent_adapter:openai",
		"apg-agent-ollama=cli.agent_adapter:ollama",
		"apg-agent-pi=cli.agent_adapter:pi",
	]:
		assert command in setup_py
