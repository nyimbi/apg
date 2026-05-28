"""
APG AI agent adapter shims.

These entry points speak the generated APG agent-runtime protocol. They do not
assume raw vendor CLIs accept APG envelopes; a provider command must be
configured explicitly for execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Mapping, TextIO


SUPPORTED_RUNTIMES = ("codex", "claude_code", "opencode", "openai", "ollama", "pi")


def _env_fragment(value: str) -> str:
	return "".join(character if character.isalnum() else "_" for character in value.upper()).strip("_")


def provider_environment_keys(runtime: str) -> list[str]:
	fragment = _env_fragment(runtime)
	return [
		f"APG_AGENT_{fragment}_PROVIDER_COMMAND",
		f"APG_AGENT_{fragment}_CLI",
		"APG_AGENT_PROVIDER_COMMAND",
	]


def _coerce_command(value: Any) -> list[str] | None:
	if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
		return list(value)
	if isinstance(value, str) and value.strip():
		return shlex.split(value)
	return None


def _provider_command(runtime: str, environ: Mapping[str, str]) -> tuple[list[str] | None, str | None]:
	for key in provider_environment_keys(runtime):
		command = _coerce_command(environ.get(key))
		if command:
			return command, key
	return None, None


def _agent_name(envelope: Mapping[str, Any]) -> str | None:
	agent = envelope.get("agent")
	if isinstance(agent, Mapping):
		name = agent.get("name")
		if isinstance(name, str):
			return name
	return None


def _read_envelope(stdin: TextIO) -> dict[str, Any]:
	raw = stdin.read()
	if not raw.strip():
		return {}
	parsed = json.loads(raw)
	if not isinstance(parsed, dict):
		raise ValueError("APG agent adapter envelope must be a JSON object.")
	return parsed


def _provider_input(envelope: Mapping[str, Any]) -> str:
	return json.dumps(envelope, sort_keys=True)


def _json_result(stdout: TextIO, payload: Mapping[str, Any]) -> None:
	stdout.write(json.dumps(payload, sort_keys=True))
	stdout.write("\n")


def run_adapter(
	runtime: str,
	stdin: TextIO | None = None,
	stdout: TextIO | None = None,
	environ: Mapping[str, str] | None = None,
) -> int:
	"""Run one APG agent adapter shim and emit a normalized JSON result."""
	stdin = stdin or sys.stdin
	stdout = stdout or sys.stdout
	environ = environ or os.environ
	runtime = runtime.lower()
	try:
		envelope = _read_envelope(stdin)
	except (json.JSONDecodeError, ValueError) as error:
		_json_result(stdout, {
			"status": "failed",
			"mode": "adapter_shim",
			"runtime": runtime,
			"message": f"Invalid APG agent adapter envelope: {error}",
		})
		return 2

	command, command_source = _provider_command(runtime, environ)
	agent_name = _agent_name(envelope)
	if not command:
		_json_result(stdout, {
			"status": "adapter_required",
			"mode": "adapter_shim",
			"runtime": runtime,
			"agent": agent_name,
			"message": (
				f"{runtime} APG adapter shim is installed, but no provider command is configured."
			),
			"provider_environment_keys": provider_environment_keys(runtime),
			"input": envelope.get("input"),
		})
		return 0

	try:
		timeout = float(environ.get("APG_AGENT_PROVIDER_TIMEOUT", "120"))
	except ValueError:
		timeout = 120.0

	try:
		completed = subprocess.run(
			command,
			input=_provider_input(envelope),
			capture_output=True,
			text=True,
			check=False,
			timeout=timeout,
			cwd=environ.get("APG_AGENT_WORKDIR") or None,
		)
	except FileNotFoundError as error:
		_json_result(stdout, {
			"status": "failed",
			"mode": "adapter_shim",
			"runtime": runtime,
			"agent": agent_name,
			"message": str(error),
			"provider_command": command,
			"provider_source": command_source,
			"error": "provider_command_not_found",
		})
		return 1
	except subprocess.TimeoutExpired as error:
		_json_result(stdout, {
			"status": "failed",
			"mode": "adapter_shim",
			"runtime": runtime,
			"agent": agent_name,
			"message": f"Provider command timed out after {error.timeout} seconds.",
			"provider_command": command,
			"provider_source": command_source,
			"error": "provider_timeout",
		})
		return 1

	provider_stdout = completed.stdout.strip()
	provider_stderr = completed.stderr.strip()
	parsed_output: Any = None
	if provider_stdout:
		try:
			parsed_output = json.loads(provider_stdout)
		except json.JSONDecodeError:
			parsed_output = provider_stdout
	_json_result(stdout, {
		"status": "completed" if completed.returncode == 0 else "failed",
		"mode": "adapter_shim",
		"runtime": runtime,
		"agent": agent_name,
		"message": (
			"Provider command completed." if completed.returncode == 0 else "Provider command failed."
		),
		"provider_command": command,
		"provider_source": command_source,
		"returncode": completed.returncode,
		"stdout": provider_stdout,
		"stderr": provider_stderr,
		"parsed": parsed_output,
	})
	return 0 if completed.returncode == 0 else 1


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Run an APG AI agent adapter shim.")
	parser.add_argument("runtime", choices=SUPPORTED_RUNTIMES)
	args = parser.parse_args(argv)
	return run_adapter(args.runtime)


def codex() -> int:
	return run_adapter("codex")


def claude_code() -> int:
	return run_adapter("claude_code")


def opencode() -> int:
	return run_adapter("opencode")


def openai() -> int:
	return run_adapter("openai")


def ollama() -> int:
	return run_adapter("ollama")


def pi() -> int:
	return run_adapter("pi")


if __name__ == "__main__":
	raise SystemExit(main())
