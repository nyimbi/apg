"""Provider-neutral AI agent integration adapters.

APG treats AI agents as first-class citizens, but the concrete runner is
expected to change quickly.  This module keeps volatile Codex, Claude Code,
OpenCode, Pi, and future runtime wiring behind adapters.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentInvocation:
	"""A normalized request sent to an AI agent backend."""

	prompt: str
	cwd: str | None = None
	context: dict[str, Any] = field(default_factory=dict)
	files: list[str] = field(default_factory=list)
	timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AgentRunResult:
	"""A normalized response from an AI agent backend."""

	success: bool
	backend: str
	output: str = ""
	error: str = ""
	command: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBackendSpec:
	"""Configuration for an agent runtime adapter."""

	name: str
	kind: str
	model: str | None = None
	command: str | None = None
	args: list[str] = field(default_factory=list)
	env_token: str | None = None
	endpoint: str | None = None
	supports_workspace: bool = False
	metadata: dict[str, Any] = field(default_factory=dict)


class AgentBackend(Protocol):
	"""Runtime adapter protocol."""

	spec: AgentBackendSpec

	def available(self) -> bool:
		"""Return whether the backend can run in the current environment."""

	async def run(self, invocation: AgentInvocation) -> AgentRunResult:
		"""Execute one invocation."""


class LocalEchoBackend:
	"""Deterministic fallback backend for tests and offline execution."""

	def __init__(self, spec: AgentBackendSpec | None = None):
		self.spec = spec or AgentBackendSpec(name="local", kind="local")

	def available(self) -> bool:
		return True

	async def run(self, invocation: AgentInvocation) -> AgentRunResult:
		payload = {
			"prompt": invocation.prompt,
			"context": invocation.context,
			"files": invocation.files,
		}
		return AgentRunResult(
			success=True,
			backend=self.spec.name,
			output=json.dumps(payload, sort_keys=True),
			metadata={"mode": "local"},
		)


class CLIAgentBackend:
	"""Adapter for local terminal coding agents such as Codex and OpenCode."""

	def __init__(self, spec: AgentBackendSpec):
		if not spec.command:
			raise ValueError(f"CLI backend '{spec.name}' requires command")
		self.spec = spec

	def available(self) -> bool:
		return shutil.which(self.spec.command or "") is not None

	def build_command(self, invocation: AgentInvocation) -> list[str]:
		values = {
			"prompt": invocation.prompt,
			"model": self.spec.model or "",
			"cwd": invocation.cwd or "",
		}
		command = [self.spec.command or ""]
		for arg in self.spec.args:
			if arg == "{files}":
				command.extend(invocation.files)
			else:
				command.append(arg.format(**values))
		return command

	async def run(self, invocation: AgentInvocation) -> AgentRunResult:
		command = self.build_command(invocation)
		if not self.available():
			return AgentRunResult(
				success=False,
				backend=self.spec.name,
				error=f"Command not found: {self.spec.command}",
				command=command,
			)

		cwd = invocation.cwd or os.getcwd()
		process = await asyncio.create_subprocess_exec(
			*command,
			cwd=cwd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		try:
			stdout, stderr = await asyncio.wait_for(
				process.communicate(),
				timeout=invocation.timeout_seconds,
			)
		except asyncio.TimeoutError:
			process.kill()
			await process.wait()
			return AgentRunResult(
				success=False,
				backend=self.spec.name,
				error=f"Timed out after {invocation.timeout_seconds}s",
				command=command,
			)

		return AgentRunResult(
			success=process.returncode == 0,
			backend=self.spec.name,
			output=stdout.decode(errors="replace"),
			error=stderr.decode(errors="replace"),
			command=command,
			metadata={"returncode": process.returncode, "cwd": str(Path(cwd))},
		)


class HTTPAgentBackend:
	"""Small stdlib HTTP adapter for API-hosted chat agents."""

	def __init__(self, spec: AgentBackendSpec):
		if not spec.endpoint:
			raise ValueError(f"HTTP backend '{spec.name}' requires endpoint")
		self.spec = spec

	def available(self) -> bool:
		return not self.spec.env_token or bool(os.environ.get(self.spec.env_token))

	async def run(self, invocation: AgentInvocation) -> AgentRunResult:
		return await asyncio.to_thread(self._run_sync, invocation)

	def _run_sync(self, invocation: AgentInvocation) -> AgentRunResult:
		if not self.available():
			return AgentRunResult(
				success=False,
				backend=self.spec.name,
				error=f"Missing API token env var: {self.spec.env_token}",
			)

		body = json.dumps({
			"model": self.spec.model,
			"messages": [
				{"role": "system", "content": invocation.context.get("system", "")},
				{"role": "user", "content": invocation.prompt},
			],
			"metadata": invocation.context,
		}).encode()
		headers = {"Content-Type": "application/json"}
		if self.spec.env_token:
			headers["Authorization"] = f"Bearer {os.environ[self.spec.env_token]}"

		request = urllib.request.Request(
			self.spec.endpoint or "",
			data=body,
			headers=headers,
			method="POST",
		)
		try:
			with urllib.request.urlopen(request, timeout=invocation.timeout_seconds) as response:
				output = response.read().decode(errors="replace")
			return AgentRunResult(success=True, backend=self.spec.name, output=output)
		except urllib.error.URLError as exc:
			return AgentRunResult(success=False, backend=self.spec.name, error=str(exc))


class AgentIntegrationRegistry:
	"""Registry of named AI agent runtime adapters."""

	def __init__(self):
		self._specs: dict[str, AgentBackendSpec] = {}

	def register(self, spec: AgentBackendSpec) -> None:
		self._specs[spec.name] = spec

	def names(self) -> list[str]:
		return sorted(self._specs)

	def spec(self, name: str) -> AgentBackendSpec:
		return self._specs[name]

	def create(self, name: str) -> AgentBackend:
		spec = self.spec(name)
		if spec.kind == "cli":
			return CLIAgentBackend(spec)
		if spec.kind == "http":
			return HTTPAgentBackend(spec)
		if spec.kind == "local":
			return LocalEchoBackend(spec)
		raise ValueError(f"Unsupported agent backend kind: {spec.kind}")

	async def run(self, backend_name: str, invocation: AgentInvocation) -> AgentRunResult:
		return await self.create(backend_name).run(invocation)


def default_agent_integration_registry() -> AgentIntegrationRegistry:
	"""Return APG's built-in volatile AI runtime adapters."""

	registry = AgentIntegrationRegistry()
	registry.register(AgentBackendSpec(name="local", kind="local"))
	registry.register(AgentBackendSpec(
		name="codex",
		kind="cli",
		command="codex",
		args=["exec", "{prompt}"],
		supports_workspace=True,
		metadata={"family": "coding_agent"},
	))
	registry.register(AgentBackendSpec(
		name="claude_code",
		kind="cli",
		command="claude",
		args=["-p", "{prompt}"],
		supports_workspace=True,
		metadata={"family": "coding_agent"},
	))
	registry.register(AgentBackendSpec(
		name="opencode",
		kind="cli",
		command="opencode",
		args=["run", "{prompt}"],
		supports_workspace=True,
		metadata={"family": "coding_agent"},
	))
	registry.register(AgentBackendSpec(
		name="pi",
		kind="http",
		model="inflection_3_pi",
		endpoint="https://api.inflection.ai/v1/chat/completions",
		env_token="INFLECTION_API_KEY",
		metadata={"family": "chat_agent"},
	))
	return registry


DEFAULT_AGENT_INTEGRATIONS = default_agent_integration_registry()


__all__ = [
	"AgentBackend",
	"AgentBackendSpec",
	"AgentIntegrationRegistry",
	"AgentInvocation",
	"AgentRunResult",
	"CLIAgentBackend",
	"DEFAULT_AGENT_INTEGRATIONS",
	"HTTPAgentBackend",
	"LocalEchoBackend",
	"default_agent_integration_registry",
]
