"""AI agent runtime integration adapter tests."""

import asyncio

from agents import (
	AgentBackendSpec,
	AgentInvocation,
	BaseAgent,
	DEFAULT_AGENT_INTEGRATIONS,
)
from agents.integrations import CLIAgentBackend
from agents.base_agent import AgentTask


def test_default_agent_integration_registry_has_fast_changing_runtimes():
	names = DEFAULT_AGENT_INTEGRATIONS.names()

	assert "codex" in names
	assert "claude_code" in names
	assert "opencode" in names
	assert "pi" in names


def test_cli_backend_builds_workspace_command_without_running_external_tool():
	backend = CLIAgentBackend(AgentBackendSpec(
		name="example",
		kind="cli",
		command="agentctl",
		args=["run", "--model", "{model}", "{prompt}", "{files}"],
		model="provider:model",
	))

	command = backend.build_command(AgentInvocation(
		prompt="fix the parser",
		files=["compiler/parser.py", "tests/test_parser.py"],
	))

	assert command == [
		"agentctl",
		"run",
		"--model",
		"provider:model",
		"fix the parser",
		"compiler/parser.py",
		"tests/test_parser.py",
	]


def test_base_agent_executes_through_configured_local_backend():
	async def run_task():
		agent = BaseAgent("agent_1", config={"runtime": "local"})
		task = AgentTask(
			name="Local backend smoke",
			description="Summarize execution context",
			requirements={"type": "smoke", "files": ["README.md"]},
		)
		await agent.receive_task(task)
		await asyncio.sleep(0.05)
		return task

	task = asyncio.run(run_task())

	assert task.status == "completed"
	assert task.result["backend"] == "local"
	assert task.result["success"] is True
	assert "README.md" in task.result["output"]
