#!/usr/bin/env python3
"""Setuptools compatibility shim for editable installs."""

from setuptools import setup


setup(
	entry_points={
		"console_scripts": [
			"apg=cli.main:cli",
			"apg-agent-codex=cli.agent_adapter:codex",
			"apg-agent-claude-code=cli.agent_adapter:claude_code",
			"apg-agent-claude=cli.agent_adapter:claude_code",
			"apg-agent-opencode=cli.agent_adapter:opencode",
			"apg-agent-openai=cli.agent_adapter:openai",
			"apg-agent-ollama=cli.agent_adapter:ollama",
			"apg-agent-pi=cli.agent_adapter:pi",
		],
	},
)
