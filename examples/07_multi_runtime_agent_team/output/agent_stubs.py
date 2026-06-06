"""Typed agent stub classes generated from APG agent declarations.

Each class wraps the agent metadata and provides an async invoke()
that delegates to the declared runtime via the APG adapter protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from typing import Any, Optional


class AgentContext:
    """Runtime context for an agent invocation."""
    def __init__(self, tenant_id: str = 'default', user_id: str = 'anonymous',
                 session_id: str = '', **kwargs: Any) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = kwargs


class AgentBase:
    """Base class for APG agent stubs."""
    name: str = ''
    role: str = ''
    model: str = ''
    runtime: str = 'codex'

    async def invoke(self, prompt: str, context: Optional[AgentContext] = None) -> str:
        env_key = f'APG_AGENT_{self.runtime.upper()}_PROVIDER_COMMAND'
        cmd = os.environ.get(env_key) or os.environ.get('APG_AGENT_PROVIDER_COMMAND')
        if not cmd:
            raise RuntimeError(
                f'Agent {self.name!r}: no provider command configured. '
                f'Set {env_key} to wire up the {self.runtime} runtime.'
            )
        payload = {
            'agent': {'name': self.name, 'role': self.role, 'model': self.model},
            'input': prompt,
            'context': {
                'tenant_id': getattr(context, 'tenant_id', 'default'),
                'user_id': getattr(context, 'user_id', 'anonymous'),
            } if context else {},
        }
        result = await asyncio.to_thread(
            subprocess.run, shlex.split(cmd),
            input=json.dumps(payload), capture_output=True, text=True, timeout=120
        )
        out = result.stdout.strip()
        try:
            return json.loads(out).get('output', out)
        except Exception:
            return out


class Researcher(AgentBase):
    """Agent stub for Researcher — researcher."""
    name = 'Researcher'
    role = 'researcher'
    model = 'openai:gpt-4.1-mini'
    runtime = 'codex'
    system = 'Gather structured evidence and cite source IDs.'
    capabilities = []
    tools = ['web.search', 'docs.read']


class Coder(AgentBase):
    """Agent stub for Coder — implementation_engineer."""
    name = 'Coder'
    role = 'implementation_engineer'
    model = 'claude:sonnet'
    runtime = 'claude_code'
    system = 'Implement focused code changes from accepted plans.'
    capabilities = []
    tools = ['repo.edit', 'tests.run']


class LocalReviewer(AgentBase):
    """Agent stub for LocalReviewer — local_review."""
    name = 'LocalReviewer'
    role = 'local_review'
    model = 'ollama:llama3.1'
    runtime = 'ollama'
    system = 'Review diffs locally without sending code to external services.'
    capabilities = []
    tools = []


# Registry of all declared agents
AGENTS = {
    'Researcher': Researcher,
    'Coder': Coder,
    'LocalReviewer': LocalReviewer,
}
