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


class Planner(AgentBase):
    name = 'Planner'
    role = 'support planner'
    model = 'openai:gpt-4.1-mini'
    runtime = 'codex'
    system = "Break the customer's support request into a concrete resolution plan."
    capabilities = ()
    tools = ('tickets.read', 'docs.search', 'product.lookup')


class Writer(AgentBase):
    name = 'Writer'
    role = 'support writer'
    model = 'openai:gpt-4.1-mini'
    runtime = 'codex'
    system = 'Write concise, empathetic customer-facing replies based on the resolution plan.'
    capabilities = ()
    tools = ('tickets.update', 'templates.fetch')


class Reviewer(AgentBase):
    name = 'Reviewer'
    role = 'quality reviewer'
    model = 'openai:gpt-4.1-mini'
    runtime = 'codex'
    system = 'Review the draft reply for accuracy, tone, and completeness. Flag any issues.'
    capabilities = ()
    tools = ('knowledge.verify', 'compliance.check')


# Registry of all declared agents
AGENTS = {
    'Planner': Planner,
    'Writer': Writer,
    'Reviewer': Reviewer,
}
