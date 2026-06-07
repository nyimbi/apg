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


class ERPAnalyst(AgentBase):
    name = 'ERPAnalyst'
    role = 'ERP business analyst'
    model = 'openai:gpt-4.1'
    runtime = 'codex'
    system = 'Analyse ERP data across Finance, Procurement, HR, and CRM. Produce concise insights with KPI context'
    capabilities = ('erp_entities', 'executive_dashboard', 'board_reporting')
    tools = ('gl.query', 'procurement.analysis', 'inventory.forecast', 'hr.analytics')


class PrivacyAgent(AgentBase):
    name = 'PrivacyAgent'
    role = 'privacy-safe HR advisor'
    model = 'ollama:llama3.3'
    runtime = 'codex'
    system = 'Handle HR queries locally. No data transmitted externally. GDPR-compliant responses.'
    capabilities = ('erp_entities',)
    tools = ('hr.query_local', 'payroll.summarise_local')


# Registry of all declared agents
AGENTS = {
    'ERPAnalyst': ERPAnalyst,
    'PrivacyAgent': PrivacyAgent,
}
