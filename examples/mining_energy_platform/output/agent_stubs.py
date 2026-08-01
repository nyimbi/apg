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


class MineOptimisationAgent(AgentBase):
    name = 'MineOptimisationAgent'
    role = 'production optimisation and scheduling'
    model = 'ollama:llama3.2'
    runtime = 'codex'
    system = 'You are a mining production optimisation agent. Analyse ore grade distributions, equipment availabil'
    capabilities = ('mining_prod', 'mining_equip', 'mining_safety')
    tools = ('resource_block_model_api', 'equipment_dispatch_api', 'grade_control_api', 'safety_permit_api')


class EnergyDispatchAgent(AgentBase):
    name = 'EnergyDispatchAgent'
    role = 'real-time energy dispatch and grid balancing'
    model = 'ollama:llama3.2'
    runtime = 'codex'
    system = 'You are an energy system operator assistant. Monitor grid frequency, load balancing, and renewable o'
    capabilities = ('energy_gen', 'energy_dist', 'energy_ren')
    tools = ('scada_api', 'dispatch_api', 'frequency_monitor', 'load_forecast_api', 'battery_api')


class SafetyComplianceAgent(AgentBase):
    name = 'SafetyComplianceAgent'
    role = 'safety and environmental compliance monitoring'
    model = 'ollama:llama3.2'
    runtime = 'codex'
    system = 'You are a safety and environmental compliance monitor. Continuously watch for permit expirations, sa'
    capabilities = ('mining_safety', 'mining_prod')
    tools = ('incident_database_api', 'permit_register_api', 'environmental_monitor_api', 'regulatory_calendar_api')


class SmartMeterAnalyticsAgent(AgentBase):
    name = 'SmartMeterAnalyticsAgent'
    role = 'metering analytics and non-technical loss detection'
    model = 'ollama:llama3.2'
    runtime = 'codex'
    system = 'You are a smart metering analytics agent. Detect anomalies in interval data that indicate tampering,'
    capabilities = ('energy_metr', 'energy_dist')
    tools = ('mdm_api', 'billing_api', 'field_workforce_api', 'demand_forecast_api')


# Registry of all declared agents
AGENTS = {
    'MineOptimisationAgent': MineOptimisationAgent,
    'EnergyDispatchAgent': EnergyDispatchAgent,
    'SafetyComplianceAgent': SafetyComplianceAgent,
    'SmartMeterAnalyticsAgent': SmartMeterAnalyticsAgent,
}
