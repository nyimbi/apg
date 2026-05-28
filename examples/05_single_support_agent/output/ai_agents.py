"""
AI Agent Composition Runtime
============================

Generated from first-class APG AI agent declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AIAgentSpec:
    name: str
    role: Optional[str]
    model: Optional[str]
    runtime: Optional[str]
    system: Optional[str]
    capabilities: List[str]
    tools: List[str]
    memory: Optional[Dict[str, Optional[str]]]
    inputs: List[str]
    outputs: List[str]
    handoffs: List[Dict[str, str]]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    ui: Dict[str, Any]
    theme: Dict[str, Any]


@dataclass(frozen=True)
class AgentTeamSpec:
    name: str
    agents: List[str]
    capabilities: List[str]
    flow: List[Dict[str, str]]
    policy: Dict[str, Any]
    configuration: Dict[str, Any]
    rules: List[Dict[str, Any]]
    ui: Dict[str, Any]
    theme: Dict[str, Any]


AI_AGENT_DATA: Dict[str, Dict[str, Any]] = {'TriageAgent': {'role': 'support_triage', 'model': 'openai:gpt-4.1-mini', 'runtime': 'codex', 'system': 'Classify incoming tickets and produce a concise next action.', 'capabilities': [], 'tools': ['tickets.read', 'knowledge.search'], 'memory': {'kind': 'vector', 'name': 'support_memory'}, 'inputs': ['ticket'], 'outputs': ['triage_plan'], 'handoffs': [], 'configuration': {}, 'rules': [{'name': 'ticket_required', 'when': 'ticket missing', 'action': 'reject'}], 'ui': {}, 'theme': {}}}
AI_TEAM_DATA: Dict[str, Dict[str, Any]] = {}
AI_AGENT_RUNTIME_DATA: Dict[str, Dict[str, Any]] = {'local': {'kind': 'local', 'aliases': ['offline', 'test'], 'supports_workspace': False, 'requires_token': False, 'family': 'deterministic'}, 'codex': {'kind': 'cli', 'aliases': ['codex_cli', 'openai_codex'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent'}, 'claude_code': {'kind': 'cli', 'aliases': ['claude', 'claude-code'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent'}, 'opencode': {'kind': 'cli', 'aliases': ['open_code'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent'}, 'openai': {'kind': 'http', 'aliases': ['openai_chat'], 'supports_workspace': False, 'requires_token': True, 'family': 'chat_agent'}, 'ollama': {'kind': 'http', 'aliases': ['local_llm'], 'supports_workspace': False, 'requires_token': False, 'family': 'local_model'}, 'pi': {'kind': 'http', 'aliases': ['inflection_pi'], 'supports_workspace': False, 'requires_token': True, 'family': 'chat_agent'}}
AI_AGENT_RUNTIME_ALIASES: Dict[str, str] = {'offline': 'local', 'test': 'local', 'codex_cli': 'codex', 'openai_codex': 'codex', 'claude': 'claude_code', 'claude-code': 'claude_code', 'open_code': 'opencode', 'openai_chat': 'openai', 'local_llm': 'ollama', 'inflection_pi': 'pi'}


AI_AGENTS: Dict[str, AIAgentSpec] = {
    name: AIAgentSpec(name=name, **data)
    for name, data in AI_AGENT_DATA.items()
}
AI_AGENT_TEAMS: Dict[str, AgentTeamSpec] = {
    name: AgentTeamSpec(name=name, **data)
    for name, data in AI_TEAM_DATA.items()
}


def get_agent(name: str) -> AIAgentSpec:
    return AI_AGENTS[name]


def get_team(name: str) -> AgentTeamSpec:
    return AI_AGENT_TEAMS[name]


def describe_agent(name: str) -> Dict[str, Any]:
    agent = get_agent(name)
    return {
        "name": agent.name,
        "role": agent.role,
        "model": agent.model,
        "runtime": agent.runtime,
        "system": agent.system,
        "capabilities": list(agent.capabilities),
        "tools": list(agent.tools),
        "memory": dict(agent.memory) if agent.memory else None,
        "inputs": list(agent.inputs),
        "outputs": list(agent.outputs),
        "handoffs": [dict(edge) for edge in agent.handoffs],
        "configuration": dict(agent.configuration),
        "rules": [dict(rule) for rule in agent.rules],
        "ui": dict(agent.ui),
        "theme": dict(agent.theme),
    }


def list_agents() -> List[str]:
    return sorted(AI_AGENTS)


def list_agent_teams() -> List[str]:
    return sorted(AI_AGENT_TEAMS)


def list_teams() -> List[str]:
    return list_agent_teams()


def list_agent_runtimes(include_aliases: bool = False) -> List[str]:
    names = set(AI_AGENT_RUNTIME_DATA)
    if include_aliases:
        names.update(AI_AGENT_RUNTIME_ALIASES)
    return sorted(names)


def canonical_runtime(name: Optional[str]) -> str:
    runtime = name or "local"
    if runtime in AI_AGENT_RUNTIME_DATA:
        return runtime
    if runtime in AI_AGENT_RUNTIME_ALIASES:
        return AI_AGENT_RUNTIME_ALIASES[runtime]
    raise KeyError(f"Unknown AI agent runtime: {runtime}")


def describe_agent_runtimes() -> Dict[str, Dict[str, Any]]:
    return {
        name: dict(spec)
        for name, spec in AI_AGENT_RUNTIME_DATA.items()
    }


def agents_by_runtime() -> Dict[str, List[AIAgentSpec]]:
    grouped: Dict[str, List[AIAgentSpec]] = {}
    for agent in AI_AGENTS.values():
        runtime = canonical_runtime(agent.runtime)
        grouped.setdefault(runtime, []).append(agent)
    return grouped


def validate_agent_runtimes(available_runtimes: Optional[List[str]] = None) -> Dict[str, Any]:
    allowed = set(available_runtimes or list_agent_runtimes(include_aliases=True))
    errors: List[str] = []
    validated: List[str] = []
    for agent in AI_AGENTS.values():
        runtime = agent.runtime or "local"
        try:
            canonical = canonical_runtime(runtime)
        except KeyError:
            errors.append(f"{agent.name} references unknown runtime {runtime}")
            continue
        if runtime not in allowed and canonical not in allowed:
            errors.append(f"{agent.name} references unavailable runtime {runtime}")
            continue
        validated.append(agent.name)
    return {"errors": errors, "validated_agents": sorted(validated)}


def _invocation_input(payload: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(payload, dict):
        return {}
    if "input" in payload:
        return payload["input"]
    if "message" in payload:
        return payload["message"]
    return dict(payload)


def invoke_agent(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    agent = get_agent(name)
    runtime = canonical_runtime(agent.runtime)
    runtime_spec = dict(AI_AGENT_RUNTIME_DATA[runtime])
    requires_adapter = runtime != "local"
    return {
        "agent": agent.name,
        "role": agent.role,
        "model": agent.model,
        "runtime": runtime,
        "runtime_spec": runtime_spec,
        "status": "adapter_required" if requires_adapter else "completed",
        "mode": "planned" if requires_adapter else "local",
        "input": _invocation_input(payload),
        "system": agent.system,
        "capabilities": list(agent.capabilities),
        "tools": list(agent.tools),
        "configuration": dict(agent.configuration),
        "handoffs": [dict(edge) for edge in agent.handoffs],
        "output": {
            "message": (
                f"{agent.name} is ready for a {runtime} adapter."
                if requires_adapter
                else f"{agent.name} handled the request locally."
            ),
            "requires_adapter": requires_adapter,
        },
    }


def invoke_team(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    team = get_team(name)
    invocations = [
        invoke_agent(agent_name, payload)
        for agent_name in team.agents
    ]
    return {
        "team": team.name,
        "status": "planned" if any(item["output"]["requires_adapter"] for item in invocations) else "completed",
        "policy": dict(team.policy),
        "configuration": dict(team.configuration),
        "flow": [dict(edge) for edge in team.flow],
        "invocations": invocations,
    }


def describe_team(name: str) -> Dict[str, Any]:
    team = get_team(name)
    return {
        "name": team.name,
        "agents": [describe_agent(agent) for agent in team.agents],
        "agent_names": list(team.agents),
        "capabilities": list(team.capabilities),
        "flow": [dict(edge) for edge in team.flow],
        "policy": dict(team.policy),
        "configuration": dict(team.configuration),
        "rules": [dict(rule) for rule in team.rules],
        "ui": dict(team.ui),
        "theme": dict(team.theme),
    }
