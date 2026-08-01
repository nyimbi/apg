"""
AI Agent Composition Runtime
============================

Generated from first-class APG AI agent declarations.
"""

from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess

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


AI_AGENT_DATA: Dict[str, Dict[str, Any]] = {'MineOptimisationAgent': {'role': 'production optimisation and scheduling', 'model': 'ollama:llama3.2', 'runtime': None, 'system': 'You are a mining production optimisation agent. Analyse ore grade distributions, equipment availability, and scheduling constraints to maximise metal recovery while minimising unit costs. Recommend blast designs, dig sequences, and equipment deployment plans. Always prioritise safety permits as hard constraints.', 'capabilities': ['mining_prod', 'mining_equip', 'mining_safety'], 'tools': ['resource_block_model_api', 'equipment_dispatch_api', 'grade_control_api', 'safety_permit_api'], 'memory': {'kind': 'vector', 'name': 'mine_operational_memory'}, 'inputs': [], 'outputs': [], 'handoffs': [], 'configuration': {'temperature': 0.2, 'max_turns': 8}, 'rules': [{'name': 'no_dispatch_without_valid_permit', 'when': 'work_type == blasting and permit_status != approved', 'action': 'deny'}, {'name': 'safety_constraint_overrides_output', 'when': 'safety_alert_active == true', 'action': 'halt_production'}], 'ui': {}, 'theme': {}}, 'EnergyDispatchAgent': {'role': 'real-time energy dispatch and grid balancing', 'model': 'ollama:llama3.2', 'runtime': None, 'system': 'You are an energy system operator assistant. Monitor grid frequency, load balancing, and renewable output in real time. Recommend dispatch adjustments, load shed actions, and battery dispatch commands to maintain grid stability within N-1 security standards. Escalate all actions above 10 MW to human operators.', 'capabilities': ['energy_gen', 'energy_dist', 'energy_ren'], 'tools': ['scada_api', 'dispatch_api', 'frequency_monitor', 'load_forecast_api', 'battery_api'], 'memory': {'kind': 'working', 'name': 'grid_state_memory'}, 'inputs': [], 'outputs': [], 'handoffs': [], 'configuration': {'temperature': 0.1, 'max_turns': 5}, 'rules': [{'name': 'large_action_requires_human_approval', 'when': 'dispatch_change_mw_gt == 10 and human_approved != true', 'action': 'require_review'}, {'name': 'emergency_load_shed_auto_allowed', 'when': 'frequency_hz_lt == 49.5', 'action': 'allow'}], 'ui': {}, 'theme': {}}, 'SafetyComplianceAgent': {'role': 'safety and environmental compliance monitoring', 'model': 'ollama:llama3.2', 'runtime': None, 'system': 'You are a safety and environmental compliance monitor. Continuously watch for permit expirations, safety incident patterns, environmental threshold breaches, and regulatory reporting deadlines. Raise alerts proactively, draft regulatory notifications, and recommend corrective actions based on root cause analysis of safety data.', 'capabilities': ['mining_safety', 'mining_prod'], 'tools': ['incident_database_api', 'permit_register_api', 'environmental_monitor_api', 'regulatory_calendar_api'], 'memory': {'kind': 'episodic', 'name': 'safety_incident_memory'}, 'inputs': [], 'outputs': [], 'handoffs': [], 'configuration': {'temperature': 0.2, 'max_turns': 6}, 'rules': [{'name': 'fatality_requires_immediate_escalation', 'when': 'incident_severity == fatal', 'action': 'notify_immediately'}, {'name': 'permit_expiry_24h_warning', 'when': 'permit_expires_hours_lt == 24', 'action': 'alert'}], 'ui': {}, 'theme': {}}, 'SmartMeterAnalyticsAgent': {'role': 'metering analytics and non-technical loss detection', 'model': 'ollama:llama3.2', 'runtime': None, 'system': 'You are a smart metering analytics agent. Detect anomalies in interval data that indicate tampering, meter faults, or non-technical losses. Identify demand response opportunities and forecast load profiles for grid planning. Flag unusual consumption patterns for field investigation.', 'capabilities': ['energy_metr', 'energy_dist'], 'tools': ['mdm_api', 'billing_api', 'field_workforce_api', 'demand_forecast_api'], 'memory': {'kind': 'vector', 'name': 'meter_data_memory'}, 'inputs': [], 'outputs': [], 'handoffs': [], 'configuration': {'temperature': 0.2, 'max_turns': 6}, 'rules': [{'name': 'tamper_auto_alert', 'when': 'tamper_confidence_pct_gt == 80', 'action': 'notify_immediately'}, {'name': 'no_disconnection_without_notice', 'when': 'operation == remote_disconnect and notice_sent != true', 'action': 'deny'}], 'ui': {}, 'theme': {}}}
AI_TEAM_DATA: Dict[str, Dict[str, Any]] = {}
AI_AGENT_RUNTIME_DATA: Dict[str, Dict[str, Any]] = {'local': {'kind': 'local', 'aliases': ['offline', 'test'], 'supports_workspace': False, 'requires_token': False, 'family': 'deterministic'}, 'codex': {'kind': 'cli', 'aliases': ['codex_cli', 'openai_codex'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent', 'command_candidates': [['apg-agent-codex']]}, 'claude_code': {'kind': 'cli', 'aliases': ['claude', 'claude-code'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent', 'command_candidates': [['apg-agent-claude-code'], ['apg-agent-claude']]}, 'opencode': {'kind': 'cli', 'aliases': ['open_code'], 'supports_workspace': True, 'requires_token': False, 'family': 'coding_agent', 'command_candidates': [['apg-agent-opencode']]}, 'openai': {'kind': 'http', 'aliases': ['openai_chat'], 'supports_workspace': False, 'requires_token': True, 'family': 'chat_agent', 'command_candidates': [['apg-agent-openai']]}, 'ollama': {'kind': 'http', 'aliases': ['local_llm'], 'supports_workspace': False, 'requires_token': False, 'family': 'local_model', 'command_candidates': [['apg-agent-ollama']]}, 'pi': {'kind': 'http', 'aliases': ['inflection_pi'], 'supports_workspace': False, 'requires_token': True, 'family': 'chat_agent', 'command_candidates': [['apg-agent-pi']]}}
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


def _env_fragment(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.upper()).strip("_")


def runtime_adapter_environment_keys(runtime: str, agent_name: Optional[str] = None) -> List[str]:
    keys: List[str] = []
    if agent_name:
        keys.append(f"APG_AGENT_{_env_fragment(agent_name)}_COMMAND")
    keys.extend([
        f"APG_AGENT_RUNTIME_{_env_fragment(runtime)}_COMMAND",
        f"APG_AGENT_{_env_fragment(runtime)}_COMMAND",
        "APG_AGENT_RUNTIME_COMMAND",
    ])
    return keys


def _coerce_command(value: Any) -> Optional[List[str]]:
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return list(value)
    if isinstance(value, str) and value.strip():
        return shlex.split(value)
    return None


def runtime_adapter_command_candidates(runtime: str) -> List[List[str]]:
    runtime_spec = AI_AGENT_RUNTIME_DATA.get(canonical_runtime(runtime), {})
    candidates = runtime_spec.get("command_candidates", [])
    commands: List[List[str]] = []
    for candidate in candidates:
        command = _coerce_command(candidate)
        if command:
            commands.append(command)
    return commands


def _adapter_command(agent: AIAgentSpec, runtime: str) -> tuple[Optional[List[str]], Optional[str]]:
    configured = (
        agent.configuration.get("adapter_command")
        or agent.configuration.get("runtime_command")
        or agent.configuration.get("agent_command")
    )
    command = _coerce_command(configured)
    if command:
        return command, "agent.configuration"
    for key in runtime_adapter_environment_keys(runtime, agent.name):
        command = _coerce_command(os.environ.get(key))
        if command:
            return command, key
    for candidate in runtime_adapter_command_candidates(runtime):
        resolved = shutil.which(candidate[0])
        if resolved:
            return [resolved, *candidate[1:]], f"runtime.{runtime}.command_candidates"
    return None, None


def _adapter_timeout(agent: AIAgentSpec) -> float:
    configured = agent.configuration.get("adapter_timeout", agent.configuration.get("timeout"))
    raw_value = os.environ.get("APG_AGENT_RUNTIME_TIMEOUT", configured)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 120.0


def _agent_invocation_base(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "agent": agent.name,
        "role": agent.role,
        "model": agent.model,
        "runtime": runtime,
        "runtime_spec": dict(runtime_spec),
        "input": _invocation_input(payload),
        "system": agent.system,
        "capabilities": list(agent.capabilities),
        "tools": list(agent.tools),
        "configuration": dict(agent.configuration),
        "handoffs": [dict(edge) for edge in agent.handoffs],
    }


def _external_invocation_envelope(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "agent": describe_agent(agent.name),
        "runtime": runtime,
        "runtime_spec": dict(runtime_spec),
        "input": _invocation_input(payload),
        "payload": dict(payload) if isinstance(payload, dict) else {},
    }


def _run_external_agent(agent: AIAgentSpec, runtime: str, runtime_spec: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    command, command_source = _adapter_command(agent, runtime)
    if not command:
        return None
    envelope = _external_invocation_envelope(agent, runtime, runtime_spec, payload)
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(envelope, sort_keys=True),
            capture_output=True,
            text=True,
            check=False,
            timeout=_adapter_timeout(agent),
            cwd=os.environ.get("APG_AGENT_WORKDIR") or None,
        )
    except FileNotFoundError as error:
        return {
            "status": "failed",
            "mode": "external",
            "output": {
                "message": str(error),
                "requires_adapter": False,
                "adapter_command": command,
                "adapter_source": command_source,
                "error": "adapter_command_not_found",
            },
        }
    except subprocess.TimeoutExpired as error:
        return {
            "status": "failed",
            "mode": "external",
            "output": {
                "message": f"External runtime adapter timed out after {error.timeout} seconds.",
                "requires_adapter": False,
                "adapter_command": command,
                "adapter_source": command_source,
                "error": "adapter_timeout",
            },
        }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed_output: Any = None
    if stdout:
        try:
            parsed_output = json.loads(stdout)
        except json.JSONDecodeError:
            parsed_output = stdout
    adapter_status = "completed" if completed.returncode == 0 else "failed"
    adapter_mode = "external"
    adapter_message = "External runtime adapter completed." if completed.returncode == 0 else "External runtime adapter failed."
    if isinstance(parsed_output, dict):
        parsed_status = parsed_output.get("status")
        if parsed_status in {"completed", "failed", "adapter_required"}:
            adapter_status = parsed_status
        parsed_mode = parsed_output.get("mode")
        if isinstance(parsed_mode, str) and parsed_mode:
            adapter_mode = parsed_mode
        parsed_message = parsed_output.get("message")
        if isinstance(parsed_message, str) and parsed_message:
            adapter_message = parsed_message
    adapter_requires = adapter_status == "adapter_required"
    return {
        "status": adapter_status,
        "mode": adapter_mode,
        "output": {
            "message": adapter_message,
            "requires_adapter": adapter_requires,
            "adapter_command": command,
            "adapter_source": command_source,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed_output,
        },
    }


def invoke_agent(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    agent = get_agent(name)
    runtime = canonical_runtime(agent.runtime)
    runtime_spec = dict(AI_AGENT_RUNTIME_DATA[runtime])
    requires_adapter = runtime != "local"
    base = _agent_invocation_base(agent, runtime, runtime_spec, payload)
    if requires_adapter:
        external = _run_external_agent(agent, runtime, runtime_spec, payload)
        if external is not None:
            base.update(external)
            return base
    base.update({
        "status": "adapter_required" if requires_adapter else "completed",
        "mode": "adapter_missing" if requires_adapter else "local",
        "output": {
            "message": (
                f"{agent.name} requires a configured {runtime} adapter command before invocation."
                if requires_adapter
                else f"{agent.name} handled the request locally."
            ),
            "requires_adapter": requires_adapter,
            "adapter_environment_keys": runtime_adapter_environment_keys(runtime, agent.name) if requires_adapter else [],
            "adapter_command_candidates": runtime_adapter_command_candidates(runtime) if requires_adapter else [],
        },
    })
    return base


def invoke_team(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    team = get_team(name)
    invocations = [
        invoke_agent(agent_name, payload)
        for agent_name in team.agents
    ]
    if any(item["status"] == "failed" for item in invocations):
        status = "failed"
    elif any(item["status"] == "adapter_required" for item in invocations):
        status = "adapter_required"
    else:
        status = "completed"
    return {
        "team": team.name,
        "status": status,
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
