# AI Agent Composition

AI agents are first-class APG entities. A single `agent` declares an LLM-backed worker. A `swarm`, `team`, or `agent_team` composes agents into an executable handoff graph.

The syntax is intentionally terse but readable: short field names, explicit model selection, and arrows for handoffs.

## Minimal Example

```apg
module support version 1.0.0 {
    description: "Support response crew";
}

agent Planner {
    role: "planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Break the ticket into concrete work.";
    tools: [tickets.read, docs.search];
    memory: vector support_memory;
    input: ticket;
    output: plan;
}

agent Writer {
    role: "writer";
    model: "openai:gpt-4.1-mini";
    system: "Write concise customer-facing replies.";
    tools: [tickets.update];
    input: plan;
    output: reply;
}

swarm SupportCrew {
    agents: [Planner, Writer];
    flow: Planner -> Writer;
}
```

## Entity Model

`agent` declares a runnable AI agent:

| Field | Required | Meaning |
| --- | --- | --- |
| `model` | yes | Provider and model identifier, for example `openai:gpt-4.1-mini`. |
| `runtime` / `runner` | no | Execution adapter, for example `codex`, `claude_code`, `opencode`, `pi`, or `local`. |
| `role` | recommended | Short human-readable responsibility. |
| `system` | recommended | System instruction used by the generated runtime manifest. |
| `tools` | no | Tool or capability references available to the agent. |
| `memory` | no | Memory backend hint, for example `vector support_memory`. |
| `input` / `inputs` | no | Named input contract values. |
| `output` / `outputs` | no | Named output contract values. |
| `handoff` / `handoffs` | no | Agent-local directed handoff, for example `handoff: Reviewer;`. |

`swarm`, `team`, and `agent_team` declare the same composition shape. Prefer `swarm` when the group is an autonomous multi-agent system and `team` when the group is a simple pipeline.

```apg
swarm ResearchCrew {
    agent Researcher {
        model: "openai:gpt-4.1-mini";
        system: "Find source-backed facts.";
        tools: [web.search, docs.read];
    }

    agent Reviewer {
        model: "openai:gpt-4.1-mini";
        system: "Check claims and flag weak evidence.";
    }

    flow: Researcher -> Reviewer;
}
```

Nested agents are lifted into the module as first-class agent declarations, then the swarm records the team membership and flow.

## Semantic Rules

The compiler validates agent composition before generation:

- Every AI agent must declare `model`.
- Each team must contain at least one agent.
- Every `agents` entry must resolve to a declared AI agent.
- Every `flow` or `handoff` source and target must resolve to a declared AI agent.
- AI agents do not require the older generic-agent `process` method; their runtime contract is the generated agent spec.

## Generated Runtime

When APG source contains first-class AI agents, code generation emits `ai_agents.py`.

The file contains:

- `AIAgentSpec`: immutable agent metadata.
- `AgentTeamSpec`: immutable team metadata.
- `AI_AGENTS`: registry keyed by agent name.
- `AI_AGENT_TEAMS`: registry keyed by team name.
- `get_agent(name)`, `get_team(name)`, and `describe_team(name)` helpers.

The runtime manifest is dependency-free. Provider SDK wiring belongs in the selected AI capability integration, while `ai_agents.py` remains the stable contract between APG syntax and application code.

## Runtime Integration

AI tooling changes faster than APG grammar should. Keep the APG declaration stable and swap the execution adapter:

```apg
agent CodeReviewer {
    role: "reviewer";
    model: "openai:gpt-5.4";
    runtime: codex;
    system: "Review the branch and return concrete defects.";
    tools: [repo.read, tests.run];
}

agent Refactorer {
    role: "executor";
    model: "anthropic:claude-sonnet";
    runtime: claude_code;
    system: "Make the requested change and run verification.";
}

agent OSSCoder {
    role: "executor";
    model: "provider:selected";
    runtime: opencode;
}

agent CustomerCoach {
    role: "support";
    model: "inflection:inflection_3_pi";
    runtime: pi;
}
```

The built-in `agents.integrations` registry includes:

| Runtime | Kind | Purpose |
| --- | --- | --- |
| `local` | in-process | Deterministic offline execution for tests and simple automation. |
| `codex` | CLI | Workspace-aware coding-agent execution through a local `codex` command. |
| `claude_code` | CLI | Workspace-aware coding-agent execution through a local `claude` command. |
| `opencode` | CLI | Workspace-aware execution through a local `opencode` command. |
| `pi` | HTTP | Chat-agent execution through the Inflection API using `inflection_3_pi` when `INFLECTION_API_KEY` is present. |

Register additional adapters in Python rather than adding grammar keywords:

```python
from agents.integrations import AgentBackendSpec, DEFAULT_AGENT_INTEGRATIONS

DEFAULT_AGENT_INTEGRATIONS.register(AgentBackendSpec(
    name="my_agent_tool",
    kind="cli",
    command="my-agent",
    args=["run", "--model", "{model}", "{prompt}"],
    supports_workspace=True,
))
```

The adapter contract normalizes every runner into `AgentInvocation` and `AgentRunResult`. APG code should depend on those stable objects, not on a provider-specific SDK response shape.

## Capability Selection

Any first-class AI agent or team selects the `ai/llm_integration` capability. A vector memory declaration also selects `data/vector_database`.

This keeps the language surface concise while still making deployment dependencies explicit in the generated application.

## Design Guidance

Use:

```apg
flow: Intake -> Research -> Resolve;
```

instead of verbose step objects when the handoff is linear.

Use:

```apg
handoff: Escalation when confidence < 0.6;
```

when an individual agent can branch to another agent.

Use quoted strings for human instructions and bare names for APG references. This keeps agent declarations compact without hiding the difference between prose and symbols.
