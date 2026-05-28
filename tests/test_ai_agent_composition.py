"""
First-class AI agent composition tests.
"""

import json
import sys
import types

from compiler.ast_builder import AIAgentDeclaration, AgentTeamDeclaration
from compiler.compiler import APGCompiler
from compiler.parser import APGParser
from compiler.semantic_analyzer import SemanticAnalyzer


AI_AGENT_SOURCE = """
module support version 1.0.0 {
    description: "Support response crew";
}

agent Planner {
    role: "planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Break the ticket into concrete work.";
    capabilities: [planning, ticket_triage];
    tools: [tickets.read, docs.search];
    memory: vector support_memory;
    input: ticket;
    output: plan;
    config: {temperature: 0.2, max_turns: 4};
    rules: [{name: "ticket_required", when: "ticket missing", action: "reject"}];
    ui: {view: "PlannerConsole", route: "/support/planner"};
    theme: {name: "support_ops", density: compact};
}

agent Writer {
    role: "writer";
    model: "openai:gpt-4.1-mini";
    system: "Write concise customer-facing replies.";
    tools: [tickets.update];
}

swarm SupportCrew {
    agents: [Planner, Writer];
    capabilities: [support_response];
    flow: Planner -> Writer;
    config: {handoff_mode: sequential};
    rules: [{name: "review_low_confidence", when: "confidence < 0.6", action: "human_review"}];
    ui: {view: "SupportCrewDashboard", route: "/support/crew"};
    theme: {name: "support_ops"};
}
"""


def test_ai_agent_composition_parses_to_first_class_ast():
    result = APGParser().parse_string(AI_AGENT_SOURCE, "support.apg")

    assert result["success"] is True
    assert result["ast"].name == "support"
    assert [entity.name for entity in result["ast"].entities] == [
        "Planner",
        "Writer",
        "SupportCrew",
    ]

    planner = result["ast"].entities[0]
    crew = result["ast"].entities[2]
    assert isinstance(planner, AIAgentDeclaration)
    assert planner.model == "openai:gpt-4.1-mini"
    assert planner.runtime == "codex"
    assert planner.capabilities == ["planning", "ticket_triage"]
    assert planner.tools == ["tickets.read", "docs.search"]
    assert planner.memory.kind == "vector"
    assert planner.memory.name == "support_memory"
    assert planner.configuration == {"temperature": 0.2, "max_turns": 4}
    assert planner.rules == [{"name": "ticket_required", "when": "ticket missing", "action": "reject"}]
    assert planner.ui == {"view": "PlannerConsole", "route": "/support/planner"}
    assert planner.theme == {"name": "support_ops", "density": "compact"}

    assert isinstance(crew, AgentTeamDeclaration)
    assert crew.agents == ["Planner", "Writer"]
    assert crew.capabilities == ["support_response"]
    assert [(edge.source, edge.target) for edge in crew.flow] == [("Planner", "Writer")]
    assert crew.configuration == {"handoff_mode": "sequential"}
    assert crew.rules == [{"name": "review_low_confidence", "when": "confidence < 0.6", "action": "human_review"}]
    assert crew.ui == {"view": "SupportCrewDashboard", "route": "/support/crew"}
    assert crew.theme == {"name": "support_ops"}


def test_agent_composition_semantics_reject_unknown_handoffs():
    source = """
    agent Planner {
        model: "openai:gpt-4.1-mini";
    }

    team BrokenCrew {
        agents: [Planner, MissingAgent];
        flow: Planner -> MissingAgent;
    }
    """
    ast = APGParser().parse_string(source, "broken.apg")["ast"]
    result = SemanticAnalyzer().analyze(ast)

    assert result["success"] is False
    messages = [str(error) for error in result["errors"]]
    assert any("MissingAgent" in message for message in messages)


def test_ai_agent_composition_generates_runtime_manifest():
    result = APGCompiler().compile_string(AI_AGENT_SOURCE, "support.apg")

    assert result.success is True
    assert "ai_agents.py" in result.generated_files

    runtime = result.generated_files["ai_agents.py"]
    assert "AI_AGENTS" in runtime
    assert "AI_AGENT_TEAMS" in runtime
    assert "AI_AGENT_RUNTIME_DATA" in runtime
    assert "validate_agent_runtimes" in runtime
    assert "'Planner'" in runtime
    assert "'SupportCrew'" in runtime
    assert "openai:gpt-4.1-mini" in runtime
    assert "'runtime': 'codex'" in runtime
    assert "'capabilities': ['planning', 'ticket_triage']" in runtime
    assert "'configuration': {'temperature': 0.2, 'max_turns': 4}" in runtime
    assert "'rules': [{'name': 'ticket_required'" in runtime
    assert "'ui': {'view': 'SupportCrewDashboard', 'route': '/support/crew'}" in runtime
    assert "'theme': {'name': 'support_ops'}" in runtime

    namespace = {}
    exec(compile(runtime, "ai_agents.py", "exec"), namespace)

    assert namespace["list_agents"]() == ["Planner", "Writer"]
    assert namespace["list_agent_teams"]() == ["SupportCrew"]
    assert namespace["list_teams"]() == ["SupportCrew"]
    assert "codex" in namespace["list_agent_runtimes"]()
    assert namespace["canonical_runtime"]("claude") == "claude_code"
    assert namespace["describe_agent"]("Planner")["runtime"] == "codex"
    team_description = namespace["describe_team"]("SupportCrew")
    assert team_description["capabilities"] == ["support_response"]
    assert team_description["agent_names"] == ["Planner", "Writer"]
    assert team_description["agents"][0]["name"] == "Planner"
    assert team_description["configuration"] == {"handoff_mode": "sequential"}
    assert json.loads(json.dumps(team_description))["name"] == "SupportCrew"
    assert namespace["agents_by_runtime"]()["codex"][0].name == "Planner"
    assert namespace["validate_agent_runtimes"]()["errors"] == []
    assert namespace["validate_agent_runtimes"](["local"])["errors"] == [
        "Planner references unavailable runtime codex"
    ]


def test_ai_agent_runtime_catalog_supports_fast_moving_agent_tools():
    source = """
    agent CodexAgent {
        model: "openai:gpt-5";
        runtime: codex;
        capability: coding;
    }

    agent ClaudeAgent {
        model: "anthropic:claude";
        runner: claude;
        capabilities: [code_review];
    }

    agent OpenCodeAgent {
        model: "opencode/default";
        runtime: open_code;
        capability: terminal_coding;
    }

    agent PiAgent {
        model: "inflection:pi";
        runtime: pi;
        capability: conversation;
    }
    """

    ast = APGParser().parse_string(source, "runtimes.apg")["ast"]
    semantic_result = SemanticAnalyzer().analyze(ast)
    assert semantic_result["success"] is True
    assert [agent.capabilities for agent in ast.entities] == [
        ["coding"],
        ["code_review"],
        ["terminal_coding"],
        ["conversation"],
    ]

    result = APGCompiler().compile_string(source, "runtimes.apg")
    assert result.success is True

    namespace = {}
    exec(compile(result.generated_files["ai_agents.py"], "ai_agents.py", "exec"), namespace)

    assert namespace["canonical_runtime"]("codex") == "codex"
    assert namespace["canonical_runtime"]("claude") == "claude_code"
    assert namespace["canonical_runtime"]("open_code") == "opencode"
    assert namespace["canonical_runtime"]("pi") == "pi"
    assert set(namespace["agents_by_runtime"]()) >= {"codex", "claude_code", "opencode", "pi"}


def test_generated_app_manifest_includes_ai_agents_and_teams():
    result = APGCompiler().compile_string(AI_AGENT_SOURCE, "support.apg")

    assert result.success is True

    ai_agents = types.ModuleType("ai_agents")
    sys.modules["ai_agents"] = ai_agents
    try:
        exec(compile(result.generated_files["ai_agents.py"], "ai_agents.py", "exec"), ai_agents.__dict__)

        app = types.ModuleType("app")
        exec(compile(result.generated_files["app.py"], "app.py", "exec"), app.__dict__)
        manifest = app.describe_application()
    finally:
        sys.modules.pop("ai_agents", None)

    assert manifest["ai_agents"] == ["Planner", "Writer"]
    assert manifest["ai_agent_teams"] == ["SupportCrew"]
    assert manifest["ai_agent_descriptions"]["Planner"]["runtime"] == "codex"
    assert manifest["ai_agent_team_descriptions"]["SupportCrew"]["agent_names"] == [
        "Planner",
        "Writer",
    ]
    assert json.loads(json.dumps(manifest))["ai_agent_team_descriptions"]["SupportCrew"]["name"] == "SupportCrew"
