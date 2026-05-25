"""
First-class AI agent composition tests.
"""

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
}

swarm SupportCrew {
    agents: [Planner, Writer];
    flow: Planner -> Writer;
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
    assert planner.tools == ["tickets.read", "docs.search"]
    assert planner.memory.kind == "vector"
    assert planner.memory.name == "support_memory"

    assert isinstance(crew, AgentTeamDeclaration)
    assert crew.agents == ["Planner", "Writer"]
    assert [(edge.source, edge.target) for edge in crew.flow] == [("Planner", "Writer")]


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
    assert "'Planner'" in runtime
    assert "'SupportCrew'" in runtime
    assert "openai:gpt-4.1-mini" in runtime
