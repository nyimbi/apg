"""Integrated code-generation coverage for Python-first APG artifacts."""

from __future__ import annotations

from compiler.ast_builder import (
    AIAgentDeclaration,
    AgentTeamDeclaration,
    CapabilityDeclaration,
    EntityType,
    ModuleDeclaration,
)
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator


def _composition_module() -> ModuleDeclaration:
    return ModuleDeclaration(
        name="commerce_ops",
        version="2.0.0",
        description="Composable commerce operations",
        entities=[
            AIAgentDeclaration(
                entity_type=EntityType.AI_AGENT,
                name="Planner",
                role="planner",
                model="codex:gpt-5.4",
                runtime="codex",
                system_prompt="Plan order fulfillment work.",
                capabilities=["orders.plan"],
                tools=["repo.search"],
                configuration={"temperature": 0.2},
                rules=[{"when": "order.priority == 'high'", "then": "escalate"}],
                ui={"screen": "planner_console"},
                theme={"accent": "blue"},
            ),
            AIAgentDeclaration(
                entity_type=EntityType.AI_AGENT,
                name="Auditor",
                role="auditor",
                model="claude-code:sonnet",
                runtime="claude_code",
                system_prompt="Review fulfillment decisions.",
                capabilities=["orders.audit"],
            ),
            AgentTeamDeclaration(
                entity_type=EntityType.AGENT_TEAM,
                name="FulfillmentTeam",
                agents=["Planner", "Auditor"],
                capabilities=["orders.plan", "orders.audit"],
                configuration={"handoff": "review_required"},
                rules=[{"when": "plan.ready", "then": "handoff:Auditor"}],
                ui={"screen": "team_board"},
                theme={"density": "compact"},
            ),
            CapabilityDeclaration(
                entity_type=EntityType.CAPABILITY,
                name="OrderFulfillment",
                provides=["orders.fulfill"],
                requires=["inventory.reserve"],
                configuration={"sla_minutes": 30},
                rules=[{"when": "stock.available", "then": "reserve"}],
                rule_engine={"mode": "deterministic"},
                ui={"screen": "fulfillment_dashboard"},
                theme={"palette": "operations"},
                runtime={"stream": "bytewax"},
                erp_modules=["sales", "inventory"],
                components={"screens": ["orders", "shipments"]},
                business_rules=[{"name": "reserve_before_ship"}],
                approvals={"manager": "required"},
                master_data={"entities": ["customer", "item"]},
                i18n={"default": "en"},
                streaming={"engine": "bytewax"},
            ),
        ],
    )


def test_default_python_generation_integrates_agents_and_capabilities():
    generator = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False))

    files = generator.generate(_composition_module())

    assert set(files) == {
        "README.md",
        "__init__.py",
        "agent_stubs.py",
        "ai_agents.py",
        "apg_capabilities.py",
        "app.py",
        "requirements.txt",
        "static/apg.css",
        "static/htmx.min.js",
        "static/sortable.min.js",
    }
    assert "standard library" in files["requirements.txt"]
    assert "Planner" in files["ai_agents.py"]
    assert "FulfillmentTeam" in files["ai_agents.py"]
    assert "OrderFulfillment" in files["apg_capabilities.py"]
    assert "bytewax" in files["apg_capabilities.py"]

    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")


def test_hybrid_generation_adds_python_entity_catalog_not_legacy_views():
    generator = PythonCodeGenerator(
        CodeGenConfig(use_composable_templates=True, template_output_mode="hybrid")
    )

    files = generator.generate(_composition_module())

    assert "entities.py" in files
    assert "views.py" not in files
    assert "model_views.py" not in files
    assert "Flask-AppBuilder" not in files["entities.py"]
    assert "flask_appbuilder" not in files["entities.py"]
    assert "Planner" in files["entities.py"]
    assert "OrderFulfillment" in files["entities.py"]
    compile(files["entities.py"], "entities.py", "exec")
