"""Regression coverage for executable Python code-generation defaults."""

import re

from compiler.ast_builder import (
    BlockStatement,
    CallExpression,
    EntityDeclaration,
    EntityType,
    Expression,
    ExpressionStatement,
    IdentifierExpression,
    ListExpression,
    LiteralExpression,
    MethodDeclaration,
    ModuleDeclaration,
    PropertyDeclaration,
    TypeAnnotation,
)
from compiler.code_generator import CodeGenConfig, PythonCodeGenerator


def _generate_python_files() -> dict[str, str]:
    module = ModuleDeclaration(
        name="runtime_defaults",
        entities=[
            EntityDeclaration(
                entity_type=EntityType.FORM,
                name="RuntimeProbe",
                properties=[
                    PropertyDeclaration("enabled", TypeAnnotation("bool"), LiteralExpression(True, "boolean")),
                ],
                methods=[
                    MethodDeclaration(
                        "ready",
                        return_type=TypeAnnotation("bool"),
                        is_async=True,
                    ),
                    MethodDeclaration("reset"),
                    MethodDeclaration(
                        "record",
                        body=BlockStatement(
                            [
                                ExpressionStatement(
                                    CallExpression(
                                        IdentifierExpression("str"),
                                        [LiteralExpression("observed", "string")],
                                    )
                                )
                            ]
                        ),
                    ),
                ],
            ),
            EntityDeclaration(
                entity_type=EntityType.DIGITAL_TWIN,
                name="MachineTwin",
                properties=[
                    PropertyDeclaration("temperature", TypeAnnotation("float"), LiteralExpression(20.0, "float")),
                ],
            ),
            EntityDeclaration(
                entity_type=EntityType.WORKFLOW,
                name="ProvisionWorkflow",
                properties=[
                    PropertyDeclaration(
                        "steps",
                        TypeAnnotation("str", is_list=True),
                        ListExpression(
                            [
                                LiteralExpression("reserve", "string"),
                                LiteralExpression("activate", "string"),
                            ]
                        ),
                    ),
                ],
            ),
            EntityDeclaration(
                entity_type=EntityType.AGENT,
                name="ProbeAgent",
                methods=[MethodDeclaration("ping")],
            ),
        ],
    )
    generator = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False))
    return generator.generate(module)


def test_python_generator_emits_executable_defaults_without_framework_scaffolding():
    files = _generate_python_files()
    generated_python = "\n\n".join(
        content for path, content in files.items() if path.endswith(".py")
    )

    assert "TODO: Implement" not in generated_python
    assert "placeholder implementation" not in generated_python
    assert "None  # TODO" not in generated_python
    assert not re.search(r"^\s*pass\s*$", generated_python, re.MULTILINE)
    assert "Flask-AppBuilder" not in generated_python
    assert "flask_appbuilder" not in generated_python
    assert "django" not in generated_python.lower()

    assert "app.py" in files
    assert "requirements.txt" in files
    assert "APG Python Application" in files["app.py"]
    assert "standard library" in files["requirements.txt"]

    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")


def test_unknown_expression_default_is_valid_python_literal():
    generator = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False))

    assert generator._generate_expression(Expression()) == "None"


def test_hybrid_template_mode_uses_python_entity_catalog(monkeypatch):
    module = ModuleDeclaration(
        name="hybrid_runtime",
        entities=[
            EntityDeclaration(
                entity_type=EntityType.AGENT,
                name="Planner",
                properties=[PropertyDeclaration("role", TypeAnnotation("str"))],
                methods=[MethodDeclaration("plan")],
            )
        ],
    )
    generator = PythonCodeGenerator(
        CodeGenConfig(use_composable_templates=True, template_output_mode="hybrid")
    )

    files = generator.generate(module)

    assert "entities.py" in files
    assert "views.py" not in files
    assert "model_views.py" not in files
    assert "Flask-AppBuilder" not in files["entities.py"]
    assert "flask_appbuilder" not in files["entities.py"]
    compile(files["entities.py"], "entities.py", "exec")


def test_legacy_framework_generation_helpers_are_removed():
    removed_helpers = {
        "_generate_legacy_flask_app",
        "_generate_requirements",
        "_generate_flask_app",
        "_generate_views",
        "_generate_config",
        "_generate_model_views",
        "_generate_table_model_view",
        "_generate_templates",
        "_generate_base_template",
        "_generate_agent_dashboard_template",
    }

    for helper_name in removed_helpers:
        assert not hasattr(PythonCodeGenerator, helper_name)
