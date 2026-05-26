"""
Regression coverage for executable legacy code-generation defaults.
"""

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


def _generate_legacy_files() -> dict[str, str]:
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


def test_legacy_generator_emits_executable_defaults_without_todos_or_passes():
    files = _generate_legacy_files()
    generated_python = "\n\n".join(
        content for path, content in files.items() if path.endswith(".py")
    )

    assert "TODO: Implement" not in generated_python
    assert "placeholder implementation" not in generated_python
    assert "None  # TODO" not in generated_python
    assert not re.search(r"^\s*pass\s*$", generated_python, re.MULTILINE)

    assert "await asyncio.sleep(0)" in generated_python
    assert "return False" in generated_python
    assert "return None" in generated_python
    assert "setattr(self, key, value)" in generated_python
    assert "self._step_results[str(step)]" in generated_python
    assert "return {'status': 'executed', 'method': 'ping'}" in generated_python

    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")


def test_unknown_expression_default_is_valid_python_literal():
    generator = PythonCodeGenerator(CodeGenConfig(use_composable_templates=False))

    assert generator._generate_expression(Expression()) == "None"
