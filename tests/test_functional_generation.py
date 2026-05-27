"""Functional smoke coverage for Python-first APG generation."""

from __future__ import annotations

from compiler.compiler import compile_apg_string


FUNCTIONAL_SOURCE = """
module task_manager version 1.0.0 {
    description: "Task Management System with APG";
}

agent TaskManager {
    role: "task_manager";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Coordinate task intake and completion.";
}
"""


def test_functional_generation_emits_executable_python_manifest(tmp_path):
    result = compile_apg_string(FUNCTIONAL_SOURCE)

    assert result.success is True
    assert result.target_language == "python"
    assert set(result.generated_files) >= {"app.py", "__init__.py", "requirements.txt", "ai_agents.py"}

    output_dir = tmp_path / "generated"
    for relative_path, content in result.generated_files.items():
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    app_source = (output_dir / "app.py").read_text(encoding="utf-8")
    requirements = (output_dir / "requirements.txt").read_text(encoding="utf-8")
    ai_agents = (output_dir / "ai_agents.py").read_text(encoding="utf-8")

    assert "APG Python Application" in app_source
    assert "standard library" in requirements
    assert "TaskManager" in ai_agents
    assert "Flask-AppBuilder" not in app_source
    assert "flask_appbuilder" not in app_source
    assert "http://localhost:8080" not in app_source

    namespace: dict[str, object] = {}
    exec(compile(app_source, "app.py", "exec"), namespace)
    manifest = namespace["describe_application"]()

    assert manifest["name"] == "task_manager"
    assert manifest["version"] == "1.0.0"
    assert manifest["entities"] == [
        {
            "methods": [],
            "name": "TaskManager",
            "properties": [],
            "type": "ai_agent",
        }
    ]
