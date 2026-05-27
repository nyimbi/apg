"""Executable-default checks for composable templates."""

import json
from pathlib import Path

from jinja2 import Template

from templates.composable.base_template import BaseTemplate, BaseTemplateManager, BaseTemplateType
from templates.composable.capability import (
    Capability,
    CapabilityCategory,
    CapabilityIntegration,
    CapabilityManager,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSABLE_ROOT = REPO_ROOT / "templates" / "composable"


def test_generated_capability_structure_has_executable_defaults(tmp_path):
    capability = Capability(
        name="Risk Signals",
        category=CapabilityCategory.ANALYTICS,
        description="Detects risk signals from operational events.",
        version="1.0.0",
        features=["Rule Evaluation", "Status Reporting"],
        configuration={"threshold": 0.7},
        compatible_bases=["python_web"],
        integration=CapabilityIntegration(config_additions={"RISK_THRESHOLD": 0.7}),
    )
    manager = CapabilityManager(tmp_path)

    assert manager.create_capability_structure(tmp_path / "analytics" / "risk_signals", capability)

    integration = (tmp_path / "analytics" / "risk_signals" / "integration.py.template").read_text()
    readme = (tmp_path / "analytics" / "risk_signals" / "README.md").read_text()
    api = (tmp_path / "analytics" / "risk_signals" / "API.md").read_text()
    features = (tmp_path / "analytics" / "risk_signals" / "FEATURES.md").read_text()

    generated = "\n\n".join([integration, readme, api, features])
    assert "TODO:" not in generated
    assert "pass" not in integration
    assert "get_status" in integration
    assert "RiskSignalsCapability" in readme
    assert "/analytics/risk_signals/status" in api
    assert "APG_RUNTIME_URL" in api
    assert "http://localhost:8080" not in api
    assert "Executable Risk Signals support for rule evaluation." in features
    compile(integration, "integration.py.template", "exec")


def test_base_template_fallback_renders_executable_health_descriptor(tmp_path):
    template = BaseTemplate(
        name="CLI Tool",
        type=BaseTemplateType.CLI_TOOL,
        description="Command-line application",
        framework="python",
        capabilities_supported=[],
        default_capabilities=[],
        structure={},
        requirements=[],
    )
    manager = BaseTemplateManager(tmp_path)

    raw_template = manager._generate_app_template(template)
    rendered = Template(raw_template).render(
        project_name="Ops CLI",
        capabilities=["auth/basic_authentication"],
    )

    assert "TODO:" not in raw_template
    namespace = {}
    exec(compile(rendered, "app.py", "exec"), namespace)

    app = namespace["create_app"]({"debug": True})
    health = namespace["health_check"]()
    assert app["base_template"] == "cli_tool"
    assert app["capabilities"] == ["auth/basic_authentication"]
    assert health["status"] == "healthy"


def test_checked_in_base_metadata_is_python_first():
    offenders: list[str] = []
    for path in sorted(COMPOSABLE_ROOT.glob("bases/*/base.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("framework") != "python":
            offenders.append(f"{path.relative_to(REPO_ROOT)}: framework")
        if metadata.get("requirements") != []:
            offenders.append(f"{path.relative_to(REPO_ROOT)}: requirements")

    assert offenders == []


def test_checked_in_composable_templates_do_not_emit_placeholder_bodies():
    searched = [
        COMPOSABLE_ROOT / "capability.py",
        COMPOSABLE_ROOT / "base_template.py",
        *COMPOSABLE_ROOT.glob("bases/*/app.py.template"),
        *COMPOSABLE_ROOT.glob("bases/*/config.py.template"),
        *COMPOSABLE_ROOT.glob("capabilities/*/*/README.md"),
        *COMPOSABLE_ROOT.glob("capabilities/*/*/API.md"),
        *COMPOSABLE_ROOT.glob("capabilities/*/*/integration.py.template"),
    ]

    offenders: list[str] = []
    for path in searched:
        text = path.read_text()
        if any(marker in text for marker in [
            "TODO: Implement",
            "TODO: Add usage examples",
            "TODO: Add more examples",
        ]):
            offenders.append(str(path.relative_to(REPO_ROOT)))
        if path.name == "integration.py.template":
            for line_number, line in enumerate(text.splitlines(), 1):
                if line.strip() == "pass":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")
            compile(text, str(path.relative_to(REPO_ROOT)), "exec")
        if path.match("*/bases/*/app.py.template"):
            rendered = Template(text).render(
                project_name="Template Smoke",
                base_template=path.parent.name,
                version="1.0.0",
                project_description="Template smoke test",
                capabilities=["auth"],
            )
            compile(rendered, str(path.relative_to(REPO_ROOT)), "exec")
        if path.match("*/bases/*/config.py.template"):
            rendered = Template(text).render(
                project_name="Template Smoke",
                base_template=path.parent.name,
                secret_key="dev-secret",
                database_url="sqlite:///app.db",
                capabilities=["auth"],
            )
            compile(rendered, str(path.relative_to(REPO_ROOT)), "exec")

    assert offenders == []
