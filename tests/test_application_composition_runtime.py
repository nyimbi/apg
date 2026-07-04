"""Regression coverage for first-class APG application composition."""

from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path

from compiler.ast_builder import ASTBuilder, ApplicationDeclaration
from compiler.compiler import APGCompiler
from compiler.parser import APGParser


def _write_generated_files(target: Path, generated_files: dict[str, str]) -> None:
    for filename, content in generated_files.items():
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


APPLICATION_SOURCE = """
module enterprise_suite version 1.0.0 {}

app EnterpriseSuite {
    description: "Composable ERP shell";
    capabilities: [GeneralLedger];
    routes: ["/finance", "/ops"];
    components: {
        ledger_workbench: {capability: ledger_service, route: "/finance"}
    };
    screens: {
        Home: {route: "/", capability: GeneralLedger}
    };
    theme: {name: enterprise_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: container};
}

capability GeneralLedger {
    contract: {
        id: general_ledger,
        provides: [ledger_service],
        rules: [{name: "balanced", when: "debits != credits", action: "deny"}],
        ui: {shell: python}
    };
    approvals: {levels: 1, approvers: [controller]};
}
"""


def test_application_declaration_parses_as_first_class_composition():
    parse_result = APGParser().parse_string(APPLICATION_SOURCE, "application.apg")
    assert parse_result["success"] is True

    ast = ASTBuilder().build_ast(parse_result["parse_tree"], "application.apg")
    applications = [
        entity for entity in ast.entities
        if isinstance(entity, ApplicationDeclaration)
    ]

    assert len(applications) == 1
    application = applications[0]
    assert application.name == "EnterpriseSuite"
    assert application.description == "Composable ERP shell"
    assert application.capabilities == ["GeneralLedger"]
    assert application.routes == ["/finance", "/ops"]
    assert application.components["ledger_workbench"]["route"] == "/finance"
    assert application.theme["tokens"]["accent"] == "#174EA6"
    assert application.runtime == {"target": "python", "deployment": "container"}


def test_application_composition_compiles_to_executable_runtime_manifest():
    result = APGCompiler().compile_string(APPLICATION_SOURCE, "application.apg")
    assert result.success is True
    assert "apg_application.py" in result.generated_files

    namespace: dict[str, object] = {}
    exec(
        compile(result.generated_files["apg_application.py"], "apg_application.py", "exec"),
        namespace,
    )

    assert namespace["list_applications"]() == ["EnterpriseSuite"]
    description = namespace["describe_application_composition"]("EnterpriseSuite")
    assert description["capabilities"] == ["GeneralLedger"]
    assert description["routes"] == ["/finance", "/ops"]
    assert namespace["application_component_catalog"]()["EnterpriseSuite.ledger_workbench"]["spec"] == {
        "capability": "ledger_service",
        "route": "/finance",
    }
    assert namespace["application_screens"]("EnterpriseSuite")[0]["route"] == "/"
    assert namespace["application_route_index"]()["/"]["name"] == "Home"
    assert namespace["application_route_index"]()["/finance"]["application"] == "EnterpriseSuite"
    graph_edges = {
        (edge["source"], edge["relation"], edge["target"])
        for edge in namespace["application_dependency_graph"]()["edges"]
    }
    assert ("application:EnterpriseSuite", "uses_capability", "capability:GeneralLedger") in graph_edges
    assert ("application:EnterpriseSuite", "exposes_route", "route:/finance") in graph_edges
    assert ("application_screen:EnterpriseSuite.Home", "mounted_at", "route:/") in graph_edges
    assert namespace["validate_application_compositions"](
        available_capabilities=["GeneralLedger"],
    ) == {"errors": [], "warnings": []}
    assert json.loads(json.dumps(description))["name"] == "EnterpriseSuite"


def test_generated_app_manifest_includes_application_composition():
    result = APGCompiler().compile_string(APPLICATION_SOURCE, "application.apg")
    assert result.success is True

    application_module = types.ModuleType("apg_application")
    capabilities_module = types.ModuleType("apg_capabilities")
    sys.modules["apg_application"] = application_module
    sys.modules["apg_capabilities"] = capabilities_module
    try:
        exec(
            compile(result.generated_files["apg_application.py"], "apg_application.py", "exec"),
            application_module.__dict__,
        )
        exec(
            compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"),
            capabilities_module.__dict__,
        )

        app = types.ModuleType("app")
        exec(compile(result.generated_files["app.py"], "app.py", "exec"), app.__dict__)
        manifest = app.describe_application()
        component = app.component_manifest()
        validation = app.validate_application()
        applications_payload = app._route_payload("/applications")[1]
        app_screen = app._application_screen("/")
        app_screen_status, app_screen_html = app._application_screen_payload("/")
        ui_status, ui_html = app._ui_payload("/ui")
        capability_status, capability_html = app._ui_payload("/ui/capabilities/GeneralLedger")
        rule_status, rule_html = app._ui_post_payload(
            "/ui/capabilities/GeneralLedger/rules/evaluate",
            {"record": {"context_json": '{"debits": 10, "credits": 5}'}},
        )
        config_status, config_html = app._ui_post_payload(
            "/ui/capabilities/GeneralLedger/configuration/resolve",
            {"record": {"configuration_json": "{}"}},
        )
        approval_status, approval_html = app._ui_post_payload(
            "/ui/capabilities/GeneralLedger/approval/plan",
            {"record": {"context_json": "{}"}},
        )
        openapi = app.openapi_document()
    finally:
        sys.modules.pop("apg_application", None)
        sys.modules.pop("apg_capabilities", None)

    assert manifest["application_compositions"] == ["EnterpriseSuite"]
    assert manifest["application_composition_descriptions"]["EnterpriseSuite"]["routes"] == ["/finance", "/ops"]
    assert manifest["application_routes"]["/"]["name"] == "Home"
    assert component["application_compositions"] == ["EnterpriseSuite"]
    assert component["application_dependency_graph"]["edges"]
    assert component["application_routes"]["/finance"]["application"] == "EnterpriseSuite"
    assert validation["valid"] is True
    assert validation["checks"]["application_compositions"] == {"errors": [], "warnings": []}
    assert applications_payload["applications"]["EnterpriseSuite"]["capabilities"] == ["GeneralLedger"]
    assert app_screen["application"] == "EnterpriseSuite"
    assert app_screen_status == 200
    assert "EnterpriseSuite" in app_screen_html
    assert "GeneralLedger" in app_screen_html
    assert ui_status == 200
    assert "Application Routes" in ui_html
    assert 'href="/finance"' in ui_html
    assert 'href="/ui/capabilities/GeneralLedger"' in ui_html
    assert "Capabilities" in ui_html
    assert "GeneralLedger" in ui_html
    assert capability_status == 200
    assert "GeneralLedger" in capability_html
    assert rule_status == 200
    assert "&quot;decision&quot;: &quot;deny&quot;" in rule_html["html"]
    assert config_status == 200
    assert "&quot;capability&quot;: &quot;GeneralLedger&quot;" in config_html["html"]
    assert approval_status == 200
    assert "&quot;required&quot;: true" in approval_html["html"]
    assert "/" in openapi["paths"]
    assert "/finance" in openapi["paths"]


def test_generated_package_reexports_application_composition_helpers(tmp_path):
    result = APGCompiler().compile_string(APPLICATION_SOURCE, "application.apg")
    assert result.success is True

    package_dir = tmp_path / "enterprise_suite_generated"
    package_dir.mkdir()
    _write_generated_files(package_dir, result.generated_files)

    spec = importlib.util.spec_from_file_location(
        "enterprise_suite_generated",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["enterprise_suite_generated"] = module
    try:
        spec.loader.exec_module(module)
        applications = module.list_applications()
        application = module.describe_application_composition("EnterpriseSuite")
        graph = module.application_dependency_graph()
        routes = module.application_route_index()
    finally:
        sys.modules.pop("enterprise_suite_generated", None)
        for name in list(sys.modules):
            if name.startswith("enterprise_suite_generated."):
                sys.modules.pop(name, None)

    assert applications == ["EnterpriseSuite"]
    assert application["capabilities"] == ["GeneralLedger"]
    assert graph["edges"]
    assert routes["/"]["name"] == "Home"
    assert "list_applications" in module.__all__
    assert "application_dependency_graph" in module.__all__
    assert "application_route_index" in module.__all__
    assert "application_screens" in module.__all__
    assert "validate_application_compositions" in module.__all__
