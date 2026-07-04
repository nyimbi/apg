"""Executable compiler baseline for first-class APG capability declarations."""

from __future__ import annotations

import json
import importlib.util
import re
import socket
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

from compiler.ast_builder import ASTBuilder, CapabilityDeclaration
from compiler.compiler import APGCompiler
from compiler.parser import APGParser
from compiler.semantic_analyzer import SemanticAnalyzer


def _write_generated_files(target: Path, generated_files: dict[str, str]) -> None:
    for filename, content in generated_files.items():
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


CAPABILITY_SOURCE = """
module erp_ops version 1.0.0 {
    description: "ERP capability composition";
}

capability GeneralLedger {
    contract: {
        id: general_ledger,
        name: "General Ledger",
        version: "1.0.0",
        provides: [journal_entries, chart_of_accounts, financial_periods],
        requires: [audit_log],
        configuration: {currency: "KES", fiscal_calendar: "monthly"},
        rules: [{name: "balanced_journal", when: "debits != credits", action: "deny"}],
        ui: {shell: react, routes: [{name: "Journals", path: "/finance/gl/journals", component: "JournalScreen"}]},
        theme: {name: finance_ops, tokens: {accent: "#126E82"}},
        runtime: {backend: python, streaming: {processor: bytewax}}
    };
    erp_modules: [finance, general_ledger, accounts_payable];
    components: {posting: {capability: journal_entries, permissions: [post, reverse]}};
    business_rules: [{name: "posting_period_open", when: "period.closed", action: "deny", priority: 100}];
    approvals: {levels: 2, approvers: [controller, cfo]};
    master_data: {entities: [account, cost_center], deduplication: strict};
    i18n: {supported_languages: [en, sw, ha, yo, zu], default_language: en, fallback_language: en};
    streaming: {processor: bytewax, state: ledger_posting_state};
}
"""


DEPENDENCY_SOURCE = """
capability AuditLog {
    contract: {
        id: audit_log,
        provides: [audit_log],
        configuration: {retention_days: 365}
    };
    master_data: {entities: [audit_event]};
    streaming: {processor: bytewax, state: audit_log_state};
}

capability GeneralLedger {
    contract: {
        id: general_ledger,
        provides: [journal_entries],
        requires: [audit_log],
        configuration: {currency: "KES"}
    };
    master_data: {entities: [account]};
    streaming: {processor: bytewax, state: ledger_state};
}
"""


SCREEN_SOURCE = """
capability OperationsWorkbench {
    contract: {
        id: operations_workbench,
        provides: [operations_ui],
        configuration: {tenant_scoped: true},
        ui: {shell: python},
        theme: {name: ops_theme}
    };
    screens: {
        Dashboard: {
            route: "/ops",
            layout: dashboard,
            contains: [KpiStrip, ApprovalQueue],
            composes: [LedgerTable],
            binds: [ledger.entries],
            actions: [approve, reject],
            events: [{on: "select", do: "filter", target: LedgerTable}],
            relationships: [
                {from: KpiStrip, to: LedgerTable, via: filters},
                ApprovalQueue -> LedgerTable
            ]
        }
    };
}
"""


INVALID_I18N_SOURCE = """
capability InvalidLocalization {
    contract: {
        id: invalid_localization,
        provides: [localized_ui],
        configuration: {tenant_scoped: true}
    };
    i18n: {supported_languages: [en, "zz"], default_language: "zz", fallback_language: "qq"};
    master_data: {entities: [translation]};
    streaming: {processor: bytewax, state: invalid_i18n_state};
}
"""


def test_capability_declaration_parses_to_first_class_ast():
    parse_result = APGParser().parse_string(CAPABILITY_SOURCE, "erp_ops.apg")
    assert parse_result["success"] is True, parse_result["errors"]
    ast = ASTBuilder().build_ast(parse_result["parse_tree"], "erp_ops.apg")

    assert ast.name == "erp_ops"
    assert len(ast.entities) == 1

    capability = ast.entities[0]
    assert isinstance(capability, CapabilityDeclaration)
    assert capability.name == "GeneralLedger"
    assert capability.provides == ["journal_entries", "chart_of_accounts", "financial_periods"]
    assert capability.requires == ["audit_log"]
    assert capability.configuration == {"currency": "KES", "fiscal_calendar": "monthly"}
    assert capability.erp_modules == ["finance", "general_ledger", "accounts_payable"]
    assert capability.rules == [{"name": "balanced_journal", "when": "debits != credits", "action": "deny"}]
    assert capability.business_rules == [
        {"name": "posting_period_open", "when": "period.closed", "action": "deny", "priority": 100}
    ]
    assert capability.ui["routes"][0]["component"] == "JournalScreen"
    assert capability.theme["tokens"]["accent"] == "#126E82"
    assert capability.i18n["supported_languages"] == ["en", "sw", "ha", "yo", "zu"]
    assert capability.streaming == {"processor": "bytewax", "state": "ledger_posting_state"}


def test_capability_screens_compile_to_executable_composition_manifest():
    parse_result = APGParser().parse_string(SCREEN_SOURCE, "screens.apg")
    assert parse_result["success"] is True, parse_result["errors"]
    ast = ASTBuilder().build_ast(parse_result["parse_tree"], "screens.apg")
    capability = ast.entities[0]

    assert isinstance(capability, CapabilityDeclaration)
    assert capability.screens["Dashboard"]["route"] == "/ops"
    assert capability.screens["Dashboard"]["relationships"][0]["via"] == "filters"

    result = APGCompiler().compile_string(SCREEN_SOURCE, "screens.apg")
    assert result.success is True, result.errors

    namespace = {}
    exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), namespace)

    screens = namespace["capability_screens"]("OperationsWorkbench")
    assert len(screens) == 1
    dashboard = screens[0]
    assert dashboard["name"] == "Dashboard"
    assert dashboard["path"] == "/ops"
    assert dashboard["layout"] == "dashboard"
    assert dashboard["contains"] == ["KpiStrip", "ApprovalQueue"]
    assert dashboard["composes"] == ["LedgerTable"]
    assert dashboard["binds"] == ["ledger.entries"]
    assert dashboard["actions"] == ["approve", "reject"]
    assert dashboard["events"] == [{"on": "select", "do": "filter", "target": "LedgerTable"}]
    assert dashboard["relationships"] == [
        {"from": "KpiStrip", "to": "LedgerTable", "via": "filters"},
        {"type": "relates_to", "from": "ApprovalQueue", "to": "LedgerTable"},
    ]
    assert namespace["ui_route_index"]()["/ops"]["name"] == "Dashboard"

    graph_edges = {
        (edge["source"], edge["relation"], edge["target"])
        for edge in namespace["composition_graph"]()["edges"]
    }
    assert ("screen:OperationsWorkbench.Dashboard", "contains", "component:KpiStrip") in graph_edges
    assert ("screen:OperationsWorkbench.Dashboard", "composes", "component:LedgerTable") in graph_edges
    assert ("screen:OperationsWorkbench.Dashboard", "binds_to", "binding:ledger.entries") in graph_edges
    assert ("component:KpiStrip", "filters", "component:LedgerTable") in graph_edges
    assert ("component:ApprovalQueue", "relates_to", "component:LedgerTable") in graph_edges


def test_capability_semantics_require_executable_contract_shape():
    source = """
    capability BrokenCapability {
        contract: {id: broken, provides: [ledger, ledger]};
    }
    """
    parse_result = APGParser().parse_string(source, "broken_capability.apg")
    assert parse_result["success"] is True, parse_result["errors"]
    ast = ASTBuilder().build_ast(parse_result["parse_tree"], "broken_capability.apg")
    result = SemanticAnalyzer().analyze(ast)

    assert result["success"] is False
    assert any("duplicate provided services" in str(error) for error in result["errors"])


def test_capability_declaration_generates_runtime_manifest():
    result = APGCompiler().compile_string(CAPABILITY_SOURCE, "erp_ops.apg")

    assert result.success is True
    assert "apg_capabilities.py" in result.generated_files

    runtime = result.generated_files["apg_capabilities.py"]
    readme = result.generated_files["README.md"]
    assert "CAPABILITIES" in runtime
    assert "CapabilitySpec" in runtime
    assert not re.search(r"^\s*pass\s*$", runtime, re.MULTILINE)
    assert "GeneralLedger" in runtime
    assert "'erp_modules': ['finance', 'general_ledger', 'accounts_payable']" in runtime
    assert "'supported_languages': ['en', 'sw', 'ha', 'yo', 'zu']" in runtime
    assert "'processor': 'bytewax'" in runtime
    assert "`GeneralLedger` - provides journal_entries, chart_of_accounts, financial_periods" in readme
    assert "GET /streaming" in readme
    assert "POST /capabilities/{Capability}/rules/evaluate" in readme
    assert "GET /finance/gl/journals" in readme

    namespace = {}
    exec(compile(runtime, "apg_capabilities.py", "exec"), namespace)

    assert namespace["list_capabilities"]() == ["GeneralLedger"]
    capability = namespace["get_capability"]("GeneralLedger")
    assert capability.provides == ["journal_entries", "chart_of_accounts", "financial_periods"]
    capability_description = namespace["describe_capability"]("GeneralLedger")
    assert capability_description["name"] == "GeneralLedger"
    assert capability_description["provides"] == ["journal_entries", "chart_of_accounts", "financial_periods"]
    assert capability_description["configuration"] == {"currency": "KES", "fiscal_calendar": "monthly"}
    assert capability_description["theme"]["tokens"]["accent"] == "#126E82"
    assert namespace["describe_capabilities"]()["GeneralLedger"]["erp_modules"] == [
        "finance",
        "general_ledger",
        "accounts_payable",
    ]
    assert json.loads(json.dumps(capability_description))["name"] == "GeneralLedger"
    assert namespace["capabilities_by_erp_module"]()["general_ledger"][0].name == "GeneralLedger"
    assert namespace["capability_names_by_erp_module"]() == {
        "accounts_payable": ["GeneralLedger"],
        "finance": ["GeneralLedger"],
        "general_ledger": ["GeneralLedger"],
    }
    grouped_descriptions = namespace["describe_capabilities_by_erp_module"]()
    assert grouped_descriptions["finance"][0]["name"] == "GeneralLedger"
    assert json.loads(json.dumps(grouped_descriptions))["general_ledger"][0]["name"] == "GeneralLedger"
    assert namespace["provided_services"]()["journal_entries"] == ["GeneralLedger"]

    assert namespace["capability_configuration"]("GeneralLedger") == {
        "currency": "KES",
        "fiscal_calendar": "monthly",
    }
    assert namespace["capability_configuration"](
        "GeneralLedger",
        {"currency": "USD", "posting": {"batch_size": 250}},
    ) == {
        "currency": "USD",
        "fiscal_calendar": "monthly",
        "posting": {"batch_size": 250},
    }
    assert namespace["configuration_value"]("GeneralLedger", "currency") == "KES"
    config_validation = namespace["validate_capability_configuration"]("GeneralLedger")
    assert config_validation["errors"] == []
    assert config_validation["warnings"] == []
    assert namespace["approval_policy"]("GeneralLedger") == {
        "levels": 2,
        "approvers": ["controller", "cfo"],
        "thresholds": {},
        "segregation_of_duties": False,
        "escalation": None,
    }
    assert namespace["approval_plan"]("GeneralLedger") == {
        "capability": "GeneralLedger",
        "required": True,
        "levels": 2,
        "approvers": ["controller", "cfo"],
        "segregation_of_duties": False,
        "escalation": None,
    }
    assert namespace["master_data_entities"]("GeneralLedger") == ["account", "cost_center"]
    assert namespace["master_data_index"]() == {
        "account": ["GeneralLedger"],
        "cost_center": ["GeneralLedger"],
    }
    assert namespace["validate_master_data_contracts"]() == {"errors": [], "warnings": []}

    assert namespace["capability_theme"]("GeneralLedger") == {
        "name": "finance_ops",
        "tokens": {"accent": "#126E82"},
        "components": {},
        "allow_tenant_overrides": True,
    }
    assert namespace["theme_token"]("GeneralLedger", "accent") == "#126E82"
    assert namespace["theme_token"](
        "GeneralLedger",
        "accent",
        tenant_overrides={"tokens": {"accent": "#0F766E", "danger": "#B91C1C"}},
    ) == "#0F766E"
    assert namespace["capability_theme"](
        "GeneralLedger",
        {"tokens": {"accent": "#0F766E", "danger": "#B91C1C"}},
    )["tokens"] == {"accent": "#0F766E", "danger": "#B91C1C"}
    assert namespace["capability_languages"]("GeneralLedger") == ["en", "sw", "ha", "yo", "zu"]
    assert namespace["resolve_language"]("GeneralLedger", "sw") == "sw"
    assert namespace["resolve_language"]("GeneralLedger", "fr") == "en"
    assert namespace["validate_capability_i18n"]() == {"errors": [], "warnings": []}
    assert len(namespace["african_language_codes"]()) >= 40
    assert {"ha", "ig", "sw", "yo", "zu"} <= set(namespace["african_language_codes"]())
    assert set(namespace["african_language_codes"]()) <= set(namespace["supported_language_codes"]())
    assert namespace["capability_streaming"]("GeneralLedger") == {
        "capability": "GeneralLedger",
        "processor": "bytewax",
        "input": None,
        "output": None,
        "state": "ledger_posting_state",
        "window": None,
        "config": {"processor": "bytewax", "state": "ledger_posting_state"},
    }
    assert namespace["streaming_processor_index"]() == {"bytewax": ["GeneralLedger"]}
    assert namespace["streaming_state_index"]() == {"ledger_posting_state": ["GeneralLedger"]}
    assert namespace["validate_streaming_contracts"]() == {"errors": [], "warnings": []}
    health = namespace["capability_health"]("GeneralLedger")
    health_report = namespace["capability_health_report"]()
    assert health["healthy"] is True
    assert health["status"] == "ok"
    assert health["configuration"]["errors"] == []
    assert health["rules"]["names"] == ["posting_period_open", "balanced_journal"]
    assert health["approvals"]["approvers"] == ["controller", "cfo"]
    assert health["ui"]["route_index"]["/finance/gl/journals"]["component"] == "JournalScreen"
    assert health["theme"]["tokens"]["accent"] == "#126E82"
    assert health["streaming"]["processor"] == "bytewax"
    assert health["master_data"] == ["account", "cost_center"]
    assert health["languages"] == ["en", "sw", "ha", "yo", "zu"]
    assert health["components"]["posting"]["capability"] == "journal_entries"
    assert health_report["healthy"] is True
    assert health_report["capabilities"]["GeneralLedger"]["status"] == "ok"

    rule_names = [rule["name"] for rule in namespace["capability_rules"]("GeneralLedger")]
    assert rule_names == ["posting_period_open", "balanced_journal"]

    denied_by_contract = namespace["evaluate_capability_rules"](
        "GeneralLedger",
        {"debits": 100, "credits": 90, "period": {"closed": False}},
    )
    assert denied_by_contract["decision"] == "deny"
    assert denied_by_contract["matched_rules"] == ["balanced_journal"]
    assert denied_by_contract["actions"][0]["action"] == "deny"

    denied_by_business_rule = namespace["evaluate_capability_rules"](
        "GeneralLedger",
        {"debits": 100, "credits": 100, "period": {"closed": True}},
    )
    assert denied_by_business_rule["decision"] == "deny"
    assert denied_by_business_rule["matched_rules"] == ["posting_period_open"]

    allowed = namespace["evaluate_capability_rules"](
        "GeneralLedger",
        {"debits": 100, "credits": 100, "period": {"closed": False}},
    )
    assert allowed["decision"] == "allow"
    assert allowed["matched_rules"] == []

    assert namespace["capability_screens"]("GeneralLedger") == [
        {
            "id": "GeneralLedger.Journals",
            "capability": "GeneralLedger",
            "name": "Journals",
            "path": "/finance/gl/journals",
            "component": "JournalScreen",
            "permission": None,
            "nav_group": None,
            "shell": "react",
            "theme": "finance_ops",
        }
    ]
    assert namespace["ui_route_index"]()["/finance/gl/journals"]["component"] == "JournalScreen"

    assert namespace["capability_components"]("GeneralLedger") == {
        "posting": {"capability": "journal_entries", "permissions": ["post", "reverse"]}
    }
    assert namespace["component_catalog"]() == {
        "GeneralLedger.posting": {
            "id": "GeneralLedger.posting",
            "capability": "GeneralLedger",
            "name": "posting",
            "service": "journal_entries",
            "permissions": ["post", "reverse"],
            "spec": {"capability": "journal_entries", "permissions": ["post", "reverse"]},
        }
    }
    assert namespace["component_permissions"]("GeneralLedger", "posting") == ["post", "reverse"]
    assert namespace["component_service_bindings"]() == {
        "journal_entries": ["GeneralLedger.posting"]
    }
    assert namespace["validate_component_contracts"]() == {"errors": [], "warnings": []}

    graph = namespace["composition_graph"]()
    graph_edges = {(edge["source"], edge["relation"], edge["target"]) for edge in graph["edges"]}
    assert ("capability:GeneralLedger", "has_screen", "screen:GeneralLedger.Journals") in graph_edges
    assert ("screen:GeneralLedger.Journals", "renders", "component:JournalScreen") in graph_edges
    assert ("capability:GeneralLedger", "belongs_to", "erp_module:general_ledger") in graph_edges
    assert ("component:posting", "binds_to", "service:journal_entries") in graph_edges
    assert ("component:posting", "requires_permission", "permission:post") in graph_edges
    assert ("component:posting", "requires_permission", "permission:reverse") in graph_edges
    assert ("capability:GeneralLedger", "streams_with", "stream_processor:bytewax") in graph_edges
    assert ("capability:GeneralLedger", "stores_stream_state", "stream_state:ledger_posting_state") in graph_edges

    validation = namespace["validate_capability_contracts"]()
    assert validation["errors"] == []
    assert validation["warnings"] == ["GeneralLedger requires external service audit_log"]


def test_generated_rule_engine_evaluates_arithmetic_and_configuration_thresholds():
    source = """
capability WarehouseInventory {
    contract: {
        id: warehouse_inventory,
        provides: [stock_balances],
        configuration: {approval_threshold: 50000},
        rules: [
            {name: "no_negative_stock", when: "on_hand - reserved < 0", action: "deny"},
            {name: "high_value_approval", when: "amount > approval_threshold", action: "require_review"},
            {name: "budget_required", when: "budget_code missing", action: "deny"}
        ]
    };
}
"""
    result = APGCompiler().compile_string(source, "warehouse.apg")
    assert result.success is True, result.errors
    namespace = {}
    exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), namespace)

    negative_stock = namespace["evaluate_capability_rules"](
        "WarehouseInventory",
        {"on_hand": 4, "reserved": 5, "amount": 100, "budget_code": "OPS"},
    )
    assert negative_stock["decision"] == "deny"
    assert negative_stock["matched_rules"] == ["no_negative_stock"]

    high_value = namespace["evaluate_capability_rules"](
        "WarehouseInventory",
        {"on_hand": 10, "reserved": 1, "amount": 75000, "budget_code": "OPS"},
    )
    assert high_value["decision"] == "require_review"
    assert high_value["matched_rules"] == ["high_value_approval"]
    assert high_value["effective_context"]["approval_threshold"] == 50000

    missing_budget = namespace["evaluate_capability_rules"](
        "WarehouseInventory",
        {"on_hand": 10, "reserved": 1, "amount": 100},
    )
    assert missing_budget["decision"] == "deny"
    assert missing_budget["matched_rules"] == ["budget_required"]

    allowed = namespace["evaluate_capability_rules"](
        "WarehouseInventory",
        {"on_hand": 10, "reserved": 1, "amount": 100, "budget_code": "OPS"},
    )
    assert allowed["decision"] == "allow"
    assert allowed["matched_rules"] == []


def test_generated_app_manifest_includes_capability_descriptions():
    result = APGCompiler().compile_string(CAPABILITY_SOURCE, "erp_ops.apg")

    assert result.success is True

    capabilities = types.ModuleType("apg_capabilities")
    sys.modules["apg_capabilities"] = capabilities
    try:
        exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), capabilities.__dict__)

        app = types.ModuleType("app")
        exec(compile(result.generated_files["app.py"], "app.py", "exec"), app.__dict__)
        manifest = app.describe_application()
        validation = app.validate_application()
    finally:
        sys.modules.pop("apg_capabilities", None)

    assert manifest["capabilities"] == ["GeneralLedger"]
    assert manifest["capability_descriptions"]["GeneralLedger"]["configuration"] == {
        "currency": "KES",
        "fiscal_calendar": "monthly",
    }
    assert manifest["capability_descriptions_by_erp_module"]["finance"][0]["name"] == "GeneralLedger"
    assert manifest["capability_dependency_graph"] == {"GeneralLedger": []}
    assert manifest["capability_load_order"]["unresolved"] == {"GeneralLedger": ["audit_log"]}
    assert manifest["ui_routes"]["/finance/gl/journals"]["component"] == "JournalScreen"
    assert manifest["streaming_processors"] == {"bytewax": ["GeneralLedger"]}
    assert json.loads(json.dumps(manifest))["capability_descriptions"]["GeneralLedger"]["name"] == "GeneralLedger"
    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["checks"]["capability_contracts"]["errors"] == []
    assert validation["checks"]["capability_dependencies"]["warnings"] == [
        "GeneralLedger requires external service audit_log"
    ]
    assert "capability_contracts: GeneralLedger requires external service audit_log" in validation["warnings"]
    assert json.loads(json.dumps(validation))["name"] == "erp_ops"


def test_generated_app_manifest_includes_capability_composition_topology():
    result = APGCompiler().compile_string(SCREEN_SOURCE, "screens.apg")
    assert result.success is True

    capabilities = types.ModuleType("apg_capabilities")
    sys.modules["apg_capabilities"] = capabilities
    try:
        exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), capabilities.__dict__)

        app = types.ModuleType("app")
        exec(compile(result.generated_files["app.py"], "app.py", "exec"), app.__dict__)
        manifest = app.describe_application()
    finally:
        sys.modules.pop("apg_capabilities", None)

    assert manifest["ui_routes"]["/ops"]["name"] == "Dashboard"
    graph_edges = {
        (edge["source"], edge["relation"], edge["target"])
        for edge in manifest["composition_graph"]["edges"]
    }
    assert ("screen:OperationsWorkbench.Dashboard", "contains", "component:KpiStrip") in graph_edges
    assert ("component:KpiStrip", "filters", "component:LedgerTable") in graph_edges
    assert json.loads(json.dumps(manifest))["ui_routes"]["/ops"]["component"] == "Dashboard"


def test_generated_app_executes_capability_operations_over_http(tmp_path):
    result = APGCompiler().compile_string(CAPABILITY_SOURCE, "erp_ops.apg")
    assert result.success is True

    app_dir = tmp_path / "generated_erp_ops"
    app_dir.mkdir()
    _write_generated_files(app_dir, result.generated_files)

    smoke = subprocess.run(
        [sys.executable, "smoke_test.py"],
        cwd=app_dir,
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0, smoke.stderr
    smoke_report = json.loads(smoke.stdout)
    assert smoke_report["checks"]["capability_health"]["healthy"] is True

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    process = subprocess.Popen(
        [sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=app_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        for _attempt in range(80):
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
                    assert response.status == 200
                break
            except OSError:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
                time.sleep(0.05)
        else:
            raise AssertionError("generated app did not answer /health")

        with urllib.request.urlopen(f"{base_url}/finance/gl/journals", timeout=1) as response:
            screen_content_type = response.headers["Content-Type"]
            screen_html = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{base_url}/theme.css", timeout=1) as response:
            theme_content_type = response.headers["Content-Type"]
            theme_css = response.read().decode("utf-8")
        with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=1) as response:
            openapi = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/component.json", timeout=1) as response:
            component = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/streaming", timeout=1) as response:
            streaming = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/capabilities/GeneralLedger/streaming", timeout=1) as response:
            capability_streaming = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/capabilities/GeneralLedger/health", timeout=1) as response:
            capability_health = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base_url}/capabilities/health", timeout=1) as response:
            capability_health_report = json.loads(response.read().decode("utf-8"))

        request = urllib.request.Request(
            f"{base_url}/capabilities/GeneralLedger/rules/evaluate",
            data=json.dumps({
                "context": {"debits": 100, "credits": 90, "period": {"closed": False}}
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            denied = json.loads(response.read().decode("utf-8"))

        request = urllib.request.Request(
            f"{base_url}/rules/evaluate",
            data=json.dumps({
                "capability": "GeneralLedger",
                "context": {"debits": 100, "credits": 100, "period": {"closed": False}},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            allowed = json.loads(response.read().decode("utf-8"))

        request = urllib.request.Request(
            f"{base_url}/capabilities/GeneralLedger/configuration/resolve",
            data=json.dumps({"overrides": {"currency": "USD"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            resolved_config = json.loads(response.read().decode("utf-8"))

        request = urllib.request.Request(
            f"{base_url}/capabilities/GeneralLedger/configuration/validate",
            data=json.dumps({"configuration": {"posting": {"batch_size": 250}}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            config_validation = json.loads(response.read().decode("utf-8"))

        request = urllib.request.Request(
            f"{base_url}/capabilities/GeneralLedger/approval/plan",
            data=json.dumps({"context": {"amount": 1000}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            approval = json.loads(response.read().decode("utf-8"))
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    assert denied["decision"] == "deny"
    assert denied["matched_rules"] == ["balanced_journal"]
    assert allowed["decision"] == "allow"
    assert allowed["matched_rules"] == []
    assert screen_content_type.startswith("text/html")
    assert "GeneralLedger" in screen_html
    assert "JournalScreen" in screen_html
    assert "finance_ops" in screen_html
    assert 'href="/theme.css"' in screen_html
    assert "#126E82" in screen_html
    assert theme_content_type.startswith("text/css")
    assert "--apg-theme-finance-ops-accent: #126E82;" in theme_css
    assert "--apg-accent: var(--apg-theme-finance-ops-accent);" in theme_css
    assert component["kind"] == "apg.application"
    assert component["capabilities"] == ["GeneralLedger"]
    assert component["interfaces"]["theme"] == "/theme.css"
    assert component["ui_routes"]["/finance/gl/journals"]["component"] == "JournalScreen"
    assert component["streaming_processors"] == {"bytewax": ["GeneralLedger"]}
    assert "/component.json" in openapi["paths"]
    assert "/finance/gl/journals" in openapi["paths"]
    assert "/streaming" in openapi["paths"]
    assert "/theme.css" in openapi["paths"]
    assert "/capabilities/GeneralLedger/streaming" in openapi["paths"]
    assert "/capabilities/GeneralLedger/health" in openapi["paths"]
    assert "/capabilities/health" in openapi["paths"]
    assert "/capabilities/GeneralLedger/rules/evaluate" in openapi["paths"]
    assert "/capabilities/GeneralLedger/configuration/resolve" in openapi["paths"]
    assert "/capabilities/GeneralLedger/configuration/validate" in openapi["paths"]
    assert "/capabilities/GeneralLedger/approval/plan" in openapi["paths"]
    assert openapi["paths"]["/streaming"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StreamingTopology"
    }
    assert openapi["paths"]["/capabilities/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CapabilityHealthReport"
    }
    assert openapi["paths"]["/capabilities/GeneralLedger/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CapabilityHealth"
    }
    assert openapi["paths"]["/capabilities/GeneralLedger/rules/evaluate"]["post"]["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RuleEvaluationRequest"
    }
    assert openapi["paths"]["/capabilities/GeneralLedger/configuration/resolve"]["post"]["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/CapabilityConfigurationResponse"
    }
    assert openapi["paths"]["/capabilities/GeneralLedger/approval/plan"]["post"]["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ApprovalPlanRequest"
    }
    assert streaming["processor"] == "bytewax"
    assert streaming["processors"] == {"bytewax": ["GeneralLedger"]}
    assert streaming["states"] == {"ledger_posting_state": ["GeneralLedger"]}
    assert streaming["streams"]["GeneralLedger"]["state"] == "ledger_posting_state"
    assert capability_streaming["processor"] == "bytewax"
    assert capability_streaming["state"] == "ledger_posting_state"
    assert capability_health["healthy"] is True
    assert capability_health["status"] == "ok"
    assert capability_health["rules"]["names"] == ["posting_period_open", "balanced_journal"]
    assert capability_health_report["healthy"] is True
    assert capability_health_report["capabilities"]["GeneralLedger"]["streaming"]["processor"] == "bytewax"
    assert resolved_config == {
        "capability": "GeneralLedger",
        "configuration": {"currency": "USD", "fiscal_calendar": "monthly"},
    }
    assert config_validation["errors"] == []
    assert config_validation["warnings"] == ["GeneralLedger has undeclared configuration posting"]
    assert approval == {
        "capability": "GeneralLedger",
        "required": True,
        "levels": 2,
        "approvers": ["controller", "cfo"],
        "segregation_of_duties": False,
        "escalation": None,
    }


def test_generated_package_reexports_grouped_capability_descriptions(tmp_path):
    result = APGCompiler().compile_string(CAPABILITY_SOURCE, "erp_ops.apg")
    assert result.success is True

    package_dir = tmp_path / "erp_ops_generated"
    package_dir.mkdir()
    _write_generated_files(package_dir, result.generated_files)

    spec = importlib.util.spec_from_file_location(
        "erp_ops_generated",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["erp_ops_generated"] = module
    try:
        spec.loader.exec_module(module)
        component = module.component_manifest()
        grouped = module.describe_capabilities_by_erp_module()
        validation = module.validate_application()
        screens = module.capability_screens("GeneralLedger")
        theme = module.capability_theme("GeneralLedger")
        health = module.capability_health("GeneralLedger")
        rules = module.capability_rules("GeneralLedger")
        rule_decision = module.evaluate_capability_rules(
            "GeneralLedger",
            {"debits": 100, "credits": 90, "period": {"closed": False}},
        )
        config = module.capability_configuration("GeneralLedger", {"currency": "USD"})
        config_validation = module.validate_capability_configuration("GeneralLedger", config)
        approval = module.approval_plan("GeneralLedger", {"amount": 2500})
    finally:
        sys.modules.pop("erp_ops_generated", None)
        for name in list(sys.modules):
            if name.startswith("erp_ops_generated."):
                sys.modules.pop(name, None)

    assert grouped["finance"][0]["name"] == "GeneralLedger"
    assert component["capabilities"] == ["GeneralLedger"]
    assert "/finance/gl/journals" in component["interfaces"]["http"]["paths"]
    assert "list_capabilities" in component["interfaces"]["python"]["exports"]
    assert "capability_health" in component["interfaces"]["python"]["exports"]
    assert "capability_health_report" in component["interfaces"]["python"]["exports"]
    assert "capability_rules" in component["interfaces"]["python"]["exports"]
    assert "evaluate_capability_rules" in component["interfaces"]["python"]["exports"]
    assert "capability_configuration" in component["interfaces"]["python"]["exports"]
    assert "validate_capability_configuration" in component["interfaces"]["python"]["exports"]
    assert "approval_plan" in component["interfaces"]["python"]["exports"]
    assert module.capability_names_by_erp_module()["general_ledger"] == ["GeneralLedger"]
    assert module.capability_dependency_graph() == {"GeneralLedger": []}
    assert module.capability_streaming("GeneralLedger")["processor"] == "bytewax"
    assert module.streaming_processor_index() == {"bytewax": ["GeneralLedger"]}
    assert module.streaming_state_index() == {"ledger_posting_state": ["GeneralLedger"]}
    assert module.ui_route_index()["/finance/gl/journals"]["component"] == "JournalScreen"
    assert screens[0]["component"] == "JournalScreen"
    assert theme["tokens"]["accent"] == "#126E82"
    assert health["healthy"] is True
    assert health["status"] == "ok"
    assert [rule["name"] for rule in rules] == ["posting_period_open", "balanced_journal"]
    assert rule_decision["decision"] == "deny"
    assert rule_decision["matched_rules"] == ["balanced_journal"]
    assert config["currency"] == "USD"
    assert config_validation["errors"] == []
    assert approval["approvers"] == ["controller", "cfo"]
    assert module.theme_token("GeneralLedger", "accent") == "#126E82"
    assert len(module.african_language_codes()) >= 40
    assert "sw" in module.supported_language_codes()
    assert validation["valid"] is True
    assert "african_language_codes" in module.__all__
    assert "capability_screens" in module.__all__
    assert "capability_health" in module.__all__
    assert "capability_health_report" in module.__all__
    assert "capability_configuration" in module.__all__
    assert "capability_languages" in module.__all__
    assert "capability_rules" in module.__all__
    assert "capability_streaming" in module.__all__
    assert "capability_theme" in module.__all__
    assert "evaluate_capability_rules" in module.__all__
    assert "describe_capabilities_by_erp_module" in module.__all__
    assert "composition_graph" in module.__all__
    assert "approval_plan" in module.__all__
    assert "supported_language_codes" in module.__all__
    assert "streaming_state_index" in module.__all__
    assert "theme_token" in module.__all__
    assert "validate_capability_configuration" in module.__all__
    assert "validate_application" in module.__all__
    assert json.loads(json.dumps(grouped))["accounts_payable"][0]["name"] == "GeneralLedger"


def test_capability_dependency_planning_uses_provides_requires_contracts():
    result = APGCompiler().compile_string(DEPENDENCY_SOURCE, "dependencies.apg")

    assert result.success is True
    namespace = {}
    exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), namespace)

    assert namespace["provided_services"]() == {
        "audit_log": ["AuditLog"],
        "journal_entries": ["GeneralLedger"],
    }
    assert namespace["service_providers"]("audit_log") == ["AuditLog"]
    assert namespace["required_services"]("GeneralLedger") == ["audit_log"]
    assert namespace["capability_dependency_graph"]() == {
        "AuditLog": [],
        "GeneralLedger": ["AuditLog"],
    }
    assert namespace["unresolved_required_services"]() == {}
    assert namespace["capability_load_order"]() == {
        "order": ["AuditLog", "GeneralLedger"],
        "cycles": [],
        "unresolved": {},
    }
    assert namespace["validate_capability_dependencies"]() == {"errors": [], "warnings": []}


def test_generated_capability_i18n_rejects_unknown_language_codes():
    result = APGCompiler().compile_string(INVALID_I18N_SOURCE, "invalid_i18n.apg")

    assert result.success is True

    namespace = {}
    exec(compile(result.generated_files["apg_capabilities.py"], "apg_capabilities.py", "exec"), namespace)

    validation = namespace["validate_capability_i18n"]()
    assert validation["warnings"] == []
    assert validation["errors"] == [
        "InvalidLocalization unsupported language code zz",
        "InvalidLocalization unknown default language zz",
        "InvalidLocalization unknown fallback language qq",
        "InvalidLocalization fallback language qq is not supported",
    ]


def test_generated_app_cli_fails_when_validation_or_self_test_fails(tmp_path):
    result = APGCompiler().compile_string(INVALID_I18N_SOURCE, "invalid_i18n.apg")
    assert result.success is True

    app_dir = tmp_path / "invalid_i18n_app"
    app_dir.mkdir()
    _write_generated_files(app_dir, result.generated_files)

    validation = subprocess.run(
        [sys.executable, "app.py", "--validate"],
        cwd=app_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    self_test = subprocess.run(
        [sys.executable, "app.py", "--self-test"],
        cwd=app_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    validation_report = json.loads(validation.stdout)
    self_test_report = json.loads(self_test.stdout)

    assert validation.returncode == 1
    assert validation_report["valid"] is False
    assert "capability_i18n: InvalidLocalization unsupported language code zz" in validation_report["errors"]
    assert self_test.returncode == 1
    assert self_test_report["passed"] is False
    assert self_test_report["checks"]["validation"]["valid"] is False

    smoke = subprocess.run(
        [sys.executable, "smoke_test.py"],
        cwd=app_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    smoke_report = json.loads(smoke.stdout)

    assert smoke.returncode == 1
    assert smoke_report["passed"] is False
