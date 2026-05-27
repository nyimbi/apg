"""Executable compiler baseline for first-class APG capability declarations."""

from __future__ import annotations

from compiler.ast_builder import ASTBuilder, CapabilityDeclaration
from compiler.compiler import APGCompiler
from compiler.parser import APGParser
from compiler.semantic_analyzer import SemanticAnalyzer


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
    assert "CAPABILITIES" in runtime
    assert "CapabilitySpec" in runtime
    assert "GeneralLedger" in runtime
    assert "'erp_modules': ['finance', 'general_ledger', 'accounts_payable']" in runtime
    assert "'supported_languages': ['en', 'sw', 'ha', 'yo', 'zu']" in runtime
    assert "'processor': 'bytewax'" in runtime

    namespace = {}
    exec(compile(runtime, "apg_capabilities.py", "exec"), namespace)

    assert namespace["list_capabilities"]() == ["GeneralLedger"]
    capability = namespace["get_capability"]("GeneralLedger")
    assert capability.provides == ["journal_entries", "chart_of_accounts", "financial_periods"]
    assert namespace["capabilities_by_erp_module"]()["general_ledger"][0].name == "GeneralLedger"
    assert namespace["provided_services"]()["journal_entries"] == ["GeneralLedger"]

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

    graph = namespace["composition_graph"]()
    graph_edges = {(edge["source"], edge["relation"], edge["target"]) for edge in graph["edges"]}
    assert ("capability:GeneralLedger", "has_screen", "screen:GeneralLedger.Journals") in graph_edges
    assert ("screen:GeneralLedger.Journals", "renders", "component:JournalScreen") in graph_edges
    assert ("capability:GeneralLedger", "belongs_to", "erp_module:general_ledger") in graph_edges
    assert ("component:posting", "binds_to", "service:journal_entries") in graph_edges

    validation = namespace["validate_capability_contracts"]()
    assert validation["errors"] == []
    assert validation["warnings"] == ["GeneralLedger requires external service audit_log"]
