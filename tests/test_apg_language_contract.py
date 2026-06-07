"""APG grammar contract coverage for platform language goals."""

from __future__ import annotations

import re
from pathlib import Path

from compiler.ast_builder import ASTBuilder
from compiler.compiler import APGCompiler
from compiler.parser import APGParser


GRAMMAR = Path(__file__).resolve().parents[1] / "spec" / "apg.g4"


def _rule_body(rule_name: str) -> str:
	text = GRAMMAR.read_text(encoding="utf-8")
	match = re.search(rf"^{rule_name}\n\s*:(.*?)\n\s*;", text, flags=re.MULTILINE | re.DOTALL)
	assert match, f"{rule_name} rule not found"
	return match.group(1)


def test_grammar_promotes_composable_capabilities_to_first_class_entities():
	entity_type = _rule_body("entity_type")

	for keyword in [
		"'capability'",
		"'capability_contract'",
		"'capability_pack'",
		"'composition'",
		"'contract'",
		"'rule_set'",
		"'guardrail'",
	]:
		assert keyword in entity_type

	for rule_name in [
		"capability_contract_block",
		"rule_engine_block",
		"ui_contract_block",
		"theme_contract_block",
	]:
		assert re.search(rf"^{rule_name}\b", GRAMMAR.read_text(encoding="utf-8"), flags=re.MULTILINE)


def test_parser_accepts_first_class_entity_keywords_from_grammar():
	parser = APGParser()
	builder = ASTBuilder()
	for source, expected_name, expected_type in [
		('twin Machine { sync: opc; }', "Machine", "digital_twin"),
		('screen Dashboard { route: "/"; }', "Dashboard", "screen"),
		('app ERP { }', "ERP", "app"),
		('flow Fulfillment { }', "Fulfillment", "flow"),
		('agent_runtime CodexRuntime { runner: codex; }', "CodexRuntime", "agent_runtime"),
	]:
		result = parser.parse_string(source, "<entity-keyword>")
		assert result["success"] is True, (source, [str(error) for error in result["errors"]])
		ast = builder.build_ast(result["parse_tree"], "<entity-keyword>")
		assert [(entity.name, entity.entity_type.value) for entity in ast.entities] == [
			(expected_name, expected_type)
		]

	compile_source = """
		twin Machine { }
		screen Dashboard { }
		app ERP { }
		flow Fulfillment { }
		agent_runtime CodexRuntime { }
	"""
	compile_result = APGCompiler().compile_string(compile_source, "entity_keywords.apg")
	assert compile_result.success is True, compile_result.errors
	namespace = {}
	exec(compile(compile_result.generated_files["app.py"], "app.py", "exec"), namespace)
	assert [
		(entity["name"], entity["type"])
		for entity in namespace["list_entities"]()
	] == [
		("Machine", "digital_twin"),
		("Dashboard", "screen"),
		("ERP", "app"),
		("Fulfillment", "flow"),
		("CodexRuntime", "agent_runtime"),
	]

	result = parser.parse_string("invalid_entity Broken { }", "<invalid>")
	assert result["success"] is False
	assert any("Unknown entity declaration" in str(error) for error in result["errors"])


def test_grammar_supports_rapid_erp_component_and_rule_composition():
	text = GRAMMAR.read_text(encoding="utf-8")
	entity_type = _rule_body("entity_type")
	erp_domain = _rule_body("erp_domain")
	component_member = _rule_body("erp_component_member")
	rule_member = _rule_body("rule_contract_member")

	for keyword in [
		"'erp_module'",
		"'erp_component'",
		"'ledger'",
		"'finance'",
		"'procurement'",
		"'inventory'",
		"'payroll'",
		"'fixed_assets'",
		"'project_accounting'",
	]:
		assert keyword in entity_type

	for domain in [
		"'general_ledger'",
		"'accounts_payable'",
		"'accounts_receivable'",
		"'purchase_orders'",
		"'supplier_management'",
		"'materials_planning'",
		"'time_attendance'",
		"'budgeting'",
		"'tax'",
	]:
		assert domain in erp_domain

	for contract in [
		"'data_model'",
		"'apis'",
		"'workflows'",
		"'rules'",
		"'approvals'",
		"'permissions'",
		"'audit'",
		"'effective_dates'",
		"'master_data'",
		"'ui'",
		"'theme'",
	]:
		assert contract in component_member

	for rule_field in [
		"'priority'",
		"'applies_to'",
		"'effective_from'",
		"'effective_to'",
		"'exception'",
		"'approval'",
		"'audit'",
	]:
		assert rule_field in rule_member

	for rule_name in [
		"erp_component_block",
		"erp_component_set",
		"erp_rule_set",
		"approval_contract",
		"permission_contract",
		"audit_contract",
		"effective_date_contract",
		"master_data_contract",
	]:
		assert re.search(rf"^{rule_name}\b", text, flags=re.MULTILINE)


def test_grammar_promotes_ai_agent_composition_to_first_class_language():
	entity_type = _rule_body("entity_type")
	runtime_ref = _rule_body("agent_runtime_ref")

	for keyword in [
		"'agent_runtime'",
		"'agent_tool'",
		"'agent_memory'",
		"'agent_handoff'",
		"'model'",
		"'tool'",
		"'memory_store'",
		"'handoff'",
	]:
		assert keyword in entity_type

	for runtime in ["'codex'", "'claude_code'", "'claude'", "'opencode'", "'open_code'", "'pi'"]:
		assert runtime in runtime_ref


def test_grammar_keeps_streaming_runtime_bytewax_native():
	processor = _rule_body("stream_processor").lower()

	assert "'bytewax'" in processor
	assert "'bytewax_streams'" in processor


def test_grammar_advertises_python_not_framework_targets():
	ui_shell = _rule_body("ui_shell")
	runtime_backend = _rule_body("runtime_backend")

	assert "'python'" in ui_shell
	assert "'python'" in runtime_backend
	for framework in ["'flask_appbuilder'", "'fastapi'", "'django'"]:
		assert framework not in ui_shell


def test_grammar_supports_screen_composition_and_relationships():
	text = GRAMMAR.read_text(encoding="utf-8")
	entity_member = _rule_body("entity_member")
	ui_member = _rule_body("ui_contract_member")
	screen_member = _rule_body("screen_contract_member")
	screen_relationship_member = _rule_body("screen_relationship_member")

	assert "screen_contract_block" in entity_member
	assert "'screens'" in ui_member

	for rule_name in [
		"screen_contract_block",
		"screen_set",
		"screen_binding",
		"screen_contract",
		"screen_element_list",
		"screen_event_list",
		"screen_relationship_list",
		"screen_relation_edge",
	]:
		assert re.search(rf"^{rule_name}\b", text, flags=re.MULTILINE)

	for field in [
		"'route'",
		"'layout'",
		"'contains'",
		"'composes'",
		"'binds'",
		"'actions'",
		"'events'",
		"'relationships'",
		"'permissions'",
		"'rules'",
		"'theme'",
	]:
		assert field in screen_member

	for field in ["'from'", "'to'", "'via'", "'type'", "'when'"]:
		assert field in screen_relationship_member


def test_grammar_includes_broad_african_language_codes():
	# language_code now uses member_name | STRING, accepting any identifier (including
	# all language codes like 'sw', 'af', 'en', etc.) rather than an explicit enumeration.
	# This avoids creating ~100 implicit keyword tokens that shadow IDENTIFIER.
	language_code = _rule_body("language_code")
	assert "member_name" in language_code
	assert "STRING" in language_code
