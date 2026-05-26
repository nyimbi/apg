"""Executable semantic checks for central APG compiler objects."""

from compiler.ast_builder import (
	BlockStatement,
	EntityDeclaration,
	EntityType,
	Expression,
	IdentifierExpression,
	LiteralExpression,
	MethodDeclaration,
	ModuleDeclaration,
	Parameter,
	ReturnStatement,
	Statement,
	TypeAnnotation,
)
from compiler.semantic_analyzer import SemanticAnalyzer
from capabilities import discover_subcapabilities


def _analyze_method(method: MethodDeclaration):
	module = ModuleDeclaration(
		name="semantic_runtime",
		entities=[
			EntityDeclaration(
				entity_type=EntityType.FORM,
				name="SemanticProbe",
				methods=[method],
			)
		],
	)
	return SemanticAnalyzer().analyze(module)


def test_statement_and_expression_base_nodes_are_explicit_ast_categories():
	assert Statement().node_category == "statement"
	assert Expression().node_category == "expression"


def test_semantic_analyzer_rejects_literal_return_type_mismatch():
	result = _analyze_method(
		MethodDeclaration(
			"count",
			return_type=TypeAnnotation("int"),
			body=BlockStatement([
				ReturnStatement(LiteralExpression("not a number", "string"))
			]),
		)
	)

	assert result["success"] is False
	assert any("returns str, expected int" in str(error) for error in result["errors"])


def test_semantic_analyzer_accepts_parameter_return_type_match():
	result = _analyze_method(
		MethodDeclaration(
			"echo",
			parameters=[Parameter("value", TypeAnnotation("str"))],
			return_type=TypeAnnotation("str"),
			body=BlockStatement([
				ReturnStatement(IdentifierExpression("value"))
			]),
		)
	)

	assert result["success"] is True
	assert result["errors"] == []


def test_semantic_analyzer_rejects_void_method_returning_value():
	result = _analyze_method(
		MethodDeclaration(
			"reset",
			return_type=TypeAnnotation("void"),
			body=BlockStatement([
				ReturnStatement(LiteralExpression(True, "boolean"))
			]),
		)
	)

	assert result["success"] is False
	assert any("returns bool, expected void" in str(error) for error in result["errors"])


def test_subcapability_discovery_records_import_failures():
	assert discover_subcapabilities("definitely_missing_capability") == []
	assert discover_subcapabilities.last_error is not None
	assert "Unknown capability" in discover_subcapabilities.last_error
