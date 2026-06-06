"""Tests for the rule expression parser (Phase 2 of grammar analysis resolution)."""

from __future__ import annotations

import pytest
from compiler.rule_expr import (
	parse_rule_expr, extract_fields, validate_rule_fields, expr_to_dict,
	CompareNode, MissingNode, InNode, AndNode, OrNode, RuleExprParseError,
)


class TestParsing:
	def test_simple_comparison(self):
		node = parse_rule_expr("amount > 50000")
		assert isinstance(node, CompareNode)
		assert node.field == "amount"
		assert node.op == ">"
		assert node.value == 50000

	def test_equality_alias(self):
		node = parse_rule_expr("status = active")
		assert isinstance(node, CompareNode)
		assert node.op == "=="
		assert node.value == "active"

	def test_not_equal_alias(self):
		node = parse_rule_expr("status <> inactive")
		assert isinstance(node, CompareNode)
		assert node.op == "!="

	def test_missing(self):
		node = parse_rule_expr("bank_account missing")
		assert isinstance(node, MissingNode)
		assert node.field == "bank_account"
		assert not node.negated

	def test_not_missing(self):
		node = parse_rule_expr("bank_account not missing")
		assert isinstance(node, MissingNode)
		assert node.negated

	def test_in_list(self):
		node = parse_rule_expr("agency in [FDA, EMA, HC]")
		assert isinstance(node, InNode)
		assert node.field == "agency"
		assert "FDA" in node.values

	def test_and_expression(self):
		node = parse_rule_expr("amount > 50000 and stage == qualification")
		assert isinstance(node, AndNode)
		assert isinstance(node.left, CompareNode)
		assert isinstance(node.right, CompareNode)
		assert node.left.field == "amount"
		assert node.right.field == "stage"

	def test_and_case_insensitive(self):
		node = parse_rule_expr("amount > 50000 AND stage == qualification")
		assert isinstance(node, AndNode)

	def test_or_expression(self):
		node = parse_rule_expr("status == urgent or priority == high")
		assert isinstance(node, OrNode)

	def test_boolean_value(self):
		node = parse_rule_expr("approved == false")
		assert isinstance(node, CompareNode)
		assert node.value is False

	def test_empty_returns_none(self):
		assert parse_rule_expr("") is None
		assert parse_rule_expr("   ") is None

	def test_complex_real_condition(self):
		node = parse_rule_expr("ae_type = serious_adverse_event AND within_24h = false")
		assert isinstance(node, AndNode)

	def test_missing_with_and(self):
		node = parse_rule_expr("allergy_detected == true and override_reason missing")
		assert isinstance(node, AndNode)
		assert isinstance(node.right, MissingNode)
		assert node.right.field == "override_reason"


class TestFieldExtraction:
	def test_single_field(self):
		node = parse_rule_expr("amount > 50000")
		assert extract_fields(node) == {"amount"}

	def test_multiple_fields(self):
		node = parse_rule_expr("amount > 50000 and stage == qualification")
		assert extract_fields(node) == {"amount", "stage"}

	def test_missing_field(self):
		node = parse_rule_expr("bank_account missing")
		assert extract_fields(node) == {"bank_account"}

	def test_in_field(self):
		node = parse_rule_expr("agency in [FDA, EMA]")
		assert extract_fields(node) == {"agency"}


class TestFieldValidation:
	def test_known_fields_no_warnings(self):
		w = validate_rule_fields("amount > 50000 and stage == q", {"amount", "stage"})
		assert w == []

	def test_unknown_field_produces_warning(self):
		w = validate_rule_fields("amount > 50000 and stage == q", {"amount"})
		assert len(w) == 1
		assert "stage" in w[0]

	def test_empty_condition_no_warnings(self):
		assert validate_rule_fields("", {"amount"}) == []

	def test_parse_error_produces_warning(self):
		w = validate_rule_fields(">>> invalid <<<", set())
		assert len(w) >= 1


class TestSerialisation:
	def test_compare_node_to_dict(self):
		node = parse_rule_expr("amount > 50000")
		d = expr_to_dict(node)
		assert d == {"type": "compare", "field": "amount", "op": ">", "value": 50000}

	def test_and_node_to_dict(self):
		node = parse_rule_expr("amount > 50000 and stage == q")
		d = expr_to_dict(node)
		assert d["type"] == "and"
		assert d["left"]["field"] == "amount"
		assert d["right"]["field"] == "stage"

	def test_dict_is_json_serializable(self):
		import json
		node = parse_rule_expr("amount > 50000 and status missing")
		d = expr_to_dict(node)
		# Must not raise
		json.dumps(d)


class TestRealConditions:
	"""Test conditions extracted directly from examples/ .apg files."""

	_CONDITIONS = [
		"amount > 50000",
		"amount > 50000 and stage == qualification",
		"bank_account missing",
		"allergy_detected == true and override_reason missing",
		"ae_type = serious_adverse_event AND within_24h = false",
		"aml_status == flagged",
		"approved == false",
		"amount > approval_threshold_high",
		"annual_rent_gt == 5000000 and cfo_approved != true",
		"applicant_county != service_county",
		"batch_status != released",
		"budget_code missing",
	]

	@pytest.mark.parametrize("cond", _CONDITIONS)
	def test_real_condition_parses(self, cond: str):
		node = parse_rule_expr(cond)
		assert node is not None
		assert extract_fields(node)
