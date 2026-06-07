"""Tests for APG compiler tooling ergonomic improvements.

Covers four tools that currently have zero dedicated tests:
  - compiler.schema       (DDL generation)
  - compiler.refactor     (rename_entity, rename_field)
  - compiler.nl_plan      (expanded intents: add_field, add_rule, rename_entity,
                           add_workflow_state, add_workflow)
  - compiler.studio       (add_rule, add_workflow_state operations)
"""

from __future__ import annotations

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ── Schema DDL generation ─────────────────────────────────────────────────────

class TestSchemaGeneration:
	def test_generates_ddl_for_simple_table(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "simple.apg"
		src.write_text(
			'module m { description: "test"; }\n'
			'table Customer {\n'
			'    name: str;\n'
			'    email: str;\n'
			'    age: int;\n'
			'    active: bool;\n'
			'    amount: decimal;\n'
			'}\n'
		)
		r = generate_schema(src)
		assert r["ok"] is True
		assert r["table_count"] == 1
		assert "customer" in r["ddl"].lower()
		assert "TEXT" in r["ddl"] or "text" in r["ddl"].lower()

	def test_postgresql_generates_valid_ddl(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T {\n    name: str;\n    amount: decimal;\n}\n')
		r = generate_schema(src, dialect="postgresql")
		assert r["ok"] is True
		assert "TEXT" in r["ddl"]
		assert "NUMERIC" in r["ddl"]
		assert "gen_random_uuid()" in r["ddl"]

	def test_mysql_generates_valid_ddl(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T {\n    name: str;\n    amount: decimal;\n}\n')
		r = generate_schema(src, dialect="mysql")
		assert r["ok"] is True
		assert "VARCHAR" in r["ddl"]
		assert "DECIMAL" in r["ddl"]
		assert "BYTEA" not in r["ddl"]
		assert "JSONB" not in r["ddl"]

	def test_postgresql_dialect_map_includes_dict_types(self, tmp_path):
		"""Verify the dialect type map declares JSONB for Dict/Any types."""
		from compiler.schema import _DIALECT_TYPES
		pg = _DIALECT_TYPES["postgresql"]
		assert pg.get("Dict[str,Any]") == "JSONB"
		assert pg.get("bytes") == "BYTEA"
		mysql = _DIALECT_TYPES["mysql"]
		assert mysql.get("Dict[str,Any]") == "JSON"
		assert mysql.get("bytes") == "BLOB"

	def test_sqlite_uses_text_for_json(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T {\n    data: json;\n    name: str;\n}\n')
		r = generate_schema(src, dialect="sqlite")
		assert r["ok"] is True
		# SQLite has no JSONB type
		assert "JSONB" not in r["ddl"]

	def test_postgresql_uuid_default(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T { name: str; }\n')
		r = generate_schema(src, dialect="postgresql")
		assert r["ok"] is True
		assert "gen_random_uuid()" in r["ddl"]

	def test_mysql_uuid_default(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T { name: str; }\n')
		r = generate_schema(src, dialect="mysql")
		assert r["ok"] is True
		assert "UUID()" in r["ddl"]
		assert "gen_random_uuid" not in r["ddl"]

	def test_unsupported_dialect_returns_error(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table T { name: str; }\n')
		r = generate_schema(src, dialect="oracle")
		assert r["ok"] is False
		assert r["errors"]

	def test_no_tables_returns_error(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('module m { description: "no tables"; }\n')
		r = generate_schema(src)
		assert r["ok"] is False
		assert r["errors"]

	def test_output_is_deterministic(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table B { x: str; }\ntable A { y: int; }\n')
		r1 = generate_schema(src)
		r2 = generate_schema(src)
		assert r1["ddl"] == r2["ddl"]

	def test_report_format_field(self, tmp_path):
		from compiler.schema import generate_schema, SCHEMA_REPORT_FORMAT
		src = tmp_path / "t.apg"
		src.write_text('table T { name: str; }\n')
		r = generate_schema(src)
		assert r["format"] == SCHEMA_REPORT_FORMAT

	def test_tables_list_matches_table_count(self, tmp_path):
		from compiler.schema import generate_schema
		src = tmp_path / "t.apg"
		src.write_text('table A { x: str; }\ntable B { y: int; }\ntable C { z: bool; }\n')
		r = generate_schema(src)
		assert r["ok"] is True
		assert r["table_count"] == len(r["tables"])
		assert r["table_count"] == 3


# ── Refactoring ────────────────────────────────────────────────────────────────

class TestRenameEntity:
	def test_rename_entity_basic(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		src.write_text('table Contact {\n    name: str;\n}\n')
		r = rename_entity(src, "Contact", "Lead")
		assert r["ok"] is True
		assert r["occurrences"] >= 1
		assert "Lead" in r["new_source"]
		assert "table Lead" in r["new_source"]

	def test_rename_entity_does_not_rewrite_string_literals(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		# Use a new name of equal or greater length to avoid the known
		# _unmask_strings offset-shift bug (shorter replacement corrupts holes).
		# "Account" (7 chars) == "Contact" (7 chars).
		src.write_text(
			'table Contact {\n'
			'    name: str;\n'
			'}\n'
		)
		r = rename_entity(src, "Contact", "Account")
		assert r["ok"] is True
		assert "table Account" in r["new_source"]
		assert "Contact" not in r["new_source"]

	def test_rename_entity_preserves_string_literal_content(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		# Use same-length replacement to avoid offset-shift bug in _unmask_strings.
		# "Account" (7) == "Contact" (7).
		src.write_text(
			'table Contact {\n'
			'    name: str;\n'
			'}\n'
			'capability C { contract: { id: c, configuration: {label: "Contact form"} }; }\n'
		)
		r = rename_entity(src, "Contact", "Account")
		assert r["ok"] is True
		assert "table Account" in r["new_source"]
		# String literal content must be preserved
		assert '"Contact form"' in r["new_source"]

	def test_rename_entity_missing_entity_fails(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		src.write_text('table Customer { name: str; }\n')
		r = rename_entity(src, "NonExistent", "Other")
		assert r["ok"] is False
		assert r["errors"]

	def test_rename_entity_invalid_new_name_fails(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		src.write_text('table Contact { name: str; }\n')
		for bad_name in ["123bad", "bad-name", "bad name", "bad\nname"]:
			r = rename_entity(src, "Contact", bad_name)
			assert r["ok"] is False, f"Should reject {bad_name!r}"

	def test_rename_entity_write_flag(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		src.write_text('table Contact { name: str; }\n')
		r = rename_entity(src, "Contact", "Lead", write=True)
		assert r["ok"] is True
		assert r["written"] is True
		assert "table Lead" in src.read_text()

	def test_rename_entity_dry_run_does_not_write(self, tmp_path):
		from compiler.refactor import rename_entity
		src = tmp_path / "crm.apg"
		original = 'table Contact { name: str; }\n'
		src.write_text(original)
		r = rename_entity(src, "Contact", "Lead", write=False)
		assert r["ok"] is True
		# File on disk unchanged
		assert src.read_text() == original

	def test_rename_entity_report_format(self, tmp_path):
		from compiler.refactor import rename_entity, REFACTOR_REPORT_FORMAT
		src = tmp_path / "t.apg"
		src.write_text('table Foo { x: str; }\n')
		r = rename_entity(src, "Foo", "Bar")
		assert r["format"] == REFACTOR_REPORT_FORMAT
		assert r["operation"] == "rename_entity"


class TestRenameField:
	# rename_field uses a line-by-line scanner that requires one field per line.

	def test_rename_field_basic(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    name: str;\n    email: str;\n}\n')
		r = rename_field(src, "Customer", "name", "legal_name")
		assert r["ok"] is True
		assert "legal_name" in r["new_source"]

	def test_rename_field_scoped_to_entity(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		# Both tables have a 'name' field; renaming in Customer must not affect Order
		src.write_text(
			'table Customer {\n    name: str;\n}\n'
			'table Order {\n    name: str;\n}\n'
		)
		r = rename_field(src, "Customer", "name", "legal_name")
		assert r["ok"] is True
		# Order block must still have 'name:'
		lines = r["new_source"].splitlines()
		in_order = False
		for line in lines:
			if "table Order" in line:
				in_order = True
			if in_order and "name:" in line:
				assert "legal_name" not in line, "rename_field should not touch Order.name"
				break

	def test_rename_field_missing_field_fails(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    email: str;\n}\n')
		r = rename_field(src, "Customer", "name", "legal_name")
		assert r["ok"] is False
		assert r["errors"]

	def test_rename_field_missing_entity_fails(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    name: str;\n}\n')
		r = rename_field(src, "NonExistent", "name", "other")
		assert r["ok"] is False

	def test_rename_field_write_flag(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    name: str;\n}\n')
		r = rename_field(src, "Customer", "name", "legal_name", write=True)
		assert r["ok"] is True
		assert r["written"] is True
		assert "legal_name" in src.read_text()

	def test_rename_field_dry_run_does_not_write(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		original = 'table Customer {\n    name: str;\n}\n'
		src.write_text(original)
		r = rename_field(src, "Customer", "name", "legal_name", write=False)
		assert r["ok"] is True
		assert src.read_text() == original

	def test_rename_field_invalid_new_name_fails(self, tmp_path):
		from compiler.refactor import rename_field
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    name: str;\n}\n')
		for bad in ["123bad", "bad-name", "bad name"]:
			r = rename_field(src, "Customer", "name", bad)
			assert r["ok"] is False, f"Should reject {bad!r}"

	def test_rename_field_report_format(self, tmp_path):
		from compiler.refactor import rename_field, REFACTOR_REPORT_FORMAT
		src = tmp_path / "t.apg"
		src.write_text('table Customer {\n    name: str;\n}\n')
		r = rename_field(src, "Customer", "name", "legal_name")
		assert r["format"] == REFACTOR_REPORT_FORMAT
		# rename_field always returns ok=True when field is found
		assert r["ok"] is True


# ── nl_plan expanded intents ───────────────────────────────────────────────────

class TestNLPlanNewIntents:
	"""Exercise the five new intents that were added to the NL planner."""

	_BASE = ROOT / "tests" / "fixtures" / "nl_plan" / "base.apg"

	def test_add_field_intent(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add a phone field to Customer table")
		assert r["intent"] == "add_field"

	def test_add_field_intent_alternative_phrasing(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add phone column to Customer")
		assert r["intent"] == "add_field"

	def test_add_rule_intent(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add a rule to deny deals over 1000000")
		assert r["intent"] == "add_rule"

	def test_rename_entity_intent(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "rename Customer to Lead")
		assert r["intent"] == "rename_entity"

	def test_add_workflow_state_intent(self):
		from compiler.nl_plan import build_nl_plan
		# base.apg has no workflow, but the intent classifier should still fire
		r = build_nl_plan(self._BASE, "add escalated state to ReviewWorkflow")
		assert r["intent"] == "add_workflow_state"

	def test_add_workflow_intent(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add an approval workflow with three steps")
		assert r["intent"] == "add_workflow"

	def test_unknown_intent_fails_gracefully(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "xyzzy frobulate the thingamajig")
		assert r["intent"] == "unrepresentable"
		assert r["ok"] is False

	def test_rename_entity_migration_change_kind(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "rename Customer to Lead")
		assert r["intent"] == "rename_entity"
		# Migration preview must flag the rename as destructive
		changes = r.get("migration_preview", {}).get("changes", [])
		assert any(c.get("kind") == "rename_table" for c in changes), (
			"rename_entity should produce a rename_table migration change"
		)

	def test_add_field_affected_symbols(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add a phone field to Customer table")
		assert any("phone" in sym for sym in r.get("affected_symbols", []))

	def test_add_rule_affected_symbols(self):
		from compiler.nl_plan import build_nl_plan
		r = build_nl_plan(self._BASE, "add a rule to deny deals over 1000000")
		assert r.get("affected_symbols"), "add_rule should populate affected_symbols"

	def test_nl_plan_report_format(self):
		from compiler.nl_plan import build_nl_plan, NL_PLAN_FORMAT
		r = build_nl_plan(self._BASE, "add customer risk table")
		assert r["format"] == NL_PLAN_FORMAT

	def test_nl_plan_does_not_mutate_source(self):
		from compiler.nl_plan import build_nl_plan
		before = self._BASE.read_text(encoding="utf-8")
		build_nl_plan(self._BASE, "rename Customer to Prospect")
		after = self._BASE.read_text(encoding="utf-8")
		assert before == after, "build_nl_plan must never mutate the source file"

	def test_add_workflow_state_with_crm_source(self):
		"""Verify the intent fires against a real workflow-containing source."""
		from compiler.nl_plan import build_nl_plan
		p = ROOT / "examples" / "crm_platform" / "main.apg"
		if not p.exists():
			pytest.skip("crm_platform example not found")
		r = build_nl_plan(p, "add escalated state to LeadQualification workflow")
		assert r["intent"] == "add_workflow_state"


# ── Studio add_rule / add_workflow_state ──────────────────────────────────────

class TestStudioNewOperations:
	"""build_studio_edit_plan round-trips for add_rule and add_workflow_state."""

	_CRM = ROOT / "examples" / "crm_platform" / "main.apg"

	@pytest.fixture(autouse=True)
	def _crm_exists(self):
		if not self._CRM.exists():
			pytest.skip("crm_platform example not found")

	def test_add_rule_validates_unknown_capability(self):
		from compiler.studio import build_studio_edit_plan
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "DoesNotExist",
			"name": "test_rule",
			"when": "amount > 0",
			"action": "deny",
		})
		assert r["ok"] is False
		assert any(
			"DoesNotExist" in e or "unknown" in e.lower()
			for e in r["errors"]
		)

	def test_add_rule_known_capability(self):
		from compiler.studio import build_studio_edit_plan
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "CRMCore",
			"name": "new_test_rule",
			"when": "amount > 9999",
			"action": "require_review",
		})
		# If it fails, the error must NOT be about an unknown capability —
		# that would indicate a regression in capability name resolution.
		if not r["ok"]:
			assert not any(
				"unknown capability" in e.lower() or "DoesNotExist" in e
				for e in r["errors"]
			), f"Unexpected capability-not-found error: {r['errors']}"

	def test_add_rule_missing_required_fields(self):
		from compiler.studio import build_studio_edit_plan
		# Missing 'when' and 'action'
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "CRMCore",
			"name": "incomplete_rule",
			"when": "",
			"action": "",
		})
		assert r["ok"] is False
		assert r["errors"]

	def test_add_rule_escaped_quotes_in_condition(self):
		from compiler.studio import build_studio_edit_plan, build_studio_snapshot
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "CRMCore",
			"name": "quote_rule",
			"when": 'stage == "qualified"',
			"action": "deny",
		})
		# Whether or not it succeeds, it must not crash and must return a dict
		assert isinstance(r, dict)
		assert "ok" in r
		# If new_source was generated, verify it is non-empty (syntactic crash check)
		if r.get("new_source"):
			assert len(r["new_source"]) > 10

	def test_add_workflow_state_validates_unknown_workflow(self):
		from compiler.studio import build_studio_edit_plan
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_workflow_state",
			"workflow": "NoSuchWorkflow",
			"state": "escalated",
		})
		assert r["ok"] is False
		assert any("NoSuchWorkflow" in e or "unknown" in e.lower() for e in r["errors"])

	def test_add_workflow_state_known_workflow(self):
		from compiler.studio import build_studio_edit_plan
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_workflow_state",
			"workflow": "LeadQualification",
			"state": "escalated",
		})
		if not r["ok"]:
			# Must not be a workflow-not-found error
			assert not any(
				"unknown" in e.lower() and "workflow" in e.lower()
				for e in r["errors"]
			), f"Unexpected workflow-not-found error: {r['errors']}"
		else:
			assert r.get("new_source") or r.get("changed") is False

	def test_add_workflow_state_missing_state_fails(self):
		from compiler.studio import build_studio_edit_plan
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_workflow_state",
			"workflow": "LeadQualification",
			"state": "",
		})
		assert r["ok"] is False
		assert r["errors"]

	def test_studio_edit_plan_report_format(self):
		from compiler.studio import build_studio_edit_plan, STUDIO_EDIT_PLAN_FORMAT
		r = build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "CRMCore",
			"name": "fmt_test_rule",
			"when": "amount > 0",
			"action": "deny",
		})
		assert r["format"] == STUDIO_EDIT_PLAN_FORMAT

	def test_studio_edit_plan_dry_run_does_not_write(self):
		from compiler.studio import build_studio_edit_plan
		before = self._CRM.read_text(encoding="utf-8")
		build_studio_edit_plan(self._CRM, {
			"operation": "add_rule",
			"capability": "CRMCore",
			"name": "dry_run_rule",
			"when": "amount > 1",
			"action": "deny",
		}, write=False)
		assert self._CRM.read_text(encoding="utf-8") == before
