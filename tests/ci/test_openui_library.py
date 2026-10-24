# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""Conformance tests for the APG OpenUI library and prompt generator.

Covers:
  1. base and entity libraries are conformant (spec 3.1)
  2. validate_library rejects each defect it claims to catch — one canary per rule
  3. positional argument order equals schema property key order (spec 2.3)
  4. prompt generation is deterministic and complete
  5. entity specialisation maps APG types, drops implicit fields, honours limits
  6. the serialized spec is JSON-safe with no dangling component refs
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from compiler.openui_library import (
	BASE_COMPONENTS,
	COMPONENT_REF,
	Component,
	FunctionSpec,
	LibrarySpec,
	Prop,
	build_library,
	validate_library,
)
from compiler.openui_prompt import generate_system_prompt, prompt_report
from compiler.semantic_model import build_semantic_model


REPO_ROOT = Path(__file__).resolve().parents[2]
CLINIC = REPO_ROOT / "examples" / "clinic.apg"


@pytest.fixture(scope="module")
def clinic_model() -> dict:
	return build_semantic_model(CLINIC)


@pytest.fixture(scope="module")
def entity_library(clinic_model: dict) -> LibrarySpec:
	return build_library(clinic_model, entities=["Patient", "Appointment"])


# ---------------------------------------------------------------------------
# 1. Conformance of the shipped libraries
# ---------------------------------------------------------------------------


def test_base_library_is_conformant():
	assert validate_library(build_library()) == []


def test_entity_library_is_conformant(entity_library: LibrarySpec):
	assert validate_library(entity_library) == []


def test_every_shipped_component_has_a_description():
	for component in BASE_COMPONENTS:
		assert component.description.strip(), component.name


# ---------------------------------------------------------------------------
# 2. Canaries — each rule must reject its own defect
# ---------------------------------------------------------------------------


def test_validation_rejects_required_after_optional():
	broken = LibrarySpec(
		components=(
			Component("Page", "Root", (Prop("title", "string"), Prop("children", type="any", required=True))),
		),
		root="Page",
	)
	errors = validate_library(broken)
	assert any("follows an optional one" in error for error in errors), errors


def test_validation_accepts_required_with_default_after_optional():
	# A required prop carrying a default counts as optional (spec 3.1), so this is valid.
	ok = LibrarySpec(
		components=(
			Component(
				"Page",
				"Root",
				(Prop("title", "string"), Prop("count", "number", required=True, has_default=True, default=0)),
			),
		),
		root="Page",
	)
	assert validate_library(ok) == []


def test_validation_rejects_reserved_component_name():
	broken = LibrarySpec(components=(Component("Action", "Clashes with an expression form"),), root="Action")
	assert any("reserved name" in error for error in validate_library(broken))


def test_validation_rejects_lowercase_component_name():
	broken = LibrarySpec(components=(Component("card", "Lowercase"),), root="card")
	assert any("uppercase" in error for error in validate_library(broken))


def test_validation_rejects_unknown_root():
	broken = LibrarySpec(components=(Component("Page", "Root"),), root="Missing")
	assert any("not a component" in error for error in validate_library(broken))


def test_validation_rejects_reserved_function_name():
	broken = LibrarySpec(
		components=(Component("Page", "Root"),),
		root="Page",
		functions=(FunctionSpec("Sort", "Shadows a built-in"),),
	)
	assert any("reserved name" in error for error in validate_library(broken))


def test_validation_rejects_unknown_group_member():
	broken = LibrarySpec(
		components=(Component("Page", "Root"),),
		root="Page",
		groups=(("Layout", ("Page", "Ghost"), ""),),
	)
	assert any("unknown component" in error for error in validate_library(broken))


# ---------------------------------------------------------------------------
# 3. Positional order is the schema key order
# ---------------------------------------------------------------------------


def test_signature_order_matches_schema_order(entity_library: LibrarySpec):
	spec = entity_library.to_spec()
	for name, entry in spec["components"].items():
		signature = entry["signature"]
		params = signature[signature.index("(") + 1 : -1]
		declared = [prop.name for prop in entity_library.component(name).props]
		if not params:
			assert declared == []
			continue
		# Prop names are the token before `?` or `:` in each top-level argument.
		assert declared[0] in params.split(",")[0], name
		assert len(declared) == len(spec["schema"]["$defs"][name]["properties"]), name


def test_required_props_come_first_in_schema():
	for component in build_library().components:
		required = [prop.name for prop in component.props if prop.required]
		declared = [prop.name for prop in component.props]
		if required:
			assert declared[: len(required)] == required, component.name


# ---------------------------------------------------------------------------
# 4. Prompt generation
# ---------------------------------------------------------------------------


def test_prompt_is_deterministic(entity_library: LibrarySpec):
	first = generate_system_prompt(entity_library, tools=["list_patients"])
	second = generate_system_prompt(entity_library, tools=["list_patients"])
	assert first == second


def test_prompt_lists_every_component(entity_library: LibrarySpec):
	prompt = generate_system_prompt(entity_library)
	report = prompt_report(entity_library, prompt)
	assert report["ok"] is True, report["missing_signatures"]
	assert report["components"] == len(entity_library.components)


def test_prompt_report_canary(entity_library: LibrarySpec):
	"""The report must flag a signature the prompt does not contain."""
	report = prompt_report(entity_library, "nothing here")
	assert report["ok"] is False
	assert len(report["missing_signatures"]) == len(entity_library.components)


def test_prompt_mentions_root_and_positional_args(entity_library: LibrarySpec):
	prompt = generate_system_prompt(entity_library)
	assert "`root`" in prompt
	assert "positional" in prompt.lower()
	assert entity_library.root in prompt


def test_prompt_omits_data_sections_without_tools(entity_library: LibrarySpec):
	prompt = generate_system_prompt(entity_library, tools=None)
	assert "### Queries and mutations" not in prompt
	assert "### State variables" not in prompt


def test_prompt_includes_data_sections_with_tools(entity_library: LibrarySpec):
	prompt = generate_system_prompt(entity_library, tools=["list_patients"])
	assert "### Queries and mutations" in prompt
	assert "### State variables" in prompt
	assert "`list_patients`" in prompt


def test_prompt_includes_preamble_and_extra_rules(entity_library: LibrarySpec):
	prompt = generate_system_prompt(
		entity_library,
		preamble="You build clinic screens.",
		additional_rules=("Never show patient identifiers.",),
	)
	assert prompt.startswith("You build clinic screens.")
	assert "Never show patient identifiers." in prompt


# ---------------------------------------------------------------------------
# 5. Entity specialisation
# ---------------------------------------------------------------------------


def test_entity_components_are_generated(entity_library: LibrarySpec):
	names = {component.name for component in entity_library.components}
	assert {"PatientRow", "PatientTable", "AppointmentRow", "AppointmentTable"} <= names


def test_status_field_produces_a_board(entity_library: LibrarySpec):
	names = {component.name for component in entity_library.components}
	# Appointment.status exists; Patient has no status field.
	assert {"AppointmentBoard", "AppointmentColumn", "AppointmentCard"} <= names
	assert "PatientBoard" not in names


def test_entity_row_drops_implicit_fields(entity_library: LibrarySpec):
	row = entity_library.component("PatientRow")
	names = {prop.name for prop in row.props}
	assert "id" not in names
	assert "created_at" not in names
	assert "full_name" in names


def test_entity_row_maps_apg_types(clinic_model: dict):
	library = build_library(clinic_model, entities=["Patient"])
	row = {prop.name: prop for prop in library.component("PatientRow").props}
	assert row["date_of_birth"].type == "string"
	assert row["full_name"].type == "string"


def test_entity_selection_honours_explicit_subset(entity_library: LibrarySpec):
	groups = {group[0] for group in entity_library.groups}
	assert {"Patient", "Appointment"} <= groups
	assert "Bill" not in groups


def test_entity_limit_is_applied(clinic_model: dict):
	capped = build_library(clinic_model, max_entities=2)
	entity_groups = [group[0] for group in capped.groups if group[0] not in {"Layout", "Content", "Data", "Forms", "Actions"}]
	assert len(entity_groups) == 2


def test_system_table_is_never_specialised(clinic_model: dict):
	library = build_library(clinic_model, max_entities=50)
	assert "securityRow" not in {component.name for component in library.components}


def test_library_without_model_has_no_entity_groups():
	library = build_library()
	groups = {group[0] for group in library.groups}
	assert groups == {"Layout", "Content", "Data", "Forms", "Actions"}


# ---------------------------------------------------------------------------
# 6. Serialized spec
# ---------------------------------------------------------------------------


def test_spec_is_json_serializable(entity_library: LibrarySpec):
	text = json.dumps(entity_library.to_spec(), sort_keys=True)
	assert json.loads(text)["id"] == "apg"


def test_spec_has_no_dangling_component_refs(entity_library: LibrarySpec):
	document = entity_library.to_spec()
	defs = document["schema"]["$defs"]
	assert COMPONENT_REF not in defs

	def walk(node):
		if isinstance(node, dict):
			ref = node.get("$ref")
			if isinstance(ref, str):
				yield ref.rsplit("/", 1)[-1]
			for value in node.values():
				yield from walk(value)
		elif isinstance(node, list):
			for item in node:
				yield from walk(item)

	for target in walk(defs):
		assert target in defs, target


def test_any_type_has_no_json_schema_type(entity_library: LibrarySpec):
	document = entity_library.to_spec()
	column = document["schema"]["$defs"]["Column"]
	assert "type" not in column["properties"]["data"]["items"]


def test_binding_prop_renders_in_signature_only(entity_library: LibrarySpec):
	signature = entity_library.component("Input").signature()
	assert "$binding<string>" in signature
	# The schema carries no binding marker (spec: fixtures README).
	assert "binding" not in json.dumps(entity_library.to_spec()["schema"]["$defs"]["Input"])


def test_output_rules_name_only_components_that_exist():
	"""The entity hint must not name a composite the base library lacks."""
	base_prompt = generate_system_prompt(build_library())
	assert "PatientRow" not in base_prompt

	entity_prompt = generate_system_prompt(build_library(build_semantic_model(CLINIC), entities=["Patient"]))
	assert "PatientRow" in entity_prompt


def test_component_groups_cover_every_component(entity_library: LibrarySpec):
	grouped = {group[1][index] for group in entity_library.groups for index in range(len(group[1]))}
	declared = {component.name for component in entity_library.components}
	assert declared <= grouped
