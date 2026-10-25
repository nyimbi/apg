"""OpenUI Lang 1.0 conformance suite (level 1: program cases).

Drives the parser, evaluator, streamer, and patch merger against the archived
spec fixtures in ``tests/fixtures/openui/1.0``. The fixtures are the normative
test suite shipped with the language (spec Appendix B), so this file is the gate
on the whole ``compiler.openui_lang`` package: a regression anywhere in lexing,
statement splitting, evaluation, validation, or streaming lands here.

Comparison follows Appendix B exactly:

* ``errors`` is a multiset of ``{code, statementId}``; an expected entry with no
  ``statementId`` matches any statement.
* ``unresolved`` and ``orphans`` are sets.
* Everything else is compared as-is.
* Streaming cases must match both streamed and as a batch parse of the joined
  chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from compiler.openui_lang import Library, Program
from compiler.openui_lang.program import StreamParser, merge_patch

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "openui" / "1.0"
LIBRARY_PATH = FIXTURES / "library.json"


def _library() -> Library:
	return Library.from_spec(json.loads(LIBRARY_PATH.read_text()))


def _upper(text: str) -> str:
	return text.upper()


FUNCTIONS = {"Upper": _upper}


def _cases() -> list[Path]:
	cases = []
	for expected in sorted(FIXTURES.glob("*/*/expected.json")):
		cases.append(expected.parent)
	return cases


CASES = _cases()
IDS = [case.relative_to(FIXTURES).as_posix() for case in CASES]


def _load(path: Path) -> Any:
	return json.loads(path.read_text())


def _run_case(case: Path) -> dict[str, Any]:
	library = _library()
	chunks_path = case / "chunks.json"
	input_path = case / "input.oui"
	patch_path = case / "patch.oui"

	if chunks_path.exists():
		chunks = _load(chunks_path)
		streamed = StreamParser().run(chunks, library, functions=FUNCTIONS)
		batch = Program.parse("".join(chunks)).evaluate(library, functions=FUNCTIONS)
		assert streamed.to_dict(include_state=False) == batch.to_dict(include_state=False), (
			"streaming must equal parsing (spec 4)"
		)
		result = streamed
		output = result.to_dict()
	else:
		source = input_path.read_text()
		if patch_path.exists():
			merged = merge_patch(source, patch_path.read_text(), library)
			output = Program.parse(merged.text).evaluate(library, functions=FUNCTIONS).to_dict()
			output["statements"] = merged.statements
		else:
			result = Program.parse(source).evaluate(library, functions=FUNCTIONS)
			output = result.to_dict()
			output["errorObjects"] = result.error_objects()
	return output


def _compare(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
	failures: list[str] = []

	if "root" in expected:
		want = expected["root"]
		got = actual.get("root")
		if _normalise(got) != _normalise(want):
			failures.append(f"root mismatch\n  expected: {json.dumps(want, sort_keys=True)}\n  actual:   {json.dumps(got, sort_keys=True)}")

	if "errors" in expected:
		want = expected["errors"]
		got = actual.get("errors", [])
		if not _errors_match(got, want):
			failures.append(f"errors mismatch\n  expected: {json.dumps(want, sort_keys=True)}\n  actual:   {json.dumps(got, sort_keys=True)}")

	if "unresolved" in expected:
		if set(actual.get("unresolved", [])) != set(expected["unresolved"]):
			failures.append(f"unresolved mismatch: expected {expected['unresolved']}, actual {actual.get('unresolved')}")

	if "orphans" in expected:
		if set(actual.get("orphans", [])) != set(expected["orphans"]):
			failures.append(f"orphans mismatch: expected {expected['orphans']}, actual {actual.get('orphans')}")

	if "incomplete" in expected:
		if actual.get("incomplete") != expected["incomplete"]:
			failures.append(f"incomplete mismatch: expected {expected['incomplete']}, actual {actual.get('incomplete')}")

	if "state" in expected:
		if actual.get("state") != expected["state"]:
			failures.append(f"state mismatch: expected {expected['state']}, actual {actual.get('state')}")

	if "statements" in expected:
		if actual.get("statements") != expected["statements"]:
			failures.append(f"statements mismatch: expected {expected['statements']}, actual {actual.get('statements')}")

	if "errorObjects" in expected:
		actual_objects = actual.get("errorObjects", [])
		for wanted in expected["errorObjects"]:
			if not any(_object_matches(actual_object, wanted) for actual_object in actual_objects):
				failures.append(f"no error object matches {wanted}; actual: {actual_objects}")

	return failures


def _normalise(value: Any) -> Any:
	"""Compare component props with absent and null equal (Appendix B)."""
	if isinstance(value, dict):
		return {key: _normalise(item) for key, item in value.items() if item is not None}
	if isinstance(value, list):
		return [_normalise(item) for item in value]
	return value


def _errors_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
	remaining = list(actual)
	for wanted in expected:
		for index, candidate in enumerate(remaining):
			if candidate.get("code") != wanted.get("code"):
				continue
			if "statementId" in wanted and candidate.get("statementId") != wanted["statementId"]:
				continue
			del remaining[index]
			break
		else:
			return False
	return not remaining


def _object_matches(actual: dict[str, Any], wanted: dict[str, Any]) -> bool:
	return all(actual.get(key) == value for key, value in wanted.items())


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_conformance(case: Path) -> None:
	expected = _load(case / "expected.json")
	actual = _run_case(case)
	failures = _compare(actual, expected)
	assert not failures, "\n".join(failures)


def test_fixture_count() -> None:
	"""The archived suite is complete: every area's cases are present."""
	assert len(CASES) == 58, f"expected 58 program cases, found {len(CASES)}"


def test_library_loads() -> None:
	library = _library()
	assert library.root == "Stack"
	assert len(library.components) == 12
	assert "Upper" in library.functions
	assert "CopyToClipboard" in library.actions
	assert _prop(library, "Input", "value").binding is True
	assert _prop(library, "Input", "name").binding is False
	assert _prop(library, "Stack", "children").is_component_slot is True
	assert _prop(library, "Table", "columns").is_component_slot is True
	assert _prop(library, "Button", "action").is_action_slot is True
	assert _prop(library, "Button", "share").is_action_slot is True
	assert _prop(library, "Text", "text").is_component_slot is False


def _prop(library: Library, component: str, prop: str):
	spec = library.component(component)
	assert spec is not None
	found = spec.prop(prop)
	assert found is not None
	return found
