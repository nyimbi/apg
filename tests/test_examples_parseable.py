"""Regression coverage for the numbered APG example progression."""

from __future__ import annotations

from pathlib import Path

from compiler.compiler import compile_apg_file
from compiler.parser import APGParser


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def numbered_example_sources() -> list[Path]:
	return sorted(EXAMPLES_DIR.glob("[0-9][0-9]*/main.apg"))


def test_numbered_apg_examples_are_present_and_ordered():
	sources = numbered_example_sources()
	names = [path.parent.name for path in sources]

	assert len(sources) == 20
	assert [name[:2] for name in names] == [f"{index:02d}" for index in range(1, 21)]


def test_numbered_apg_examples_parse_and_compile():
	parser = APGParser()
	failures: list[str] = []

	for source in numbered_example_sources():
		parse_result = parser.parse_file(str(source))
		if not parse_result["success"]:
			failures.append(f"{source}: parse failed: {parse_result['errors']}")
			continue

		compile_result = compile_apg_file(str(source))
		if not compile_result.success:
			failures.append(f"{source}: compile failed: {compile_result.errors}")

	assert failures == []
