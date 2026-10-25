"""OpenUI Lang: parser, evaluator, and renderer for streamed generative UI.

The language is Thesys's OpenUI Lang 1.0 (MIT). This package implements the
program model (statements), the ECMAScript value semantics, streaming, and the
conformance surface defined by the spec's fixture suite.

    from compiler.openui_lang import Library, Program

    library = Library.from_spec(spec_document)
    program = Program.parse(source, library)
    program.render()
"""

from __future__ import annotations

from .evaluator import Diagnostic, EvaluationResult, Evaluator, tree_summary
from .library import ComponentSpec, FunctionSpec, Library, PropSpec
from .parser import ParsedStatement, parse_statement
from .program import Program, StreamParser
from .source import prepare_source, split_statements
from .values import to_boolean, to_number, to_string

__all__ = [
	"ComponentSpec",
	"Diagnostic",
	"EvaluationResult",
	"Evaluator",
	"FunctionSpec",
	"Library",
	"ParsedStatement",
	"Program",
	"PropSpec",
	"StreamParser",
	"parse_statement",
	"prepare_source",
	"split_statements",
	"to_boolean",
	"to_number",
	"to_string",
	"tree_summary",
]
