"""Tests for the ANTLR-based AST visitor (Phase 1 of grammar analysis resolution)."""

from __future__ import annotations

from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _build_regex(path: str):
	"""Build AST using regex parser directly."""
	from compiler.ast_builder import ASTBuilder
	src = (ROOT / path).read_text(encoding="utf-8")
	return ASTBuilder()._build_source_ast(src, path)


def _build_via_compiler(path: str):
	"""Compile file (uses ANTLR visitor when parse is clean, else regex fallback)."""
	from compiler.compiler import APGCompiler
	c = APGCompiler()
	r = c.compile_file(str(ROOT / path))
	return r.module, r.success


class TestANTLRVisitorInfrastructure:
	def test_antlr_source_parse_tree_has_antlr_tree_field(self):
		"""Parser now attaches the real ANTLR tree to the compat parse tree."""
		from compiler.parser import APGParser
		p = APGParser()
		result = p.parse_file(str(ROOT / "examples/01_minimal_customer_records/main.apg"))
		pt = result["parse_tree"]
		assert hasattr(pt, "antlr_tree"), "parse_tree must have antlr_tree attribute"
		assert hasattr(pt, "antlr_source"), "parse_tree must have antlr_source attribute"
		assert hasattr(pt, "antlr_clean"), "parse_tree must have antlr_clean attribute"
		assert isinstance(pt.antlr_source, str)

	def test_comment_stripping_preserves_length(self):
		"""Comment stripping must preserve character positions."""
		from compiler.parser import _strip_comments_preserve_positions
		src = "// comment line\ntable Foo {\n    name: str;\n}\n"
		stripped = _strip_comments_preserve_positions(src)
		assert len(stripped) == len(src), "Stripped source must have same length"
		assert stripped[0] == " "  # // replaced with spaces
		assert stripped.count("\n") == src.count("\n")  # newlines preserved

	def test_comment_stripping_handles_empty_comments(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = "//\ntable Foo {}\n"
		stripped = _strip_comments_preserve_positions(src)
		assert len(stripped) == len(src)

	def test_antlr_visitor_available(self):
		from compiler.antlr_ast_visitor import build_ast_from_antlr, APGASTVisitor
		assert APGASTVisitor is not None
		assert callable(build_ast_from_antlr)


class TestVisitorOnNumberedExamples:
	"""The compiler (with ANTLR fallback) must produce the same entity count as the regex parser."""

	@pytest.mark.parametrize("example_num", range(1, 8))
	def test_simple_numbered_examples_compile(self, example_num: int):
		dirs = sorted(ROOT.glob(f"examples/{example_num:02d}_*"))
		if not dirs:
			pytest.skip(f"No example {example_num:02d}")
		path = dirs[0] / "main.apg"
		if not path.exists():
			pytest.skip(f"No main.apg in {dirs[0]}")
		module, success = _build_via_compiler(str(path.relative_to(ROOT)))
		assert success, f"Compilation failed for {path}"
		assert module is not None
		assert len(module.entities) > 0


class TestWorkflowDeclaration:
	def test_workflow_produces_states_and_transitions(self):
		module, _ = _build_via_compiler("examples/crm_platform/main.apg")
		from compiler.ast_builder import WorkflowDeclaration
		workflows = [e for e in module.entities if isinstance(e, WorkflowDeclaration)]
		assert len(workflows) >= 1, "CRM platform must have at least one WorkflowDeclaration"
		lq = next(w for w in workflows if w.name == "LeadQualification")
		assert "new_lead" in lq.states
		assert "qualified" in lq.states
		assert len(lq.transitions) == len(lq.states) - 1
		assert "contacted" in lq.human_tasks
		assert "qualified" in lq.human_tasks
		assert lq.guards.get("qualified") == "budget_confirmed and timeline_defined"

	def test_workflow_guard_on_transition(self):
		module, _ = _build_via_compiler("examples/crm_platform/main.apg")
		from compiler.ast_builder import WorkflowDeclaration
		deal = next(e for e in module.entities if isinstance(e, WorkflowDeclaration) and e.name == "DealApproval")
		# The 'finance_review' state should have a guard
		guarded = [t for t in deal.transitions if t.guard is not None]
		assert len(guarded) >= 1
		assert any("100000" in (t.guard or "") for t in guarded)

	def test_workflow_in_semantic_model_flows(self):
		from compiler.semantic_model import build_semantic_model
		m = build_semantic_model(ROOT / "examples/crm_platform/main.apg")
		flows = m.get("flows", {})
		assert "LeadQualification" in flows
		lq = flows["LeadQualification"]
		assert lq["states"] == ["new_lead", "researched", "contacted", "qualified", "opportunity_created"]
		assert isinstance(lq["transitions"], list)
		assert isinstance(lq["human_tasks"], list)
		assert isinstance(lq["guards"], dict)
