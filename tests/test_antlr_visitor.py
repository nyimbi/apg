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

	def test_block_comment_stripped_length_preserved(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = "/* block comment */ table Foo {}\n"
		stripped = _strip_comments_preserve_positions(src)
		assert len(stripped) == len(src)
		assert "block" not in stripped
		assert "table Foo" in stripped

	def test_block_comment_preserves_newlines(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = "/* line1\nline2\nline3 */ table Foo {}\n"
		stripped = _strip_comments_preserve_positions(src)
		assert len(stripped) == len(src)
		assert stripped.count("\n") == src.count("\n")

	def test_comment_inside_string_not_stripped(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = 'url: "http://example.com//path";\n'
		stripped = _strip_comments_preserve_positions(src)
		assert stripped == src, "Must NOT strip // inside string literal"

	def test_hash_inside_string_not_stripped(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = 'color: "#FF0000";\n'
		stripped = _strip_comments_preserve_positions(src)
		assert stripped == src, "Must NOT strip # inside string literal"

	def test_python_comment_stripped(self):
		from compiler.parser import _strip_comments_preserve_positions
		src = "# python comment\ntable Foo {}\n"
		stripped = _strip_comments_preserve_positions(src)
		assert "python" not in stripped
		assert len(stripped) == len(src)
		assert stripped.count("\n") == src.count("\n")


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


class TestImportResolution:
	"""Phase 2b: multi-file APG import resolution."""

	def test_resolve_imports_inlines_entities(self, tmp_path):
		from compiler.compiler import APGCompiler
		(tmp_path / "lib.apg").write_text(
			'module lib version 1.0.0 { description: "lib"; }\ntable User { id: str; name: str; }',
			encoding="utf-8"
		)
		(tmp_path / "main.apg").write_text(
			'module main version 1.0.0 { description: "main"; }\nimport lib;\ntable Order { user_id: str; }',
			encoding="utf-8"
		)
		r = APGCompiler().compile_file(str(tmp_path / "main.apg"))
		assert r.success
		names = {e.name for e in r.module.entities}
		assert "User" in names, "Imported entity must appear in module"
		assert "Order" in names, "Local entity must remain"

	def test_resolve_imports_missing_file_is_skipped(self, tmp_path):
		from compiler.compiler import APGCompiler
		(tmp_path / "main.apg").write_text(
			'module main version 1.0.0 { description: "main"; }\nimport nonexistent_module;\ntable Foo { id: str; }',
			encoding="utf-8"
		)
		r = APGCompiler().compile_file(str(tmp_path / "main.apg"))
		assert r.success, f"Missing import should be skipped silently: {r.errors}"
		assert any(e.name == "Foo" for e in r.module.entities)

	def test_resolve_imports_cycle_does_not_hang(self, tmp_path):
		from compiler.compiler import APGCompiler
		(tmp_path / "a.apg").write_text(
			'module a version 1.0.0 { description: "a"; }\nimport b;\ntable A { id: str; }',
			encoding="utf-8"
		)
		(tmp_path / "b.apg").write_text(
			'module b version 1.0.0 { description: "b"; }\nimport a;\ntable B { id: str; }',
			encoding="utf-8"
		)
		r = APGCompiler().compile_file(str(tmp_path / "a.apg"))
		assert r.success  # cycle must be detected and skipped, not infinite loop

	def test_resolve_imports_rejects_path_traversal(self, tmp_path):
		from compiler.compiler import APGCompiler
		(tmp_path / "main.apg").write_text(
			'module main version 1.0.0 { description: "main"; }\nimport ..secret_module;\ntable Safe { id: str; }',
			encoding="utf-8"
		)
		r = APGCompiler().compile_file(str(tmp_path / "main.apg"))
		# Should succeed (traversal silently rejected) or succeed with just Safe entity
		assert r.success
		names = {e.name for e in r.module.entities}
		assert "Safe" in names

	def test_resolve_imports_named_items_filter(self, tmp_path):
		from compiler.compiler import APGCompiler
		(tmp_path / "lib.apg").write_text(
			'module lib version 1.0.0 { description: "lib"; }\ntable UserA { id: str; }\ntable UserB { id: str; }',
			encoding="utf-8"
		)
		(tmp_path / "main.apg").write_text(
			'module main version 1.0.0 { description: "main"; }\nfrom lib import UserA;',
			encoding="utf-8"
		)
		r = APGCompiler().compile_file(str(tmp_path / "main.apg"))
		assert r.success
		names = {e.name for e in r.module.entities}
		assert "UserA" in names, "Named import must be included"
		assert "UserB" not in names, "Non-imported entity must be excluded"


class TestANTLRVisitorCorrectness:
	"""Verify ANTLR visitor produces same results as regex parser."""

	@pytest.mark.parametrize("example_num", [1, 2, 3, 4, 5])
	def test_visitor_entity_names_match_regex_parser(self, example_num):
		dirs = sorted(ROOT.glob(f"examples/{example_num:02d}_*"))
		if not dirs:
			pytest.skip(f"No example {example_num:02d}")
		path = dirs[0] / "main.apg"
		if not path.exists():
			pytest.skip()

		# Reference: regex parser
		from compiler.ast_builder import ASTBuilder
		src = path.read_text(encoding="utf-8")
		ref = ASTBuilder()._build_source_ast(src, str(path))

		# Subject: full compile path
		from compiler.compiler import APGCompiler
		r = APGCompiler().compile_file(str(path))
		assert r.success
		actual = r.module

		# Same entity names and types (order-independent)
		ref_summary = sorted((e.name, type(e).__name__) for e in ref.entities)
		actual_summary = sorted((e.name, type(e).__name__) for e in actual.entities)
		assert actual_summary == ref_summary, (
			f"Entity mismatch in example {example_num:02d}:\n"
			f"  regex: {ref_summary}\n"
			f"  compiler: {actual_summary}"
		)


class TestCrossEntityValidation:
	"""Phase 3: cross-entity reference validation."""

	def test_application_references_unknown_capability_warns(self):
		from compiler.compiler import APGCompiler
		r = APGCompiler().compile_string('''
module m version 1.0.0 { description: "test"; }

app A {
  capabilities: [DoesNotExist];
  routes: ["/x"];
}
''', "test.apg")
		assert r.success
		warn_msgs = [str(w) for w in r.warnings]
		assert any("DoesNotExist" in m for m in warn_msgs), \
			f"Expected warning about unknown capability, got: {warn_msgs}"

	def test_known_local_capability_no_warning(self):
		from compiler.compiler import APGCompiler
		r = APGCompiler().compile_string('''
module m version 1.0.0 { description: "test"; }

capability Producer {
  contract: { id: p, provides: [feature_x] };
}
capability Consumer {
  contract: { id: c, requires: [feature_x] };
}
''', "test.apg")
		cap_warnings = [str(w) for w in r.warnings if "requires" in str(w) and "Consumer" in str(w)]
		assert not cap_warnings, f"Should not warn on locally provided capability: {cap_warnings}"
