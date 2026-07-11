"""ANTLR-based AST visitor for APG.

Implements the hybrid approach: ANTLR for structural entity identification
(solving brace-counting bugs), existing _parse_source_* methods for body
content (safe, tested, backward-compatible).

This replaces the regex-based `_build_source_ast` as the authoritative
parser while reusing all body parsers unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# Add spec/ to path for ANTLR-generated files
_SPEC_DIR = str(Path(__file__).parent.parent / "spec")
if _SPEC_DIR not in sys.path:
	sys.path.insert(0, _SPEC_DIR)

try:
	from apgVisitor import apgVisitor  # type: ignore[import]
	from apgParser import apgParser    # type: ignore[import]
	_ANTLR_AVAILABLE = True
except ImportError:
	apgVisitor = object  # type: ignore[misc,assignment]
	apgParser = None
	_ANTLR_AVAILABLE = False

from .ast_builder import (
	ASTBuilder,
	ModuleDeclaration,
	EntityDeclaration,
	EntityType,
	WorkflowDeclaration,
)


def _ctx_text(ctx: Any) -> str:
	"""Return the raw getText() of an ANTLR context, or '' if None."""
	return ctx.getText() if ctx is not None else ""


def _token_text(tok: Any) -> str:
	"""Return the text of an ANTLR terminal node/token, or '' if None."""
	if tok is None:
		return ""
	try:
		return tok.getText()
	except Exception:
		return ""


class APGASTVisitor(apgVisitor):  # type: ignore[misc]
	"""Concrete ANTLR visitor that produces APG AST nodes.

	Strategy: use ANTLR for reliable entity boundary detection (fixes all
	nested-brace bugs), delegate body content to existing _parse_source_*
	methods so that no semantic behaviour changes.

	WHY THE HYBRID EXISTS — FLOORDIV/COMMENT lexer ordering bug:
	    In spec/apg.g4 the FLOORDIV token rule ('//') is declared before the
	    COMMENT rule ('// ...' line comment).  ANTLR uses first-match lexer
	    ordering, so it tokenizes every '//' as FLOORDIV and never recognises
	    line comments.  Rather than reorder the grammar (which would change
	    token indices and break generated code), we pre-strip all comment text
	    in _strip_comments_preserve_positions() before handing source to ANTLR.
	    The stripped source has the same length as the original, so all
	    ctx.start.start / ctx.stop.stop character offsets remain valid and
	    _span_text() still reconstructs the right body text.

	To add a native ANTLR body handler for a specific entity kind, add a
	`_visit_<kind>_body(body_ctx)` method and dispatch to it in visitEntity.
	"""

	def __init__(self, source_code: str, source_file: Optional[str] = None):
		self._source = source_code
		self._source_file = source_file
		# Shared ASTBuilder for body parsing — reuse all existing methods
		self._builder = ASTBuilder()

	# ── character-position text extraction ────────────────────────────────

	def _span_text(self, ctx: Any) -> str:
		"""Extract original source text using ANTLR token character positions."""
		if ctx is None:
			return ""
		try:
			return self._source[ctx.start.start: ctx.stop.stop + 1]
		except (AttributeError, TypeError):
			return ""

	def _body_text(self, body_ctx: Any) -> str:
		"""Extract entity body text (the content between outer braces)."""
		raw = self._span_text(body_ctx)
		# Strip the enclosing { } if present
		raw = raw.strip()
		if raw.startswith("{") and raw.endswith("}"):
			return raw[1:-1]
		return raw

	# ── program / module ──────────────────────────────────────────────────

	def visitProgram(self, ctx: Any) -> ModuleDeclaration:
		module = ModuleDeclaration(source_file=self._source_file)

		# Module declaration
		mod_ctx = ctx.module_declaration()
		if mod_ctx is not None:
			name_ctx = mod_ctx.module_name()
			if name_ctx is not None:
				module.name = _ctx_text(name_ctx)
			ver_ctx = mod_ctx.version_tag()
			if ver_ctx is not None:
				ver_text = _ctx_text(ver_ctx)
				# getText() returns "version1.2.3" — strip the keyword
				module.version = ver_text[7:] if ver_text.startswith("version") else ver_text

		# Import statements
		for imp_ctx in ctx.import_statement():
			try:
				import_text = self._span_text(imp_ctx)
				# Parse 'from X import A, B' or 'import X'
				import re as _re
				from_m = _re.search(r'\bfrom\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([^;]+)', import_text)
				bare_m = _re.search(r'\bimport\s+([A-Za-z_][A-Za-z0-9_.]*)', import_text) if not from_m else None
				if from_m:
					mod_name = from_m.group(1).strip()
					items_raw = from_m.group(2).strip()
					items = [i.strip() for i in items_raw.split(',') if i.strip() and i.strip() != '*']
					from .ast_builder import ImportDeclaration
					module.imports.append(ImportDeclaration(module_name=mod_name, import_items=items))
				elif bare_m:
					mod_name = bare_m.group(1).strip()
					from .ast_builder import ImportDeclaration
					module.imports.append(ImportDeclaration(module_name=mod_name, import_items=[]))
			except Exception:
				pass

		# Top-level entities
		for entity_ctx in ctx.entity():
			try:
				entity = self.visitEntity(entity_ctx)
				if entity is not None:
					module.entities.append(entity)
			except Exception:
				# Fall back to regex parser for this entity (visitProgram caller
				# will see incomplete entities and may re-try with _build_source_ast)
				pass

		return module

	# ── entity dispatch ───────────────────────────────────────────────────

	def visitEntity(self, ctx: Any) -> Optional[EntityDeclaration]:
		"""Dispatch entity to the correct body parser based on entity type keyword."""
		kind = _ctx_text(ctx.entity_type())
		name_tok = ctx.IDENTIFIER()
		if name_tok is None:
			return None
		name = _token_text(name_tok)

		body_text = self._body_text(ctx.entity_body())

		# Dispatch to existing body parsers — same logic as _build_source_ast
		if kind in {"app", "application", "composition"}:
			return self._builder._parse_source_application(
				kind, name, body_text, self._source_file
			)

		if kind == "capability":
			return self._builder._parse_source_capability(
				name, body_text, self._source_file
			)

		if kind in {"db", "database"}:
			return self._builder._parse_source_database(
				name, body_text, self._source_file
			)

		if kind in {"workflow", "flow"}:
			return self._builder._parse_source_workflow(
				name, body_text, self._source_file, kind=kind
			)

		if kind == "agent" and self._builder._is_agent_config_body(body_text):
			return self._builder._parse_source_agent(
				name, body_text, self._source_file
			)

		# Generic entity: parse properties + methods
		properties, methods = self._builder._parse_source_members(
			body_text, self._source_file
		)
		relationships = self._builder._parse_source_relationships(
			body_text, self._source_file
		)
		entity_type = self._builder._entity_type_for_source_kind(kind)
		return EntityDeclaration(
			entity_type=entity_type,
			name=name,
			properties=properties,
			methods=methods,
			relationships=relationships,
			source_file=self._source_file,
		)


def build_ast_from_antlr(
	antlr_tree: Any,
	source_code: str,
	source_file: Optional[str] = None,
) -> Optional[ModuleDeclaration]:
	"""Build a ModuleDeclaration from an ANTLR parse tree.

	Precondition — callers MUST check ``parse_tree.antlr_clean is True`` before
	calling this function.  When ANTLR reported errors, token offsets may be
	unreliable and _span_text() will produce garbage body text; the regex-based
	fallback (_build_source_ast) is safer in that case.

	The ``source_code`` argument must be the comment-stripped source that was
	fed to ANTLR (i.e. ``APGSourceParseTree.antlr_source``), NOT the raw
	original.  This is necessary because _span_text() uses ANTLR character
	offsets to slice into the string — those offsets index into the stripped
	source, not the raw source.

	Returns None if ANTLR is unavailable, the tree is None, or the visitor
	raises, allowing callers to fall back to the regex-based parser.
	"""
	if not _ANTLR_AVAILABLE or antlr_tree is None:
		return None
	try:
		visitor = APGASTVisitor(source_code, source_file)
		result = visitor.visit(antlr_tree)
		if isinstance(result, ModuleDeclaration):
			return result
		return None
	except Exception:
		return None
