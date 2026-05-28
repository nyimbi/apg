"""Constrained natural-language-to-APG patch planning.

The planner is intentionally local and deterministic. It turns a small set of
common APG edit intents into append-only DSL snippets, validates the candidate
source through the existing parser/AST/semantic stack, and returns a reviewable
plan. It does not mutate the source file or generate application output.
"""

from __future__ import annotations

import difflib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ast_builder import ASTBuilder
from .parser import APGParser, APGSyntaxError
from .semantic_analyzer import SemanticAnalyzer, SemanticError


NL_PLAN_FORMAT = "apg.nl-plan.v1"
MIGRATION_PREVIEW_FORMAT = "apg.migration-preview.v1"


@dataclass(frozen=True)
class PlannedEdit:
	intent: str
	base_name: str
	append_text: str
	affected_symbols: list[str]
	migration_changes: list[dict[str, Any]]
	confidence: str = "medium"


def build_nl_plan(source_file: Path, prompt: str) -> dict[str, Any]:
	"""Build an `apg.nl-plan.v1` report without writing generated code."""
	source_file = Path(source_file)
	source_text = source_file.read_text(encoding="utf-8")
	prompt = (prompt or "").strip()
	edit = _classify_prompt(prompt)
	if edit is None:
		lint_report = _empty_lint_report(source_file, _planner_diagnostic(source_file, prompt))
		return _report(
			source_file=source_file,
			prompt=prompt,
			intent="unrepresentable",
			confidence="low",
			dsl_patch="",
			append_text="",
			affected_symbols=[],
			lint=lint_report,
			migration_preview=_migration_preview([]),
			test_plan=[],
			errors=["Prompt cannot be represented as a bounded APG DSL edit."],
			warnings=[],
			ok=False,
		)

	candidate_text = _append_source(source_text, edit.append_text)
	dsl_patch = _unified_patch(source_file, source_text, candidate_text)
	lint_report = _lint_candidate(source_file, candidate_text)
	ok = bool(lint_report["ok"])
	errors = [] if ok else ["Generated DSL patch did not pass APG lint validation."]
	warnings = [] if ok else ["Review the nested lint diagnostics before applying this patch."]
	return _report(
		source_file=source_file,
		prompt=prompt,
		intent=edit.intent,
		confidence=edit.confidence,
		dsl_patch=dsl_patch,
		append_text=edit.append_text,
		affected_symbols=edit.affected_symbols,
		lint=lint_report,
		migration_preview=_migration_preview(edit.migration_changes),
		test_plan=_test_plan(source_file),
		errors=errors,
		warnings=warnings,
		ok=ok,
	)


def _report(
	source_file: Path,
	prompt: str,
	intent: str,
	confidence: str,
	dsl_patch: str,
	append_text: str,
	affected_symbols: list[str],
	lint: dict[str, Any],
	migration_preview: dict[str, Any],
	test_plan: list[dict[str, str]],
	errors: list[str],
	warnings: list[str],
	ok: bool,
) -> dict[str, Any]:
	return {
		"format": NL_PLAN_FORMAT,
		"ok": ok,
		"source": str(source_file),
		"prompt": prompt,
		"intent": intent,
		"confidence": confidence,
		"dsl_patch": dsl_patch,
		"append_text": append_text,
		"affected_symbols": affected_symbols,
		"lint": lint,
		"migration_preview": migration_preview,
		"test_plan": test_plan,
		"token_budget_notes": [
			"Planner used deterministic local templates instead of external AI calls.",
			"Review only the DSL patch before running compile/package/release evidence.",
		],
		"errors": errors,
		"warnings": warnings,
	}


def _classify_prompt(prompt: str) -> PlannedEdit | None:
	normalized = _normalize_words(prompt)
	if not normalized:
		return None

	if "credit memo" in normalized or "credit memos" in normalized:
		return _domain_feature_edit("credit_memo", "Credit Memo Management")
	if re.search(r"\b(agent|assistant|copilot)\b", normalized):
		base_name = _base_name_from_prompt(normalized, fallback="assistant_agent")
		return _agent_edit(base_name)
	if re.search(r"\b(capability|module|component)\b", normalized):
		base_name = _base_name_from_prompt(normalized, fallback="business_capability")
		return _capability_edit(base_name)
	if re.search(r"\b(table|entity|record|records)\b", normalized):
		base_name = _base_name_from_prompt(normalized, fallback="business_record")
		return _table_edit(base_name)
	return None


def _domain_feature_edit(base_name: str, label: str) -> PlannedEdit:
	table = _pascal_case(base_name)
	capability = f"{table}Management"
	append_text = (
		f"\n// Planned APG feature: {label}.\n"
		f"table {table} {{\n"
		"    memo_number: str;\n"
		"    customer_id: int;\n"
		"    source_invoice_id: int;\n"
		"    amount: float;\n"
		"    reason: str;\n"
		"    status: str;\n"
		"}\n"
		"\n"
		f"capability {capability} {{\n"
		"    contract: {\n"
		f"        id: {_snake_case(capability)},\n"
		"        provides: [credit_memo_management],\n"
		"        requires: [accounts_receivable, customer_master],\n"
		"        configuration: {tenant_scoped: true, default_status: \"draft\"},\n"
		"        rules: [{name: \"credit_memo_amount_positive\", when: \"amount <= 0\", action: \"deny\"}],\n"
		"        rule_engine: {mode: deterministic, audit: true},\n"
		"        ui: {shell: python, routes: [{name: \"Credit Memos\", path: \"/finance/credit-memos\", component: \"CreditMemoScreen\"}]},\n"
		"        theme: {name: finance_operations, tokens: {accent: \"#126E82\"}}\n"
		"    };\n"
		"}\n"
	)
	return PlannedEdit(
		intent="domain_feature",
		base_name=base_name,
		append_text=append_text,
		affected_symbols=[f"table.{table}", f"capability.{capability}"],
		migration_changes=[
			{"kind": "add_table", "symbol": f"table.{table}", "destructive": False},
			{"kind": "add_capability", "symbol": f"capability.{capability}", "destructive": False},
		],
		confidence="high",
	)


def _table_edit(base_name: str) -> PlannedEdit:
	table = _pascal_case(base_name)
	append_text = (
		f"\n// Planned APG table: {table}.\n"
		f"table {table} {{\n"
		"    name: str;\n"
		"    status: str;\n"
		"    created_at: str;\n"
		"}\n"
	)
	return PlannedEdit(
		intent="add_table",
		base_name=base_name,
		append_text=append_text,
		affected_symbols=[f"table.{table}"],
		migration_changes=[{"kind": "add_table", "symbol": f"table.{table}", "destructive": False}],
	)


def _capability_edit(base_name: str) -> PlannedEdit:
	capability = _pascal_case(base_name)
	append_text = (
		f"\n// Planned APG capability: {capability}.\n"
		f"capability {capability} {{\n"
		"    contract: {\n"
		f"        id: {_snake_case(capability)},\n"
		f"        provides: [{_snake_case(capability)}],\n"
		"        requires: [],\n"
		"        configuration: {tenant_scoped: true},\n"
		"        rules: [{name: \"default_review\", when: \"status == \\\"blocked\\\"\", action: \"require_review\"}],\n"
		"        rule_engine: {mode: deterministic, audit: true},\n"
		f"        ui: {{shell: python, routes: [{{name: \"{_title_words(base_name)}\", path: \"/{_kebab_case(base_name)}\", component: \"{capability}Screen\"}}]}},\n"
		"        theme: {name: operational, tokens: {accent: \"#126E82\"}}\n"
		"    };\n"
		"}\n"
	)
	return PlannedEdit(
		intent="add_capability",
		base_name=base_name,
		append_text=append_text,
		affected_symbols=[f"capability.{capability}"],
		migration_changes=[{"kind": "add_capability", "symbol": f"capability.{capability}", "destructive": False}],
	)


def _agent_edit(base_name: str) -> PlannedEdit:
	agent = _pascal_case(base_name)
	if not agent.endswith("Agent"):
		agent = f"{agent}Agent"
	role = _title_words(base_name).lower()
	append_text = (
		f"\n// Planned APG AI agent: {agent}.\n"
		f"agent {agent} {{\n"
		f"    role: \"{role}\";\n"
		"    model: \"openai:gpt-4.1-mini\";\n"
		"    runtime: codex;\n"
		f"    system: \"Assist with {role} work.\";\n"
		"}\n"
	)
	return PlannedEdit(
		intent="add_agent",
		base_name=base_name,
		append_text=append_text,
		affected_symbols=[f"agent.{agent}"],
		migration_changes=[{"kind": "add_agent", "symbol": f"agent.{agent}", "destructive": False}],
	)


def _base_name_from_prompt(normalized_prompt: str, fallback: str) -> str:
	phrase = re.sub(r"^(add|create|define|build|make)\s+", "", normalized_prompt).strip()
	phrase = re.split(r"\b(?:to|for|with|in|into|that|which|where|using)\b", phrase, maxsplit=1)[0]
	words = [
		word
		for word in re.findall(r"[a-z0-9]+", phrase)
		if word not in {"a", "an", "the", "new", "table", "entity", "record", "records", "capability", "component", "module", "agent", "assistant", "copilot"}
	]
	if not words:
		return fallback
	return "_".join(words[:4])


def _append_source(source_text: str, append_text: str) -> str:
	return source_text.rstrip() + "\n" + append_text.lstrip("\n")


def _unified_patch(source_file: Path, original: str, candidate: str) -> str:
	return "".join(difflib.unified_diff(
		original.splitlines(keepends=True),
		candidate.splitlines(keepends=True),
		fromfile=str(source_file),
		tofile=f"{source_file} (planned)",
	))


def _lint_candidate(source_file: Path, candidate_text: str) -> dict[str, Any]:
	with tempfile.TemporaryDirectory(prefix="apg-nl-plan-") as temp_dir:
		candidate_path = Path(temp_dir) / source_file.name
		candidate_path.write_text(candidate_text, encoding="utf-8")
		return _lint_file(candidate_path, display_file=source_file)


def _lint_file(file_path: Path, display_file: Path) -> dict[str, Any]:
	parser = APGParser()
	ast_builder = ASTBuilder()
	analyzer = SemanticAnalyzer()
	diagnostics: list[dict[str, Any]] = []
	semantic_model_available = False

	try:
		parse_result = parser.parse_file(str(file_path))
	except Exception as error:
		diagnostics.append(_diagnostic_from_error(error, display_file, "error"))
		return _lint_report(display_file, diagnostics, semantic_model_available)

	for error in parse_result.get("errors", []):
		diagnostics.append(_diagnostic_from_error(error, display_file, "error"))

	if parse_result.get("success"):
		try:
			ast = parse_result.get("ast") or ast_builder.build_ast(parse_result["parse_tree"], str(display_file))
			if ast is None:
				raise RuntimeError("Failed to build AST")
			semantic_model_available = True
			semantic_result = analyzer.analyze(ast)
			for error in semantic_result.get("errors", []):
				diagnostics.append(_diagnostic_from_error(error, display_file, "error"))
			for warning in semantic_result.get("warnings", []):
				diagnostics.append(_diagnostic_from_error(warning, display_file, "warning"))
		except Exception as error:
			diagnostics.append(_diagnostic_from_error(error, display_file, "error"))

	return _lint_report(display_file, diagnostics, semantic_model_available)


def _lint_report(file_path: Path, diagnostics: list[dict[str, Any]], semantic_model_available: bool) -> dict[str, Any]:
	counts = _severity_counts(diagnostics)
	file_report = {
		"format": "apg.lint-file-report.v1",
		"ok": counts["error"] == 0,
		"file": str(file_path),
		"strict": False,
		"severity_counts": counts,
		"diagnostics": diagnostics,
		"fixes_available": any(diagnostic.get("fixes") for diagnostic in diagnostics),
		"semantic_model_available": semantic_model_available,
	}
	return {
		"format": "apg.lint-report.v1",
		"ok": file_report["ok"],
		"source_mode": "file",
		"strict": False,
		"files": [str(file_path)],
		"severity_counts": counts,
		"diagnostics": diagnostics,
		"fixes_available": file_report["fixes_available"],
		"semantic_model_available": semantic_model_available,
		"file_reports": [file_report],
	}


def _empty_lint_report(file_path: Path, diagnostic: dict[str, Any]) -> dict[str, Any]:
	return _lint_report(file_path, [diagnostic], semantic_model_available=False)


def _severity_counts(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
	counts = {"error": 0, "warning": 0, "info": 0, "hint": 0}
	for diagnostic in diagnostics:
		severity = diagnostic.get("severity", "error")
		counts[severity] = counts.get(severity, 0) + 1
	return counts


def _diagnostic_from_error(error: APGSyntaxError | SemanticError | Exception, file_path: Path, severity: str) -> dict[str, Any]:
	if isinstance(error, APGSyntaxError):
		return _diagnostic(
			code="APG0001",
			title="Syntax error",
			severity=severity,
			message=error.message,
			file_path=file_path,
			line=error.line,
			column=error.column,
		)
	if isinstance(error, SemanticError):
		node = error.node
		return _diagnostic(
			code="APG0200" if error.error_type == "type" else "APG0100",
			title="Semantic warning" if error.error_type == "warning" else f"{error.error_type.title()} error",
			severity=severity,
			message=error.message,
			file_path=file_path,
			line=getattr(node, "line", None),
			column=getattr(node, "column", None),
		)
	return _diagnostic(
		code="APG9000",
		title="Internal tooling error",
		severity="error",
		message=str(error),
		file_path=file_path,
		line=1,
		column=0,
	)


def _planner_diagnostic(file_path: Path, prompt: str) -> dict[str, Any]:
	message = (
		"Prompt cannot be represented as a bounded APG DSL edit. "
		"Use concrete language such as 'add table', 'add capability', 'add agent', or a supported domain feature."
	)
	if prompt:
		message = f"{message} Prompt: {prompt!r}."
	return _diagnostic(
		code="APG1201",
		title="Unrepresentable natural-language plan",
		severity="error",
		message=message,
		file_path=file_path,
		line=1,
		column=0,
		fixes=[{"id": "use_bounded_apg_edit", "title": "Ask for a concrete APG table, capability, agent, or supported domain feature"}],
	)


def _diagnostic(
	code: str,
	title: str,
	severity: str,
	message: str,
	file_path: Path,
	line: int | None,
	column: int | None,
	fixes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
	start_line = max(0, int(line or 1) - 1)
	start_char = max(0, int(column or 0))
	return {
		"code": code,
		"title": title,
		"severity": severity,
		"message": message,
		"file": str(file_path),
		"range": {
			"start": {"line": start_line, "character": start_char},
			"end": {"line": start_line, "character": start_char + 1},
		},
		"related_locations": [],
		"fixes": fixes or [],
		"docs_url": "docs/tooling.md#apg-nl-plan",
	}


def _migration_preview(changes: list[dict[str, Any]]) -> dict[str, Any]:
	return {
		"format": MIGRATION_PREVIEW_FORMAT,
		"ok": True,
		"destructive": any(change.get("destructive") for change in changes),
		"requires_approval": any(change.get("destructive") for change in changes),
		"changes": changes,
	}


def _test_plan(source_file: Path) -> list[dict[str, str]]:
	return [
		{"phase": "lint", "command": f"apg lint {source_file} --json"},
		{"phase": "validate", "command": f"apg validate {source_file} --target python --json"},
		{"phase": "compile", "command": f"apg compile {source_file} --target python --output generated --verify"},
		{"phase": "release", "command": f"apg release {source_file} --json"},
	]


def _normalize_words(value: str) -> str:
	return re.sub(r"\s+", " ", value.lower()).strip()


def _pascal_case(value: str) -> str:
	parts = re.findall(r"[A-Za-z0-9]+", value)
	return "".join(part[:1].upper() + part[1:] for part in parts) or "PlannedItem"


def _snake_case(value: str) -> str:
	words = re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value))
	return "_".join(word.lower() for word in words) or "planned_item"


def _kebab_case(value: str) -> str:
	return _snake_case(value).replace("_", "-")


def _title_words(value: str) -> str:
	return " ".join(word.capitalize() for word in _snake_case(value).split("_")) or "Planned Item"
