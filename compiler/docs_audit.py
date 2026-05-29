"""Documentation coverage and navigation audit for APG."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOCS_AUDIT_FORMAT = "apg.docs-audit.v1"

REQUIRED_DOCS = [
	"README.md",
	"docs/README.md",
	"docs/quickstart.md",
	"docs/apg_language.md",
	"docs/apg_tutorial.md",
	"docs/apg_grammar_guide.md",
	"docs/apg_cheat_sheet.md",
	"docs/tooling.md",
	"docs/developer_guide.md",
	"docs/contributors_guide.md",
	"docs/capacity_development_guide.md",
	"docs/capability_standards.md",
	"docs/capability_contracts.md",
	"docs/repository_hygiene.md",
	"docs/progress_log.md",
]

NAVIGATION_DOCS = [
	"README.md",
	"docs/README.md",
	"docs/developer_guide.md",
	"docs/contributors_guide.md",
	"docs/capacity_development_guide.md",
	"docs/repository_hygiene.md",
]

COMMAND_DOCS = [
	"README.md",
	"docs/tooling.md",
	"docs/developer_guide.md",
	"docs/contributors_guide.md",
	"docs/capacity_development_guide.md",
	"docs/quickstart.md",
	"docs/apg_tutorial.md",
]

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
APG_COMMAND_PATTERN = re.compile(r"^\s*(?:\$ )?(?:\./\.venv/bin/|uv run )?apg\s+([a-z][a-z0-9-]*)")


def audit_docs(repo_root: Path | None = None) -> dict[str, Any]:
	"""Audit contributor-facing APG documentation."""
	root = repo_root or Path(__file__).resolve().parents[1]
	required_doc_results = _required_doc_results(root)
	link_results = _link_results(root)
	command_results = _command_results(root)
	violations = [
		*[
			{"check": "required_docs", "path": item["path"], "message": item["message"]}
			for item in required_doc_results
			if not item["ok"]
		],
		*[
			{"check": "local_links", "path": item["path"], "target": item["target"], "message": item["message"]}
			for item in link_results
			if not item["ok"]
		],
		*[
			{"check": "documented_commands", "path": item["path"], "command": item["command"], "message": item["message"]}
			for item in command_results
			if not item["ok"]
		],
	]
	return {
		"format": DOCS_AUDIT_FORMAT,
		"ok": not violations,
		"repo_root": str(root),
		"required_docs": required_doc_results,
		"local_links": link_results,
		"documented_commands": command_results,
		"summary": {
			"required_doc_count": len(required_doc_results),
			"missing_required_doc_count": sum(1 for item in required_doc_results if not item["ok"]),
			"local_link_count": len(link_results),
			"broken_local_link_count": sum(1 for item in link_results if not item["ok"]),
			"documented_command_count": len(command_results),
			"unknown_documented_command_count": sum(1 for item in command_results if not item["ok"]),
			"violation_count": len(violations),
		},
		"violations": violations,
		"blocking_gaps": violations,
	}


def _required_doc_results(root: Path) -> list[dict[str, Any]]:
	results: list[dict[str, Any]] = []
	for relative_path in REQUIRED_DOCS:
		path = root / relative_path
		ok = path.is_file()
		results.append({
			"path": relative_path,
			"ok": ok,
			"message": "document found" if ok else "required documentation file is missing",
		})
	return results


def _link_results(root: Path) -> list[dict[str, Any]]:
	results: list[dict[str, Any]] = []
	for relative_path in NAVIGATION_DOCS:
		path = root / relative_path
		if not path.is_file():
			continue
		content = path.read_text(encoding="utf-8", errors="ignore")
		for match in LINK_PATTERN.finditer(content):
			target = match.group(1).strip()
			if _is_external_or_anchor(target):
				continue
			target_path = _resolve_link(path.parent, target)
			ok = target_path.exists()
			results.append({
				"path": relative_path,
				"target": target,
				"resolved_path": str(target_path),
				"ok": ok,
				"message": "local link target exists" if ok else "local link target is missing",
			})
	return results


def _command_results(root: Path) -> list[dict[str, Any]]:
	from cli.main import cli

	registered_commands = set(cli.commands)
	results: list[dict[str, Any]] = []
	seen: set[tuple[str, str]] = set()
	for relative_path in COMMAND_DOCS:
		path = root / relative_path
		if not path.is_file():
			continue
		lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
		for line in lines:
			match = APG_COMMAND_PATTERN.search(line)
			if not match:
				continue
			command = match.group(1)
			key = (relative_path, command)
			if key in seen:
				continue
			seen.add(key)
			ok = command in registered_commands
			results.append({
				"path": relative_path,
				"command": command,
				"ok": ok,
				"message": "top-level command is registered" if ok else "documented top-level command is not registered",
			})
	return results


def _is_external_or_anchor(target: str) -> bool:
	return (
		target.startswith("#")
		or "://" in target
		or target.startswith("mailto:")
		or target.startswith("tel:")
	)


def _resolve_link(base_dir: Path, target: str) -> Path:
	path_part = target.split("#", 1)[0]
	return (base_dir / path_part).resolve()
