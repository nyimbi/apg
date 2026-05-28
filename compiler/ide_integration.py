"""APG IDE integration audit utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IDE_AUDIT_FORMAT = "apg.ide-audit.v1"

REQUIRED_VSCODE_COMMANDS = [
	"apg.compile",
	"apg.validateSyntax",
	"apg.lint",
	"apg.format",
	"apg.graph",
	"apg.explain",
	"apg.package",
	"apg.capabilities",
	"apg.restartLanguageServer",
]

REQUIRED_EXTENSION_CLI_FRAGMENTS = [
	"compile',",
	"--target', 'python'",
	"validate',",
	"lint',",
	"format',",
	"graph-suite',",
	"explain',",
	"package',",
	"capabilities', 'contracts'",
]

FORBIDDEN_EXTENSION_FRAGMENTS = [
	"flask-appbuilder",
	"apg build",
	"--target flask",
	"--target django",
]


def audit_vscode_extension(repo_root: Path | None = None) -> dict[str, Any]:
	"""Audit the checked-in VS Code extension against APG tooling contracts."""
	root = repo_root or Path(__file__).resolve().parents[1]
	extension_root = root / "vscode-extension"
	package_path = extension_root / "package.json"
	readme_path = extension_root / "README.md"
	source_path = extension_root / "src" / "extension.ts"
	checks: list[dict[str, Any]] = []

	package: dict[str, Any] = {}
	readme = ""
	source = ""
	if package_path.exists():
		try:
			package = json.loads(package_path.read_text(encoding="utf-8"))
		except json.JSONDecodeError as error:
			checks.append(_check("package-json-valid", False, f"package.json is invalid JSON: {error}"))
	else:
		checks.append(_check("package-json-exists", False, f"missing {package_path}"))

	if source_path.exists():
		source = source_path.read_text(encoding="utf-8")
	else:
		checks.append(_check("extension-source-exists", False, f"missing {source_path}"))

	if readme_path.exists():
		readme = readme_path.read_text(encoding="utf-8")
	else:
		checks.append(_check("extension-readme-exists", False, f"missing {readme_path}"))

	if package:
		commands = [
			str(command.get("command", ""))
			for command in package.get("contributes", {}).get("commands", [])
		]
		missing_commands = sorted(set(REQUIRED_VSCODE_COMMANDS).difference(commands))
		checks.append(_check(
			"required-commands",
			not missing_commands,
			"VS Code command palette exposes current APG tooling commands",
			{"missing": missing_commands, "commands": commands},
		))

		target_config = package.get("contributes", {}).get("configuration", {}).get("properties", {}).get("apg.compiler.target", {})
		checks.append(_check(
			"python-target-config",
			target_config.get("default") == "python" and target_config.get("enum") == ["python"],
			"VS Code compiler target is python-only",
			{"default": target_config.get("default"), "enum": target_config.get("enum")},
		))

		activation_events = package.get("activationEvents", [])
		checks.append(_check(
			"apg-language-activation",
			"onLanguage:apg" in activation_events,
			"VS Code extension activates for .apg files",
			{"activation_events": activation_events},
		))

		missing_contribution_paths = _missing_contribution_paths(extension_root, package)
		checks.append(_check(
			"contributed-files-exist",
			not missing_contribution_paths,
			"VS Code grammar, snippets, icon, and theme contribution files exist",
			{"missing": missing_contribution_paths},
		))

	if source:
		missing_fragments = [
			fragment
			for fragment in REQUIRED_EXTENSION_CLI_FRAGMENTS
			if fragment not in source
		]
		forbidden_fragments = [
			fragment
			for fragment in FORBIDDEN_EXTENSION_FRAGMENTS
			if fragment in source or fragment in json.dumps(package) or fragment in readme
		]
		checks.append(_check(
			"extension-cli-contracts",
			not missing_fragments,
			"Extension command implementation calls current APG CLI contracts",
			{"missing_fragments": missing_fragments},
		))
		checks.append(_check(
			"no-framework-target-drift",
			not forbidden_fragments,
			"Extension no longer advertises framework compile targets or apg build",
			{"forbidden_fragments": forbidden_fragments},
		))

	return {
		"format": IDE_AUDIT_FORMAT,
		"ok": all(check["ok"] for check in checks),
		"surface": "vscode",
		"extension_root": str(extension_root),
		"checks": checks,
		"summary": {
			"check_count": len(checks),
			"passing": sum(1 for check in checks if check["ok"]),
			"failing": sum(1 for check in checks if not check["ok"]),
		},
	}


def _missing_contribution_paths(extension_root: Path, package: dict[str, Any]) -> list[str]:
	contributes = package.get("contributes", {})
	relative_paths: list[str] = []
	for language in contributes.get("languages", []):
		if language.get("configuration"):
			relative_paths.append(str(language["configuration"]))
		icon = language.get("icon", {})
		for key in ["light", "dark"]:
			if icon.get(key):
				relative_paths.append(str(icon[key]))
	for grammar in contributes.get("grammars", []):
		if grammar.get("path"):
			relative_paths.append(str(grammar["path"]))
	for snippet in contributes.get("snippets", []):
		if snippet.get("path"):
			relative_paths.append(str(snippet["path"]))
	for theme in contributes.get("themes", []):
		if theme.get("path"):
			relative_paths.append(str(theme["path"]))
	return sorted(
		path
		for path in set(relative_paths)
		if not (extension_root / path).exists()
	)


def _check(name: str, ok: bool, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
	return {
		"name": name,
		"ok": ok,
		"message": message,
		"details": details or {},
	}
