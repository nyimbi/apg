"""Runtime helpers for the APG Custom Scripting Engine capability."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


SUPPORTED_LANGUAGES = {"python", "javascript", "apg"}
SCRIPT_STATES = {"draft", "review", "published", "retired"}
SANDBOX_STATES = {"ready", "running", "blocked", "retired"}
EXECUTION_STATUSES = {"queued", "running", "succeeded", "failed", "blocked", "cancelled"}
DANGEROUS_PERMISSIONS = {"network", "filesystem", "secrets", "subprocess", "system"}
DANGEROUS_IMPORTS = {"os", "subprocess", "socket", "requests", "urllib", "pathlib", "shutil"}
ISOLATION_MODES = {"process", "container", "wasm"}


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a deterministic short ID for local package runtime objects."""
	digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def source_checksum(source: str) -> str:
	return sha256((source or "").encode("utf-8")).hexdigest()


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def normalize_language(language: str) -> str:
	value = language.strip().lower()
	if value not in SUPPORTED_LANGUAGES:
		raise ValueError(f"unsupported_script_language:{language}")
	return value


def normalize_tags(tags: list[str] | None) -> list[str]:
	return sorted({tag.strip().lower() for tag in tags or [] if tag and tag.strip()})


def normalize_permissions(permissions: list[str] | None) -> list[str]:
	return sorted({permission.strip().lower() for permission in permissions or [] if permission and permission.strip()})


def normalize_script_state(state: str) -> str:
	value = state.strip().lower()
	if value not in SCRIPT_STATES:
		raise ValueError(f"unsupported_script_state:{state}")
	return value


def normalize_sandbox_state(state: str) -> str:
	value = state.strip().lower()
	if value not in SANDBOX_STATES:
		raise ValueError(f"unsupported_sandbox_state:{state}")
	return value


def normalize_isolation_mode(mode: str) -> str:
	value = mode.strip().lower()
	if value not in ISOLATION_MODES:
		raise ValueError(f"unsupported_isolation_mode:{mode}")
	return value


def detect_dangerous_permissions(language: str, source: str, requested_permissions: list[str]) -> list[str]:
	detected = set(requested_permissions) & DANGEROUS_PERMISSIONS
	if language == "python":
		for imported in python_imports(source):
			root = imported.split(".", 1)[0]
			if root in DANGEROUS_IMPORTS:
				if root in {"socket", "requests", "urllib"}:
					detected.add("network")
				elif root in {"os", "pathlib", "shutil"}:
					detected.add("filesystem")
				elif root == "subprocess":
					detected.add("subprocess")
		return sorted(detected)
	if "fetch(" in source or "http://" in source or "https://" in source:
		detected.add("network")
	return sorted(detected)


def python_imports(source: str) -> list[str]:
	try:
		tree = ast.parse(source or "")
	except SyntaxError:
		return []
	imports: list[str] = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			imports.extend(alias.name for alias in node.names)
		elif isinstance(node, ast.ImportFrom) and node.module:
			imports.append(node.module)
	return sorted(set(imports))


def validate_python_source(source: str) -> list[str]:
	try:
		ast.parse(source or "")
	except SyntaxError as exc:
		return [f"python_syntax_error:{exc.lineno}:{exc.offset}"]
	return []


def execution_status(exit_code: int, timed_out: bool = False, policy_blocked: bool = False) -> str:
	if policy_blocked:
		return "blocked"
	if timed_out:
		return "failed"
	if exit_code == 0:
		return "succeeded"
	return "failed"


def summarize_decision(result: dict[str, Any]) -> str:
	actions = result.get("actions") or []
	if not actions:
		return result.get("decision", "allow")
	return ",".join(action.get("reason", action.get("decision", "policy_action")) for action in actions)
