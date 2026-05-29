"""APG installation and serviceability health report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


DOCTOR_REPORT_FORMAT = "apg.doctor-report.v1"
APG_VERSION = "1.0.0"
LANGUAGE_SPECIFICATION = "v11"
TARGET_LANGUAGE = "Python"

REQUIRED_PACKAGES = {
	"antlr4": "antlr4-python3-runtime",
	"click": "click",
	"rich": "rich",
}

OPTIONAL_PACKAGES = {
	"pygls": "pygls",
	"lsprotocol": "lsprotocol",
}


def build_doctor_report(repo_root: Path | None = None) -> dict[str, Any]:
	"""Build a machine-readable APG environment health report."""
	root = repo_root or Path(__file__).resolve().parents[1]
	checks: list[dict[str, Any]] = []
	checks.append(_python_version_check())
	checks.extend(_package_checks(REQUIRED_PACKAGES, required=True))
	checks.extend(_component_checks(root))
	checks.append(_parser_artifact_check(root))
	checks.append(_capability_registry_check(root))
	checks.extend(_package_checks(OPTIONAL_PACKAGES, required=False))

	blocking_failures = [
		check for check in checks
		if check["required"] and not check["ok"]
	]
	warnings = [
		check for check in checks
		if not check["required"] and not check["available"]
	]
	return {
		"format": DOCTOR_REPORT_FORMAT,
		"ok": not blocking_failures,
		"version": APG_VERSION,
		"language_specification": LANGUAGE_SPECIFICATION,
		"target_language": TARGET_LANGUAGE,
		"repo_root": str(root),
		"python": {
			"version": sys.version.split()[0],
			"executable": sys.executable,
		},
		"checks": checks,
		"summary": {
			"check_count": len(checks),
			"required_check_count": sum(1 for check in checks if check["required"]),
			"optional_check_count": sum(1 for check in checks if not check["required"]),
			"passing_required_check_count": sum(1 for check in checks if check["required"] and check["ok"]),
			"blocking_failure_count": len(blocking_failures),
			"warning_count": len(warnings),
		},
		"blocking_failures": blocking_failures,
		"warnings": warnings,
	}


def _python_version_check() -> dict[str, Any]:
	ok = sys.version_info >= (3, 10)
	return {
		"name": "python_version",
		"kind": "runtime",
		"required": True,
		"ok": ok,
		"available": ok,
		"detail": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
		"message": "requires Python 3.10+" if not ok else "Python runtime is supported",
	}


def _package_checks(packages: dict[str, str], required: bool) -> list[dict[str, Any]]:
	checks: list[dict[str, Any]] = []
	for import_name, package_name in packages.items():
		available = importlib.util.find_spec(import_name) is not None
		checks.append({
			"name": f"package:{package_name}",
			"kind": "package",
			"required": required,
			"ok": available if required else True,
			"available": available,
			"detail": import_name,
			"message": "import available" if available else "optional package not installed" if not required else "required package not installed",
		})
	return checks


def _component_checks(root: Path) -> list[dict[str, Any]]:
	components = [
		("grammar_file", "spec/apg.g4"),
		("compiler", "compiler/compiler.py"),
		("code_generator", "compiler/code_generator.py"),
		("language_server", "language_server/server.py"),
		("vscode_extension", "vscode-extension/package.json"),
		("templates", "templates"),
		("capability_registry", "capabilities/capability_contract_registry.py"),
	]
	checks: list[dict[str, Any]] = []
	for name, relative_path in components:
		path = root / relative_path
		available = path.exists()
		checks.append({
			"name": f"component:{name}",
			"kind": "component",
			"required": True,
			"ok": available,
			"available": available,
			"path": str(path),
			"message": "component found" if available else f"component missing: {relative_path}",
		})
	return checks


def _parser_artifact_check(root: Path) -> dict[str, Any]:
	generated_dir = root / "spec"
	required_artifacts = ["apgLexer.py", "apgParser.py", "apgVisitor.py"]
	missing = [
		name for name in required_artifacts
		if not (generated_dir / name).exists()
	]
	return {
		"name": "parser_artifacts",
		"kind": "grammar",
		"required": True,
		"ok": not missing,
		"available": not missing,
		"path": str(generated_dir),
		"required_artifacts": required_artifacts,
		"missing_artifacts": missing,
		"message": "Generated parser found" if not missing else f"missing parser artifacts: {', '.join(missing)}",
	}


def _capability_registry_check(root: Path) -> dict[str, Any]:
	try:
		from capabilities.capability_contract_registry import validate_contract_registry

		report = validate_contract_registry()
	except Exception as error:  # pragma: no cover - defensive health report
		return {
			"name": "capability_contract_registry",
			"kind": "capability",
			"required": True,
			"ok": False,
			"available": False,
			"message": f"capability registry validation failed: {error}",
		}

	error_count = int(report.get("summary", {}).get("error_count", 0))
	if "error_count" in report:
		error_count = int(report["error_count"])
	contract_count = int(report.get("contract_count", 0))
	ok = bool(report.get("ok", report.get("valid"))) and contract_count > 0 and error_count == 0
	return {
		"name": "capability_contract_registry",
		"kind": "capability",
		"required": True,
		"ok": ok,
		"available": contract_count > 0,
		"contract_count": contract_count,
		"error_count": error_count,
		"message": f"{contract_count} capability contracts valid" if ok else "capability registry has validation errors",
	}
