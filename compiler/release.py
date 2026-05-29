"""Release evidence builder for APG generated applications."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

from .compiler import APGCompiler, CodeGenConfig, CompilationResult
from .code_generator import CodeGenerator
from .linting import lint_path


def build_release_report(source_file: Path, target: str = "python", catalog: Path | None = None) -> dict[str, Any]:
	"""Compile APG source and verify generated application release evidence."""
	normalized_target = CodeGenerator.normalize_target(target)
	report: dict[str, Any] = {
		"format": "apg.release-report.v1",
		"ok": False,
		"source": str(source_file),
		"target": normalized_target,
		"preflight": _empty_preflight_report(catalog),
		"generated": {},
		"evidence": {},
		"errors": [],
		"warnings": [],
	}

	if normalized_target != "python":
		report["errors"].append(f"APG release target must be 'python', not {target!r}")
		return report
	if catalog is not None:
		report["preflight"] = _catalog_preflight_report(source_file, catalog)
		if not report["preflight"]["ok"]:
			report["errors"].extend(report["preflight"]["errors"])
			return report

	compiler = APGCompiler(CodeGenConfig(target_language=normalized_target))
	result = compiler.compile_file(source_file, target_language=normalized_target)
	report["warnings"].extend(str(warning) for warning in result.warnings)
	if not result.success:
		report["errors"].extend(str(error) for error in result.errors)
		report["generated"] = _generated_summary(result)
		return report

	report["generated"] = _generated_summary(result)
	try:
		report["evidence"] = _verify_generated_files(result.generated_files)
	except Exception as error:
		report["errors"].append(f"release evidence verification failed: {error}")
		return report

	report["errors"].extend(report["evidence"].get("errors", []))
	report["warnings"].extend(report["evidence"].get("warnings", []))
	report["ok"] = not report["errors"] and bool(report["evidence"].get("self_test", {}).get("passed"))
	return report


def _empty_preflight_report(catalog: Path | None) -> dict[str, Any]:
	return {
		"checked": catalog is not None,
		"ok": catalog is None,
		"catalog": str(catalog) if catalog is not None else None,
		"catalog_kind": None,
		"errors": [],
	}


def _catalog_preflight_report(source_file: Path, catalog: Path) -> dict[str, Any]:
	lint_report = lint_path(source_file, catalog=catalog)
	catalog_report = lint_report.get("capability_catalog", {})
	errors = [
		f"{diagnostic.get('code')}: {diagnostic.get('message')}"
		for diagnostic in lint_report.get("diagnostics", [])
		if diagnostic.get("severity") == "error"
	]
	return {
		"checked": True,
		"ok": bool(lint_report.get("ok")),
		"catalog": str(catalog),
		"catalog_kind": catalog_report.get("catalog_kind"),
		"contract_count": catalog_report.get("contract_count", 0),
		"declared_capabilities": catalog_report.get("declared_capabilities", []),
		"matched_capabilities": catalog_report.get("matched_capabilities", []),
		"missing_capabilities": catalog_report.get("missing_capabilities", []),
		"errors": errors,
		"lint": lint_report,
	}


def _generated_summary(result: CompilationResult) -> dict[str, Any]:
	files = sorted(result.generated_files)
	return {
		"success": result.success,
		"target": result.target_language,
		"file_count": len(files),
		"files": files,
		"python_files": sorted(file_name for file_name in files if file_name.endswith(".py")),
		"semantic_model_artifact": "semantic_model.json" in result.generated_files,
	}


def _verify_generated_files(generated_files: dict[str, str]) -> dict[str, Any]:
	required_files = {
		"app.py",
		"__init__.py",
		"README.md",
		"requirements.txt",
		"semantic_model.json",
		"smoke_test.py",
	}
	errors: list[str] = []
	warnings: list[str] = []
	missing = sorted(required_files.difference(generated_files))
	for file_name in missing:
		errors.append(f"missing generated artifact {file_name}")

	python_compile_errors: list[str] = []
	for file_name, content in sorted(generated_files.items()):
		if not file_name.endswith(".py"):
			continue
		try:
			compile(content, file_name, "exec")
		except SyntaxError as error:
			python_compile_errors.append(f"{file_name}: {error}")
	errors.extend(python_compile_errors)

	with tempfile.TemporaryDirectory(prefix="apg-release-") as temporary_dir_name:
		temporary_dir = Path(temporary_dir_name)
		for file_name, content in generated_files.items():
			path = temporary_dir / file_name
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(content, encoding="utf-8")
		app = _import_generated_app(temporary_dir)
		try:
			self_test = app.self_test()
			component_manifest = app.component_manifest()
			semantic_model = app.semantic_model()
			openapi_contract = app.validate_openapi_contract()
			component_contract = app.validate_component_manifest_contract()
			route_contract = app.validate_route_dispatch_contract()
		finally:
			_cleanup_generated_modules()

	if not self_test.get("passed"):
		errors.append("generated self-test did not pass")
	for contract_name, contract in {
		"openapi_contract": openapi_contract,
		"component_manifest": component_contract,
		"route_dispatch": route_contract,
	}.items():
		for error in contract.get("errors", []):
			errors.append(f"{contract_name}: {error}")
		for warning in contract.get("warnings", []):
			warnings.append(f"{contract_name}: {warning}")
	if semantic_model.get("format") != "apg.semantic-model.v1":
		errors.append("semantic_model() did not return apg.semantic-model.v1")
	if not component_manifest.get("interfaces", {}).get("semantic_model"):
		errors.append("component manifest does not expose semantic_model interface")

	return {
		"errors": errors,
		"warnings": warnings,
		"python_compile_errors": python_compile_errors,
		"self_test": {
			"passed": bool(self_test.get("passed")),
			"status": self_test.get("status"),
			"route_count": len(self_test.get("routes", [])),
		},
		"semantic_model": {
			"format": semantic_model.get("format"),
			"ok": semantic_model.get("ok"),
			"symbol_count": len(semantic_model.get("symbols", {})),
			"agent_count": len(semantic_model.get("agents", {})),
			"capability_count": len(semantic_model.get("capabilities", {})),
			"table_count": len(semantic_model.get("tables", {})),
		},
		"component_manifest": {
			"kind": component_manifest.get("kind"),
			"target": component_manifest.get("target"),
			"http_path_count": len(component_manifest.get("interfaces", {}).get("http", {}).get("paths", [])),
			"semantic_model": component_manifest.get("interfaces", {}).get("semantic_model"),
			"artifact_count": len(component_manifest.get("deployment", {}).get("artifacts", [])),
		},
		"contracts": {
			"openapi": {
				"errors": openapi_contract.get("errors", []),
				"path_count": openapi_contract.get("path_count"),
				"schema_count": openapi_contract.get("schema_count"),
			},
			"component_manifest": {
				"errors": component_contract.get("errors", []),
				"artifact_count": component_contract.get("artifact_count"),
				"command_count": component_contract.get("command_count"),
			},
			"route_dispatch": {
				"errors": route_contract.get("errors", []),
				"route_count": route_contract.get("route_count"),
				"method_count": route_contract.get("method_count"),
			},
		},
	}


def _import_generated_app(generated_dir: Path) -> Any:
	_cleanup_generated_modules()
	sys.path.insert(0, str(generated_dir))
	try:
		spec = importlib.util.spec_from_file_location("app", generated_dir / "app.py")
		if spec is None or spec.loader is None:
			raise RuntimeError("could not load generated app.py")
		module = importlib.util.module_from_spec(spec)
		sys.modules["app"] = module
		spec.loader.exec_module(module)
		return module
	finally:
		try:
			sys.path.remove(str(generated_dir))
		except ValueError:
			pass


def _cleanup_generated_modules() -> None:
	for module_name in ("app", "ai_agents", "apg_capabilities", "apg_application"):
		sys.modules.pop(module_name, None)
