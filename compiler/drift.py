"""Cross-tool semantic drift verification for APG."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .compiler import APGCompiler, CodeGenConfig
from .semantic_model import build_semantic_model


DRIFT_REPORT_FORMAT = "apg.drift-report.v1"
DRIFT_AUDIT_FORMAT = "apg.drift-audit.v1"


def build_drift_report(source_file: Path) -> dict[str, Any]:
	"""Compare compiler, generated artifact, and generated runtime semantic models."""
	source_file = Path(source_file)
	report: dict[str, Any] = {
		"format": DRIFT_REPORT_FORMAT,
		"ok": False,
		"source": str(source_file),
		"surfaces": {},
		"comparisons": [],
		"summary": {},
		"errors": [],
		"warnings": [],
	}

	try:
		compiler_model = build_semantic_model(source_file)
	except Exception as error:
		report["errors"].append(f"compiler semantic model failed: {error}")
		return _finalize(report)

	compiler = APGCompiler(CodeGenConfig(target_language="python"))
	result = compiler.compile_file(source_file, target_language="python")
	report["warnings"].extend(str(warning) for warning in result.warnings)
	if not result.success:
		report["errors"].extend(str(error) for error in result.errors)
		return _finalize(report)

	try:
		artifact_model = json.loads(result.generated_files["semantic_model.json"])
	except KeyError:
		report["errors"].append("generated semantic_model.json artifact is missing")
		return _finalize(report)
	except json.JSONDecodeError as error:
		report["errors"].append(f"generated semantic_model.json is invalid JSON: {error}")
		return _finalize(report)

	try:
		runtime_model = _runtime_semantic_model(result.generated_files)
	except Exception as error:
		report["errors"].append(f"generated runtime semantic_model() failed: {error}")
		return _finalize(report)

	models = {
		"compiler": compiler_model,
		"generated_artifact": artifact_model,
		"generated_runtime": runtime_model,
	}
	report["surfaces"] = {
		name: _surface_summary(model)
		for name, model in models.items()
	}
	report["comparisons"] = [
		_compare_models("compiler", "generated_artifact", compiler_model, artifact_model),
		_compare_models("compiler", "generated_runtime", compiler_model, runtime_model),
		_compare_models("generated_artifact", "generated_runtime", artifact_model, runtime_model),
	]
	report["errors"].extend(
		f"{comparison['left']} != {comparison['right']}: {', '.join(comparison['differences'][:8])}"
		for comparison in report["comparisons"]
		if not comparison["ok"]
	)
	return _finalize(report)


def audit_drift_fixtures(fixture_root: Path | None = None) -> dict[str, Any]:
	"""Run drift reports over the checked-in semantic drift fixture catalog."""
	repo_root = Path(__file__).resolve().parent.parent
	fixture_root = fixture_root or repo_root / "tests" / "fixtures" / "drift"
	catalog_path = fixture_root / "catalog.json"
	reports: list[dict[str, Any]] = []
	errors: list[str] = []
	fixtures: list[dict[str, Any]] = []

	if not catalog_path.exists():
		return {
			"format": DRIFT_AUDIT_FORMAT,
			"ok": False,
			"fixture_catalog": str(catalog_path),
			"fixtures": [],
			"reports": [],
			"summary": {"fixture_count": 0, "passed": 0, "failed": 0},
			"errors": [f"semantic drift fixture catalog missing: {catalog_path}"],
			"warnings": [],
		}

	try:
		catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
		fixtures = list(catalog.get("fixtures", []))
	except json.JSONDecodeError as error:
		return {
			"format": DRIFT_AUDIT_FORMAT,
			"ok": False,
			"fixture_catalog": str(catalog_path),
			"fixtures": [],
			"reports": [],
			"summary": {"fixture_count": 0, "passed": 0, "failed": 0},
			"errors": [f"semantic drift fixture catalog is invalid JSON: {error}"],
			"warnings": [],
		}

	for fixture in fixtures:
		source = repo_root / str(fixture.get("source", ""))
		expected_ok = bool(fixture.get("expected_ok", True))
		if not source.exists():
			errors.append(f"fixture source missing: {source}")
			continue
		drift_report = build_drift_report(source)
		reports.append({
			"name": fixture.get("name") or source.parent.name,
			"source": str(source),
			"expected_ok": expected_ok,
			"actual_ok": drift_report["ok"],
			"ok": drift_report["ok"] == expected_ok,
			"summary": drift_report["summary"],
			"errors": drift_report["errors"],
		})

	for item in reports:
		if not item["ok"]:
			errors.append(f"drift fixture {item['name']} expected ok={item['expected_ok']} but got ok={item['actual_ok']}")

	passed = sum(1 for item in reports if item["ok"])
	return {
		"format": DRIFT_AUDIT_FORMAT,
		"ok": not errors and passed == len(fixtures),
		"fixture_catalog": str(catalog_path),
		"fixtures": fixtures,
		"reports": reports,
		"summary": {
			"fixture_count": len(fixtures),
			"passed": passed,
			"failed": len(fixtures) - passed,
		},
		"errors": errors,
		"warnings": [],
	}


def compare_semantic_models(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
	"""Return normalized semantic differences between two model payloads."""
	return _diff_values(_normalize_model(left), _normalize_model(right))


def _compare_models(left_name: str, right_name: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
	differences = compare_semantic_models(left, right)
	return {
		"left": left_name,
		"right": right_name,
		"ok": not differences,
		"difference_count": len(differences),
		"differences": differences,
	}


def _runtime_semantic_model(generated_files: dict[str, str]) -> dict[str, Any]:
	with tempfile.TemporaryDirectory(prefix="apg-drift-") as temporary_dir_name:
		temporary_dir = Path(temporary_dir_name)
		for file_name, content in generated_files.items():
			path = temporary_dir / file_name
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(content, encoding="utf-8")
		app = _import_generated_app(temporary_dir)
		try:
			return app.semantic_model()
		finally:
			_cleanup_generated_modules()


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


def _normalize_model(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": model.get("format"),
		"ok": model.get("ok"),
		"app": model.get("app", {}),
		"symbols": {
			symbol_id: {
				"kind": symbol.get("kind"),
				"name": symbol.get("name"),
			}
			for symbol_id, symbol in sorted(model.get("symbols", {}).items())
		},
		"tables": model.get("tables", {}),
		"views": model.get("views", {}),
		"flows": model.get("flows", {}),
		"operations": model.get("operations", {}),
		"rules": model.get("rules", {}),
		"roles": model.get("roles", {}),
		"security": model.get("security", {}),
		"agents": model.get("agents", {}),
		"llms": model.get("llms", {}),
		"capabilities": model.get("capabilities", {}),
		"composition": model.get("composition", {}),
		"contracts": model.get("contracts", {}),
		"deployment": {"target": model.get("deployment", {}).get("target")},
		"packages": model.get("packages", {}),
		"graphs": model.get("graphs", {}),
		"diagnostics": [
			{
				"code": diagnostic.get("code"),
				"severity": diagnostic.get("severity"),
				"message": diagnostic.get("message"),
			}
			for diagnostic in model.get("diagnostics", [])
		],
	}


def _surface_summary(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": model.get("format"),
		"ok": model.get("ok"),
		"app": model.get("app", {}).get("name"),
		"symbol_count": len(model.get("symbols", {})),
		"table_count": len(model.get("tables", {})),
		"agent_count": len(model.get("agents", {})),
		"capability_count": len(model.get("capabilities", {})),
		"diagnostic_count": len(model.get("diagnostics", [])),
	}


def _diff_values(left: Any, right: Any, path: str = "$") -> list[str]:
	if left == right:
		return []
	if isinstance(left, dict) and isinstance(right, dict):
		differences: list[str] = []
		for key in sorted(set(left) | set(right)):
			child_path = f"{path}.{key}"
			if key not in left:
				differences.append(f"{child_path} missing from left")
			elif key not in right:
				differences.append(f"{child_path} missing from right")
			else:
				differences.extend(_diff_values(left[key], right[key], child_path))
		return differences
	if isinstance(left, list) and isinstance(right, list):
		if len(left) != len(right):
			return [f"{path} length {len(left)} != {len(right)}"]
		differences: list[str] = []
		for index, (left_item, right_item) in enumerate(zip(left, right)):
			differences.extend(_diff_values(left_item, right_item, f"{path}[{index}]"))
		return differences
	return [f"{path} differs"]


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
	comparisons = report.get("comparisons", [])
	report["summary"] = {
		"surface_count": len(report.get("surfaces", {})),
		"comparison_count": len(comparisons),
		"drift_count": sum(1 for comparison in comparisons if not comparison.get("ok")),
		"error_count": len(report.get("errors", [])),
		"warning_count": len(report.get("warnings", [])),
	}
	report["ok"] = not report.get("errors") and all(comparison.get("ok") for comparison in comparisons)
	return report
