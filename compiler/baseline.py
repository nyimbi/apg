"""Compiler bed-down audit for curated APG examples."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .compiler import APGCompiler
from .graphs import SUPPORTED_GRAPH_KINDS, build_graph_suite
from .release import build_release_report
from .semantic_model import build_semantic_model


EXPECTED_NUMBERED_EXAMPLES = 20
BED_DOWN_DOMAINS = {
	"records": "tables",
	"screens": "screens",
	"workflows": "flows",
	"agents": "agents",
	"capabilities": "capabilities",
	"application_composition": "composition",
	"visual_theming": "theme",
	"i18n": "i18n",
	"bytewax_streaming": "bytewax",
}


def build_compiler_baseline_report(
	examples_dir: Path,
	expected_examples: int = EXPECTED_NUMBERED_EXAMPLES,
) -> dict[str, Any]:
	"""Run the APG compiler bed-down gate over numbered examples."""
	sources = sorted(examples_dir.glob("[0-9][0-9]_*/main.apg"))
	example_reports: list[dict[str, Any]] = []
	domain_sources: dict[str, list[str]] = {domain: [] for domain in BED_DOWN_DOMAINS}

	for source in sources:
		example_report = _audit_example(source)
		example_reports.append(example_report)
		_collect_domain_coverage(example_report.get("model", {}), source, domain_sources)

	domain_coverage = {
		domain: {
			"ok": bool(paths),
			"sources": paths,
		}
		for domain, paths in domain_sources.items()
	}
	checks = {
		"numbered_examples_present": len(sources) == expected_examples,
		"domain_coverage_ok": all(item["ok"] for item in domain_coverage.values()),
		"all_examples_ok": all(example["ok"] for example in example_reports),
		"python_target_only": True,
	}
	return {
		"format": "apg.compiler-baseline-report.v1",
		"ok": all(checks.values()),
		"examples_dir": str(examples_dir),
		"expected_examples": expected_examples,
		"example_count": len(sources),
		"checks": checks,
		"domains": domain_coverage,
		"summary": {
			"passed_examples": sum(1 for example in example_reports if example["ok"]),
			"failed_examples": sum(1 for example in example_reports if not example["ok"]),
			"graph_kinds": list(SUPPORTED_GRAPH_KINDS),
		},
		"examples": [
			_example_public_report(example)
			for example in example_reports
		],
	}


def _audit_example(source: Path) -> dict[str, Any]:
	errors: list[str] = []
	warnings: list[str] = []
	model: dict[str, Any] = {}
	graph_suite: dict[str, Any] = {}
	release: dict[str, Any] = {}
	compile_verify: dict[str, Any] = {}

	try:
		model = build_semantic_model(source)
	except Exception as error:
		errors.append(f"model failed: {error}")

	if model:
		for diagnostic in model.get("diagnostics", []):
			if diagnostic.get("severity") == "error":
				errors.append(f"lint/model: {diagnostic.get('message')}")
			elif diagnostic.get("severity") == "warning":
				warnings.append(str(diagnostic.get("message")))

	try:
		graph_suite = build_graph_suite(source)
	except Exception as error:
		errors.append(f"graph-suite failed: {error}")

	try:
		release = build_release_report(source, target="python")
		if not release.get("ok"):
			errors.extend(f"release: {error}" for error in release.get("errors", []))
			warnings.extend(f"release: {warning}" for warning in release.get("warnings", []))
	except Exception as error:
		errors.append(f"release failed: {error}")

	try:
		compile_verify = _compile_verify_in_temp(source)
		if not compile_verify["ok"]:
			errors.extend(f"compile-verify: {error}" for error in compile_verify.get("errors", []))
	except Exception as error:
		errors.append(f"compile-verify failed: {error}")

	checks = {
		"lint_ok": bool(model) and not any(
			diagnostic.get("severity") == "error"
			for diagnostic in model.get("diagnostics", [])
		),
		"validate_ok": bool(model) and model.get("ok") is True,
		"model_ok": model.get("format") == "apg.semantic-model.v1" and model.get("ok") is True,
		"graph_suite_ok": graph_suite.get("format") == "apg.graph-suite-report.v1" and graph_suite.get("ok") is True,
		"release_ok": release.get("format") == "apg.release-report.v1" and release.get("ok") is True,
		"compile_verify_ok": compile_verify.get("ok") is True,
	}
	return {
		"name": source.parent.name,
		"source": str(source),
		"ok": not errors and all(checks.values()),
		"checks": checks,
		"errors": errors,
		"warnings": warnings,
		"model": model,
		"graph_suite": graph_suite,
		"release": release,
		"compile_verify": compile_verify,
	}


def _compile_verify_in_temp(source: Path) -> dict[str, Any]:
	compiler = APGCompiler()
	result = compiler.compile_file(source, target_language="python")
	errors = [str(error) for error in result.errors]
	if not result.success:
		return {"ok": False, "generated_file_count": len(result.generated_files), "errors": errors}

	with tempfile.TemporaryDirectory(prefix="apg-baseline-") as temporary_dir_name:
		output_dir = Path(temporary_dir_name)
		for file_name, content in result.generated_files.items():
			file_path = output_dir / file_name
			file_path.parent.mkdir(parents=True, exist_ok=True)
			file_path.write_text(content, encoding="utf-8")
		for label, command in (
			("self-test", [sys.executable, "app.py", "--self-test"]),
			("smoke-test", [sys.executable, "smoke_test.py"]),
		):
			completed = subprocess.run(
				command,
				cwd=output_dir,
				check=False,
				capture_output=True,
				text=True,
			)
			if completed.returncode != 0:
				errors.append(
					f"{label} exited {completed.returncode}: "
					f"{completed.stdout.rstrip()} {completed.stderr.rstrip()}".strip()
				)
				return {
					"ok": False,
					"generated_file_count": len(result.generated_files),
					"errors": errors,
				}

	return {
		"ok": True,
		"generated_file_count": len(result.generated_files),
		"errors": [],
	}


def _collect_domain_coverage(
	model: dict[str, Any],
	source: Path,
	domain_sources: dict[str, list[str]],
) -> None:
	if model.get("tables"):
		domain_sources["records"].append(str(source))
	if model.get("flows"):
		domain_sources["workflows"].append(str(source))
	if model.get("agents"):
		domain_sources["agents"].append(str(source))
	if model.get("capabilities"):
		domain_sources["capabilities"].append(str(source))
	if model.get("views") or any(
		capability.get("screens")
		for capability in model.get("capabilities", {}).values()
	):
		domain_sources["screens"].append(str(source))
	if model.get("composition", {}).get("applications"):
		domain_sources["application_composition"].append(str(source))
	if _model_has_capability_value(model, "theme"):
		domain_sources["visual_theming"].append(str(source))
	if _model_has_capability_value(model, "i18n"):
		domain_sources["i18n"].append(str(source))
	if any(
		capability.get("streaming", {}).get("processor") == "bytewax"
		for capability in model.get("capabilities", {}).values()
	):
		domain_sources["bytewax_streaming"].append(str(source))


def _model_has_capability_value(model: dict[str, Any], key: str) -> bool:
	return any(
		bool(capability.get(key))
		for capability in model.get("capabilities", {}).values()
	)


def _example_public_report(example: dict[str, Any]) -> dict[str, Any]:
	model = example.get("model", {})
	graph_suite = example.get("graph_suite", {})
	release = example.get("release", {})
	return {
		"name": example["name"],
		"source": example["source"],
		"ok": example["ok"],
		"checks": example["checks"],
		"errors": example["errors"],
		"warnings": example["warnings"],
		"model": {
			"format": model.get("format"),
			"ok": model.get("ok"),
			"table_count": len(model.get("tables", {})),
			"flow_count": len(model.get("flows", {})),
			"agent_count": len(model.get("agents", {})),
			"capability_count": len(model.get("capabilities", {})),
			"diagnostic_count": len(model.get("diagnostics", [])),
		},
		"graph_suite": {
			"format": graph_suite.get("format"),
			"ok": graph_suite.get("ok"),
			"graph_kinds": graph_suite.get("graph_kinds", []),
		},
		"release": {
			"format": release.get("format"),
			"ok": release.get("ok"),
			"generated_file_count": release.get("generated", {}).get("file_count", 0),
			"self_test": release.get("evidence", {}).get("self_test", {}),
		},
		"compile_verify": example.get("compile_verify", {}),
	}
