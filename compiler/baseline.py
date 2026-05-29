"""Compiler bed-down audit for curated APG examples."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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
	refresh_outputs: bool = False,
) -> dict[str, Any]:
	"""Run the APG compiler bed-down gate over numbered examples."""
	sources = sorted(examples_dir.glob("[0-9][0-9]_*/main.apg"))
	example_reports: list[dict[str, Any]] = []
	domain_sources: dict[str, list[str]] = {domain: [] for domain in BED_DOWN_DOMAINS}

	for source in sources:
		example_report = _audit_example(source, refresh_outputs=refresh_outputs)
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
			"checked_generated_python_files": sum(
				example.get("compile_verify", {})
				.get("source_hygiene", {})
				.get("checked_python_files", 0)
				for example in example_reports
			),
			"generated_source_hygiene_violations": sum(
				len(
					example.get("compile_verify", {})
					.get("source_hygiene", {})
					.get("violations", [])
				)
				for example in example_reports
			),
			"current_output_directories": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("output_sync", {})
				.get("ok")
				is True
			),
			"stale_output_directories": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("output_sync", {})
				.get("ok")
				is False
			),
			"checked_output_runtime_passed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_runtime", {})
				.get("ok")
				is True
			),
			"checked_output_runtime_failed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_runtime", {})
				.get("ok")
				is False
			),
			"checked_output_http_passed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_http", {})
				.get("ok")
				is True
			),
			"checked_output_http_failed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_http", {})
				.get("ok")
				is False
			),
			"checked_output_domain_http_passed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_http", {})
				.get("domain_ok")
				is True
			),
			"checked_output_domain_http_failed": sum(
				1
				for example in example_reports
				if example.get("compile_verify", {})
				.get("checked_output_http", {})
				.get("domain_ok")
				is False
			),
			"graph_kinds": list(SUPPORTED_GRAPH_KINDS),
		},
		"examples": [
			_example_public_report(example)
			for example in example_reports
		],
	}


def _audit_example(source: Path, refresh_outputs: bool = False) -> dict[str, Any]:
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
		compile_verify = _compile_verify_in_temp(source, refresh_outputs=refresh_outputs)
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
		"generated_source_hygiene_ok": compile_verify.get("source_hygiene", {}).get("ok") is True,
		"checked_output_current": compile_verify.get("output_sync", {}).get("ok") is True,
		"checked_output_runtime_ok": compile_verify.get("checked_output_runtime", {}).get("ok") is True,
		"checked_output_http_ok": compile_verify.get("checked_output_http", {}).get("ok") is True,
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


def _compile_verify_in_temp(source: Path, refresh_outputs: bool = False) -> dict[str, Any]:
	compiler = APGCompiler()
	result = compiler.compile_file(source, target_language="python")
	errors = [str(error) for error in result.errors]
	if not result.success:
		return {"ok": False, "generated_file_count": len(result.generated_files), "errors": errors}

	source_hygiene = _generated_source_hygiene(result.generated_files)
	if not source_hygiene["ok"]:
		errors.extend(source_hygiene["violations"])
	if refresh_outputs:
		_refresh_output_directory(source.parent / "output", result.generated_files)
	output_sync = _checked_output_sync(source.parent / "output", result.generated_files)
	if not output_sync["ok"]:
		errors.extend(output_sync["violations"])
	checked_output_runtime = _run_generated_runtime_checks(source.parent / "output")
	if not checked_output_runtime["ok"]:
		errors.extend(checked_output_runtime["errors"])
	checked_output_http = _run_checked_output_http_checks(source.parent / "output")
	if not checked_output_http["ok"]:
		errors.extend(checked_output_http["errors"])

	with tempfile.TemporaryDirectory(prefix="apg-baseline-") as temporary_dir_name:
		output_dir = Path(temporary_dir_name)
		for file_name, content in result.generated_files.items():
			file_path = output_dir / file_name
			file_path.parent.mkdir(parents=True, exist_ok=True)
			file_path.write_text(content, encoding="utf-8")
		temp_runtime = _run_generated_runtime_checks(output_dir)
		if not temp_runtime["ok"]:
			errors.extend(temp_runtime["errors"])

	return {
		"ok": not errors,
		"generated_file_count": len(result.generated_files),
		"errors": errors,
		"source_hygiene": source_hygiene,
		"output_sync": output_sync,
		"runtime": temp_runtime,
		"checked_output_runtime": checked_output_runtime,
		"checked_output_http": checked_output_http,
	}


def _run_generated_runtime_checks(output_dir: Path) -> dict[str, Any]:
	if not output_dir.exists():
		return {
			"ok": False,
			"output_dir": str(output_dir),
			"checks": [],
			"errors": [f"{output_dir}: output directory is missing"],
		}
	checks: list[dict[str, Any]] = []
	errors: list[str] = []
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
		check = {
			"label": label,
			"returncode": completed.returncode,
			"ok": completed.returncode == 0,
		}
		checks.append(check)
		if completed.returncode != 0:
			errors.append(
				f"{output_dir}: {label} exited {completed.returncode}: "
				f"{completed.stdout.rstrip()} {completed.stderr.rstrip()}".strip()
			)
	return {
		"ok": not errors,
		"output_dir": str(output_dir),
		"checks": checks,
		"errors": errors,
	}


def _run_checked_output_http_checks(output_dir: Path) -> dict[str, Any]:
	if not output_dir.exists():
		return {
			"ok": False,
			"output_dir": str(output_dir),
			"routes": [],
			"errors": [f"{output_dir}: output directory is missing"],
		}

	port = _free_tcp_port()
	errors: list[str] = []
	routes: list[dict[str, Any]] = []
	domain_checks: list[dict[str, Any]] = []
	with tempfile.TemporaryDirectory(prefix="apg-output-http-") as data_dir_name:
		env = dict(os.environ)
		env.pop("APG_API_KEY", None)
		env["APG_DATA_FILE"] = str(Path(data_dir_name) / "records.json")
		process = subprocess.Popen(
			[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
			cwd=output_dir,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			env=env,
		)
		try:
			base_url = f"http://127.0.0.1:{port}"
			started = _wait_for_http_route(base_url + "/health", process)
			if not started:
				stdout, stderr = _terminate_process(process)
				return {
					"ok": False,
					"output_dir": str(output_dir),
					"port": port,
					"routes": routes,
					"domain_ok": False,
					"domain_checks": domain_checks,
					"errors": [
						f"{output_dir}: HTTP server did not serve /health on port {port}: "
						f"{stdout.rstrip()} {stderr.rstrip()}".strip()
					],
				}
			for route in ("/health", "/openapi.json", "/component.json", "/semantic-model.json", "/self-test"):
				result = _fetch_http_route(base_url + route)
				routes.append({"route": route, **result})
				if not result["ok"]:
					errors.append(f"{output_dir}: GET {route} failed with {result['status']}")
			domain_checks = _run_domain_http_checks(base_url)
			for check in domain_checks:
				if not check["ok"]:
					errors.append(f"{output_dir}: {check['name']} failed")
		finally:
			_terminate_process(process)

	return {
		"ok": not errors,
		"domain_ok": all(check.get("ok") for check in domain_checks),
		"output_dir": str(output_dir),
		"port": port,
		"routes": routes,
		"domain_checks": domain_checks,
		"errors": errors,
	}


def _run_domain_http_checks(base_url: str) -> list[dict[str, Any]]:
	checks: list[dict[str, Any]] = []
	entities_response = _fetch_json_route(base_url + "/entities")
	records_response = _fetch_json_route(base_url + "/records")
	relationships_response = _fetch_json_route(base_url + "/relationships")
	checks.extend([
		_json_route_check("entities_catalog", entities_response),
		_json_route_check("records_catalog", records_response),
		_json_route_check("relationships_catalog", relationships_response),
	])

	entities = entities_response.get("json", {}).get("entities", [])
	record_entities = [
		entity
		for entity in entities
		if isinstance(entity, dict)
		and entity.get("type") in {"entity", "table", "database"}
		and entity.get("fields")
	]
	if record_entities:
		record_check = _run_record_crud_http_check(base_url, record_entities[0])
		checks.append(record_check)

	workflows_response = _fetch_json_route(base_url + "/workflows")
	checks.append(_json_route_check("workflows_catalog", workflows_response))
	workflows = workflows_response.get("json", {}).get("workflows", {})
	if isinstance(workflows, dict) and workflows:
		workflow_name = sorted(workflows)[0]
		checks.append(_run_workflow_http_check(base_url, str(workflow_name)))

	capabilities_response = _fetch_json_route(base_url + "/capabilities")
	checks.append(_json_route_check("capabilities_catalog", capabilities_response))
	capabilities = capabilities_response.get("json", {}).get("capabilities", {})
	if isinstance(capabilities, dict) and capabilities:
		checks.append(_json_route_check("capabilities_health", _fetch_json_route(base_url + "/capabilities/health")))

	return checks


def _run_record_crud_http_check(base_url: str, entity: dict[str, Any]) -> dict[str, Any]:
	entity_name = str(entity.get("name"))
	payload = {
		"record": {
			str(field.get("name")): _sample_value_for_field(field)
			for field in entity.get("fields", [])
			if isinstance(field, dict) and field.get("name") not in {None, "", "id"}
		}
	}
	create = _fetch_json_route(
		base_url + f"/entities/{urllib.parse.quote(entity_name)}/records",
		method="POST",
		payload=payload,
	)
	record_id = create.get("json", {}).get("record", {}).get("id")
	listing = _fetch_json_route(base_url + f"/entities/{urllib.parse.quote(entity_name)}/records")
	read = (
		_fetch_json_route(base_url + f"/entities/{urllib.parse.quote(entity_name)}/records/{urllib.parse.quote(str(record_id))}")
		if record_id not in (None, "")
		else {"ok": False, "status": "missing_id", "json": {}}
	)
	return {
		"name": "record_crud",
		"entity": entity_name,
		"ok": create["ok"] and create["status"] == 201 and listing["ok"] and read["ok"],
		"record_id": record_id,
		"steps": {
			"create": _json_route_public_result(create),
			"list": _json_route_public_result(listing),
			"read": _json_route_public_result(read),
		},
	}


def _run_workflow_http_check(base_url: str, workflow_name: str) -> dict[str, Any]:
	result = _fetch_json_route(
		base_url + f"/workflows/{urllib.parse.quote(workflow_name)}/run",
		method="POST",
		payload={"payload": {"source": "baseline_http_probe"}},
	)
	return {
		"name": "workflow_run",
		"workflow": workflow_name,
		"ok": result["ok"] and result["status"] in {200, 201},
		"result": _json_route_public_result(result),
	}


def _sample_value_for_field(field: dict[str, Any]) -> Any:
	field_name = str(field.get("name", "value"))
	field_type = str(field.get("type", "str")).lower()
	if field_type in {"int", "integer"}:
		return 1
	if field_type in {"float", "double", "decimal", "number"}:
		return 1.0
	if field_type in {"bool", "boolean"}:
		return True
	if "email" in field_name.lower():
		return "baseline@example.com"
	if "date" in field_name.lower():
		return "2026-05-29"
	return f"baseline_{field_name}"


def _json_route_check(name: str, response: dict[str, Any]) -> dict[str, Any]:
	return {
		"name": name,
		**_json_route_public_result(response),
	}


def _json_route_public_result(response: dict[str, Any]) -> dict[str, Any]:
	return {
		"ok": response.get("ok") is True,
		"status": response.get("status"),
		"summary": _json_summary(response.get("json", {})),
	}


def _json_summary(value: Any) -> dict[str, Any]:
	if isinstance(value, dict):
		summary: dict[str, Any] = {"keys": sorted(str(key) for key in value.keys())}
		entities = value.get("entities")
		if isinstance(entities, list):
			summary["entity_count"] = len(entities)
			summary["entities"] = [
				str(entity.get("name"))
				for entity in entities[:5]
				if isinstance(entity, dict) and entity.get("name")
			]
		records = value.get("records")
		if isinstance(records, dict):
			summary["record_entities"] = sorted(str(name) for name in records)
			summary["record_counts"] = {
				str(name): len(items) if isinstance(items, list) else None
				for name, items in records.items()
			}
		if isinstance(value.get("nodes"), list):
			summary["node_count"] = len(value["nodes"])
		if isinstance(value.get("edges"), list):
			summary["edge_count"] = len(value["edges"])
		workflows = value.get("workflows")
		if isinstance(workflows, dict):
			summary["workflow_count"] = len(workflows)
			summary["workflows"] = sorted(str(name) for name in workflows)[:5]
		capabilities = value.get("capabilities")
		if isinstance(capabilities, dict):
			summary["capability_count"] = len(capabilities)
			summary["capabilities"] = sorted(str(name) for name in capabilities)[:5]
		if "healthy" in value:
			summary["healthy"] = bool(value.get("healthy"))
		if isinstance(value.get("errors"), list):
			summary["error_count"] = len(value["errors"])
		if isinstance(value.get("warnings"), list):
			summary["warning_count"] = len(value["warnings"])
		if "count" in value:
			summary["count"] = value.get("count")
		if "total" in value:
			summary["total"] = value.get("total")
		return summary
	if isinstance(value, list):
		return {"item_count": len(value)}
	return {"type": type(value).__name__}


def _free_tcp_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
		probe.bind(("127.0.0.1", 0))
		return int(probe.getsockname()[1])


def _wait_for_http_route(url: str, process: subprocess.Popen[str], timeout_seconds: float = 5.0) -> bool:
	deadline = time.monotonic() + timeout_seconds
	while time.monotonic() < deadline:
		if process.poll() is not None:
			return False
		if _fetch_http_route(url)["ok"]:
			return True
		time.sleep(0.05)
	return False


def _fetch_http_route(url: str) -> dict[str, Any]:
	try:
		with urllib.request.urlopen(url, timeout=2.0) as response:
			return {"ok": 200 <= response.status < 300, "status": response.status}
	except urllib.error.HTTPError as error:
		return {"ok": False, "status": error.code}
	except OSError:
		return {"ok": False, "status": "unreachable"}


def _fetch_json_route(
	url: str,
	method: str = "GET",
	payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
	data = None
	headers = {"Accept": "application/json"}
	if payload is not None:
		data = json.dumps(payload).encode("utf-8")
		headers["Content-Type"] = "application/json"
	request = urllib.request.Request(url, data=data, headers=headers, method=method)
	try:
		with urllib.request.urlopen(request, timeout=2.0) as response:
			body = response.read().decode("utf-8")
			parsed = json.loads(body) if body else {}
			return {"ok": 200 <= response.status < 300, "status": response.status, "json": parsed}
	except urllib.error.HTTPError as error:
		try:
			body = error.read().decode("utf-8")
			parsed = json.loads(body) if body else {}
		except (OSError, json.JSONDecodeError):
			parsed = {}
		return {"ok": False, "status": error.code, "json": parsed}
	except (OSError, json.JSONDecodeError):
		return {"ok": False, "status": "unreachable", "json": {}}


def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
	if process.poll() is None:
		process.terminate()
	try:
		return process.communicate(timeout=2.0)
	except subprocess.TimeoutExpired:
		process.kill()
		return process.communicate(timeout=2.0)


def _generated_source_hygiene(generated_files: dict[str, str]) -> dict[str, Any]:
	violations: list[str] = []
	for file_name, content in sorted(generated_files.items()):
		if not file_name.endswith(".py"):
			continue
		for forbidden in (
			"TODO: Implement",
			"placeholder implementation",
			"None  # TODO",
			"Flask-AppBuilder",
			"flask_appbuilder",
		):
			if forbidden in content:
				violations.append(f"{file_name}: generated source contains {forbidden!r}")
		if "django" in content.lower():
			violations.append(f"{file_name}: generated source contains framework target 'django'")
		if re.search(r"^\s*pass\s*$", content, re.MULTILINE):
			violations.append(f"{file_name}: generated source contains a bare pass body")
	return {
		"ok": not violations,
		"checked_python_files": sum(1 for path in generated_files if path.endswith(".py")),
		"violations": violations,
	}


def _refresh_output_directory(output_dir: Path, generated_files: dict[str, str]) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	for file_name, content in generated_files.items():
		file_path = output_dir / file_name
		file_path.parent.mkdir(parents=True, exist_ok=True)
		file_path.write_text(content, encoding="utf-8")


def _checked_output_sync(output_dir: Path, generated_files: dict[str, str]) -> dict[str, Any]:
	violations: list[str] = []
	expected_files = set(generated_files)
	if not output_dir.exists():
		return {
			"ok": False,
			"output_dir": str(output_dir),
			"expected_file_count": len(expected_files),
			"current_file_count": 0,
			"missing_files": sorted(expected_files),
			"stale_files": [],
			"extra_files": [],
			"violations": [f"{output_dir}: output directory is missing"],
		}

	current_files = {
		path.relative_to(output_dir).as_posix()
		for path in output_dir.rglob("*")
		if path.is_file() and _is_checked_output_file(path)
	}
	missing_files = sorted(expected_files - current_files)
	extra_files = sorted(current_files - expected_files)
	stale_files: list[str] = []
	for file_name in sorted(expected_files & current_files):
		if (output_dir / file_name).read_text(encoding="utf-8") != generated_files[file_name]:
			stale_files.append(file_name)

	for file_name in missing_files:
		violations.append(f"{output_dir}: missing generated file {file_name}")
	for file_name in stale_files:
		violations.append(f"{output_dir}: stale generated file {file_name}")
	for file_name in extra_files:
		violations.append(f"{output_dir}: extra generated file {file_name}")

	return {
		"ok": not violations,
		"output_dir": str(output_dir),
		"expected_file_count": len(expected_files),
		"current_file_count": len(current_files),
		"missing_files": missing_files,
		"stale_files": stale_files,
		"extra_files": extra_files,
		"violations": violations,
	}


def _is_checked_output_file(path: Path) -> bool:
	return "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name != ".DS_Store"


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
