"""Deployment evidence verifier for generated APG applications."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
SECRET_NAME_PATTERN = re.compile(r"(SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)", re.IGNORECASE)
SAFE_PLACEHOLDERS = {"", "change-me", "changeme", "example", "example-value", "<key>", "<secret>"}


def build_deployment_verification_report(path: Path) -> dict[str, Any]:
	"""Verify deployment evidence for a generated output or package directory."""
	report: dict[str, Any] = {
		"format": "apg.deployment-verification-report.v1",
		"ok": False,
		"path": str(path),
		"kind": "directory",
		"checks": {},
		"manifest": {},
		"runtime": {},
		"deployment": {},
		"topology": {},
		"errors": [],
		"warnings": [],
	}
	if not path.exists():
		report["errors"].append(f"deployment path not found: {path}")
		return report
	if not path.is_dir():
		report["errors"].append(f"deployment verifier expects a directory: {path}")
		return report

	package_manifest = _read_optional_json(path / "package_manifest.json", report, "package manifest")
	if package_manifest:
		report["kind"] = "package"
		report["manifest"] = {
			"format": package_manifest.get("format"),
			"name": package_manifest.get("name"),
			"profile": package_manifest.get("profile"),
			"base_target": package_manifest.get("base_target"),
		}
		if package_manifest.get("format") != "apg.package-manifest.v1":
			report["errors"].append("package_manifest.json did not contain apg.package-manifest.v1")
		if package_manifest.get("base_target") != "python":
			report["errors"].append("package manifest base_target must be python")

	module = _import_app(path, report)
	if module is None:
		return _finalize_report(report)

	try:
		self_test = module.self_test()
		component_manifest = module.component_manifest()
		semantic_model = module.semantic_model()
		component_contract = module.validate_component_manifest_contract()
	finally:
		_cleanup_modules()

	deployment = component_manifest.get("deployment", {}) if isinstance(component_manifest, dict) else {}
	graph_summary = semantic_model.get("graphs", {}).get("deployment", {}) if isinstance(semantic_model, dict) else {}
	report["runtime"] = {
		"self_test": {
			"passed": bool(self_test.get("passed")),
			"status": self_test.get("status"),
			"route_count": len(self_test.get("routes", [])),
		},
		"component_manifest": {
			"kind": component_manifest.get("kind"),
			"target": component_manifest.get("target"),
			"semantic_model": component_manifest.get("interfaces", {}).get("semantic_model"),
		},
		"semantic_model": {
			"format": semantic_model.get("format"),
			"ok": semantic_model.get("ok"),
		},
		"component_contract": {
			"errors": list(component_contract.get("errors", [])),
			"warnings": list(component_contract.get("warnings", [])),
		},
	}
	report["deployment"] = {
		"artifacts": list(deployment.get("artifacts", [])) if isinstance(deployment.get("artifacts"), list) else [],
		"commands": dict(deployment.get("commands", {})) if isinstance(deployment.get("commands"), dict) else {},
		"environment": list(deployment.get("environment", [])) if isinstance(deployment.get("environment"), list) else [],
	}
	report["topology"] = {
		"graph_kind": graph_summary.get("kind"),
		"nodes": graph_summary.get("nodes", 0),
		"edges": graph_summary.get("edges", 0),
		"connected": _deployment_graph_is_connected(graph_summary),
		"explainable": semantic_model.get("format") == "apg.semantic-model.v1",
	}

	_check_units(path, component_manifest, package_manifest, report)
	_check_health(path, report)
	_check_environment(path, report)
	_check_resources(path, package_manifest, report)
	_check_topology(report)
	if not self_test.get("passed"):
		report["errors"].append("runtime self_test() did not pass")
	if semantic_model.get("format") != "apg.semantic-model.v1":
		report["errors"].append("runtime semantic_model() did not return apg.semantic-model.v1")
	for error in component_contract.get("errors", []):
		report["errors"].append(f"component manifest: {error}")
	for warning in component_contract.get("warnings", []):
		report["warnings"].append(f"component manifest: {warning}")

	return _finalize_report(report)


def _read_optional_json(path: Path, report: dict[str, Any], label: str) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		report["errors"].append(f"{label} is invalid JSON: {error}")
	except OSError as error:
		report["errors"].append(f"could not read {label}: {error}")
	return {}


def _import_app(path: Path, report: dict[str, Any]) -> Any | None:
	app_path = path / "app.py"
	if not app_path.is_file():
		report["errors"].append("deployment directory is missing app.py")
		return None
	_cleanup_modules()
	sys.path.insert(0, str(path))
	try:
		spec = importlib.util.spec_from_file_location("app", app_path)
		if spec is None or spec.loader is None:
			report["errors"].append("could not create import spec for app.py")
			return None
		module = importlib.util.module_from_spec(spec)
		sys.modules["app"] = module
		spec.loader.exec_module(module)
		return module
	except Exception as error:
		report["errors"].append(f"could not load deployment app.py: {error}")
		return None
	finally:
		try:
			sys.path.remove(str(path))
		except ValueError:
			pass


def _cleanup_modules() -> None:
	for module_name in ("app", "ai_agents", "apg_capabilities", "apg_application"):
		sys.modules.pop(module_name, None)


def _check_units(
	path: Path,
	component_manifest: dict[str, Any],
	package_manifest: dict[str, Any],
	report: dict[str, Any],
) -> None:
	artifacts = report["deployment"]["artifacts"]
	units_ok = bool(component_manifest.get("target")) and "app.py" in artifacts
	if package_manifest:
		units_ok = units_ok and bool(package_manifest.get("profile"))
	if not units_ok:
		report["errors"].append("deployment units are not declared")
	for artifact in artifacts:
		if not (path / artifact).exists():
			report["errors"].append(f"deployment artifact {artifact} does not exist")
	report["checks"]["units_declared"] = units_ok


def _check_health(path: Path, report: dict[str, Any]) -> None:
	commands = report["deployment"]["commands"]
	dockerfile = path / "Dockerfile"
	dockerfile_text = dockerfile.read_text(encoding="utf-8", errors="ignore") if dockerfile.exists() else ""
	health_ok = (
		commands.get("self_test") == "python app.py --self-test"
		and commands.get("smoke_test") == "python smoke_test.py"
		and "HEALTHCHECK" in dockerfile_text
	)
	if not health_ok:
		report["errors"].append("deployment health checks are incomplete")
	report["checks"]["health_checks_declared"] = health_ok


def _check_environment(path: Path, report: dict[str, Any]) -> None:
	environment = report["deployment"]["environment"]
	env_names_ok = bool(environment) and all(isinstance(name, str) and ENV_NAME_PATTERN.match(name) for name in environment)
	if not env_names_ok:
		report["errors"].append("deployment environment variables are not named cleanly")

	secrets_absent = _secret_values_absent(path)
	if not secrets_absent:
		report["errors"].append("deployment environment contains literal secret values")
	report["checks"]["environment_variables_named"] = env_names_ok
	report["checks"]["secret_values_absent"] = secrets_absent


def _secret_values_absent(path: Path) -> bool:
	for env_path in (path / ".env", path / ".env.example"):
		if not env_path.exists():
			continue
		for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			name, value = [part.strip() for part in line.split("=", 1)]
			if SECRET_NAME_PATTERN.search(name) and value.lower() not in SAFE_PLACEHOLDERS:
				return False
	dockerfile = path / "Dockerfile"
	if dockerfile.exists():
		for raw_line in dockerfile.read_text(encoding="utf-8", errors="ignore").splitlines():
			line = raw_line.strip()
			if not line.startswith("ENV ") or "=" not in line:
				continue
			assignment = line[4:].strip()
			name, value = [part.strip() for part in assignment.split("=", 1)]
			if SECRET_NAME_PATTERN.search(name) and value.lower() not in SAFE_PLACEHOLDERS:
				return False
	return True


def _check_resources(path: Path, package_manifest: dict[str, Any], report: dict[str, Any]) -> None:
	dockerfile = path / "Dockerfile"
	dockerfile_text = dockerfile.read_text(encoding="utf-8", errors="ignore") if dockerfile.exists() else ""
	profile = package_manifest.get("profile", "python") if package_manifest else "python"
	resource_hints = {
		"dockerfile": dockerfile.exists(),
		"port": "EXPOSE " in dockerfile_text or "APG_PORT" in dockerfile_text,
		"workdir": "WORKDIR " in dockerfile_text,
		"healthcheck": "HEALTHCHECK" in dockerfile_text,
	}
	resource_ok = all(resource_hints.values())
	if profile in {"container", "web", "python"} and not resource_ok:
		report["errors"].append("deployment resource hints are incomplete")
	elif not resource_ok:
		report["warnings"].append("deployment resource hints are incomplete")
	report["checks"]["resource_hints_present"] = resource_ok
	report["resource_hints"] = resource_hints


def _check_topology(report: dict[str, Any]) -> None:
	topology = report["topology"]
	topology_ok = bool(topology["connected"] and topology["explainable"])
	if not topology_ok:
		report["errors"].append("deployment topology graph is not connected and explainable")
	report["checks"]["topology_graph_connected"] = topology_ok


def _deployment_graph_is_connected(graph_summary: dict[str, Any]) -> bool:
	nodes = int(graph_summary.get("nodes") or 0)
	edges = int(graph_summary.get("edges") or 0)
	if nodes <= 1:
		return nodes == 1
	return edges >= nodes - 1


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
	report["ok"] = not report["errors"]
	for check_name, value in {
		"units_declared": False,
		"health_checks_declared": False,
		"environment_variables_named": False,
		"secret_values_absent": False,
		"resource_hints_present": False,
		"topology_graph_connected": False,
	}.items():
		report["checks"].setdefault(check_name, value)
	return report
