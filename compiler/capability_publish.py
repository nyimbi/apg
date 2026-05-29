"""Side-effect-free capability package publish planning."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


CAPABILITY_CATALOG_FORMAT = "apg.capability-catalog.v1"
CAPABILITY_PUBLISH_APPLY_FORMAT = "apg.capability-publish-apply-report.v1"
REQUIRED_PACKAGE_FILES = {
	"app.py",
	"package_manifest.json",
	"release_report.json",
	"semantic_model.json",
}


def build_capability_publish_report(package_dir: Path) -> dict[str, Any]:
	"""Validate a package directory and return a catalog patch plan."""
	report: dict[str, Any] = {
		"format": "apg.capability-publish-report.v1",
		"ok": False,
		"package_dir": str(package_dir),
		"side_effect_free": True,
		"manifest": {},
		"release_evidence": {},
		"runtime_evidence": {},
		"capabilities": [],
		"catalog_patch": [],
		"errors": [],
		"warnings": [],
	}

	if not package_dir.exists():
		report["errors"].append(f"package directory not found: {package_dir}")
		return report
	if not package_dir.is_dir():
		report["errors"].append(f"capability publish expects a package directory: {package_dir}")
		return report

	missing = sorted(file_name for file_name in REQUIRED_PACKAGE_FILES if not (package_dir / file_name).is_file())
	for file_name in missing:
		report["errors"].append(f"missing package artifact {file_name}")
	if missing:
		return report

	manifest = _read_json(package_dir / "package_manifest.json", report, "package manifest")
	release_report = _read_json(package_dir / "release_report.json", report, "release report")
	artifact_model = _read_json(package_dir / "semantic_model.json", report, "semantic model")
	if report["errors"]:
		return report

	report["manifest"] = _manifest_summary(manifest)
	report["release_evidence"] = _release_summary(release_report)
	_validate_manifest(package_dir, manifest, report)
	_validate_release_evidence(release_report, report)

	runtime_evidence = _runtime_evidence(package_dir, report)
	report["runtime_evidence"] = runtime_evidence
	runtime_model = runtime_evidence.get("semantic_model", {})
	model = runtime_model if runtime_model.get("format") == "apg.semantic-model.v1" else artifact_model

	if artifact_model.get("format") != "apg.semantic-model.v1":
		report["errors"].append("semantic_model.json did not contain apg.semantic-model.v1")
	if runtime_model and runtime_model != artifact_model:
		report["warnings"].append("runtime semantic_model() differs from semantic_model.json")

	capabilities = _capability_records(model, manifest, package_dir)
	report["capabilities"] = capabilities
	if not capabilities:
		report["errors"].append("package semantic model does not declare publishable capabilities")
	report["catalog_patch"] = [
		{
			"op": "add_or_replace",
			"path": f"/capabilities/{record['capability']}",
			"value": record,
		}
		for record in capabilities
	]

	report["ok"] = not report["errors"]
	return report


def apply_capability_publish_report(
	package_dir: Path,
	catalog_path: Path,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Apply a valid capability publish plan to a local catalog file."""
	plan = build_capability_publish_report(package_dir)
	report: dict[str, Any] = {
		"format": CAPABILITY_PUBLISH_APPLY_FORMAT,
		"ok": False,
		"package_dir": str(package_dir),
		"catalog": str(catalog_path),
		"dry_run": dry_run,
		"written": False,
		"publish_plan_ok": bool(plan.get("ok")),
		"applied_count": 0,
		"capabilities": [],
		"catalog_summary": {},
		"errors": [],
		"warnings": list(plan.get("warnings", [])),
	}
	if not plan.get("ok"):
		report["errors"].extend(f"publish plan failed: {error}" for error in plan.get("errors", []))
		return report

	catalog = _load_or_create_catalog(catalog_path, report)
	if report["errors"]:
		return report

	before_count = len(catalog.get("capabilities", {}))
	applied = _apply_catalog_patch(catalog, plan.get("catalog_patch", []), report)
	if report["errors"]:
		return report

	report["applied_count"] = len(applied)
	report["capabilities"] = applied
	report["catalog_summary"] = {
		"format": catalog.get("format"),
		"capability_count_before": before_count,
		"capability_count_after": len(catalog.get("capabilities", {})),
	}

	if not dry_run:
		try:
			catalog_path.parent.mkdir(parents=True, exist_ok=True)
			catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
			report["written"] = True
		except OSError as error:
			report["errors"].append(f"could not write catalog: {error}")
			return report

	report["ok"] = True
	return report


def _read_json(path: Path, report: dict[str, Any], label: str) -> dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		report["errors"].append(f"{label} is invalid JSON: {error}")
	except OSError as error:
		report["errors"].append(f"could not read {label}: {error}")
	return {}


def _load_or_create_catalog(catalog_path: Path, report: dict[str, Any]) -> dict[str, Any]:
	if not catalog_path.exists():
		return {"format": CAPABILITY_CATALOG_FORMAT, "capabilities": {}}
	if not catalog_path.is_file():
		report["errors"].append(f"catalog path is not a file: {catalog_path}")
		return {}
	catalog = _read_json(catalog_path, report, "capability catalog")
	if report["errors"]:
		return {}
	if catalog.get("format") != CAPABILITY_CATALOG_FORMAT:
		report["errors"].append(f"capability catalog must use {CAPABILITY_CATALOG_FORMAT}")
	if not isinstance(catalog.get("capabilities"), dict):
		report["errors"].append("capability catalog capabilities must be an object")
	return catalog


def _apply_catalog_patch(
	catalog: dict[str, Any],
	patches: list[dict[str, Any]],
	report: dict[str, Any],
) -> list[str]:
	capabilities = catalog.setdefault("capabilities", {})
	applied: list[str] = []
	for patch in patches:
		if patch.get("op") != "add_or_replace":
			report["errors"].append(f"unsupported catalog patch op: {patch.get('op')}")
			continue
		path = str(patch.get("path", ""))
		prefix = "/capabilities/"
		if not path.startswith(prefix) or len(path) <= len(prefix):
			report["errors"].append(f"unsupported catalog patch path: {path}")
			continue
		capability_id = path[len(prefix):]
		value = deepcopy(patch.get("value", {}))
		if value.get("capability") != capability_id:
			report["errors"].append(f"catalog patch value mismatch for {capability_id}")
			continue
		capabilities[capability_id] = value
		applied.append(capability_id)
	return sorted(applied)


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": manifest.get("format"),
		"name": manifest.get("name"),
		"version": manifest.get("version"),
		"profile": manifest.get("profile"),
		"base_target": manifest.get("base_target"),
		"generated_artifact_count": len(manifest.get("generated_artifacts", [])),
		"profile_artifact_count": len(manifest.get("profile_artifacts", [])),
	}


def _release_summary(release_report: dict[str, Any]) -> dict[str, Any]:
	evidence = release_report.get("evidence", {})
	return {
		"format": release_report.get("format"),
		"ok": release_report.get("ok"),
		"target": release_report.get("target"),
		"self_test": evidence.get("self_test", {}),
		"semantic_model": evidence.get("semantic_model", {}),
		"contracts": evidence.get("contracts", {}),
	}


def _validate_manifest(package_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
	if manifest.get("format") != "apg.package-manifest.v1":
		report["errors"].append("package_manifest.json did not contain apg.package-manifest.v1")
	if manifest.get("base_target") != "python":
		report["errors"].append("package manifest base_target must be python")
	for artifact in [*manifest.get("generated_artifacts", []), *manifest.get("profile_artifacts", [])]:
		if not (package_dir / artifact).is_file():
			report["errors"].append(f"manifest references missing artifact {artifact}")


def _validate_release_evidence(release_report: dict[str, Any], report: dict[str, Any]) -> None:
	if release_report.get("format") != "apg.release-report.v1":
		report["errors"].append("release_report.json did not contain apg.release-report.v1")
	if release_report.get("target") != "python":
		report["errors"].append("release evidence target must be python")
	if not release_report.get("ok"):
		report["errors"].append("release evidence did not pass")
	evidence = release_report.get("evidence", {})
	if not evidence.get("self_test", {}).get("passed"):
		report["errors"].append("release evidence self-test did not pass")
	for contract_name, contract in evidence.get("contracts", {}).items():
		if contract.get("errors"):
			report["errors"].append(f"{contract_name} contract has errors: {contract['errors']}")


def _runtime_evidence(package_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
	module = _import_package_app(package_dir, report)
	if module is None:
		return {"loaded": False, "errors": ["could not load package app.py"]}
	try:
		self_test = module.self_test()
		component_manifest = module.component_manifest()
		semantic_model = module.semantic_model()
	except Exception as error:
		report["errors"].append(f"package entrypoint validation failed: {error}")
		return {"loaded": False, "errors": [str(error)]}
	finally:
		_cleanup_package_modules()

	if not self_test.get("passed"):
		report["errors"].append("package entrypoint self_test() did not pass")
	if component_manifest.get("kind") not in {"apg.application", "apg.generated_application"}:
		report["errors"].append("package entrypoint component_manifest() returned an unknown kind")
	if semantic_model.get("format") != "apg.semantic-model.v1":
		report["errors"].append("package entrypoint semantic_model() did not return apg.semantic-model.v1")

	return {
		"loaded": True,
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
		"semantic_model": semantic_model,
	}


def _import_package_app(package_dir: Path, report: dict[str, Any]) -> Any | None:
	_cleanup_package_modules()
	sys.path.insert(0, str(package_dir))
	try:
		spec = importlib.util.spec_from_file_location("app", package_dir / "app.py")
		if spec is None or spec.loader is None:
			report["errors"].append("could not create import spec for package app.py")
			return None
		module = importlib.util.module_from_spec(spec)
		sys.modules["app"] = module
		spec.loader.exec_module(module)
		return module
	except Exception as error:
		report["errors"].append(f"could not load package entrypoint: {error}")
		return None
	finally:
		try:
			sys.path.remove(str(package_dir))
		except ValueError:
			pass


def _cleanup_package_modules() -> None:
	for module_name in ("app", "ai_agents", "apg_capabilities", "apg_application"):
		sys.modules.pop(module_name, None)


def _capability_records(
	model: dict[str, Any],
	manifest: dict[str, Any],
	package_dir: Path,
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []
	for capability_name, capability in sorted(model.get("capabilities", {}).items()):
		records.append({
			"capability": capability_name,
			"package": manifest.get("name"),
			"version": manifest.get("version"),
			"profile": manifest.get("profile"),
			"package_dir": str(package_dir),
			"provides": list(capability.get("provides", [])),
			"requires": list(capability.get("requires", [])),
			"configuration": dict(capability.get("configuration", {})),
			"rules": list(capability.get("rules", [])),
			"rule_engine": dict(capability.get("rule_engine", {})),
			"ui": dict(capability.get("ui", {})),
			"screens": dict(capability.get("screens", {})),
			"theme": dict(capability.get("theme", {})),
			"streaming": dict(capability.get("streaming", {})),
			"release_evidence": "release_report.json",
			"manifest": "package_manifest.json",
			"entrypoint": "app.py",
		})
	return records
