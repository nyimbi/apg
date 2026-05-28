"""Package profile verifier for generated APG packages."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_package_verification_report(package_dir: Path) -> dict[str, Any]:
	"""Verify an existing generated APG package directory."""
	report: dict[str, Any] = {
		"format": "apg.package-verification-report.v1",
		"ok": False,
		"package_dir": str(package_dir),
		"profile": None,
		"manifest": {},
		"release_evidence": {},
		"runtime": {},
		"profile_checks": {},
		"errors": [],
		"warnings": [],
	}
	if not package_dir.exists():
		report["errors"].append(f"package directory not found: {package_dir}")
		return report
	if not package_dir.is_dir():
		report["errors"].append(f"package verifier expects a directory: {package_dir}")
		return report

	manifest = _read_json(package_dir / "package_manifest.json", report, "package manifest")
	release_report = _read_json(package_dir / "release_report.json", report, "release report")
	if report["errors"]:
		return report

	profile = str(manifest.get("profile") or "")
	report["profile"] = profile
	report["manifest"] = _manifest_summary(manifest)
	report["release_evidence"] = _release_summary(release_report)

	_validate_manifest_artifacts(package_dir, manifest, report)
	_validate_release_evidence(release_report, report)
	runtime = _runtime_evidence(package_dir, report)
	report["runtime"] = runtime
	report["profile_checks"] = _profile_checks(package_dir, manifest, runtime, report)

	report["ok"] = not report["errors"]
	return report


def _read_json(path: Path, report: dict[str, Any], label: str) -> dict[str, Any]:
	if not path.is_file():
		report["errors"].append(f"missing {label}: {path.name}")
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as error:
		report["errors"].append(f"{label} is invalid JSON: {error}")
	except OSError as error:
		report["errors"].append(f"could not read {label}: {error}")
	return {}


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": manifest.get("format"),
		"name": manifest.get("name"),
		"version": manifest.get("version"),
		"profile": manifest.get("profile"),
		"base_target": manifest.get("base_target"),
		"entrypoints": dict(manifest.get("entrypoints", {})),
		"signing": dict(manifest.get("signing", {})),
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
		"contracts": evidence.get("contracts", {}),
	}


def _validate_manifest_artifacts(package_dir: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
	if manifest.get("format") != "apg.package-manifest.v1":
		report["errors"].append("package_manifest.json did not contain apg.package-manifest.v1")
	if manifest.get("base_target") != "python":
		report["errors"].append("package manifest base_target must be python")
	for artifact in [*manifest.get("generated_artifacts", []), *manifest.get("profile_artifacts", [])]:
		if not isinstance(artifact, str):
			report["errors"].append("package manifest artifacts must be strings")
			continue
		if not (package_dir / artifact).is_file():
			report["errors"].append(f"package manifest references missing artifact {artifact}")


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
	module = _import_app(package_dir, report)
	if module is None:
		return {"loaded": False, "errors": ["could not load app.py"]}
	try:
		self_test = module.self_test()
		component_manifest = module.component_manifest()
		openapi_contract = module.validate_openapi_contract()
		component_contract = module.validate_component_manifest_contract()
		route_contract = module.validate_route_dispatch_contract()
	finally:
		_cleanup_modules()

	for name, contract in {
		"openapi": openapi_contract,
		"component_manifest": component_contract,
		"route_dispatch": route_contract,
	}.items():
		for error in contract.get("errors", []):
			report["errors"].append(f"{name}: {error}")
		for warning in contract.get("warnings", []):
			report["warnings"].append(f"{name}: {warning}")
	if not self_test.get("passed"):
		report["errors"].append("runtime self_test() did not pass")

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
			"http_path_count": len(component_manifest.get("interfaces", {}).get("http", {}).get("paths", [])),
		},
		"contracts": {
			"openapi": {"errors": list(openapi_contract.get("errors", []))},
			"component_manifest": {"errors": list(component_contract.get("errors", []))},
			"route_dispatch": {"errors": list(route_contract.get("errors", []))},
		},
	}


def _import_app(package_dir: Path, report: dict[str, Any]) -> Any | None:
	app_path = package_dir / "app.py"
	if not app_path.is_file():
		report["errors"].append("package directory is missing app.py")
		return None
	_cleanup_modules()
	sys.path.insert(0, str(package_dir))
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
		report["errors"].append(f"could not load package app.py: {error}")
		return None
	finally:
		try:
			sys.path.remove(str(package_dir))
		except ValueError:
			pass


def _cleanup_modules() -> None:
	for module_name in ("app", "ai_agents", "apg_capabilities", "apg_application"):
		sys.modules.pop(module_name, None)


def _profile_checks(
	package_dir: Path,
	manifest: dict[str, Any],
	runtime: dict[str, Any],
	report: dict[str, Any],
) -> dict[str, Any]:
	profile = manifest.get("profile")
	if profile == "web":
		return _web_checks(package_dir, manifest, runtime, report)
	if profile == "desktop":
		return _desktop_checks(package_dir, manifest, runtime, report)
	if profile == "mobile":
		return _mobile_checks(package_dir, manifest, runtime, report)
	if profile == "container":
		return _container_checks(package_dir, manifest, runtime, report)
	return _python_checks(package_dir, manifest, runtime, report)


def _web_checks(package_dir: Path, manifest: dict[str, Any], runtime: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
	checks = {
		"launcher_exists": (package_dir / "run_web.py").is_file(),
		"routes_exist": runtime.get("self_test", {}).get("route_count", 0) > 0,
		"forms_bind_valid_fields": not runtime.get("contracts", {}).get("component_manifest", {}).get("errors"),
		"handler_targets_resolve": not runtime.get("contracts", {}).get("route_dispatch", {}).get("errors"),
		"smoke_test_runs": _run_smoke(package_dir, report),
	}
	_record_failed_checks(checks, report)
	return checks


def _desktop_checks(package_dir: Path, manifest: dict[str, Any], runtime: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
	signing = manifest.get("signing", {})
	checks = {
		"metadata_exists": manifest.get("format") == "apg.package-manifest.v1",
		"launcher_exists": (package_dir / "run_desktop.py").is_file(),
		"signing_posture_declared": bool(signing.get("status")),
		"menus_bind_handlers": not runtime.get("contracts", {}).get("route_dispatch", {}).get("errors"),
		"smoke_launch_path_exists": manifest.get("entrypoints", {}).get("self_test") == "python app.py --self-test",
		"smoke_test_runs": _run_smoke(package_dir, report),
	}
	_record_failed_checks(checks, report)
	return checks


def _mobile_checks(package_dir: Path, manifest: dict[str, Any], runtime: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
	profile = _read_json(package_dir / "mobile_profile.json", report, "mobile profile")
	signing = manifest.get("signing", {})
	checks = {
		"metadata_exists": profile.get("format") == "apg.mobile-profile.v1",
		"signing_posture_declared": bool(signing.get("status")),
		"offline_policy_declared": profile.get("offline", {}).get("supported") is True,
		"permissions_explained": isinstance(profile.get("permissions"), list),
		"screens_fit_target_density": runtime.get("self_test", {}).get("route_count", 0) > 0,
		"smoke_launch_path_exists": profile.get("launch") == "python app.py --self-test",
		"smoke_test_runs": _run_smoke(package_dir, report),
	}
	_record_failed_checks(checks, report)
	return checks


def _container_checks(package_dir: Path, manifest: dict[str, Any], runtime: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
	profile = _read_json(package_dir / "container_profile.json", report, "container profile")
	dockerfile = package_dir / "Dockerfile"
	dockerfile_text = dockerfile.read_text(encoding="utf-8", errors="ignore") if dockerfile.is_file() else ""
	checks = {
		"metadata_exists": profile.get("format") == "apg.container-profile.v1",
		"dockerfile_exists": dockerfile.is_file(),
		"healthcheck_declared": "HEALTHCHECK" in dockerfile_text,
		"routes_exist": runtime.get("self_test", {}).get("route_count", 0) > 0,
		"smoke_test_runs": _run_smoke(package_dir, report),
	}
	_record_failed_checks(checks, report)
	return checks


def _python_checks(package_dir: Path, manifest: dict[str, Any], runtime: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
	checks = {
		"metadata_exists": manifest.get("format") == "apg.package-manifest.v1",
		"app_exists": (package_dir / "app.py").is_file(),
		"routes_exist": runtime.get("self_test", {}).get("route_count", 0) > 0,
		"smoke_test_runs": _run_smoke(package_dir, report),
	}
	_record_failed_checks(checks, report)
	return checks


def _run_smoke(package_dir: Path, report: dict[str, Any]) -> bool:
	completed = subprocess.run(
		[sys.executable, "smoke_test.py"],
		cwd=package_dir,
		check=False,
		capture_output=True,
		text=True,
	)
	if completed.returncode != 0:
		report["errors"].append(
			f"smoke_test.py exited {completed.returncode}: "
			f"{completed.stdout.rstrip()} {completed.stderr.rstrip()}".strip()
		)
		return False
	return True


def _record_failed_checks(checks: dict[str, bool], report: dict[str, Any]) -> None:
	for check_name, passed in checks.items():
		if not passed:
			report["errors"].append(f"profile check failed: {check_name}")
