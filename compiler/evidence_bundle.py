"""Release evidence bundle builder for APG applications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capability_publish import build_capability_publish_report
from .deployment_verifier import build_deployment_verification_report
from .package_verifier import build_package_verification_report
from .packager import build_package_report
from .release import build_release_report


def build_release_evidence_bundle(
	source_file: Path,
	target: str = "web",
	out_dir: Path | None = None,
	include_capability_publish: bool = True,
) -> dict[str, Any]:
	"""Build and verify the complete APG release evidence chain."""
	report: dict[str, Any] = {
		"format": "apg.release-evidence-bundle.v1",
		"ok": False,
		"source": str(source_file),
		"target": target,
		"output_dir": str(out_dir or "dist"),
		"release": {},
		"package": {},
		"package_verification": {},
		"deployment_verification": {},
		"capability_publish": {},
		"checks": {},
		"errors": [],
		"warnings": [],
	}

	release_report = build_release_report(source_file, target="python")
	report["release"] = _release_summary(release_report)
	_collect_errors("release", release_report, report)
	_collect_warnings("release", release_report, report)
	if not release_report.get("ok"):
		return _finalize(report)

	package_report = build_package_report(source_file, target=target, out_dir=out_dir)
	report["package"] = _package_summary(package_report)
	_collect_errors("package", package_report, report)
	_collect_warnings("package", package_report, report)
	package_dir_text = package_report.get("output_dir") or ""
	if not package_report.get("ok") or not package_dir_text:
		return _finalize(report)

	package_dir = Path(package_dir_text)
	package_verification = build_package_verification_report(package_dir)
	report["package_verification"] = _verification_summary(package_verification)
	_collect_errors("package_verification", package_verification, report)
	_collect_warnings("package_verification", package_verification, report)

	deployment_verification = build_deployment_verification_report(package_dir)
	report["deployment_verification"] = _verification_summary(deployment_verification)
	_collect_errors("deployment_verification", deployment_verification, report)
	_collect_warnings("deployment_verification", deployment_verification, report)

	if include_capability_publish:
		capability_publish = build_capability_publish_report(package_dir)
		report["capability_publish"] = _capability_publish_summary(capability_publish)
		_collect_errors("capability_publish", capability_publish, report)
		_collect_warnings("capability_publish", capability_publish, report)

	return _finalize(report)


def _release_summary(report: dict[str, Any]) -> dict[str, Any]:
	evidence = report.get("evidence", {})
	return {
		"format": report.get("format"),
		"ok": report.get("ok"),
		"target": report.get("target"),
		"generated_file_count": report.get("generated", {}).get("file_count", 0),
		"self_test": evidence.get("self_test", {}),
		"contracts": evidence.get("contracts", {}),
	}


def _package_summary(report: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": report.get("format"),
		"ok": report.get("ok"),
		"target": report.get("target"),
		"output_dir": report.get("output_dir"),
		"manifest_path": report.get("manifest_path"),
		"file_count": len(report.get("files", [])),
		"checks": report.get("checks", {}),
	}


def _verification_summary(report: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": report.get("format"),
		"ok": report.get("ok"),
		"profile": report.get("profile"),
		"kind": report.get("kind"),
		"checks": report.get("checks", report.get("profile_checks", {})),
		"runtime": report.get("runtime", {}),
		"topology": report.get("topology", {}),
	}


def _capability_publish_summary(report: dict[str, Any]) -> dict[str, Any]:
	return {
		"format": report.get("format"),
		"ok": report.get("ok"),
		"side_effect_free": report.get("side_effect_free"),
		"capability_count": len(report.get("capabilities", [])),
		"catalog_patch_count": len(report.get("catalog_patch", [])),
		"capabilities": [
			record.get("capability")
			for record in report.get("capabilities", [])
		],
	}


def _collect_errors(prefix: str, source: dict[str, Any], target: dict[str, Any]) -> None:
	for error in source.get("errors", []):
		target["errors"].append(f"{prefix}: {error}")


def _collect_warnings(prefix: str, source: dict[str, Any], target: dict[str, Any]) -> None:
	for warning in source.get("warnings", []):
		target["warnings"].append(f"{prefix}: {warning}")


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
	report["checks"] = {
		"release_ok": bool(report.get("release", {}).get("ok")),
		"package_ok": bool(report.get("package", {}).get("ok")),
		"package_verification_ok": bool(report.get("package_verification", {}).get("ok")),
		"deployment_verification_ok": bool(report.get("deployment_verification", {}).get("ok")),
		"capability_publish_plan_ok": (
			not report.get("capability_publish")
			or bool(report.get("capability_publish", {}).get("ok"))
		),
	}
	report["ok"] = all(report["checks"].values()) and not report["errors"]
	return report
