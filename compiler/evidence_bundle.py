"""Release evidence bundle builder for APG applications."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .capability_publish import build_capability_publish_report
from .deployment_verifier import build_deployment_verification_report
from .package_verifier import build_package_verification_report
from .packager import build_package_report
from .release import build_release_report

RELEASE_EVIDENCE_FIXTURE_AUDIT_FORMAT = "apg.release-evidence-fixture-audit.v1"
DEFAULT_RELEASE_EVIDENCE_FIXTURE_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "verifiers" / "catalog.json"


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


def audit_release_evidence_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run the checked-in release/package/deployment verifier fixture catalog."""
	catalog_file = Path(catalog_path or DEFAULT_RELEASE_EVIDENCE_FIXTURE_CATALOG)
	catalog_root = catalog_file.parent
	repo_root = Path(__file__).resolve().parents[1]
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_targets = sorted(str(target) for target in catalog.get("targets_required", []))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []
	covered_targets: set[str] = set()
	covered_tags: set[str] = set()

	with tempfile.TemporaryDirectory(prefix="apg_evidence_fixtures_") as temp_root:
		temp_dir = Path(temp_root)
		for fixture in catalog.get("fixtures", []):
			report = _audit_release_evidence_fixture(catalog_root, repo_root, temp_dir, fixture)
			fixture_reports.append(report)
			if report["ok"]:
				covered_targets.update(report["targets"])
				covered_tags.update(report["tags"])
			else:
				blocking_gaps.append({
					"id": report["id"],
					"source": report["source"],
					"errors": report["errors"],
				})

	missing_targets = sorted(set(required_targets).difference(covered_targets))
	for target in missing_targets:
		blocking_gaps.append({
			"id": f"missing_target:{target}",
			"source": str(catalog_file),
			"errors": [f"required release evidence target {target!r} is not covered by a passing fixture"],
		})

	missing_tags = sorted(set(required_tags).difference(covered_tags))
	for tag in missing_tags:
		blocking_gaps.append({
			"id": f"missing_tag:{tag}",
			"source": str(catalog_file),
			"errors": [f"required verifier fixture tag {tag!r} is not covered by a passing fixture"],
		})

	return {
		"format": RELEASE_EVIDENCE_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"fixture_catalog": str(catalog_file),
		"targets_required": required_targets,
		"targets_covered": sorted(covered_targets),
		"missing_targets": missing_targets,
		"tags_required": required_tags,
		"tags_covered": sorted(covered_tags),
		"missing_tags": missing_tags,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"target_run_count": sum(len(report["targets"]) for report in fixture_reports),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"failing_fixture_count": sum(1 for report in fixture_reports if not report["ok"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _audit_release_evidence_fixture(
	catalog_root: Path,
	repo_root: Path,
	temp_dir: Path,
	fixture: dict[str, Any],
) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	source = (catalog_root / str(fixture["source"])).resolve()
	if not source.exists():
		source = (repo_root / str(fixture["source"])).resolve()
	targets = [str(target).lower() for target in fixture.get("targets", [])]
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	expected_checks = [str(check) for check in fixture.get("expected_checks", [])]
	expected_capabilities = {str(capability) for capability in fixture.get("expected_capabilities", [])}
	include_capability_publish = bool(fixture.get("include_capability_publish", True))
	expected_ok = bool(fixture.get("expected_ok", True))
	errors: list[str] = []
	target_reports: list[dict[str, Any]] = []

	for target in targets:
		out_dir = temp_dir / fixture_id / target
		try:
			report = build_release_evidence_bundle(
				source,
				target=target,
				out_dir=out_dir,
				include_capability_publish=include_capability_publish,
			)
		except Exception as error:
			errors.append(f"{target}: {error}")
			continue

		if bool(report.get("ok")) != expected_ok:
			errors.append(f"{target}: expected ok={expected_ok}, got {report.get('ok')}")
		if report.get("target") != target:
			errors.append(f"{target}: bundle target mismatch")
		for check in expected_checks:
			if report.get("checks", {}).get(check) is not True:
				errors.append(f"{target}: expected check {check} to pass")
		if report.get("release", {}).get("format") != "apg.release-report.v1":
			errors.append(f"{target}: release report format mismatch")
		if report.get("package", {}).get("format") != "apg.package-report.v1":
			errors.append(f"{target}: package report format mismatch")
		if report.get("package_verification", {}).get("format") != "apg.package-verification-report.v1":
			errors.append(f"{target}: package verification format mismatch")
		if report.get("deployment_verification", {}).get("format") != "apg.deployment-verification-report.v1":
			errors.append(f"{target}: deployment verification format mismatch")
		if report.get("package_verification", {}).get("profile") != target:
			errors.append(f"{target}: package verification profile mismatch")

		capability_publish = report.get("capability_publish", {})
		if include_capability_publish:
			if capability_publish.get("format") != "apg.capability-publish-report.v1":
				errors.append(f"{target}: capability publish format mismatch")
			if capability_publish.get("side_effect_free") is not True:
				errors.append(f"{target}: capability publish plan is not side-effect-free")
			actual_capabilities = {str(capability) for capability in capability_publish.get("capabilities", [])}
			if not expected_capabilities.issubset(actual_capabilities):
				errors.append(f"{target}: capability publish plan missing {sorted(expected_capabilities - actual_capabilities)}")

		target_reports.append({
			"target": target,
			"ok": bool(report.get("ok")),
			"checks": report.get("checks", {}),
			"package_profile": report.get("package_verification", {}).get("profile"),
			"capability_publish_side_effect_free": capability_publish.get("side_effect_free"),
			"warnings": report.get("warnings", []),
		})

	return {
		"id": fixture_id,
		"source": str(source),
		"targets": targets,
		"tags": tags,
		"target_reports": target_reports,
		"ok": not errors,
		"errors": errors,
	}


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
