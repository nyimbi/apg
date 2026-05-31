"""Capability lifecycle audit for APG development-cycle evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capabilities.capability_contract_registry import (
	CapabilityContractRecord,
	load_contract_registry,
	validate_contract_registry,
)


CAPABILITY_LIFECYCLE_AUDIT_FORMAT = "apg.capability-lifecycle-audit.v1"
REQUIRED_LIFECYCLE_DOCS = [
	"SPECIFICATION.md",
	"PLAN.md",
	"README.md",
	"cap_spec.md",
]
REQUIRED_IMPLEMENTATION_ARTIFACTS = [
	"capability_contract.py",
	"models.py",
	"service.py",
	"api.py",
	"views.py",
	"app.py",
	"semantic_model.json",
	"package_manifest.json",
	"release_report.json",
]
MINIMUM_THEME_TOKEN_COUNT = 8


def audit_capability_lifecycle(root: Path | str | None = None) -> dict[str, Any]:
	"""Audit that every capability exposes methodical development-cycle evidence."""
	registry_report = validate_contract_registry(root)
	errors: list[str] = list(registry_report["errors"])
	warnings: list[str] = []
	records: list[dict[str, Any]] = []

	registry: dict[str, CapabilityContractRecord] = {}
	if registry_report["valid"]:
		registry = load_contract_registry(root)

	for capability_id, record in sorted(registry.items()):
		capability_report = _audit_record(record)
		records.append(capability_report)
		errors.extend(capability_report["errors"])
		warnings.extend(capability_report["warnings"])

	complete_count = sum(1 for record in records if record["complete"])
	return {
		"format": CAPABILITY_LIFECYCLE_AUDIT_FORMAT,
		"ok": not errors,
		"contract_validation": {
			"valid": bool(registry_report["valid"]),
			"contract_count": registry_report["contract_count"],
			"error_count": registry_report["error_count"],
		},
		"summary": {
			"capability_count": len(records),
			"complete_lifecycle_count": complete_count,
			"incomplete_lifecycle_count": len(records) - complete_count,
			"specification_count": sum(1 for record in records if record["development_cycle"]["specification"]),
			"plan_count": sum(1 for record in records if record["development_cycle"]["plan"]),
			"readme_count": sum(1 for record in records if record["development_cycle"]["readme"]),
			"cap_spec_count": sum(1 for record in records if record["development_cycle"]["cap_spec"]),
			"implementation_count": sum(1 for record in records if record["development_cycle"]["implementation"]),
			"test_surface_count": sum(1 for record in records if record["development_cycle"]["tests"]),
			"release_evidence_count": sum(1 for record in records if record["development_cycle"]["release_evidence"]),
			"code_review_ready_count": sum(1 for record in records if record["development_cycle"]["code_review_ready"]),
			"error_count": len(errors),
			"warning_count": len(warnings),
		},
		"records": records,
		"errors": errors,
		"warnings": warnings,
		"blocking_gaps": [
			{"surface": "capability_lifecycle", "error": error}
			for error in errors
		],
	}


def _audit_record(record: CapabilityContractRecord) -> dict[str, Any]:
	package_dir = record.path.parent
	errors: list[str] = []
	warnings: list[str] = []
	documents = _document_evidence(package_dir, errors)
	implementation = _implementation_evidence(package_dir, errors)
	tests = _test_evidence(package_dir, errors)
	release_evidence = _release_evidence(package_dir, record, errors, warnings)
	review_evidence = _review_evidence(record, documents, implementation, tests, release_evidence, errors)
	development_cycle = {
		"specification": documents["SPECIFICATION.md"]["ok"],
		"plan": documents["PLAN.md"]["ok"],
		"readme": documents["README.md"]["ok"],
		"cap_spec": documents["cap_spec.md"]["ok"],
		"implementation": implementation["complete"],
		"tests": tests["ok"],
		"release_evidence": release_evidence["ok"],
		"code_review_ready": review_evidence["ok"],
	}
	return {
		"capability": record.capability_id,
		"display_name": record.display_name,
		"category": _category(record.path),
		"package_dir": str(package_dir),
		"complete": all(development_cycle.values()) and not errors,
		"development_cycle": development_cycle,
		"documents": documents,
		"implementation": implementation,
		"tests": tests,
		"release_evidence": release_evidence,
		"review_evidence": review_evidence,
		"errors": errors,
		"warnings": warnings,
	}


def _document_evidence(package_dir: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
	documents: dict[str, dict[str, Any]] = {}
	for file_name in REQUIRED_LIFECYCLE_DOCS:
		path = package_dir / file_name
		ok = path.is_file()
		size = path.stat().st_size if ok else 0
		if not ok:
			errors.append(f"{package_dir.name}: missing lifecycle document {file_name}")
		elif size == 0:
			errors.append(f"{package_dir.name}: empty lifecycle document {file_name}")
			ok = False
		documents[file_name] = {
			"ok": ok,
			"path": str(path),
			"bytes": size,
		}
	return documents


def _implementation_evidence(package_dir: Path, errors: list[str]) -> dict[str, Any]:
	present: list[str] = []
	missing: list[str] = []
	for file_name in REQUIRED_IMPLEMENTATION_ARTIFACTS:
		path = package_dir / file_name
		if path.is_file():
			present.append(file_name)
		else:
			missing.append(file_name)
	if missing:
		errors.append(f"{package_dir.name}: missing implementation artifacts: {', '.join(missing)}")
	return {
		"complete": not missing,
		"present": present,
		"missing": missing,
	}


def _test_evidence(package_dir: Path, errors: list[str]) -> dict[str, Any]:
	nested_tests = list((package_dir / "tests").glob("test_*.py")) if (package_dir / "tests").is_dir() else []
	test_files = sorted(
		[
			*package_dir.glob("test_*.py"),
			*nested_tests,
		]
	)
	if not test_files:
		errors.append(f"{package_dir.name}: missing focused capability tests")
	return {
		"ok": bool(test_files),
		"test_file_count": len(test_files),
		"test_files": [str(path) for path in test_files],
	}


def _release_evidence(
	package_dir: Path,
	record: CapabilityContractRecord,
	errors: list[str],
	warnings: list[str],
) -> dict[str, Any]:
	release_path = package_dir / "release_report.json"
	manifest_path = package_dir / "package_manifest.json"
	semantic_path = package_dir / "semantic_model.json"
	release_report = _read_json(release_path, errors, "release report")
	manifest = _read_json(manifest_path, errors, "package manifest")
	semantic_model = _read_json(semantic_path, errors, "semantic model")
	ok = True

	if release_report.get("format") != "apg.release-report.v1":
		errors.append(f"{record.capability_id}: release_report.json must use apg.release-report.v1")
		ok = False
	if release_report.get("ok") is not True:
		errors.append(f"{record.capability_id}: release_report.json must be ok")
		ok = False
	self_test = release_report.get("evidence", {}).get("self_test", {})
	if self_test.get("passed") is not True:
		errors.append(f"{record.capability_id}: release self-test evidence must pass")
		ok = False
	if semantic_model.get("format") != "apg.semantic-model.v1":
		errors.append(f"{record.capability_id}: semantic_model.json must use apg.semantic-model.v1")
		ok = False
	if manifest.get("format") != "apg.package-manifest.v1":
		errors.append(f"{record.capability_id}: package_manifest.json must use apg.package-manifest.v1")
		ok = False
	if manifest.get("name") and manifest.get("name") != record.capability_id:
		warnings.append(
			f"{record.capability_id}: package manifest name differs from capability id "
			f"({manifest.get('name')})"
		)

	return {
		"ok": ok,
		"release_report_ok": release_report.get("ok") is True,
		"self_test_passed": self_test.get("passed") is True,
		"semantic_model_format": semantic_model.get("format"),
		"manifest_format": manifest.get("format"),
	}


def _review_evidence(
	record: CapabilityContractRecord,
	documents: dict[str, dict[str, Any]],
	implementation: dict[str, Any],
	tests: dict[str, Any],
	release_evidence: dict[str, Any],
	errors: list[str],
) -> dict[str, Any]:
	contract = record.contract
	rule_count = len(contract["rule_engine"]["rules"])
	route_count = len(contract["ui"]["routes"])
	theme_token_count = len(contract["theme"]["tokens"])
	required_configuration = contract["configuration_schema"].get("required", [])
	ok = (
		all(item["ok"] for item in documents.values())
		and implementation["complete"]
		and tests["ok"]
		and release_evidence["ok"]
		and rule_count > 0
		and route_count > 0
		and theme_token_count >= MINIMUM_THEME_TOKEN_COUNT
		and bool(required_configuration)
	)
	if rule_count == 0:
		errors.append(f"{record.capability_id}: code review gate requires at least one rule")
	if route_count == 0:
		errors.append(f"{record.capability_id}: code review gate requires at least one UI route")
	if theme_token_count < MINIMUM_THEME_TOKEN_COUNT:
		errors.append(f"{record.capability_id}: code review gate requires semantic theme tokens")
	if not required_configuration:
		errors.append(f"{record.capability_id}: code review gate requires configuration schema keys")
	return {
		"ok": ok,
		"rule_count": rule_count,
		"route_count": route_count,
		"theme_token_count": theme_token_count,
		"configuration_required_count": len(required_configuration),
	}


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except FileNotFoundError:
		errors.append(f"missing {label}: {path}")
	except json.JSONDecodeError as error:
		errors.append(f"invalid {label} JSON at {path}: {error}")
	except OSError as error:
		errors.append(f"could not read {label} at {path}: {error}")
	return {}


def _category(path: Path) -> str:
	parts = path.parts
	if "capabilities" not in parts:
		return ""
	index = parts.index("capabilities")
	return parts[index + 1] if index + 1 < len(parts) else ""
