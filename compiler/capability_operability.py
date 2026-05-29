"""Capability operability audit for APG contract surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capabilities.capability_contract_registry import (
	CapabilityContractRecord,
	evaluate_rules,
	load_contract_registry,
	validate_contract_registry,
)


CAPABILITY_OPERABILITY_AUDIT_FORMAT = "apg.capability-operability-audit.v1"
RECOMMENDED_PACKAGE_ARTIFACTS = [
	"cap_spec.md",
	"models.py",
	"service.py",
	"api.py",
	"views.py",
	"app.py",
	"semantic_model.json",
	"package_manifest.json",
	"release_report.json",
	"tests",
]
RULE_PROBE_CONTEXTS = {
	"read_allowed": {
		"tenant_context_present": True,
		"operation_type": "read",
		"policy_attached": True,
		"risk_level": "low",
		"review_recorded": True,
	},
	"write_without_tenant": {
		"tenant_context_present": False,
		"operation_type": "write",
		"policy_attached": False,
		"risk_level": "low",
		"review_recorded": True,
	},
	"high_risk_without_review": {
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"risk_level": "high",
		"review_recorded": False,
	},
}
VALID_DECISIONS = {"allow", "deny", "require_review"}


def audit_capability_operability(
	root: Path | str | None = None,
	strict_package_artifacts: bool = False,
) -> dict[str, Any]:
	"""Audit capability contracts, rule execution, UI/theme surfaces, and package evidence."""
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
		if strict_package_artifacts:
			if capability_report["package_artifacts"]["missing"]:
				errors.append(
					f"{capability_id}: missing package artifacts: "
					f"{', '.join(capability_report['package_artifacts']['missing'])}"
				)
		else:
			if capability_report["package_artifacts"]["missing"]:
				warnings.append(
					f"{capability_id}: missing package artifacts: "
					f"{', '.join(capability_report['package_artifacts']['missing'])}"
				)

	operable_contract_count = sum(1 for record in records if record["operable"])
	complete_package_count = sum(1 for record in records if record["package_artifacts"]["complete"])
	package_gap_count = sum(len(record["package_artifacts"]["missing"]) for record in records)
	return {
		"format": CAPABILITY_OPERABILITY_AUDIT_FORMAT,
		"ok": not errors,
		"strict_package_artifacts": strict_package_artifacts,
		"contract_validation": {
			"valid": bool(registry_report["valid"]),
			"contract_count": registry_report["contract_count"],
			"error_count": registry_report["error_count"],
		},
		"summary": {
			"capability_count": len(records),
			"operable_contract_count": operable_contract_count,
			"inoperable_contract_count": len(records) - operable_contract_count,
			"complete_package_count": complete_package_count,
			"partial_package_count": len(records) - complete_package_count,
			"package_gap_count": package_gap_count,
			"error_count": len(errors),
			"warning_count": len(warnings),
		},
		"records": records,
		"errors": errors,
		"warnings": warnings,
		"blocking_gaps": [
			{"surface": "capability_operability", "error": error}
			for error in errors
		],
	}


def _audit_record(record: CapabilityContractRecord) -> dict[str, Any]:
	contract = record.contract
	errors: list[str] = []
	rule_probes = _rule_probes(record.capability_id, errors)
	package_artifacts = _package_artifacts(record.path.parent)
	route_count = len(contract["ui"]["routes"])
	rule_count = len(contract["rule_engine"]["rules"])
	return {
		"capability": record.capability_id,
		"display_name": record.display_name,
		"category": _category(record.path),
		"path": str(record.path),
		"operable": not errors,
		"contract_surfaces": {
			"configuration": sorted(contract["configuration"].keys()),
			"configuration_schema_required": list(contract["configuration_schema"].get("required", [])),
			"rule_count": rule_count,
			"route_count": route_count,
			"ui_shell": contract["ui"]["shell"],
			"theme": contract["theme"]["name"],
			"theme_tokens": sorted(contract["theme"]["tokens"].keys()),
		},
		"rule_probes": rule_probes,
		"package_artifacts": package_artifacts,
		"errors": errors,
	}


def _rule_probes(capability_id: str, errors: list[str]) -> list[dict[str, Any]]:
	probes: list[dict[str, Any]] = []
	for name, context in RULE_PROBE_CONTEXTS.items():
		try:
			result = evaluate_rules(capability_id, dict(context))
			decision = result.get("decision") if isinstance(result, dict) else None
			matched_rules = result.get("matched_rules") if isinstance(result, dict) else None
			actions = result.get("actions") if isinstance(result, dict) else None
			ok = (
				decision in VALID_DECISIONS
				and isinstance(matched_rules, list)
				and isinstance(actions, list)
			)
			if not ok:
				errors.append(f"{capability_id}: rule probe {name} returned invalid result shape")
			probes.append({
				"name": name,
				"ok": ok,
				"decision": decision,
				"matched_rule_count": len(matched_rules) if isinstance(matched_rules, list) else 0,
				"action_count": len(actions) if isinstance(actions, list) else 0,
			})
		except Exception as exc:  # pragma: no cover - defensive audit shape
			errors.append(f"{capability_id}: rule probe {name} failed: {exc}")
			probes.append({
				"name": name,
				"ok": False,
				"decision": None,
				"matched_rule_count": 0,
				"action_count": 0,
			})
	return probes


def _package_artifacts(package_dir: Path) -> dict[str, Any]:
	present: list[str] = []
	missing: list[str] = []
	for artifact in RECOMMENDED_PACKAGE_ARTIFACTS:
		path = package_dir / artifact
		if path.exists():
			present.append(artifact)
		else:
			missing.append(artifact)
	return {
		"package_dir": str(package_dir),
		"complete": not missing,
		"present": present,
		"missing": missing,
	}


def _category(path: Path) -> str:
	parts = path.parts
	if "capabilities" not in parts:
		return ""
	index = parts.index("capabilities")
	return parts[index + 1] if index + 1 < len(parts) else ""
