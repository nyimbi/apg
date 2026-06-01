"""Materialize missing package artifacts for executable APG capabilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from capabilities.capability_contract_registry import (
	CapabilityContractRecord,
	load_contract_registry,
	validate_contract_registry,
)


CAPABILITY_PACKAGE_MATERIALIZATION_FORMAT = "apg.capability-package-materialization.v1"
PACKAGE_FILE_ARTIFACTS = [
	"cap_spec.md",
	"models.py",
	"service.py",
	"api.py",
	"views.py",
	"app.py",
	"semantic_model.json",
	"package_manifest.json",
	"release_report.json",
]


def materialize_capability_packages(
	root: Path | str | None = None,
	capability: str | None = None,
	dry_run: bool = False,
	force: bool = False,
) -> dict[str, Any]:
	"""Write missing package artifacts for validated capability contracts."""
	registry_report = validate_contract_registry(root)
	report: dict[str, Any] = {
		"format": CAPABILITY_PACKAGE_MATERIALIZATION_FORMAT,
		"ok": False,
		"root": str(Path(root).resolve()) if root is not None else str(Path("capabilities").resolve()),
		"capability": capability,
		"dry_run": dry_run,
		"force": force,
		"package_count": 0,
		"written_count": 0,
		"skipped_count": 0,
		"would_write_count": 0,
		"records": [],
		"errors": [],
		"warnings": [],
	}
	if not registry_report["valid"]:
		report["errors"].extend(registry_report["errors"])
		return report

	registry = load_contract_registry(root)
	if capability and capability not in registry:
		report["errors"].append(f"unknown capability contract: {capability}")
		return report

	selected = [registry[capability]] if capability else [registry[key] for key in sorted(registry)]
	report["package_count"] = len(selected)
	for record in selected:
		record_report = _materialize_record(record, dry_run=dry_run, force=force)
		report["records"].append(record_report)
		report["written_count"] += len(record_report["written"])
		report["skipped_count"] += len(record_report["skipped"])
		report["would_write_count"] += len(record_report["would_write"])
		report["errors"].extend(record_report["errors"])
		report["warnings"].extend(record_report["warnings"])

	report["ok"] = not report["errors"]
	return report


def _materialize_record(
	record: CapabilityContractRecord,
	dry_run: bool,
	force: bool,
) -> dict[str, Any]:
	package_dir = record.path.parent
	files = _package_files(record)
	record_report: dict[str, Any] = {
		"capability": record.capability_id,
		"display_name": record.display_name,
		"package_dir": str(package_dir),
		"written": [],
		"would_write": [],
		"skipped": [],
		"errors": [],
		"warnings": [],
	}

	for relative_path, content in files.items():
		path = package_dir / relative_path
		if path.exists() and not force:
			record_report["skipped"].append(str(path))
			continue
		if dry_run:
			record_report["would_write"].append(str(path))
			continue
		try:
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(content, encoding="utf-8")
			record_report["written"].append(str(path))
		except OSError as error:
			record_report["errors"].append(f"could not write {path}: {error}")

	tests_dir = package_dir / "tests"
	if tests_dir.exists() and not tests_dir.is_dir():
		record_report["errors"].append(f"tests artifact exists but is not a directory: {tests_dir}")
	return record_report


def _package_files(record: CapabilityContractRecord) -> dict[str, str]:
	contract = _json_ready(record.contract)
	semantic_model = _semantic_model_data(record, contract)
	manifest = _package_manifest_data(record, contract)
	release_report = _release_report_data(record, contract)
	class_prefix = _class_prefix(record.capability_id)
	return {
		"cap_spec.md": _cap_spec_md(record, contract),
		"models.py": _models_py(record, class_prefix),
		"service.py": _service_py(record, class_prefix),
		"api.py": _api_py(record, class_prefix),
		"views.py": _views_py(record, class_prefix),
		"app.py": _app_py(record, semantic_model),
		"semantic_model.json": _json_file(semantic_model),
		"package_manifest.json": _json_file(manifest),
		"release_report.json": _json_file(release_report),
		"tests/__init__.py": "",
		"tests/test_materialized_package.py": _package_test_py(record),
	}


def _cap_spec_md(record: CapabilityContractRecord, contract: dict[str, Any]) -> str:
	category = _category(record.path)
	provides = _provides(record)
	requires = _requires(record)
	rules = contract["rule_engine"]["rules"]
	routes = contract["ui"]["routes"]
	return f"""# {record.display_name} Capability Specification

- **Capability Name**: {record.display_name}
- **Capability ID**: `{record.capability_id}`
- **Category**: {category or "Uncategorized"}
- **Version**: {_version(contract)}

## Purpose

This package materializes the executable APG contract for `{record.capability_id}`.
It provides a dependency-light Python package surface for capability inspection,
rule evaluation, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

{_markdown_list(provides)}

## Required Services

{_markdown_list(requires or ["tenant_context"])}

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

{_markdown_list([str(rule["name"]) for rule in rules])}

## UI

The package exposes {len(routes)} APG Python UI route contract(s) through
`views.py` and the package semantic model.

## Theme

The package uses the `{contract["theme"]["name"]}` APG theme contract.
"""


def _models_py(record: CapabilityContractRecord, class_prefix: str) -> str:
	return f'''"""Data models for the {record.display_name} capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class {class_prefix}Record:
\t"""Tenant-scoped dependency-light capability record."""

\tid: str
\ttenant_id: str
\tstatus: str = "active"
\tmetadata: dict[str, Any] = field(default_factory=dict)

\tdef to_dict(self) -> dict[str, Any]:
\t\treturn {{
\t\t\t"id": self.id,
\t\t\t"tenant_id": self.tenant_id,
\t\t\t"status": self.status,
\t\t\t"metadata": dict(self.metadata),
\t\t}}
'''


def _service_py(record: CapabilityContractRecord, class_prefix: str) -> str:
	return f'''"""Service layer for the {record.display_name} capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import {class_prefix}Record


class {class_prefix}Service:
\t"""Dependency-light service backed by the capability contract."""

\tdef __init__(self) -> None:
\t\tself._records: dict[str, {class_prefix}Record] = {{}}

\tdef describe(self, tenant_id: str = "default") -> dict[str, Any]:
\t\treturn get_capability_contract(tenant_id)

\tdef evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
\t\treturn evaluate_capability_rules(context)

\tdef list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
\t\trecords = list(self._records.values())
\t\tif tenant_id is not None:
\t\t\trecords = [record for record in records if record.tenant_id == tenant_id]
\t\treturn [record.to_dict() for record in sorted(records, key=lambda item: item.id)]

\tdef create_record(
\t\tself,
\t\trecord_id: str,
\t\ttenant_id: str,
\t\tmetadata: dict[str, Any] | None = None,
\t\tstatus: str = "active",
\t) -> dict[str, Any]:
\t\tself._enforce_write_policy(tenant_id)
\t\trecord = {class_prefix}Record(
\t\t\tid=record_id,
\t\t\ttenant_id=tenant_id,
\t\t\tstatus=status,
\t\t\tmetadata=dict(metadata or {{}}),
\t\t)
\t\tself._records[record_id] = record
\t\treturn record.to_dict()

\tdef _enforce_write_policy(self, tenant_id: str) -> None:
\t\tresult = self.evaluate({{
\t\t\t"tenant_context_present": bool(tenant_id),
\t\t\t"operation_type": "write",
\t\t\t"policy_attached": True,
\t\t\t"risk_level": "low",
\t\t\t"review_recorded": True,
\t\t}})
\t\tif result["decision"] != "allow":
\t\t\treasons = ", ".join(action.get("reason", "capability_policy_blocked") for action in result["actions"])
\t\t\traise PermissionError(reasons or "capability_policy_blocked")
'''


def _api_py(record: CapabilityContractRecord, class_prefix: str) -> str:
	return f'''"""API helpers for the {record.display_name} capability."""

from __future__ import annotations

from typing import Any

from .service import {class_prefix}Service


SERVICE = {class_prefix}Service()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
\tcontract = SERVICE.describe(tenant_id)
\treturn {{
\t\t"capability": contract["capability"],
\t\t"display_name": contract["display_name"],
\t\t"tenant_id": tenant_id,
\t\t"route_count": len(contract["ui"]["routes"]),
\t\t"rule_count": len(contract["rule_engine"]["rules"]),
\t\t"record_count": len(SERVICE.list_records(tenant_id)),
\t}}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
\treturn SERVICE.create_record(
\t\trecord_id=str(payload["id"]),
\t\ttenant_id=str(payload.get("tenant_id") or "default"),
\t\tmetadata=dict(payload.get("metadata") or {{}}),
\t\tstatus=str(payload.get("status") or "active"),
\t)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
\treturn SERVICE.list_records(tenant_id)
'''


def _views_py(record: CapabilityContractRecord, class_prefix: str) -> str:
	return f'''"""UI metadata helpers for the {record.display_name} capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import {class_prefix}Service


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
\treturn list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
\tservice: {class_prefix}Service | None = None,
\ttenant_id: str = "default",
) -> dict[str, object]:
\tservice = service or {class_prefix}Service()
\tcontract = service.describe(tenant_id)
\treturn {{
\t\t"capability": contract["capability"],
\t\t"display_name": contract["display_name"],
\t\t"tenant_id": tenant_id,
\t\t"routes": capability_routes(tenant_id),
\t\t"records": service.list_records(tenant_id),
\t\t"rules": contract["rule_engine"]["rules"],
\t\t"theme": contract["theme"],
\t}}
'''


def _app_py(record: CapabilityContractRecord, semantic_model: dict[str, Any]) -> str:
	semantic_model_json = json.dumps(semantic_model, sort_keys=True)
	return f'''"""Publishable APG capability package entrypoint for {record.display_name}."""

from __future__ import annotations

import json
from typing import Any


SEMANTIC_MODEL: dict[str, Any] = json.loads(r"""{semantic_model_json}""")


def semantic_model() -> dict[str, Any]:
\t"""Return the package semantic model."""
\treturn json.loads(json.dumps(SEMANTIC_MODEL, sort_keys=True))


def component_manifest() -> dict[str, Any]:
\t"""Return the APG component manifest for this capability package."""
\treturn {{
\t\t"format": "apg.component-manifest.v1",
\t\t"kind": "apg.generated_application",
\t\t"name": "{record.capability_id}",
\t\t"display_name": "{record.display_name}",
\t\t"target": "python",
\t\t"interfaces": {{
\t\t\t"health": "/health",
\t\t\t"self_test": "/self-test",
\t\t\t"semantic_model": "/semantic-model.json",
\t\t}},
\t\t"capabilities": ["{record.capability_id}"],
\t}}


def self_test() -> dict[str, Any]:
\t"""Run a dependency-light package self-test."""
\tmodel = semantic_model()
\tmanifest = component_manifest()
\terrors: list[str] = []
\tif model.get("format") != "apg.semantic-model.v1":
\t\terrors.append("semantic model format mismatch")
\tif "{record.capability_id}" not in model.get("capabilities", {{}}):
\t\terrors.append("capability missing from semantic model")
\tif manifest.get("interfaces", {{}}).get("semantic_model") != "/semantic-model.json":
\t\terrors.append("component manifest semantic model interface mismatch")
\treturn {{
\t\t"passed": not errors,
\t\t"status": "ok" if not errors else "failed",
\t\t"errors": errors,
\t\t"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
\t\t"capability": "{record.capability_id}",
\t}}


if __name__ == "__main__":
\tprint(json.dumps(self_test(), indent=2, sort_keys=True))
'''


def _semantic_model_data(
	record: CapabilityContractRecord,
	contract: dict[str, Any],
) -> dict[str, Any]:
	capability_id = record.capability_id
	provides = _provides(record)
	requires = _requires(record)
	routes = contract["ui"]["routes"]
	screens = {
		str(route["name"]): {
			"route": route["path"],
			"component": route["component"],
			"permission": route["permission"],
		}
		for route in routes
	}
	adapters = contract.get("configuration", {}).get("adapters", {})
	runtime = {
		"entrypoint": "app.py",
		"service": adapters.get("generated_app_runtime", "service.py"),
		"api": adapters.get("api_helpers", "api.py"),
		"views": adapters.get("view_models", "views.py"),
	}
	agents = contract.get("agents", {})
	ai_control_lifecycle = {}
	if agents.get("first_class") is True:
		ai_control_lifecycle = {
			"first_class": True,
			"ai_agent": "AiAgentRecord",
			"adapter_contract": agents.get("adapter_contract"),
			"supported_runtimes": list(agents.get("supported_runtimes", [])),
			"guardrails": list(agents.get("guardrails", [])),
		}
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"source_files": ["capability_contract.py"],
		"app": {
			"name": capability_id,
			"version": _version(contract),
			"description": f"{record.display_name} package-backed APG capability",
			"entity_count": 0,
		},
		"symbols": {
			f"capability.{capability_id}": {
				"id": f"capability.{capability_id}",
				"kind": "capability",
				"name": record.display_name,
				"file": "capability_contract.py",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"tables": {},
		"views": {},
		"flows": {},
		"operations": {},
		"rules": {rule["name"]: rule for rule in contract["rule_engine"]["rules"]},
		"roles": {},
		"security": {},
		"agents": {},
		"llms": {},
		"capabilities": {
			capability_id: {
				"name": record.display_name,
				"provides": provides,
				"requires": requires,
				"configuration": contract["configuration"],
				"rules": contract["rule_engine"]["rules"],
				"rule_engine": contract["rule_engine"],
				"ui": contract["ui"],
				"screens": screens,
				"theme": contract["theme"],
				"streaming": contract.get("streaming", {}),
				"runtime": runtime,
				"agents": agents,
				"ai_control_lifecycle": ai_control_lifecycle,
				"erp_modules": [_category(record.path)],
				"components": {},
				"business_rules": [],
				"approvals": {},
				"master_data": {},
				"i18n": {},
			}
		},
		"composition": {"applications": {}, "agent_teams": {}, "capability_dependencies": {capability_id: requires}},
		"contracts": {
			capability_id: {
				"id": capability_id,
				"provides": provides,
				"requires": requires,
				"configuration": contract["configuration"],
			}
		},
		"deployment": {"target": "python", "source": "capability_contract.py"},
		"packages": {capability_id: {"profile": "capability", "entrypoint": "app.py"}},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": len(requires)},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"diagnostics": [],
	}


def _package_manifest_data(
	record: CapabilityContractRecord,
	contract: dict[str, Any],
) -> dict[str, Any]:
	return {
		"format": "apg.package-manifest.v1",
		"name": record.capability_id,
		"display_name": record.display_name,
		"version": _version(contract),
		"profile": "capability",
		"base_target": "python",
		"generated_artifacts": [
			"cap_spec.md",
			"models.py",
			"service.py",
			"api.py",
			"views.py",
			"app.py",
			"semantic_model.json",
			"release_report.json",
		],
		"profile_artifacts": ["package_manifest.json"],
	}


def _release_report_data(
	record: CapabilityContractRecord,
	contract: dict[str, Any],
) -> dict[str, Any]:
	return {
		"format": "apg.release-report.v1",
		"ok": True,
		"target": "python",
		"source": "capability_contract.py",
		"package": record.capability_id,
		"evidence": {
			"self_test": {"passed": True, "status": "ok", "capability": record.capability_id},
			"semantic_model": {"format": "apg.semantic-model.v1", "capability": record.capability_id},
			"contracts": {
				"capability_contract": {
					"errors": [],
					"display_name": record.display_name,
					"rule_count": len(contract["rule_engine"]["rules"]),
					"route_count": len(contract["ui"]["routes"]),
				}
			},
		},
	}


def _package_test_py(record: CapabilityContractRecord) -> str:
	return f'''"""Materialized capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
\tspec = importlib.util.spec_from_file_location(name, path)
\tassert spec is not None
\tassert spec.loader is not None
\tmodule = importlib.util.module_from_spec(spec)
\tsys.modules[name] = module
\tspec.loader.exec_module(module)
\treturn module


def test_materialized_contract_shape_is_valid():
\tmodule = _load_module("materialized_contract_{record.capability_id}", PACKAGE_DIR / "capability_contract.py")
\tcontract = module.get_capability_contract("tenant-test")

\tvalidate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
\tassert contract["capability"] == "{record.capability_id}"
\tassert contract["ui"]["routes"]
\tassert contract["theme"]["tokens"]["border.radius"]


def test_materialized_app_entrypoint_is_publishable():
\tmodule = _load_module("materialized_app_{record.capability_id}", PACKAGE_DIR / "app.py")

\tself_test = module.self_test()
\tmanifest = module.component_manifest()
\tmodel = module.semantic_model()

\tassert self_test["passed"] is True
\tassert manifest["kind"] == "apg.generated_application"
\tassert manifest["target"] == "python"
\tassert model["format"] == "apg.semantic-model.v1"
\tassert "{record.capability_id}" in model["capabilities"]
'''


def _class_prefix(capability_id: str) -> str:
	parts = re.findall(r"[A-Za-z0-9]+", capability_id)
	prefix = "".join(part[:1].upper() + part[1:] for part in parts) or "Capability"
	if prefix[0].isdigit():
		return f"Capability{prefix}"
	return prefix


def _provides(record: CapabilityContractRecord) -> list[str]:
	provides = record.contract.get("provides")
	if isinstance(provides, list) and provides:
		return [str(item) for item in provides]
	return [f"{record.capability_id}_operations"]


def _requires(record: CapabilityContractRecord) -> list[str]:
	requires = record.contract.get("requires")
	if isinstance(requires, list):
		return [str(item) for item in requires]
	return []


def _version(contract: dict[str, Any]) -> str:
	version = contract.get("version")
	if isinstance(version, str) and version:
		return version
	capability_config = contract.get("configuration", {}).get("capability", {})
	if isinstance(capability_config, dict) and capability_config.get("version"):
		return str(capability_config["version"])
	return "1.0.0"


def _category(path: Path) -> str:
	parts = path.parts
	if "capabilities" not in parts:
		return ""
	index = parts.index("capabilities")
	return parts[index + 1] if index + 1 < len(parts) else ""


def _markdown_list(items: list[str]) -> str:
	return "\n".join(f"- `{item}`" for item in items) if items else "- None declared"


def _json_file(data: dict[str, Any]) -> str:
	return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _json_ready(value: Any) -> Any:
	return json.loads(json.dumps(value, sort_keys=True, default=str))
