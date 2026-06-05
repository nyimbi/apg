#!/usr/bin/env python3
"""APG capability registry commands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

from capabilities.capability_contract_registry import (
	evaluate_rules,
	get_contract,
	load_contract_registry,
	validate_contract_registry,
)
from compiler.capability_publish import (
	apply_capability_publish_report,
	build_capability_catalog_report,
	build_capability_publish_report,
)
from compiler.capability_operability import audit_capability_operability
from compiler.capability_materializer import materialize_capability_packages
from compiler.capability_implementation import audit_capability_implementation
from compiler.capability_lifecycle import audit_capability_lifecycle


CAPABILITY_SCAFFOLD_FORMAT = "apg.capability-scaffold-report.v1"
CAPABILITY_INSPECT_FORMAT = "apg.capability-inspect-report.v1"
CAPABILITY_RULE_EVALUATION_FORMAT = "apg.capability-rule-evaluation-report.v1"
SAFE_SEGMENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def _contract_records(category: str | None = None) -> list[dict[str, Any]]:
	registry = load_contract_registry()
	records: list[dict[str, Any]] = []
	for record in sorted(registry.values(), key=lambda item: item.capability_id):
		path_parts = record.path.parts
		category_name = ""
		if "capabilities" in path_parts:
			index = path_parts.index("capabilities")
			if index + 1 < len(path_parts):
				category_name = path_parts[index + 1]
		if category and category_name != category:
			continue
		contract = record.contract
		records.append({
			"capability": record.capability_id,
			"display_name": record.display_name,
			"category": category_name,
			"path": str(record.path),
			"routes": len(contract["ui"]["routes"]),
			"rules": len(contract["rule_engine"]["rules"]),
			"theme": contract["theme"]["name"],
			"ui_shell": contract["ui"]["shell"],
		})
	return records


def _contracts_report(category: str | None = None) -> dict[str, Any]:
	records = _contract_records(category=category)
	return {
		"format": "apg.capability-contracts.v1",
		"ok": True,
		"category": category,
		"contract_count": len(records),
		"records": records,
	}


def _inspect_report(capability_id: str, tenant_id: str) -> dict[str, Any]:
	errors: list[str] = []
	try:
		contract = get_contract(capability_id, tenant_id=tenant_id)
	except KeyError:
		contract = {}
		errors.append(f"unknown capability contract: {capability_id}")
	except Exception as exc:  # pragma: no cover - defensive report shape
		contract = {}
		errors.append(str(exc))

	if errors:
		return {
			"format": CAPABILITY_INSPECT_FORMAT,
			"ok": False,
			"capability": capability_id,
			"tenant_id": tenant_id,
			"errors": errors,
		}

	return {
		"format": CAPABILITY_INSPECT_FORMAT,
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract.get("display_name", contract["capability"]),
		"tenant_id": tenant_id,
		"summary": {
			"configuration_sections": sorted(contract["configuration"].keys()),
			"rule_count": len(contract["rule_engine"]["rules"]),
			"route_count": len(contract["ui"]["routes"]),
			"theme": contract["theme"]["name"],
			"ui_shell": contract["ui"]["shell"],
		},
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rules": contract["rule_engine"]["rules"],
		"ui": contract["ui"],
		"theme": contract["theme"],
		"errors": [],
	}


def _rule_evaluation_report(
	capability_id: str,
	tenant_id: str,
	context_json: str | None,
	context_file: Path | None,
) -> dict[str, Any]:
	errors: list[str] = []
	context = _load_rule_context(context_json, context_file, errors)
	if errors:
		return {
			"format": CAPABILITY_RULE_EVALUATION_FORMAT,
			"ok": False,
			"capability": capability_id,
			"tenant_id": tenant_id,
			"context": context,
			"errors": errors,
		}
	try:
		result = evaluate_rules(capability_id, context, tenant_id=tenant_id)
	except KeyError:
		return {
			"format": CAPABILITY_RULE_EVALUATION_FORMAT,
			"ok": False,
			"capability": capability_id,
			"tenant_id": tenant_id,
			"context": context,
			"errors": [f"unknown capability contract: {capability_id}"],
		}
	except Exception as exc:  # pragma: no cover - defensive report shape
		return {
			"format": CAPABILITY_RULE_EVALUATION_FORMAT,
			"ok": False,
			"capability": capability_id,
			"tenant_id": tenant_id,
			"context": context,
			"errors": [str(exc)],
		}
	return {
		"format": CAPABILITY_RULE_EVALUATION_FORMAT,
		"ok": True,
		"capability": capability_id,
		"tenant_id": tenant_id,
		"context": context,
		"decision": result["decision"],
		"matched_rules": result["matched_rules"],
		"actions": result["actions"],
		"result": result,
		"errors": [],
	}


def _load_rule_context(
	context_json: str | None,
	context_file: Path | None,
	errors: list[str],
) -> dict[str, Any]:
	if context_json and context_file:
		errors.append("use only one of --context-json or --context-file")
		return {}
	if context_file:
		try:
			text = context_file.read_text(encoding="utf-8")
		except OSError as exc:
			errors.append(f"could not read context file: {exc}")
			return {}
	elif context_json:
		text = context_json
	else:
		text = "{}"
	try:
		parsed = json.loads(text)
	except json.JSONDecodeError as exc:
		errors.append(f"context must be valid JSON: {exc}")
		return {}
	if not isinstance(parsed, dict):
		errors.append("context JSON must be an object")
		return {}
	return parsed


def _scaffold_report(
	domain: str,
	code: str,
	name: str | None,
	out_dir: Path,
	force: bool,
) -> dict[str, Any]:
	"""Create a valid spec-backed APG capability package skeleton."""
	errors: list[str] = []
	warnings: list[str] = []
	domain = _normalize_segment(domain)
	code = _normalize_segment(code)
	if not SAFE_SEGMENT_PATTERN.match(domain):
		errors.append("domain must start with a lowercase letter and use lowercase letters, numbers, or underscores")
	if not SAFE_SEGMENT_PATTERN.match(code):
		errors.append("code must start with a lowercase letter and use lowercase letters, numbers, or underscores")

	capability_dir = out_dir / domain / code
	display_name = name or _display_name(code)
	capability_id = f"{domain}_{code}"
	files = _scaffold_files(domain, code, display_name)
	written: list[str] = []
	skipped: list[str] = []

	if errors:
		return _final_scaffold_report(domain, code, display_name, capability_dir, written, skipped, errors, warnings)

	for relative_path, content in files.items():
		path = capability_dir / relative_path
		if path.exists() and not force:
			skipped.append(str(path))
			continue
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(content, encoding="utf-8")
		written.append(str(path))

	if skipped:
		errors.append("target files already exist; rerun with --force to overwrite")

	report = _final_scaffold_report(domain, code, display_name, capability_dir, written, skipped, errors, warnings)
	report["capability"] = capability_id
	return report


def _final_scaffold_report(
	domain: str,
	code: str,
	display_name: str,
	capability_dir: Path,
	written: list[str],
	skipped: list[str],
	errors: list[str],
	warnings: list[str],
) -> dict[str, Any]:
	return {
		"format": CAPABILITY_SCAFFOLD_FORMAT,
		"ok": not errors,
		"domain": domain,
		"code": code,
		"capability": f"{domain}_{code}" if domain and code else "",
		"display_name": display_name,
		"path": str(capability_dir),
		"written": written,
		"skipped": skipped,
		"errors": errors,
		"warnings": warnings,
		"next_steps": [
			f"Review {capability_dir / 'cap_spec.md'}",
			f"Run python -m pytest -q {capability_dir / 'tests'}",
			f"Run apg capabilities validate-contracts --json after moving the scaffold under the repository capabilities root",
		],
	}


def _normalize_segment(value: str) -> str:
	return value.strip().lower().replace("-", "_")


def _display_name(code: str) -> str:
	return code.replace("_", " ").replace("-", " ").title()


def _scaffold_files(domain: str, code: str, display_name: str) -> dict[str, str]:
	capability_id = f"{domain}_{code}"
	class_prefix = "".join(part.title() for part in capability_id.split("_"))
	return {
		"__init__.py": _init_py(display_name, class_prefix),
		"cap_spec.md": _cap_spec_md(domain, code, display_name, capability_id),
		"capability_contract.py": _capability_contract_py(),
		"models.py": _models_py(class_prefix),
		"service.py": _service_py(class_prefix),
		"api.py": _api_py(class_prefix),
		"views.py": _views_py(class_prefix),
		"app.py": _app_py(capability_id, display_name),
		"semantic_model.json": _json_file(_semantic_model_data(capability_id, display_name, domain)),
		"package_manifest.json": _json_file(_package_manifest_data(capability_id, display_name)),
		"release_report.json": _json_file(_release_report_data(capability_id, display_name)),
		"tests/__init__.py": "",
		"tests/test_capability_contract.py": _contract_test_py(domain, code, capability_id, display_name),
		"pyproject.toml": _pyproject_toml(domain, code, display_name, capability_id),
		"CHANGELOG.md": _changelog_md(display_name, capability_id),
		"README.md": _scaffold_readme_md(display_name, capability_id, domain),
		"py.typed": "",
		"__main__.py": _main_py(domain, code, capability_id),
		"domain/__init__.py": _domain_init_py(display_name),
		"domain/adapters.py": _domain_adapters_py(display_name, class_prefix),
		"domain/rules.py": _domain_rules_py(display_name, capability_id, class_prefix),
		"domain/events.py": _domain_events_py(display_name, capability_id),
		"database/__init__.py": _db_init_py(display_name),
		"database/store.py": _db_store_py(display_name),
		"database/schema.sql": _db_schema_sql(capability_id, code),
		"alembic.ini": _alembic_ini_content(),
		"alembic/env.py": _alembic_env_py(),
		"alembic/script.py.mako": _alembic_mako(),
		"alembic/README": "Alembic migration environment.",
		"alembic/versions/0001_initial.py": _alembic_initial_migration(capability_id),
	}


def _init_py(display_name: str, class_prefix: str) -> str:
	return f'''"""APG capability package for {display_name}."""

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import {class_prefix}Service

__all__ = ["{class_prefix}Service", "evaluate_capability_rules", "get_capability_contract"]
'''


def _cap_spec_md(domain: str, code: str, display_name: str, capability_id: str) -> str:
	return f"""# {display_name} Capability Specification

- **Capability Name**: {display_name}
- **Capability ID**: `{capability_id}`
- **Category**: {domain.replace("_", " ").title()}
- **Code**: `{code}`
- **Version**: 1.0.0

## Purpose

Describe the business or platform outcome this capability owns.

## Provided Services

- `{capability_id}_operations`

## Required Services

- `audit_events`
- `tenant_context`

## Configuration

- `tenant_id`: tenant context required for every operation.
- `execution.require_tenant_context`: guards operations without tenant scope.
- `execution.policy_enforced`: requires policy attachment for writes.

## Rules

The generated `capability_contract.py` uses APG's standard deterministic
contract rules: tenant context required, policy required for writes, and
high-risk operations requiring review.

## UI

The scaffold exposes dashboard, operations, rules, and settings routes through
the APG Python UI shell.

## Theme

The scaffold uses the standard APG capability theme tokens. Add capability
specific tokens as UI behavior becomes concrete.
"""


def _capability_contract_py() -> str:
	return '''"""Executable APG capability contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from capabilities.capability_contract_factory import (
\tbuild_spec_capability_contract,
\tevaluate_contract_rules,
)


CAPABILITY_PATH = Path(__file__).resolve().parent


def get_capability_contract(
\ttenant_id: str = "default",
\toverrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
\t"""Return the executable APG capability contract."""
\treturn build_spec_capability_contract(CAPABILITY_PATH, tenant_id, overrides)


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
\t"""Evaluate standard deterministic capability rules."""
\tcontract = get_capability_contract(str(context.get("tenant_id") or "default"))
\treturn evaluate_contract_rules(contract["rule_engine"]["rules"], context)
'''


def _models_py(class_prefix: str) -> str:
	return f'''"""Data models for the {class_prefix} capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class {class_prefix}Record:
\t"""Dependency-light capability state record."""

\tid: str
\ttenant_id: str
\tstatus: str = "active"
\tmetadata: dict[str, Any] = field(default_factory=dict)

\tdef to_dict(self) -> dict[str, Any]:
\t\t"""Return a JSON-ready representation of this capability record."""
\t\treturn {{
\t\t\t"id": self.id,
\t\t\t"tenant_id": self.tenant_id,
\t\t\t"status": self.status,
\t\t\t"metadata": dict(self.metadata),
\t\t}}
'''


def _service_py(class_prefix: str) -> str:
	return f'''"""Service layer for the {class_prefix} capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import {class_prefix}Record


class {class_prefix}Service:
\t"""Executable dependency-light service backed by the capability contract."""

\tdef __init__(self) -> None:
\t\tself._records: dict[str, {class_prefix}Record] = {{}}

\tdef describe(self, tenant_id: str = "default") -> dict[str, Any]:
\t\treturn get_capability_contract(tenant_id)

\tdef evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
\t\treturn evaluate_capability_rules(context)

\tdef list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
\t\trecords = self._records.values()
\t\tif tenant_id is not None:
\t\t\trecords = [record for record in records if record.tenant_id == tenant_id]
\t\treturn [record.to_dict() for record in sorted(records, key=lambda item: item.id)]

\tdef get_record(self, record_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
\t\trecord = self._records.get(record_id)
\t\tif record is None:
\t\t\treturn None
\t\tif tenant_id is not None and record.tenant_id != tenant_id:
\t\t\treturn None
\t\treturn record.to_dict()

\tdef create_record(
\t\tself,
\t\trecord_id: str,
\t\ttenant_id: str,
\t\tmetadata: dict[str, Any] | None = None,
\t\tstatus: str = "active",
\t\tpolicy_attached: bool = True,
\t\trisk_level: str = "low",
\t\treview_recorded: bool = True,
\t) -> dict[str, Any]:
\t\t"""Create one tenant-scoped record after contract-rule evaluation."""
\t\tself._enforce_write_policy(tenant_id, policy_attached, risk_level, review_recorded)
\t\tif record_id in self._records:
\t\t\traise ValueError(f"record already exists: {{record_id}}")
\t\trecord = {class_prefix}Record(
\t\t\tid=record_id,
\t\t\ttenant_id=tenant_id,
\t\t\tstatus=status,
\t\t\tmetadata=dict(metadata or {{}}),
\t\t)
\t\tself._records[record_id] = record
\t\treturn record.to_dict()

\tdef update_status(
\t\tself,
\t\trecord_id: str,
\t\tstatus: str,
\t\ttenant_id: str | None = None,
\t\tpolicy_attached: bool = True,
\t\trisk_level: str = "low",
\t\treview_recorded: bool = True,
\t) -> dict[str, Any]:
\t\t"""Update one record status after contract-rule evaluation."""
\t\trecord = self._records.get(record_id)
\t\tif record is None:
\t\t\traise KeyError(f"unknown record: {{record_id}}")
\t\tif tenant_id is not None and record.tenant_id != tenant_id:
\t\t\traise KeyError(f"unknown record for tenant: {{record_id}}")
\t\tself._enforce_write_policy(record.tenant_id, policy_attached, risk_level, review_recorded)
\t\trecord.status = status
\t\treturn record.to_dict()

\tdef _enforce_write_policy(
\t\tself,
\t\ttenant_id: str,
\t\tpolicy_attached: bool,
\t\trisk_level: str,
\t\treview_recorded: bool,
\t) -> None:
\t\tresult = self.evaluate({{
\t\t\t"tenant_context_present": bool(tenant_id),
\t\t\t"operation_type": "write",
\t\t\t"policy_attached": policy_attached,
\t\t\t"risk_level": risk_level,
\t\t\t"review_recorded": review_recorded,
\t\t}})
\t\tif result["decision"] == "deny":
\t\t\treasons = ", ".join(action.get("reason", "policy_denied") for action in result["actions"])
\t\t\traise PermissionError(reasons or "capability_write_denied")
\t\tif result["decision"] == "require_review":
\t\t\treasons = ", ".join(action.get("reason", "review_required") for action in result["actions"])
\t\t\traise PermissionError(reasons or "capability_review_required")
'''


def _api_py(class_prefix: str) -> str:
	return f'''"""API helpers for the {class_prefix} capability."""

from __future__ import annotations

from typing import Any

from .service import {class_prefix}Service


SERVICE = {class_prefix}Service()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
\t"""Return dependency-light capability status for generated integrations."""
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
\t"""Create a tenant-scoped record from a JSON-like payload."""
\treturn SERVICE.create_record(
\t\trecord_id=str(payload["id"]),
\t\ttenant_id=str(payload.get("tenant_id") or "default"),
\t\tmetadata=dict(payload.get("metadata") or {{}}),
\t\tstatus=str(payload.get("status") or "active"),
\t\tpolicy_attached=bool(payload.get("policy_attached", True)),
\t\trisk_level=str(payload.get("risk_level") or "low"),
\t\treview_recorded=bool(payload.get("review_recorded", True)),
\t)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
\t"""List records for generated integrations or smoke tests."""
\treturn SERVICE.list_records(tenant_id)


def get_record(record_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
\t"""Return one record if it exists for the requested tenant."""
\treturn SERVICE.get_record(record_id, tenant_id)


def update_record_status(
\trecord_id: str,
\tstatus: str,
\ttenant_id: str | None = None,
\tpolicy_attached: bool = True,
\trisk_level: str = "low",
\treview_recorded: bool = True,
) -> dict[str, Any]:
\t"""Update one record status through the rule-guarded service path."""
\treturn SERVICE.update_status(
\t\trecord_id,
\t\tstatus,
\t\ttenant_id,
\t\tpolicy_attached=policy_attached,
\t\trisk_level=risk_level,
\t\treview_recorded=review_recorded,
\t)
'''


def _views_py(class_prefix: str) -> str:
	return f'''"""UI metadata helpers for the {class_prefix} capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import {class_prefix}Service


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
\t"""Return APG Python UI route metadata."""
\treturn list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
\tservice: {class_prefix}Service | None = None,
\ttenant_id: str = "default",
) -> dict[str, object]:
\t"""Return a dependency-light dashboard view model."""
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


def _app_py(capability_id: str, display_name: str) -> str:
	semantic_model_json = json.dumps(_semantic_model_data(capability_id, display_name, capability_id.split("_", 1)[0]), sort_keys=True)
	return f'''"""Publishable APG capability package entrypoint for {display_name}."""

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
\t\t"name": "{capability_id}",
\t\t"display_name": "{display_name}",
\t\t"target": "python",
\t\t"interfaces": {{
\t\t\t"health": "/health",
\t\t\t"self_test": "/self-test",
\t\t\t"semantic_model": "/semantic-model.json",
\t\t}},
\t\t"capabilities": ["{capability_id}"],
\t}}


def self_test() -> dict[str, Any]:
\t"""Run a dependency-light package self-test."""
\tmodel = semantic_model()
\tmanifest = component_manifest()
\terrors: list[str] = []
\tif model.get("format") != "apg.semantic-model.v1":
\t\terrors.append("semantic model format mismatch")
\tif "{capability_id}" not in model.get("capabilities", {{}}):
\t\terrors.append("capability missing from semantic model")
\tif manifest.get("interfaces", {{}}).get("semantic_model") != "/semantic-model.json":
\t\terrors.append("component manifest semantic model interface mismatch")
\treturn {{
\t\t"passed": not errors,
\t\t"status": "ok" if not errors else "failed",
\t\t"errors": errors,
\t\t"routes": ["/health", "/self-test", "/component.json", "/semantic-model.json"],
\t}}


if __name__ == "__main__":
\tprint(json.dumps(self_test(), indent=2, sort_keys=True))
'''


def _semantic_model_data(capability_id: str, display_name: str, domain: str) -> dict[str, Any]:
	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"source_files": ["cap_spec.md"],
		"app": {
			"name": capability_id,
			"version": "1.0.0",
			"description": f"{display_name} scaffolded capability package",
			"entity_count": 1,
		},
		"symbols": {
			f"capability.{capability_id}": {
				"id": f"capability.{capability_id}",
				"kind": "capability",
				"name": display_name,
				"file": "cap_spec.md",
				"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
				"references": [],
			}
		},
		"tables": {},
		"views": {},
		"flows": {},
		"operations": {},
		"rules": {},
		"roles": {},
		"security": {},
		"agents": {},
		"llms": {},
		"capabilities": {
			capability_id: {
				"name": display_name,
				"provides": [f"{capability_id}_operations"],
				"requires": ["audit_events", "tenant_context"],
				"configuration": {"tenant_scoped": True, "policy_enforced": True},
				"rules": [
					{"name": "tenant_context_required", "when": "tenant_context_present == false", "action": "deny"},
					{"name": "operation_policy_required", "when": "operation_type == write and policy_attached == false", "action": "deny"},
				],
				"rule_engine": {"type": "deterministic"},
				"ui": {
					"shell": "apg_python",
					"routes": [
						{"name": "dashboard", "path": f"/{capability_id.replace('_', '-')}/dashboard", "component": "CapabilityDashboard"}
					],
				},
				"theme": {"name": f"{capability_id}_operations", "tokens": {"border.radius": "8px"}},
				"runtime": {"entrypoint": "app.py", "service": "service.py", "api": "api.py", "views": "views.py"},
				"erp_modules": [domain],
				"components": {},
				"business_rules": [],
				"approvals": {},
				"master_data": {},
				"i18n": {},
				"streaming": {},
				"screens": {},
			}
		},
		"composition": {"applications": {}, "agent_teams": {}, "capability_dependencies": {capability_id: ["audit_events", "tenant_context"]}},
		"contracts": {
			capability_id: {
				"id": capability_id,
				"provides": [f"{capability_id}_operations"],
				"requires": ["audit_events", "tenant_context"],
				"configuration": {"tenant_scoped": True, "policy_enforced": True},
			}
		},
		"deployment": {"target": "python", "source": "cap_spec.md"},
		"packages": {capability_id: {"profile": "capability", "entrypoint": "app.py"}},
		"graphs": {
			"capability": {"kind": "capability", "nodes": 1, "edges": 0},
			"package": {"kind": "package", "nodes": 2, "edges": 1},
		},
		"diagnostics": [],
	}


def _package_manifest_data(capability_id: str, display_name: str) -> dict[str, Any]:
	return {
		"format": "apg.package-manifest.v1",
		"name": capability_id,
		"display_name": display_name,
		"version": "1.0.0",
		"profile": "capability",
		"base_target": "python",
		"generated_artifacts": [
			"__init__.py",
			"cap_spec.md",
			"capability_contract.py",
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


def _release_report_data(capability_id: str, display_name: str) -> dict[str, Any]:
	return {
		"format": "apg.release-report.v1",
		"ok": True,
		"target": "python",
		"source": "cap_spec.md",
		"package": capability_id,
		"evidence": {
			"self_test": {"passed": True, "status": "ok", "capability": capability_id},
			"semantic_model": {"format": "apg.semantic-model.v1", "capability": capability_id},
			"contracts": {"capability_contract": {"errors": [], "display_name": display_name}},
		},
	}


def _json_file(data: dict[str, Any]) -> str:
	return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _contract_test_py(domain: str, code: str, capability_id: str, display_name: str) -> str:
	return f'''"""Scaffolded capability contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

from capabilities.capability_contract_registry import validate_contract_shape


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "capability_contract.py"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "scaffolded_{domain}_{code}"


def _load_contract_module():
\tspec = importlib.util.spec_from_file_location("scaffolded_capability_contract", CONTRACT_PATH)
\tassert spec is not None
\tassert spec.loader is not None
\tmodule = importlib.util.module_from_spec(spec)
\tsys.modules[spec.name] = module
\tspec.loader.exec_module(module)
\treturn module


def _load_package_module(name: str):
\tif PACKAGE_NAME not in sys.modules:
\t\tpackage = types.ModuleType(PACKAGE_NAME)
\t\tpackage.__path__ = [str(PACKAGE_DIR)]
\t\tsys.modules[PACKAGE_NAME] = package
\tspec = importlib.util.spec_from_file_location(
\t\tf"{{PACKAGE_NAME}}.{{name}}",
\t\tPACKAGE_DIR / f"{{name}}.py",
\t)
\tassert spec is not None
\tassert spec.loader is not None
\tmodule = importlib.util.module_from_spec(spec)
\tsys.modules[spec.name] = module
\tspec.loader.exec_module(module)
\treturn module


def test_scaffolded_capability_contract_is_valid():
\tmodule = _load_contract_module()
\tcontract = module.get_capability_contract("tenant-test")

\tvalidate_contract_shape(contract, CONTRACT_PATH)
\tassert contract["capability"] == "{capability_id}"
\tassert contract["display_name"] == "{display_name}"
\tassert contract["configuration"]["tenant_id"] == "tenant-test"
\tassert contract["ui"]["routes"]
\tassert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_scaffolded_capability_rules_are_executable():
\tmodule = _load_contract_module()
\tresult = module.evaluate_capability_rules({{"tenant_context_present": False}})

\tassert result["decision"] == "deny"
\tassert "tenant_context_required" in result["matched_rules"]


def test_scaffolded_service_api_and_views_are_executable():
\tservice_module = _load_package_module("service")
\tapi_module = _load_package_module("api")
\tviews_module = _load_package_module("views")

\tservice = service_module.{''.join(part.title() for part in capability_id.split('_'))}Service()
\trecord = service.create_record("demo-1", "tenant-test", {{"amount": 42}})
\tupdated = service.update_status("demo-1", "approved", tenant_id="tenant-test")
\tdashboard = views_module.dashboard_model(service, tenant_id="tenant-test")
\tapi_record = api_module.create_record({{"id": "api-1", "tenant_id": "tenant-test"}})

\tassert record["metadata"]["amount"] == 42
\tassert updated["status"] == "approved"
\tassert dashboard["records"][0]["id"] == "demo-1"
\tassert api_record["id"] == "api-1"
\tassert api_module.capability_status("tenant-test")["record_count"] == 1


def test_scaffolded_service_enforces_write_rules():
\tservice_module = _load_package_module("service")
\tservice = service_module.{''.join(part.title() for part in capability_id.split('_'))}Service()

\ttry:
\t\tservice.create_record("blocked-1", "", policy_attached=False)
\texcept PermissionError as exc:
\t\tassert "tenant_context_required" in str(exc)
\telse:
\t\traise AssertionError("expected tenant and policy denial")
'''


@click.group(name="capabilities")
def capabilities() -> None:
	"""Inspect executable APG capability contracts."""


@capabilities.command(name="list")
@click.option("--category", default=None, help="Filter by top-level capability category")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-contracts.v1 JSON")
def list_capabilities(category: str | None, as_json: bool) -> None:
	"""List executable capability contracts."""
	report = _contracts_report(category=category)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		click.echo(f"Capability contracts: {report['contract_count']}")
		for record in report["records"]:
			click.echo(
				f"  {record['capability']:<32} "
				f"category={record['category']:<12} "
				f"rules={record['rules']:<2} "
				f"routes={record['routes']:<2} "
				f"theme={record['theme']}"
			)


@capabilities.command(name="contracts")
@click.option("--category", default=None, help="Filter by top-level capability category")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-contracts.v1 JSON")
def contracts(category: str | None, as_json: bool) -> None:
	"""List executable capability contract metadata."""
	report = _contracts_report(category=category)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		click.echo(f"Capability contracts: {report['contract_count']}")
		for record in report["records"]:
			click.echo(
				f"  {record['capability']:<32} "
				f"rules={record['rules']:<2} "
				f"routes={record['routes']:<2} "
				f"theme={record['theme']}"
			)


@capabilities.command(name="inspect")
@click.argument("capability_id")
@click.option("--tenant-id", default="default", help="Tenant id used to resolve tenant-scoped configuration")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-inspect-report.v1 JSON")
def inspect_capability(capability_id: str, tenant_id: str, as_json: bool) -> None:
	"""Inspect one capability's configuration, rules, UI, and theme contract."""
	report = _inspect_report(capability_id, tenant_id)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		if report["ok"]:
			summary = report["summary"]
			click.echo(f"Capability: {report['capability']} ({report['display_name']})")
			click.echo(f"  tenant: {report['tenant_id']}")
			click.echo(f"  configuration sections: {', '.join(summary['configuration_sections'])}")
			click.echo(f"  rules: {summary['rule_count']}")
			click.echo(f"  routes: {summary['route_count']} via {summary['ui_shell']}")
			click.echo(f"  theme: {summary['theme']}")
		else:
			click.echo(f"Capability inspect FAILED: {capability_id}")
			for error in report["errors"]:
				click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="evaluate-rules")
@click.argument("capability_id")
@click.option("--tenant-id", default="default", help="Tenant id used to load the capability contract")
@click.option("--context-json", default=None, help="JSON object used as the rule-evaluation context")
@click.option("--context-file", type=click.Path(path_type=Path), default=None, help="Path to a JSON object used as the rule-evaluation context")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-rule-evaluation-report.v1 JSON")
def evaluate_capability_rules(
	capability_id: str,
	tenant_id: str,
	context_json: str | None,
	context_file: Path | None,
	as_json: bool,
) -> None:
	"""Evaluate one capability's deterministic rule engine."""
	report = _rule_evaluation_report(capability_id, tenant_id, context_json, context_file)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		if report["ok"]:
			click.echo(f"Capability rule decision: {report['decision']}")
			click.echo(f"  capability: {report['capability']}")
			click.echo(f"  tenant: {report['tenant_id']}")
			click.echo(f"  matched rules: {', '.join(report['matched_rules']) or 'none'}")
		else:
			click.echo(f"Capability rule evaluation FAILED: {capability_id}")
			for error in report["errors"]:
				click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="validate-contracts")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-contract-validation.v1 JSON")
def validate_contracts(as_json: bool) -> None:
	"""Validate every executable capability contract."""
	registry_report = validate_contract_registry()
	report = {
		"format": "apg.capability-contract-validation.v1",
		"ok": bool(registry_report["valid"]),
		**registry_report,
	}
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		if report["ok"]:
			click.echo(f"Validated {report['contract_count']} capability contracts")
		else:
			click.echo(
				f"Capability contract validation failed with "
				f"{report['error_count']} error(s)"
			)
			for error in report["errors"]:
				click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="audit")
@click.option(
	"--strict-package-artifacts",
	is_flag=True,
	help="Treat missing package runtime artifacts as blocking gaps",
)
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-operability-audit.v1 JSON")
def audit(strict_package_artifacts: bool, as_json: bool) -> None:
	"""Audit capability operability across contracts, rule probes, UI, theme, and package evidence."""
	report = audit_capability_operability(strict_package_artifacts=strict_package_artifacts)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		summary = report["summary"]
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability operability audit {status}: "
			f"{summary['operable_contract_count']}/{summary['capability_count']} contracts operable, "
			f"{summary['complete_package_count']} complete package(s), "
			f"{summary['package_gap_count']} package artifact gap(s)"
		)
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"][:10]:
			click.echo(f"  warning: {warning}")
		if len(report["warnings"]) > 10:
			click.echo(f"  ... {len(report['warnings']) - 10} more warning(s)")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="implementation-audit")
@click.option(
	"--strict",
	is_flag=True,
	help="Treat materialized-baseline packages as blocking implementation gaps",
)
@click.option("--root", type=click.Path(path_type=Path), default=None, help="Capability root directory to audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-implementation-audit.v1 JSON")
def implementation_audit(strict: bool, root: Path | None, as_json: bool) -> None:
	"""Audit whether capability packages still rely on materialized baseline code."""
	report = audit_capability_implementation(root=root, strict=strict)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		summary = report["summary"]
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability implementation audit {status}: "
			f"{summary['domain_specific_count']} domain-specific, "
			f"{summary['mixed_implementation_count']} mixed, "
			f"{summary['contract_only_count']} contract-only, "
			f"{summary['materialized_baseline_count']} materialized baseline package(s)"
		)
		for warning in report["warnings"][:10]:
			click.echo(f"  warning: {warning}")
		if len(report["warnings"]) > 10:
			click.echo(f"  ... {len(report['warnings']) - 10} more warning(s)")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="lifecycle-audit")
@click.option("--root", type=click.Path(path_type=Path), default=None, help="Capability root directory to audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-lifecycle-audit.v1 JSON")
def lifecycle_audit(root: Path | None, as_json: bool) -> None:
	"""Audit capability specification, plan, implementation, test, release, and review evidence."""
	report = audit_capability_lifecycle(root=root)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		summary = report["summary"]
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability lifecycle audit {status}: "
			f"{summary['complete_lifecycle_count']}/{summary['capability_count']} complete, "
			f"{summary['test_surface_count']} test surface(s), "
			f"{summary['release_evidence_count']} release evidence record(s)"
		)
		for warning in report["warnings"][:10]:
			click.echo(f"  warning: {warning}")
		if len(report["warnings"]) > 10:
			click.echo(f"  ... {len(report['warnings']) - 10} more warning(s)")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="materialize-packages")
@click.option("--root", type=click.Path(path_type=Path), default=None, help="Capability root directory to materialize")
@click.option("--capability", "capability_id", default=None, help="Materialize one capability id")
@click.option("--dry-run", is_flag=True, help="Report files that would be written without changing the workspace")
@click.option("--force", is_flag=True, help="Overwrite existing materialized files")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-package-materialization.v1 JSON")
def materialize_packages(
	root: Path | None,
	capability_id: str | None,
	dry_run: bool,
	force: bool,
	as_json: bool,
) -> None:
	"""Materialize missing package artifacts for executable capability contracts."""
	report = materialize_capability_packages(
		root=root,
		capability=capability_id,
		dry_run=dry_run,
		force=force,
	)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		action = "would write" if dry_run else "wrote"
		count = report["would_write_count"] if dry_run else report["written_count"]
		click.echo(
			f"Capability package materialization {status}: "
			f"{report['package_count']} package(s), {action} {count} file(s), "
			f"skipped {report['skipped_count']} existing file(s)"
		)
		for record in report["records"][:10]:
			record_count = len(record["would_write"]) if dry_run else len(record["written"])
			click.echo(f"  {record['capability']}: {record_count} file(s)")
		if len(report["records"]) > 10:
			click.echo(f"  ... {len(report['records']) - 10} more package(s)")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="scaffold")
@click.argument("domain")
@click.argument("code")
@click.option("--name", default=None, help="Human-readable capability name")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("capabilities"), help="Capability root directory")
@click.option("--force", is_flag=True, help="Overwrite scaffold files if they already exist")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-scaffold-report.v1 JSON")
def scaffold(domain: str, code: str, name: str | None, out_dir: Path, force: bool, as_json: bool) -> None:
	"""Create a valid spec-backed APG capability package skeleton."""
	report = _scaffold_report(domain, code, name, out_dir, force)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability scaffold {status}: {report['capability']} -> {report['path']} "
			f"({len(report['written'])} file(s) written)"
		)
		for path in report["written"]:
			click.echo(f"  wrote: {path}")
		for path in report["skipped"]:
			click.echo(f"  skipped: {path}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="publish-plan")
@click.argument("package_dir", type=click.Path(path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-publish-report.v1 JSON")
def publish_plan(package_dir: Path, as_json: bool) -> None:
	"""Validate a package and emit a side-effect-free capability catalog patch."""
	report = build_capability_publish_report(package_dir)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability publish-plan {status}: "
			f"{len(report['capabilities'])} capability(ies), "
			f"{len(report['catalog_patch'])} catalog patch op(s)"
		)
		for record in report["capabilities"]:
			click.echo(f"  {record['capability']} -> {record['package']}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="catalog")
@click.argument("catalog_path", type=click.Path(path_type=Path))
@click.option("--capability", "capability_id", default=None, help="Inspect one capability id from the catalog")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-catalog-report.v1 JSON")
def catalog(catalog_path: Path, capability_id: str | None, as_json: bool) -> None:
	"""Validate and inspect a local APG capability catalog."""
	report = build_capability_catalog_report(catalog_path, capability=capability_id)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(
			f"Capability catalog {status}: "
			f"{report['capability_count']} capability(ies) in {report['catalog']}"
		)
		for record in report["records"]:
			click.echo(
				f"  {record['capability']:<32} "
				f"package={record['package']} "
				f"version={record['version']} "
				f"routes={record['route_count']} "
				f"rules={record['rule_count']}"
			)
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command(name="publish-apply")
@click.argument("package_dir", type=click.Path(path_type=Path))
@click.option("--catalog", "catalog_path", type=click.Path(path_type=Path), required=True, help="Local capability catalog JSON file to update")
@click.option("--dry-run", is_flag=True, help="Validate and plan the catalog update without writing")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.capability-publish-apply-report.v1 JSON")
def publish_apply(package_dir: Path, catalog_path: Path, dry_run: bool, as_json: bool) -> None:
	"""Apply a valid capability package publish plan to a local catalog."""
	report = apply_capability_publish_report(package_dir, catalog_path, dry_run=dry_run)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		mode = "dry-run" if dry_run else "write"
		click.echo(
			f"Capability publish-apply {status}: "
			f"{len(report['capabilities'])} capability(ies), "
			f"catalog={report['catalog']} mode={mode}"
		)
		for capability_id in report["capabilities"]:
			click.echo(f"  applied: {capability_id}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


@capabilities.command("search")
@click.argument("keyword")
@click.option("--domain", default=None, help="Filter by domain (intel, fintech, fin, etc.)")
@click.option("--limit", default=10, help="Maximum results to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def capabilities_search(keyword: str, domain: str | None, limit: int, as_json: bool):
    """Search capabilities by keyword, domain, service name, or method."""
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from capabilities.manifest import find_capabilities, get_domain

    if domain:
        results = [c for c in get_domain(domain)
                   if keyword.lower() in (c["id"] + " " + c["display_name"] + " " + " ".join(c.get("provides", []))).lower()]
    else:
        results = find_capabilities(keyword, limit=limit)

    results = results[:limit]

    if as_json:
        click.echo(json.dumps([{
            "id": r["id"], "display_name": r["display_name"],
            "domain": r["domain"], "provides": r["provides"][:3],
            "install": r["install"]
        } for r in results], indent=2))
        return

    if not results:
        click.echo(f"No capabilities found for '{keyword}'")
        return

    click.echo(f"\nFound {len(results)} capabilities matching '{keyword}':\n")
    for cap in results:
        provides_preview = ", ".join(cap.get("provides", [])[:3])
        click.echo(f"  {cap['id']:40s}  {cap['display_name']}")
        click.echo(f"    Domain: {cap['domain']}  |  {cap['service_method_count']} methods")
        click.echo(f"    Provides: {provides_preview}")
        click.echo(f"    Install:  {cap['install']}")
        click.echo()


@capabilities.command("manifest")
@click.option("--capability", "-c", default=None, help="Show details for a specific capability ID")
@click.option("--domain", "-d", default=None, help="List all capabilities in a domain")
@click.option("--stats", is_flag=True, help="Show quality statistics")
def capabilities_manifest(capability: str | None, domain: str | None, stats: bool):
    """Inspect the APG Capability Manifest."""
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from capabilities.manifest import (
        get_capability, get_domain, all_capabilities, capability_count
    )
    import statistics

    if capability:
        cap = get_capability(capability)
        if not cap:
            click.echo(f"Capability '{capability}' not found", err=True)
            return
        click.echo(json.dumps(cap, indent=2))
        return

    if domain:
        caps = get_domain(domain)
        click.echo(f"\n{domain.upper()} domain — {len(caps)} capabilities:\n")
        for c in caps:
            click.echo(f"  {c['id']:45s}  {c['service_method_count']} methods  {c['rule_count']} rules")
        return

    if stats:
        caps = all_capabilities()
        methods = [c["service_method_count"] for c in caps]
        rules = [c["rule_count"] for c in caps]
        click.echo(f"\n=== APG Capability Quality Statistics ===\n")
        click.echo(f"  Total capabilities:   {capability_count()}")
        click.echo(f"  World-class (40+):    {sum(1 for m in methods if m >= 40)} ({sum(1 for m in methods if m >= 40)*100//len(caps)}%)")
        click.echo(f"  Mean methods:         {statistics.mean(methods):.1f}")
        click.echo(f"  Median methods:       {statistics.median(methods):.0f}")
        click.echo(f"  Mean rules:           {statistics.mean(rules):.1f}")
        click.echo(f"  Min methods:          {min(methods)}")
        click.echo(f"  Max methods:          {max(methods)}")
        click.echo(f"  No streaming events:  {sum(1 for c in caps if not c.get('streaming_events'))}")
        click.echo()
        return

    # Default: show summary
    click.echo(f"\nAPG Capability Manifest — {capability_count()} capabilities\n")
    from capabilities.manifest import list_domains, get_domain as gd
    for d in list_domains():
        caps = gd(d)
        avg_methods = sum(c["service_method_count"] for c in caps) / len(caps)
        click.echo(f"  {d:20s}  {len(caps):3d} caps  avg {avg_methods:.0f} methods")
    click.echo()
    click.echo("Use: apg capabilities search <keyword>")
    click.echo("     apg capabilities manifest --capability <id>")
    click.echo("     apg capabilities manifest --domain <domain>")
    click.echo("     apg capabilities manifest --stats")


# ── New scaffold generator functions ──────────────────────────────────────────

def _pyproject_toml(domain: str, code: str, display_name: str, capability_id: str) -> str:
	pkg_name = f"apg-{domain}-{code}"
	mod_name = pkg_name.replace("-", "_")
	return f'''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{pkg_name}"
version = "1.0.0"
description = "APG {display_name} capability"
readme = "README.md"
license = {{text = "Proprietary"}}
requires-python = ">=3.11"
keywords = ["apg", "datacraft", "capability", "{domain}", "{code}"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "pydantic>=2.0",
    "uuid6>=0.4",
    "sqlalchemy>=2.0",
    "flask>=3.0",
]

[project.optional-dependencies]
streaming = ["bytewax>=0.18"]
full = ["{pkg_name}[streaming]"]

[project.urls]
Homepage = "https://www.datacraft.co.ke"

[project.entry-points."apg.capabilities"]
{capability_id} = "{mod_name}:get_capability_contract"

[project.scripts]
{pkg_name} = "{mod_name}.app:main"

[tool.setuptools]
package-dir = {{"{mod_name}" = "."}}
packages    = ["{mod_name}"]

[tool.setuptools.package-data]
"{mod_name}" = ["py.typed", "README.md"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
'''


def _changelog_md(display_name: str, capability_id: str) -> str:
	return f'''# Changelog — {capability_id}

## [1.0.0] — {__import__("datetime").date.today()}

### Added
- Initial production release of **{display_name}** capability.
'''


def _scaffold_readme_md(display_name: str, capability_id: str, domain: str) -> str:
	pkg = f"apg-{domain}-{capability_id.split('_', 1)[-1] if '_' in capability_id else capability_id}"
	mod = pkg.replace("-", "_")
	return f'''# {display_name}

## Overview

{display_name} capability for the APG platform.

## Capability ID

`{capability_id}`

## Standalone Usage

```bash
pip install {pkg}
{pkg} --port 8080
```

```python
from {mod} import get_capability_contract, evaluate_capability_rules
from {mod}.service import {capability_id.replace("_", " ").title().replace(" ", "")}Service

svc = {capability_id.replace("_", " ").title().replace(" ", "")}Service(tenant_id="my_org")
contract = get_capability_contract("my_org")
```

## Development

```bash
pytest tests/ -q
python -m build --wheel .
```
'''


def _main_py(domain: str, code: str, capability_id: str) -> str:
	return f'"""Enable: python -m apg_{domain}_{code}"""\nfrom .app import main\nmain()\n'


def _domain_init_py(display_name: str) -> str:
	return f'"""Domain logic for {display_name}."""\nfrom .adapters import get_auth_adapter, get_audit_adapter, get_notify_adapter\n'


def _domain_adapters_py(display_name: str, class_prefix: str) -> str:
	return f'''"""Adapter protocols for {display_name}."""
from __future__ import annotations
import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthAdapter(Protocol):
    async def verify_token(self, token: str) -> dict[str, Any]: ...
    async def check_permission(self, user_id: str, permission: str) -> bool: ...


class NullAuthAdapter:
    async def verify_token(self, token: str) -> dict[str, Any]:
        return {{"user_id": token or "anonymous", "tenant_id": "default", "roles": ["user"]}}
    async def check_permission(self, user_id: str, permission: str) -> bool:
        return True


@runtime_checkable
class AuditAdapter(Protocol):
    async def log_event(self, event_type: str, actor_id: str, tenant_id: str, resource_id: str, details: dict) -> None: ...


class NullAuditAdapter:
    async def log_event(self, event_type: str, actor_id: str, tenant_id: str, resource_id: str, details: dict) -> None:
        pass


@runtime_checkable
class NotifyAdapter(Protocol):
    async def send(self, recipient: str, channel: str, subject: str, body: str) -> None: ...


class NullNotifyAdapter:
    async def send(self, recipient: str, channel: str, subject: str, body: str) -> None:
        pass


def get_auth_adapter(auth_service=None) -> AuthAdapter:
    if auth_service is not None: return auth_service
    try:
        from apg_common_auth import AuthService
        return AuthService.from_env()
    except ImportError:
        return NullAuthAdapter()

def get_audit_adapter(audit_service=None) -> AuditAdapter:
    if audit_service is not None: return audit_service
    try:
        from apg_common_audl import AuditService
        return AuditService.from_env()
    except ImportError:
        return NullAuditAdapter()

def get_notify_adapter(notify_service=None) -> NotifyAdapter:
    if notify_service is not None: return notify_service
    try:
        from apg_common_ntfy import NotifyService
        return NotifyService.from_env()
    except ImportError:
        return NullNotifyAdapter()
'''


def _domain_rules_py(display_name: str, capability_id: str, class_prefix: str) -> str:
	return f'''"""Deterministic domain rules for {display_name}."""
from __future__ import annotations
from typing import Any


class RuleViolation(Exception):
    def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
        self.rule_name = rule_name
        self.reason = reason
        self.required_action = required_action
        super().__init__(f"Rule '{{rule_name}}' violated: {{reason}}")


def assert_tenant_context(context: dict[str, Any]) -> None:
    if not context.get("tenant_id"):
        raise RuleViolation("tenant_context_required", "tenant_id is required", "attach_tenant_context")


def assert_write_policy(context: dict[str, Any]) -> None:
    if context.get("operation_type") == "write" and not context.get("policy_attached"):
        raise RuleViolation("write_requires_policy", "write operations require a policy", "attach_policy")


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
    if actor_tenant != resource_tenant:
        raise RuleViolation("cross_tenant_access_denied", "cross-tenant access is denied", "use_own_tenant_resources")
'''


def _domain_events_py(display_name: str, capability_id: str) -> str:
	return f'''"""Domain events for {display_name}."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
    event_type: str
    tenant_id: str
    actor_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {{
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "capability_id": "{capability_id}",
        }}
'''


def _db_init_py(display_name: str) -> str:
	return f'"""Database store for {display_name}."""\nfrom .store import get_store, InMemoryStore, Store\n__all__ = ["get_store", "InMemoryStore", "Store"]\n'


def _db_store_py(display_name: str) -> str:
	return '''"""Database store for this capability."""
from __future__ import annotations
import json, os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    async def get(self, collection: str, id: str) -> dict[str, Any] | None: ...
    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]: ...
    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]: ...
    async def delete(self, collection: str, id: str) -> bool: ...
    async def count(self, collection: str, filters: dict[str, Any]) -> int: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        return self._data.get(collection, {}).get(id)

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._data.setdefault(collection, {})[record["id"]] = dict(record)
        return record

    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        rows = list(self._data.get(collection, {}).values())
        for k, v in filters.items():
            rows = [r for r in rows if r.get(k) == v]
        return rows[:limit]

    async def delete(self, collection: str, id: str) -> bool:
        col = self._data.get(collection, {})
        return col.pop(id, None) is not None

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        return len(await self.query(collection, filters, limit=100_000))


def get_store(db_url: str | None = None) -> Store:
    resolved = db_url or os.environ.get("APG_DATABASE_URL")
    if resolved:
        try:
            return _PostgreSQLStore(resolved)
        except Exception:
            pass
    return InMemoryStore()


class _PostgreSQLStore:
    def __init__(self, db_url: str) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
        engine = create_async_engine(db_url, echo=False)
        self._session = async_sessionmaker(engine, class_=AsyncSession)

    async def get(self, collection: str, id: str) -> dict[str, Any] | None:
        async with self._session() as s:
            row = (await s.execute("SELECT data FROM apg_records WHERE collection=:c AND id=:id", {"c": collection, "id": id})).fetchone()
            return json.loads(row[0]) if row else None

    async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        async with self._session() as s:
            await s.execute("INSERT INTO apg_records(id,collection,tenant_id,data) VALUES(:id,:c,:t,:d) ON CONFLICT(collection,id) DO UPDATE SET data=EXCLUDED.data",
                {"id": record["id"], "c": collection, "t": record.get("tenant_id","default"), "d": json.dumps(record, default=str)})
            await s.commit()
        return record

    async def query(self, collection: str, filters: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        conds = " AND ".join(f"data->>\\'{k}\\' = :{k}" for k in filters)
        where = f"WHERE collection=:_c{' AND ' + conds if conds else ''}"
        async with self._session() as s:
            rows = (await s.execute(f"SELECT data FROM apg_records {where} LIMIT :lim", {"_c": collection, "lim": limit, **filters})).fetchall()
            return [json.loads(r[0]) for r in rows]

    async def delete(self, collection: str, id: str) -> bool:
        async with self._session() as s:
            r = await s.execute("DELETE FROM apg_records WHERE collection=:c AND id=:id", {"c": collection, "id": id})
            await s.commit()
            return r.rowcount > 0

    async def count(self, collection: str, filters: dict[str, Any]) -> int:
        return len(await self.query(collection, filters, limit=100_000))
'''


def _db_schema_sql(capability_id: str, code: str) -> str:
	return f'''-- APG {capability_id} database schema
CREATE TABLE IF NOT EXISTS apg_records (
    id TEXT NOT NULL,
    collection TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT \'default\',
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection, id)
);
CREATE INDEX IF NOT EXISTS idx_apg_{code}_tenant ON apg_records (collection, tenant_id);
CREATE INDEX IF NOT EXISTS idx_apg_{code}_data ON apg_records USING gin (data);
'''


def _alembic_ini_content() -> str:
	return '''[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = %(APG_DATABASE_URL)s

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
'''


def _alembic_env_py() -> str:
	return '''"""Alembic environment."""
import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

db_url = os.environ.get("APG_DATABASE_URL", "sqlite:///./capability.db")
config.set_main_option("sqlalchemy.url", db_url)
target_metadata = None

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''


def _alembic_mako() -> str:
	return '''"""${message}
Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

def upgrade():
    ${upgrades if upgrades else "pass"}

def downgrade():
    ${downgrades if downgrades else "pass"}
'''


def _alembic_initial_migration(capability_id: str) -> str:
	return f'''"""Initial migration: apg_records JSONB store.
Revision ID: 0001
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = \'0001\'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(\'apg_records\',
        sa.Column(\'id\', sa.Text(), nullable=False),
        sa.Column(\'collection\', sa.Text(), nullable=False),
        sa.Column(\'tenant_id\', sa.Text(), server_default=\'default\'),
        sa.Column(\'data\', postgresql.JSONB(), nullable=False),
        sa.Column(\'created_at\', sa.TIMESTAMP(timezone=True), server_default=sa.text(\'now()\')),
        sa.Column(\'updated_at\', sa.TIMESTAMP(timezone=True), server_default=sa.text(\'now()\')),
        sa.PrimaryKeyConstraint(\'collection\', \'id\'),
    )
    op.create_index(\'idx_{capability_id}_tenant\', \'apg_records\', [\'collection\', \'tenant_id\'])
    op.create_index(\'idx_{capability_id}_data\', \'apg_records\', [\'data\'], postgresql_using=\'gin\')

def downgrade():
    op.drop_table(\'apg_records\')
'''
