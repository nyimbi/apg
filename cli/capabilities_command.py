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
from compiler.capability_publish import build_capability_publish_report


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
		"__init__.py": _init_py(display_name),
		"cap_spec.md": _cap_spec_md(domain, code, display_name, capability_id),
		"capability_contract.py": _capability_contract_py(),
		"models.py": _models_py(class_prefix),
		"service.py": _service_py(class_prefix),
		"api.py": _api_py(class_prefix),
		"views.py": _views_py(class_prefix),
		"tests/__init__.py": "",
		"tests/test_capability_contract.py": _contract_test_py(capability_id, display_name),
	}


def _init_py(display_name: str) -> str:
	return f'''"""APG capability package for {display_name}."""

from .capability_contract import evaluate_capability_rules, get_capability_contract

__all__ = ["evaluate_capability_rules", "get_capability_contract"]
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
'''


def _service_py(class_prefix: str) -> str:
	return f'''"""Service layer for the {class_prefix} capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


class {class_prefix}Service:
\t"""Small executable service shell backed by the capability contract."""

\tdef describe(self, tenant_id: str = "default") -> dict[str, Any]:
\t\treturn get_capability_contract(tenant_id)

\tdef evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
\t\treturn evaluate_capability_rules(context)
'''


def _api_py(class_prefix: str) -> str:
	return f'''"""API helpers for the {class_prefix} capability."""

from __future__ import annotations

from typing import Any

from .service import {class_prefix}Service


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
\t"""Return dependency-light capability status for generated integrations."""
\tcontract = {class_prefix}Service().describe(tenant_id)
\treturn {{
\t\t"capability": contract["capability"],
\t\t"display_name": contract["display_name"],
\t\t"tenant_id": tenant_id,
\t\t"route_count": len(contract["ui"]["routes"]),
\t\t"rule_count": len(contract["rule_engine"]["rules"]),
\t}}
'''


def _views_py(class_prefix: str) -> str:
	return f'''"""UI metadata helpers for the {class_prefix} capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
\t"""Return APG Python UI route metadata."""
\treturn list(get_capability_contract(tenant_id)["ui"]["routes"])
'''


def _contract_test_py(capability_id: str, display_name: str) -> str:
	return f'''"""Scaffolded capability contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "capability_contract.py"


def _load_contract_module():
\tspec = importlib.util.spec_from_file_location("scaffolded_capability_contract", CONTRACT_PATH)
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
