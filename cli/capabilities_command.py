#!/usr/bin/env python3
"""APG capability registry commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from capabilities.capability_contract_registry import (
	load_contract_registry,
	validate_contract_registry,
)
from compiler.capability_publish import build_capability_publish_report


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
