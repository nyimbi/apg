#!/usr/bin/env python3
"""APG doctor command."""

from __future__ import annotations

import json

import click
from rich.console import Console

from compiler.doctor import build_doctor_report


console = Console()


@click.command(name="doctor")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.doctor-report.v1 JSON")
def doctor(as_json: bool) -> None:
	"""Check APG installation and environment."""
	report = build_doctor_report()
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		_print_doctor_report(report)
	if not report["ok"]:
		raise click.exceptions.Exit(1)


def _print_doctor_report(report: dict[str, object]) -> None:
	console.print("[bold blue]APG Environment Check[/bold blue]")
	console.print()
	console.print(f"Version: {report['version']}")
	console.print(f"Language Specification: {report['language_specification']}")
	console.print(f"Target Language: {report['target_language']}")
	console.print(f"Python: {report['python']['version']}")
	console.print()

	checks = report["checks"]
	for title, kind in [
		("Runtime", "runtime"),
		("Required Packages", "package"),
		("APG Components", "component"),
		("Grammar Compilation", "grammar"),
		("Capability Registry", "capability"),
	]:
		kind_checks = [
			check for check in checks
			if check["kind"] == kind and (kind != "package" or check["required"])
		]
		if not kind_checks:
			continue
		console.print(f"[bold]{title}:[/bold]")
		for check in kind_checks:
			detail = f"{check['detail']}: " if check["kind"] == "package" else ""
			console.print(f"{_status_label(check)} {detail}{check['message']}")
		console.print()

	optional_checks = [
		check for check in checks
		if check["kind"] == "package" and not check["required"]
	]
	if optional_checks:
		console.print("[bold]Optional Packages:[/bold]")
		for check in optional_checks:
			console.print(f"{_status_label(check)} {check['detail']}: {check['message']}")
		console.print()

	summary = report["summary"]
	status = "complete" if report["ok"] else "failed"
	style = "green" if report["ok"] else "red"
	console.print(
		f"[{style}]APG environment check {status}: "
		f"{summary['passing_required_check_count']}/{summary['required_check_count']} "
		f"required checks passing, {summary['warning_count']} warning(s).[/{style}]"
	)


def _status_label(check: dict[str, object]) -> str:
	if check["ok"] and check["available"]:
		return "OK"
	if not check["required"]:
		return "WARN"
	return "FAIL"
