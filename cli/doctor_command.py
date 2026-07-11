#!/usr/bin/env python3
"""APG doctor command."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
from typing import Any

import click

from compiler.doctor import build_doctor_report


REQUIRED_RUNTIME_PACKAGES = {
	"flask": "flask",
	"jinja2": "Jinja2",
	"pydantic": "pydantic",
	"sqlalchemy": "sqlalchemy",
	"httpx": "httpx",
	"dateutil": "python-dateutil",
}

_STATUS_COLORS = {
	"OK": "\033[32m",
	"WARN": "\033[33m",
	"ERROR": "\033[31m",
}
_RESET = "\033[0m"


@click.command(name="doctor")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.doctor-report.v1 JSON")
def doctor(as_json: bool) -> None:
	"""Check APG installation and environment."""
	report = build_doctor_report()
	_extend_report_with_cli_checks(report)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		_print_doctor_report(report)
	error_count = _error_count(report)
	if error_count:
		raise click.exceptions.Exit(error_count)


def _print_doctor_report(report: dict[str, object]) -> None:
	click.echo("APG Environment Check")
	click.echo()
	click.echo(f"Version: {report['version']}")
	click.echo(f"Language Specification: {report['language_specification']}")
	click.echo(f"Target Language: {report['target_language']}")
	click.echo(f"Python: {report['python']['version']}")
	click.echo()

	checks = report["checks"]
	for title, kind in [
		("Runtime", "runtime"),
		("Required Packages", "package"),
		("APG Components", "component"),
		("Grammar Compilation", "grammar"),
		("Capability Registry", "capability"),
		("Configuration", "configuration"),
		("Network", "network"),
	]:
		kind_checks = [
			check for check in checks
			if check["kind"] == kind and (kind != "package" or check["required"])
		]
		if not kind_checks:
			continue
		click.echo(f"{title}:")
		for check in kind_checks:
			detail = f"{check['detail']}: " if check.get("detail") else ""
			click.echo(f"{_status_label(check)} {detail}{check['message']}")
		click.echo()

	optional_checks = [
		check for check in checks
		if check["kind"] == "package" and not check["required"]
	]
	if optional_checks:
		click.echo("Optional Packages:")
		for check in optional_checks:
			click.echo(f"{_status_label(check)} {check['detail']}: {check['message']}")
		click.echo()

	summary = report["summary"]
	status = "complete" if report["ok"] else "failed"
	click.echo(
		f"APG environment check {status}: "
		f"{summary['passing_required_check_count']}/{summary['required_check_count']} "
		f"required checks passing, {summary['warning_count']} warning(s), "
		f"{summary['blocking_failure_count']} error(s)."
	)


def _status_label(check: dict[str, object]) -> str:
	status = str(check.get("status") or _legacy_status(check))
	label = f"[{status}]"
	if sys.stdout.isatty() and status in _STATUS_COLORS:
		return f"{_STATUS_COLORS[status]}{label}{_RESET}"
	return label


def _legacy_status(check: dict[str, object]) -> str:
	if check["ok"] and check["available"]:
		return "OK"
	if not check["required"]:
		return "WARN"
	return "ERROR"


def _extend_report_with_cli_checks(report: dict[str, Any]) -> None:
	checks = report["checks"]
	checks.extend(_build_cli_checks())

	blocking_failures = [
		check for check in checks
		if check["required"] and not check["ok"]
	]
	warnings = [
		check for check in checks
		if str(check.get("status") or _legacy_status(check)) == "WARN"
	]
	report["ok"] = not blocking_failures
	report["blocking_failures"] = blocking_failures
	report["warnings"] = warnings
	report["summary"] = {
		"check_count": len(checks),
		"required_check_count": sum(1 for check in checks if check["required"]),
		"optional_check_count": sum(1 for check in checks if not check["required"]),
		"passing_required_check_count": sum(1 for check in checks if check["required"] and check["ok"]),
		"blocking_failure_count": len(blocking_failures),
		"warning_count": len(warnings),
	}


def _build_cli_checks() -> list[dict[str, Any]]:
	checks: list[dict[str, Any]] = []
	checks.append(_check_python_39())
	checks.extend(_check_required_packages())
	checks.append(_check_production_secret())
	checks.append(_check_auth_users())
	checks.append(_check_smtp_host())
	return checks


def _check_python_39() -> dict[str, Any]:
	ok = sys.version_info >= (3, 9)
	return _check(
		name="cli_python_version",
		kind="runtime",
		required=True,
		ok=ok,
		detail=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
		message="Python version is >= 3.9" if ok else "Python 3.9+ is required",
	)


def _check_required_packages() -> list[dict[str, Any]]:
	checks: list[dict[str, Any]] = []
	for import_name, package_name in REQUIRED_RUNTIME_PACKAGES.items():
		available = importlib.util.find_spec(import_name) is not None
		checks.append(
			_check(
				name=f"cli_package:{package_name}",
				kind="package",
				required=True,
				ok=available,
				detail=import_name,
				message="import available" if available else f"required package not installed: {package_name}",
			)
		)
	return checks


def _check_production_secret() -> dict[str, Any]:
	production = _truthy(os.getenv("APG_PRODUCTION"))
	secret = os.getenv("APG_SECRET_KEY", "")
	ok = not production or bool(secret.strip())
	if production:
		message = "APG_SECRET_KEY is set" if ok else "APG_SECRET_KEY is required when APG_PRODUCTION=1"
	else:
		message = "development mode does not require APG_SECRET_KEY"
	return _check(
		name="apg_secret_key",
		kind="configuration",
		required=True,
		ok=ok,
		detail="APG_SECRET_KEY",
		message=message,
	)


def _check_auth_users() -> dict[str, Any]:
	raw_users = os.getenv("APG_AUTH_USERS")
	if not raw_users:
		return _check(
			name="apg_auth_users",
			kind="configuration",
			required=True,
			ok=True,
			detail="APG_AUTH_USERS",
			message="not set",
		)
	try:
		json.loads(raw_users)
	except json.JSONDecodeError as exc:
		return _check(
			name="apg_auth_users",
			kind="configuration",
			required=True,
			ok=False,
			detail="APG_AUTH_USERS",
			message=f"invalid JSON: {exc.msg}",
		)
	return _check(
		name="apg_auth_users",
		kind="configuration",
		required=True,
		ok=True,
		detail="APG_AUTH_USERS",
		message="valid JSON",
	)


def _check_smtp_host() -> dict[str, Any]:
	host = os.getenv("APG_SMTP_HOST")
	if not host:
		return _check(
			name="apg_smtp_host",
			kind="network",
			required=True,
			ok=True,
			detail="APG_SMTP_HOST",
			message="not set",
		)
	port_text = os.getenv("APG_SMTP_PORT", "587")
	try:
		port = int(port_text)
	except ValueError:
		return _check(
			name="apg_smtp_host",
			kind="network",
			required=True,
			ok=False,
			detail="APG_SMTP_HOST",
			message=f"APG_SMTP_PORT must be an integer, got {port_text!r}",
		)
	try:
		with socket.create_connection((host, port), timeout=2):
			pass
	except OSError as exc:
		return _check(
			name="apg_smtp_host",
			kind="network",
			required=True,
			ok=False,
			detail="APG_SMTP_HOST",
			message=f"cannot connect to {host}:{port}: {exc}",
		)
	return _check(
		name="apg_smtp_host",
		kind="network",
		required=True,
		ok=True,
		detail="APG_SMTP_HOST",
		message=f"reachable at {host}:{port}",
	)


def _check(
	*,
	name: str,
	kind: str,
	required: bool,
	ok: bool,
	detail: str,
	message: str,
) -> dict[str, Any]:
	return {
		"name": name,
		"kind": kind,
		"required": required,
		"ok": ok,
		"available": ok,
		"detail": detail,
		"message": message,
		"status": "OK" if ok else "ERROR" if required else "WARN",
	}


def _truthy(value: str | None) -> bool:
	return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _error_count(report: dict[str, Any]) -> int:
	return sum(
		1 for check in report["checks"]
		if str(check.get("status") or _legacy_status(check)) == "ERROR"
	)
