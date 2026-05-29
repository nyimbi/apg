#!/usr/bin/env python3
"""APG documentation audit commands."""

from __future__ import annotations

import json

import click

from compiler.docs_audit import audit_docs


@click.group(name="docs")
def docs() -> None:
	"""Audit APG documentation coverage and navigation."""


@docs.command(name="audit")
@click.option("--json", "as_json", is_flag=True, help="Emit apg.docs-audit.v1 JSON")
def audit(as_json: bool) -> None:
	"""Verify contributor-facing documentation is present and navigable."""
	report = audit_docs()
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		summary = report["summary"]
		click.echo(
			f"APG docs audit {status}: "
			f"{summary['required_doc_count']} required docs, "
			f"{summary['broken_local_link_count']} broken local link(s), "
			f"{summary['unknown_documented_command_count']} unknown documented command(s)"
		)
		for violation in report["violations"]:
			click.echo(f"  {violation['check']}: {violation.get('path', '')} {violation['message']}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)
