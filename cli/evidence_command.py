#!/usr/bin/env python3
"""APG release evidence bundle command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from compiler.evidence_bundle import audit_release_evidence_fixtures, build_release_evidence_bundle
from compiler.packager import SUPPORTED_PACKAGE_TARGETS


@click.command(name="evidence")
@click.argument("source_file", required=False, type=click.Path(path_type=Path))
@click.option("--audit-fixtures", is_flag=True, help="Audit checked-in release evidence verifier fixtures")
@click.option("--fixtures", type=click.Path(path_type=Path), default=None, help="Release evidence fixture catalog path")
@click.option(
	"--target",
	default="web",
	type=click.Choice(SUPPORTED_PACKAGE_TARGETS, case_sensitive=False),
	help="Packaging profile to bundle evidence for",
)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("dist"), help="Evidence package output root")
@click.option(
	"--skip-capability-publish",
	is_flag=True,
	help="Skip side-effect-free capability publish planning",
)
@click.option("--json", "as_json", is_flag=True, help="Emit apg.release-evidence-bundle.v1 JSON")
def evidence(
	source_file: Path | None,
	audit_fixtures: bool,
	fixtures: Path | None,
	target: str,
	out_dir: Path,
	skip_capability_publish: bool,
	as_json: bool,
) -> None:
	"""Build package and verifier evidence for an APG application."""
	if audit_fixtures:
		if source_file is not None or skip_capability_publish:
			raise click.ClickException("--audit-fixtures cannot be combined with a source file or --skip-capability-publish")
		report = audit_release_evidence_fixtures(fixtures)
		if as_json:
			click.echo(json.dumps(report, indent=2, sort_keys=True))
		else:
			summary = report["summary"]
			status = "OK" if report["ok"] else "FAILED"
			click.echo(
				f"APG evidence fixtures {status}: "
				f"{summary['passing_fixture_count']}/{summary['fixture_count']} fixtures, "
				f"{summary['target_run_count']} target runs, "
				f"{summary['blocking_gap_count']} blocking gaps"
			)
		if not report["ok"]:
			raise click.exceptions.Exit(1)
		return
	if source_file is None:
		raise click.ClickException("Specify an APG source file or use --audit-fixtures")
	if not source_file.exists():
		raise click.ClickException(f"APG source file not found: {source_file}")
	if not source_file.is_file():
		raise click.ClickException(f"APG evidence expects a file: {source_file}")

	report = build_release_evidence_bundle(
		source_file,
		target=target,
		out_dir=out_dir,
		include_capability_publish=not skip_capability_publish,
	)
	if as_json:
		click.echo(json.dumps(report, indent=2, sort_keys=True))
	else:
		status = "OK" if report["ok"] else "FAILED"
		click.echo(f"APG evidence {status}: {source_file}, target={target}, output={report['package'].get('output_dir')}")
		for check_name, passed in report["checks"].items():
			click.echo(f"  {check_name}: {'ok' if passed else 'failed'}")
		for error in report["errors"]:
			click.echo(f"  error: {error}")
		for warning in report["warnings"]:
			click.echo(f"  warning: {warning}")
	if not report["ok"]:
		raise click.exceptions.Exit(1)


if __name__ == "__main__":
	evidence()
