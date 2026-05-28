"""APG diagnostic registry and fixture audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DIAGNOSTIC_AUDIT_FORMAT = "apg.diagnostic-audit.v1"


DIAGNOSTIC_REGISTRY: dict[str, dict[str, str]] = {
	"APG0001": {
		"title": "Syntax error",
		"severity": "error",
		"area": "syntax",
		"trigger": "Source cannot be parsed.",
		"example_fix": "Show syntax location and nearest valid construct.",
		"meaning": "The parser could not accept the source at the reported location.",
		"next_step": "Fix the syntax near the diagnostic range, then rerun lint or parser-golden.",
	},
	"APG0100": {
		"title": "Semantic warning",
		"severity": "warning",
		"area": "semantic",
		"trigger": "The semantic analyzer found a non-blocking model concern.",
		"example_fix": "Review the referenced declaration and decide whether the warning is expected.",
		"meaning": "The semantic analyzer found a non-blocking model concern.",
		"next_step": "Review the referenced declaration and decide whether the warning is expected for this baseline.",
	},
	"APG0101": {
		"title": "Duplicate top-level declaration",
		"severity": "error",
		"area": "naming",
		"trigger": "Duplicate top-level declaration in the same namespace.",
		"example_fix": "Rename one symbol.",
		"meaning": "Two declarations share a namespace and cannot be addressed unambiguously.",
		"next_step": "Rename or remove one declaration and rerun lint.",
	},
	"APG0200": {
		"title": "Type error",
		"severity": "error",
		"area": "tables",
		"trigger": "A declaration or expression does not satisfy current semantic type rules.",
		"example_fix": "Correct the referenced type, field, or expression before compiling.",
		"meaning": "A declaration or expression does not satisfy the current semantic type rules.",
		"next_step": "Correct the referenced type, field, or expression before compiling.",
	},
	"APG0201": {
		"title": "Unknown field type",
		"severity": "error",
		"area": "tables",
		"trigger": "Field references unknown type where no custom type is allowed.",
		"example_fix": "Create enum/table/type or choose known scalar.",
		"meaning": "The field type cannot be resolved to a scalar or declared APG type.",
		"next_step": "Create the target type or use a known scalar type.",
	},
	"APG0202": {
		"title": "Calculated field references unknown field",
		"severity": "error",
		"area": "tables",
		"trigger": "Calculated field references unknown field.",
		"example_fix": "Create field or fix expression.",
		"meaning": "A calculated expression names a field outside the table model.",
		"next_step": "Correct the expression or declare the missing field.",
	},
	"APG0301": {
		"title": "Unknown relationship target table",
		"severity": "error",
		"area": "relationships",
		"trigger": "Relationship target table does not exist.",
		"example_fix": "Create table or correct target.",
		"meaning": "A relationship points at a table that the semantic model cannot find.",
		"next_step": "Create the target table or correct the relationship target.",
	},
	"APG0302": {
		"title": "Unknown relationship target field",
		"severity": "error",
		"area": "relationships",
		"trigger": "Relationship target field does not exist.",
		"example_fix": "Create field or correct target.",
		"meaning": "A relationship points at a field that is absent from the target table.",
		"next_step": "Create the target field or correct the relationship.",
	},
	"APG0303": {
		"title": "Unresolved lookup path",
		"severity": "error",
		"area": "relationships",
		"trigger": "Lookup path cannot be resolved.",
		"example_fix": "Add relationship or change binding.",
		"meaning": "A lookup path cannot be traversed through known table relationships.",
		"next_step": "Add the missing relationship or change the binding path.",
	},
	"APG0304": {
		"title": "Broken multi-hop lookup chain",
		"severity": "error",
		"area": "relationships",
		"trigger": "Multi-hop lookup chain breaks at an intermediate segment.",
		"example_fix": "Add missing relationship.",
		"meaning": "A lookup path segment cannot be resolved before the final field.",
		"next_step": "Add the intermediate relationship or shorten the lookup path.",
	},
	"APG0401": {
		"title": "Unknown view subject table",
		"severity": "error",
		"area": "ui",
		"trigger": "View subject table does not exist.",
		"example_fix": "Create table or correct `for` target.",
		"meaning": "A view is bound to a table absent from the semantic model.",
		"next_step": "Create the table or correct the view subject.",
	},
	"APG0402": {
		"title": "Invalid view binding",
		"severity": "error",
		"area": "ui",
		"trigger": "Database-backed view binding is not a field, calculated field, or lookup path.",
		"example_fix": "Replace binding or create valid field/path.",
		"meaning": "A view binding cannot be backed by known data.",
		"next_step": "Bind to a field, calculated field, or valid lookup path.",
	},
	"APG0403": {
		"title": "Unresolved handler target",
		"severity": "error",
		"area": "ui",
		"trigger": "Handler target does not resolve.",
		"example_fix": "Create operation/flow/agent/contract target.",
		"meaning": "A UI handler references a target the semantic model cannot find.",
		"next_step": "Create the target or correct the handler reference.",
	},
	"APG0404": {
		"title": "Unknown component",
		"severity": "warning",
		"area": "ui",
		"trigger": "Component is unknown to the registered component catalog.",
		"example_fix": "Use known component or register one.",
		"meaning": "A visual component is not currently cataloged.",
		"next_step": "Register the component or use a known component.",
	},
	"APG0501": {
		"title": "Invalid rule equality operator",
		"severity": "error",
		"area": "rules",
		"trigger": "Rule expression uses single `=` instead of `==`.",
		"example_fix": "Rewrite equality operator.",
		"meaning": "A rule expression uses assignment syntax where comparison is required.",
		"next_step": "Replace `=` with `==` in the rule expression.",
	},
	"APG0502": {
		"title": "Rule references unknown field",
		"severity": "error",
		"area": "rules",
		"trigger": "Rule references unknown field.",
		"example_fix": "Correct field or lookup path.",
		"meaning": "A rule references data that is not in scope.",
		"next_step": "Correct the field name or declare the required field/path.",
	},
	"APG0601": {
		"title": "Invalid flow transition",
		"severity": "error",
		"area": "flows",
		"trigger": "Flow transition references undeclared or unreachable state where strict mode is enabled.",
		"example_fix": "Add transition or state directive.",
		"meaning": "A workflow transition does not connect valid states.",
		"next_step": "Declare the state or adjust the transition graph.",
	},
	"APG0602": {
		"title": "Human task missing assignee",
		"severity": "warning",
		"area": "flows",
		"trigger": "Human task has no assignee/participant.",
		"example_fix": "Add participant or assignment.",
		"meaning": "A human task cannot be routed to a participant.",
		"next_step": "Add an assignee, participant, or role mapping.",
	},
	"APG0701": {
		"title": "Unknown permission resource",
		"severity": "error",
		"area": "security",
		"trigger": "Permission references unknown resource.",
		"example_fix": "Create resource or correct permission subject.",
		"meaning": "A permission points to a resource absent from the semantic model.",
		"next_step": "Create the resource or correct the permission.",
	},
	"APG0702": {
		"title": "Secret literal in source",
		"severity": "error",
		"area": "security",
		"trigger": "Secret literal appears in source.",
		"example_fix": "Replace with env/secret binding.",
		"meaning": "A DSL file appears to contain a secret value directly.",
		"next_step": "Replace the literal with an environment or secret binding.",
	},
	"APG0801": {
		"title": "Unknown deployment unit target",
		"severity": "error",
		"area": "deployment",
		"trigger": "Deployment unit target is unknown.",
		"example_fix": "Use supported unit kind.",
		"meaning": "A deployment unit does not map to a supported package profile.",
		"next_step": "Use a supported deployment unit target.",
	},
	"APG0802": {
		"title": "Unsupported compiler target",
		"severity": "error",
		"area": "deployment",
		"trigger": "Package target does not match app targets.",
		"example_fix": "Add app target or change package target.",
		"meaning": "APG currently compiles to generated Python artifacts only.",
		"next_step": "Use --target python and express desktop, mobile, web, or deployment needs as packaging profiles.",
	},
	"APG0901": {
		"title": "Unknown capability include",
		"severity": "error",
		"area": "capability",
		"trigger": "Composition includes unknown capability key.",
		"example_fix": "Register capability or correct key.",
		"meaning": "An application composition references a capability that is not declared or cataloged.",
		"next_step": "Declare the capability or correct the key.",
	},
	"APG0902": {
		"title": "Unknown cross-capability contract",
		"severity": "error",
		"area": "capability",
		"trigger": "Cross-capability connection references unknown event/API/command.",
		"example_fix": "Declare contract or correct reference.",
		"meaning": "A capability connection cannot resolve both sides of the contract.",
		"next_step": "Declare the API/event/command or correct the reference.",
	},
	"APG0903": {
		"title": "Private table access across capabilities",
		"severity": "error",
		"area": "capability",
		"trigger": "Capability attempts shared private-table access.",
		"example_fix": "Use API/event/projection contract.",
		"meaning": "A capability boundary is bypassing its public contract.",
		"next_step": "Expose the data through an API, event, projection, or public capability contract.",
	},
	"APG1001": {
		"title": "Unresolved agent skill target",
		"severity": "error",
		"area": "agents",
		"trigger": "Agent skill target does not resolve.",
		"example_fix": "Create operation/flow/contract target.",
		"meaning": "An AI agent skill points at an unknown operation, flow, or contract.",
		"next_step": "Create the target or correct the agent skill.",
	},
	"APG1002": {
		"title": "Write-capable agent skill lacks permission",
		"severity": "error",
		"area": "agents",
		"trigger": "Agent has write-capable skill with no permission.",
		"example_fix": "Add permission or remove skill.",
		"meaning": "An AI agent can mutate state without an explicit permission declaration.",
		"next_step": "Add the permission or remove the write-capable skill.",
	},
	"APG1101": {
		"title": "Migration plan contains destructive drop",
		"severity": "warning",
		"area": "migration",
		"trigger": "Migration plan contains destructive drop.",
		"example_fix": "Require explicit migration approval.",
		"meaning": "A migration would remove a table or field.",
		"next_step": "Approve the drop explicitly or provide a rename hint.",
	},
	"APG1102": {
		"title": "Migration planner found a rename candidate",
		"severity": "info",
		"area": "migration",
		"trigger": "Dropped and added symbols appear to be a rename.",
		"example_fix": "Confirm with a rename hint.",
		"meaning": "The migration planner found a non-destructive rename possibility.",
		"next_step": "Confirm the rename or treat it as add/drop.",
	},
	"APG1103": {
		"title": "Migration requires data backfill",
		"severity": "warning",
		"area": "migration",
		"trigger": "Required field addition or nullability change requires existing rows to be populated.",
		"example_fix": "Add a default or backfill step.",
		"meaning": "Existing records need values before the schema can be safely enforced.",
		"next_step": "Add a default, backfill plan, or staged nullable field.",
	},
	"APG1104": {
		"title": "Capability ownership transfer requires review",
		"severity": "warning",
		"area": "migration",
		"trigger": "Capability-owned table moved between capability contracts.",
		"example_fix": "Review ownership boundary and data access contract.",
		"meaning": "A table moved from one capability owner to another.",
		"next_step": "Review API/event contracts and approve ownership transfer.",
	},
	"APG1105": {
		"title": "Unsupported migration backend",
		"severity": "error",
		"area": "migration",
		"trigger": "Requested migration backend is unknown.",
		"example_fix": "Use a supported migration backend.",
		"meaning": "The migration planner cannot target the requested backend profile.",
		"next_step": "Use postgresql, mysql, sqlite, or compatible.",
	},
	"APG1106": {
		"title": "Field type change may require data conversion",
		"severity": "warning",
		"area": "migration",
		"trigger": "Field type changed between semantic models.",
		"example_fix": "Add a staged conversion or new field plus backfill.",
		"meaning": "Existing values may not convert safely to the new type.",
		"next_step": "Plan conversion, validation, and rollback before execution.",
	},
	"APG1201": {
		"title": "Unrepresentable natural-language plan",
		"severity": "error",
		"area": "natural-language",
		"trigger": "Natural-language plan cannot be represented as DSL diff.",
		"example_fix": "Ask for narrower DSL-scoped change.",
		"meaning": "The request is too vague or outside bounded APG DSL operations.",
		"next_step": "Ask for a concrete table, capability, agent, or supported domain feature.",
	},
	"APG9000": {
		"title": "Internal tooling error",
		"severity": "error",
		"area": "internal",
		"trigger": "Tooling raised an unexpected exception.",
		"example_fix": "Capture command, source file, and error text.",
		"meaning": "A tooling exception occurred while parsing, validating, or explaining APG source.",
		"next_step": "Capture the command, source file, and error text as a compiler/tooling defect.",
	},
}


def diagnostic_registry() -> dict[str, dict[str, str]]:
	"""Return the stable diagnostic registry sorted by code."""
	return {code: dict(DIAGNOSTIC_REGISTRY[code]) for code in sorted(DIAGNOSTIC_REGISTRY)}


def explain_diagnostic(code: str) -> dict[str, str]:
	"""Return registry explanation for a diagnostic code, or a stable fallback."""
	normalized = code.upper()
	return dict(DIAGNOSTIC_REGISTRY.get(normalized, {
		"title": normalized,
		"severity": "info",
		"area": "unknown",
		"trigger": "Diagnostic code is not registered.",
		"example_fix": "Add registry coverage if this code is stable.",
		"meaning": "No built-in explanation is registered for this diagnostic code yet.",
		"next_step": "Inspect the diagnostic message and add registry coverage if this code is stable.",
	}))


def audit_diagnostic_fixtures(fixture_root: Path | None = None) -> dict[str, Any]:
	"""Audit diagnostic fixture coverage and registry shape."""
	fixture_root = fixture_root or Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "diagnostics"
	catalog_path = fixture_root / "catalog.json"
	diagnostics: list[dict[str, Any]] = []
	fixtures: list[dict[str, Any]] = []

	if catalog_path.exists():
		try:
			catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
			fixtures = list(catalog.get("fixtures", []))
		except json.JSONDecodeError as error:
			diagnostics.append(_audit_diagnostic("APG9000", "error", f"Diagnostic fixture catalog is invalid JSON: {error}"))
	else:
		diagnostics.append(_audit_diagnostic("APG9000", "error", f"Diagnostic fixture catalog missing: {catalog_path}"))

	registry_codes = set(DIAGNOSTIC_REGISTRY)
	fixture_codes = {str(fixture.get("code", "")).upper() for fixture in fixtures}
	missing_fixture_codes = sorted(registry_codes - fixture_codes)
	unknown_fixture_codes = sorted(fixture_codes - registry_codes)
	invalid_registry_codes = sorted(code for code in registry_codes if not re.fullmatch(r"APG\d{4}", code))
	severity_mismatches: list[dict[str, str]] = []

	for fixture in fixtures:
		code = str(fixture.get("code", "")).upper()
		expected = DIAGNOSTIC_REGISTRY.get(code)
		if not expected:
			continue
		fixture_severity = str(fixture.get("severity", ""))
		if fixture_severity != expected["severity"]:
			severity_mismatches.append({
				"code": code,
				"registry": expected["severity"],
				"fixture": fixture_severity,
			})

	for code in missing_fixture_codes:
		diagnostics.append(_audit_diagnostic("APG9000", "error", f"Diagnostic fixture missing for registered code {code}."))
	for code in unknown_fixture_codes:
		diagnostics.append(_audit_diagnostic("APG9000", "error", f"Fixture references unregistered diagnostic code {code}."))
	for code in invalid_registry_codes:
		diagnostics.append(_audit_diagnostic("APG9000", "error", f"Registry code is not APGdddd format: {code}."))
	for mismatch in severity_mismatches:
		diagnostics.append(_audit_diagnostic(
			"APG9000",
			"error",
			f"Fixture severity for {mismatch['code']} is {mismatch['fixture']}, expected {mismatch['registry']}.",
		))

	return {
		"format": DIAGNOSTIC_AUDIT_FORMAT,
		"ok": not any(diagnostic["severity"] == "error" for diagnostic in diagnostics),
		"fixture_catalog": str(catalog_path),
		"registry": diagnostic_registry(),
		"registry_codes": sorted(registry_codes),
		"fixture_codes": sorted(fixture_codes),
		"missing_fixture_codes": missing_fixture_codes,
		"unknown_fixture_codes": unknown_fixture_codes,
		"severity_mismatches": severity_mismatches,
		"fixtures": fixtures,
		"summary": {
			"registry_count": len(registry_codes),
			"fixture_count": len(fixtures),
			"missing_fixture_count": len(missing_fixture_codes),
			"unknown_fixture_count": len(unknown_fixture_codes),
			"severity_mismatch_count": len(severity_mismatches),
		},
		"diagnostics": diagnostics,
	}


def _audit_diagnostic(code: str, severity: str, message: str) -> dict[str, Any]:
	return {
		"code": code,
		"title": explain_diagnostic(code)["title"],
		"severity": severity,
		"message": message,
		"file": "",
		"range": {
			"start": {"line": 0, "character": 0},
			"end": {"line": 0, "character": 1},
		},
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#diagnostic-specification",
	}
