"""Semantic-model migration planning for APG applications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .semantic_model import build_semantic_model


MIGRATION_PLAN_FORMAT = "apg.migration-plan.v1"
MIGRATION_FIXTURE_AUDIT_FORMAT = "apg.migration-fixture-audit.v1"
SUPPORTED_MIGRATION_BACKENDS = ("postgresql", "mysql", "sqlite", "compatible")
DEFAULT_MIGRATION_FIXTURE_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "migrations" / "catalog.json"


def build_migration_plan(
	previous: Path | dict[str, Any],
	current: Path | dict[str, Any],
	backend: str = "postgresql",
	rename_hints: dict[str, str] | None = None,
) -> dict[str, Any]:
	"""Compare two APG semantic models or APG sources and emit a migration plan."""
	previous_model = _load_model(previous)
	current_model = _load_model(current)
	backend = (backend or "postgresql").lower()
	rename_hints = rename_hints or {}
	diagnostics: list[dict[str, Any]] = []
	changes: list[dict[str, Any]] = []

	if backend not in SUPPORTED_MIGRATION_BACKENDS:
		diagnostics.append(_diagnostic(
			"APG1105",
			"Unsupported migration backend",
			"error",
			f"Migration backend must be one of {', '.join(SUPPORTED_MIGRATION_BACKENDS)}, not {backend!r}.",
			fixes=[{"id": "use_supported_backend", "title": "Use --backend postgresql, mysql, sqlite, or compatible"}],
		))

	changes.extend(_table_changes(previous_model, current_model, rename_hints))
	changes.extend(_capability_ownership_changes(previous_model, current_model))
	diagnostics.extend(_diagnostics_for_changes(changes))

	destructive = any(change.get("destructive") for change in changes)
	requires_approval = destructive or any(change.get("requires_approval") for change in changes)
	ok = not destructive and not any(diagnostic["severity"] == "error" for diagnostic in diagnostics)

	return {
		"format": MIGRATION_PLAN_FORMAT,
		"ok": ok,
		"backend": backend,
		"previous": _model_source(previous_model),
		"current": _model_source(current_model),
		"changes": changes,
		"destructive": destructive,
		"requires_approval": requires_approval,
		"summary": _summary(changes),
		"diagnostics": diagnostics,
	}


def audit_migration_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run the checked-in migration planner fixture catalog."""
	catalog_file = Path(catalog_path or DEFAULT_MIGRATION_FIXTURE_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []
	covered_tags: set[str] = set()

	for fixture in catalog.get("fixtures", []):
		report = _audit_migration_fixture(catalog_root, fixture)
		fixture_reports.append(report)
		if report["ok"]:
			covered_tags.update(report["tags"])
		else:
			blocking_gaps.append({
				"id": report["id"],
				"previous": report["previous"],
				"current": report["current"],
				"errors": report["errors"],
			})

	missing_tags = sorted(set(required_tags).difference(covered_tags))
	for tag in missing_tags:
		blocking_gaps.append({
			"id": f"missing_tag:{tag}",
			"previous": str(catalog_file),
			"current": str(catalog_file),
			"errors": [f"required migration fixture tag {tag!r} is not covered by a passing fixture"],
		})

	return {
		"format": MIGRATION_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"fixture_catalog": str(catalog_file),
		"tags_required": required_tags,
		"tags_covered": sorted(covered_tags),
		"missing_tags": missing_tags,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"failing_fixture_count": sum(1 for report in fixture_reports if not report["ok"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _audit_migration_fixture(catalog_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	previous = (catalog_root / str(fixture["previous"])).resolve()
	current = (catalog_root / str(fixture["current"])).resolve()
	backend = str(fixture.get("backend", "postgresql"))
	rename_hints = {str(key): str(value) for key, value in dict(fixture.get("rename_hints", {})).items()}
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	errors: list[str] = []
	report: dict[str, Any] | None = None

	try:
		report = build_migration_plan(previous, current, backend=backend, rename_hints=rename_hints)
	except Exception as error:
		errors.append(str(error))

	if report is None:
		return {
			"id": fixture_id,
			"previous": str(previous),
			"current": str(current),
			"backend": backend,
			"tags": tags,
			"change_kinds": [],
			"diagnostic_codes": [],
			"ok": False,
			"errors": errors,
		}

	_expected_bool(fixture, report, "ok", errors)
	_expected_bool(fixture, report, "destructive", errors)
	_expected_bool(fixture, report, "requires_approval", errors)

	changes = {
		(str(change.get("kind")), str(change.get("symbol"))): change
		for change in report.get("changes", [])
	}
	for expected_change in fixture.get("expected_changes", []):
		kind = str(expected_change.get("kind"))
		symbol = str(expected_change.get("symbol"))
		change = changes.get((kind, symbol))
		if change is None:
			errors.append(f"missing change {kind} {symbol}")
			continue
		for key, expected_value in expected_change.items():
			if key in {"kind", "symbol"}:
				continue
			if change.get(key) != expected_value:
				errors.append(f"change {kind} {symbol} expected {key}={expected_value!r}, got {change.get(key)!r}")

	diagnostic_codes = {str(diagnostic.get("code")) for diagnostic in report.get("diagnostics", [])}
	for code in fixture.get("expected_diagnostics", []):
		if str(code) not in diagnostic_codes:
			errors.append(f"missing diagnostic {code}")

	return {
		"id": fixture_id,
		"previous": str(previous),
		"current": str(current),
		"backend": backend,
		"tags": tags,
		"change_kinds": sorted({str(change.get("kind")) for change in report.get("changes", [])}),
		"diagnostic_codes": sorted(diagnostic_codes),
		"ok": not errors,
		"errors": errors,
	}


def _expected_bool(fixture: dict[str, Any], report: dict[str, Any], key: str, errors: list[str]) -> None:
	expected_key = f"expected_{key}"
	if expected_key not in fixture:
		return
	expected_value = bool(fixture[expected_key])
	actual_value = bool(report.get(key))
	if actual_value != expected_value:
		errors.append(f"expected {key}={expected_value}, got {actual_value}")


def _load_model(value: Path | dict[str, Any]) -> dict[str, Any]:
	if isinstance(value, dict):
		return value

	path = Path(value)
	if path.suffix.lower() == ".json":
		return json.loads(path.read_text(encoding="utf-8"))
	return build_semantic_model(path)


def _model_source(model: dict[str, Any]) -> str:
	files = model.get("source_files") or []
	return str(files[0]) if files else ""


def _table_changes(
	previous_model: dict[str, Any],
	current_model: dict[str, Any],
	rename_hints: dict[str, str],
) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	previous_tables = previous_model.get("tables", {})
	current_tables = current_model.get("tables", {})
	previous_names = set(previous_tables)
	current_names = set(current_tables)
	added_names = current_names - previous_names
	dropped_names = previous_names - current_names
	table_renames = _table_rename_candidates(previous_tables, current_tables, dropped_names, added_names, rename_hints)
	renamed_from = {change["before"]["name"] for change in table_renames}
	renamed_to = {change["after"]["name"] for change in table_renames}
	changes.extend(table_renames)

	for table_name in sorted(added_names - renamed_to):
		changes.append(_change(
			"add_table",
			f"table.{table_name}",
			after={"name": table_name, "fields": current_tables[table_name].get("fields", {})},
		))

	for table_name in sorted(dropped_names - renamed_from):
		changes.append(_change(
			"drop_table",
			f"table.{table_name}",
			before={"name": table_name, "fields": previous_tables[table_name].get("fields", {})},
			destructive=True,
			requires_approval=True,
			alternatives=["Rename the table with a migration hint if this is not an intentional drop."],
		))

	for table_name in sorted(previous_names & current_names):
		changes.extend(_field_changes(table_name, previous_tables[table_name], current_tables[table_name], rename_hints))
		changes.extend(_table_directive_changes(table_name, previous_tables[table_name], current_tables[table_name]))

	return changes


def _table_rename_candidates(
	previous_tables: dict[str, Any],
	current_tables: dict[str, Any],
	dropped_names: set[str],
	added_names: set[str],
	rename_hints: dict[str, str],
) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	for old_name in sorted(dropped_names):
		hinted_new_name = rename_hints.get(f"table.{old_name}") or rename_hints.get(old_name)
		candidates = [hinted_new_name] if hinted_new_name else sorted(added_names)
		for new_name in candidates:
			if not new_name or new_name not in added_names:
				continue
			if hinted_new_name or _field_signature(previous_tables[old_name]) == _field_signature(current_tables[new_name]):
				changes.append(_change(
					"rename_table_candidate",
					f"table.{old_name}",
					before={"name": old_name},
					after={"name": new_name},
					requires_approval=True,
					reason="Dropped and added tables have matching field signatures." if not hinted_new_name else "Matched explicit rename hint.",
				))
				break
	return changes


def _field_changes(
	table_name: str,
	previous_table: dict[str, Any],
	current_table: dict[str, Any],
	rename_hints: dict[str, str],
) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	previous_fields = previous_table.get("fields", {})
	current_fields = current_table.get("fields", {})
	previous_names = set(previous_fields)
	current_names = set(current_fields)
	added_names = current_names - previous_names
	dropped_names = previous_names - current_names
	field_renames = _field_rename_candidates(table_name, previous_fields, current_fields, dropped_names, added_names, rename_hints)
	renamed_from = {change["before"]["name"] for change in field_renames}
	renamed_to = {change["after"]["name"] for change in field_renames}
	changes.extend(field_renames)

	for field_name in sorted(added_names - renamed_to):
		field = current_fields[field_name]
		requires_backfill = bool(field.get("required")) and field.get("default") is None
		changes.append(_change(
			"add_field",
			f"field.{table_name}.{field_name}",
			after=field,
			requires_approval=requires_backfill,
			requires_backfill=requires_backfill,
			reason="Required field has no default and existing rows need a backfill." if requires_backfill else "",
		))

	for field_name in sorted(dropped_names - renamed_from):
		changes.append(_change(
			"drop_field",
			f"field.{table_name}.{field_name}",
			before=previous_fields[field_name],
			destructive=True,
			requires_approval=True,
			alternatives=["Rename the field with a migration hint if this is not an intentional drop."],
		))

	for field_name in sorted(previous_names & current_names):
		previous_field = previous_fields[field_name]
		current_field = current_fields[field_name]
		if previous_field.get("type") != current_field.get("type"):
			changes.append(_change(
				"type_change",
				f"field.{table_name}.{field_name}",
				before={"type": previous_field.get("type")},
				after={"type": current_field.get("type")},
				destructive=True,
				requires_approval=True,
				alternatives=["Add a new field and backfill before dropping or converting the old field."],
			))
		if previous_field.get("required") != current_field.get("required"):
			now_required = bool(current_field.get("required"))
			changes.append(_change(
				"nullability_change",
				f"field.{table_name}.{field_name}",
				before={"required": previous_field.get("required")},
				after={"required": current_field.get("required")},
				destructive=now_required,
				requires_approval=now_required,
				requires_backfill=now_required and current_field.get("default") is None,
			))
		if previous_field.get("default") != current_field.get("default"):
			changes.append(_change(
				"default_change",
				f"field.{table_name}.{field_name}",
				before={"default": previous_field.get("default")},
				after={"default": current_field.get("default")},
			))
		if previous_field.get("relationship") != current_field.get("relationship"):
			changes.append(_change(
				"relationship_change",
				f"field.{table_name}.{field_name}",
				before={"relationship": previous_field.get("relationship")},
				after={"relationship": current_field.get("relationship")},
				requires_approval=True,
			))
		if previous_field.get("calculated") != current_field.get("calculated"):
			changes.append(_change(
				"calculated_field_change",
				f"field.{table_name}.{field_name}",
				before={"calculated": previous_field.get("calculated")},
				after={"calculated": current_field.get("calculated")},
				requires_approval=True,
			))
		changes.extend(_field_directive_changes(table_name, field_name, previous_field, current_field))

	return changes


def _field_rename_candidates(
	table_name: str,
	previous_fields: dict[str, Any],
	current_fields: dict[str, Any],
	dropped_names: set[str],
	added_names: set[str],
	rename_hints: dict[str, str],
) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	for old_name in sorted(dropped_names):
		hinted_new_name = (
			rename_hints.get(f"field.{table_name}.{old_name}")
			or rename_hints.get(f"{table_name}.{old_name}")
			or rename_hints.get(old_name)
		)
		candidates = [hinted_new_name] if hinted_new_name else sorted(added_names)
		for new_name in candidates:
			if not new_name or new_name not in added_names:
				continue
			if hinted_new_name or _field_core(previous_fields[old_name]) == _field_core(current_fields[new_name]):
				changes.append(_change(
					"rename_field_candidate",
					f"field.{table_name}.{old_name}",
					before={"name": old_name},
					after={"name": new_name},
					requires_approval=True,
					reason="Dropped and added fields have matching type/nullability." if not hinted_new_name else "Matched explicit rename hint.",
				))
				break
	return changes


def _field_core(field: dict[str, Any]) -> dict[str, Any]:
	return {
		"type": field.get("type"),
		"required": field.get("required"),
		"relationship": field.get("relationship"),
	}


def _field_signature(table: dict[str, Any]) -> list[tuple[str, str, bool]]:
	return sorted(
		(field_name, str(field.get("type")), bool(field.get("required")))
		for field_name, field in table.get("fields", {}).items()
	)


def _table_directive_changes(table_name: str, previous_table: dict[str, Any], current_table: dict[str, Any]) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	if previous_table.get("indexes") != current_table.get("indexes"):
		changes.append(_change(
			"index_change",
			f"table.{table_name}.indexes",
			before=previous_table.get("indexes", []),
			after=current_table.get("indexes", []),
		))
	for key in sorted((set(previous_table) | set(current_table)) - {"name", "schema", "database", "fields", "lookup_paths", "indexes"}):
		if previous_table.get(key) != current_table.get(key):
			changes.append(_change(
				"table_directive_change",
				f"table.{table_name}.{key}",
				before=previous_table.get(key),
				after=current_table.get(key),
				requires_approval=True,
			))
	return changes


def _field_directive_changes(
	table_name: str,
	field_name: str,
	previous_field: dict[str, Any],
	current_field: dict[str, Any],
) -> list[dict[str, Any]]:
	changes: list[dict[str, Any]] = []
	for key, kind in (("unique", "unique_change"), ("constraints", "constraint_change"), ("check", "check_change"), ("primary_key", "primary_key_change")):
		if previous_field.get(key) != current_field.get(key):
			changes.append(_change(
				kind,
				f"field.{table_name}.{field_name}.{key}",
				before=previous_field.get(key),
				after=current_field.get(key),
				requires_approval=key in {"constraints", "check", "primary_key"},
			))
	for key in sorted((set(previous_field) | set(current_field)) - {"type", "required", "relationship", "default", "calculated", "unique", "constraints", "check", "primary_key"}):
		if previous_field.get(key) != current_field.get(key):
			changes.append(_change(
				"field_directive_change",
				f"field.{table_name}.{field_name}.{key}",
				before=previous_field.get(key),
				after=current_field.get(key),
				requires_approval=True,
			))
	return changes


def _capability_ownership_changes(previous_model: dict[str, Any], current_model: dict[str, Any]) -> list[dict[str, Any]]:
	previous_owners = _capability_table_owners(previous_model)
	current_owners = _capability_table_owners(current_model)
	changes: list[dict[str, Any]] = []
	for table_name in sorted(set(previous_owners) | set(current_owners)):
		before = previous_owners.get(table_name)
		after = current_owners.get(table_name)
		if before and after and before != after:
			changes.append(_change(
				"capability_ownership_transfer",
				f"table.{table_name}",
				before={"owner": before},
				after={"owner": after},
				requires_approval=True,
				reason="Capability-owned table moved between capability contracts.",
			))
	return changes


def _capability_table_owners(model: dict[str, Any]) -> dict[str, str]:
	tables = set(model.get("tables", {}))
	owners: dict[str, str] = {}
	for capability_name, capability in model.get("capabilities", {}).items():
		for table_name in _owned_tables(capability):
			matched = _match_table_name(table_name, tables)
			if matched:
				owners[matched] = capability_name
	return owners


def _owned_tables(capability: dict[str, Any]) -> set[str]:
	values: set[str] = set()
	for container in (
		capability,
		capability.get("configuration", {}),
		capability.get("master_data", {}),
		capability.get("contract", {}),
	):
		if isinstance(container, dict):
			for key in ("owned_tables", "tables", "entities"):
				values.update(str(value) for value in _list_value(container.get(key)))
	return values


def _match_table_name(value: str, tables: set[str]) -> str | None:
	normalized = _normalize_name(value)
	for table_name in tables:
		if _normalize_name(table_name) == normalized:
			return table_name
	return None


def _normalize_name(value: str) -> str:
	return "".join(character.lower() for character in value if character.isalnum())


def _list_value(value: Any) -> list[Any]:
	if value is None:
		return []
	if isinstance(value, list):
		return value
	if isinstance(value, tuple):
		return list(value)
	return [value]


def _change(
	kind: str,
	symbol: str,
	before: Any = None,
	after: Any = None,
	destructive: bool = False,
	requires_approval: bool = False,
	requires_backfill: bool = False,
	reason: str = "",
	alternatives: list[str] | None = None,
) -> dict[str, Any]:
	return {
		"kind": kind,
		"symbol": symbol,
		"before": before,
		"after": after,
		"destructive": destructive,
		"requires_approval": requires_approval,
		"requires_backfill": requires_backfill,
		"reason": reason,
		"alternatives": alternatives or [],
	}


def _diagnostics_for_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
	diagnostics: list[dict[str, Any]] = []
	for change in changes:
		if change["kind"] in {"drop_table", "drop_field"}:
			diagnostics.append(_diagnostic(
				"APG1101",
				"Migration plan contains destructive drop",
				"warning",
				f"{change['kind']} affects {change['symbol']} and requires explicit approval.",
				fixes=[{"id": "confirm_or_hint_rename", "title": "Approve the drop or provide a rename hint"}],
			))
		elif change["kind"] in {"rename_table_candidate", "rename_field_candidate"}:
			diagnostics.append(_diagnostic(
				"APG1102",
				"Migration planner found a rename candidate",
				"info",
				f"{change['symbol']} may have been renamed and should be confirmed before execution.",
			))
		elif change.get("requires_backfill"):
			diagnostics.append(_diagnostic(
				"APG1103",
				"Migration requires data backfill",
				"warning",
				f"{change['symbol']} requires a backfill before production execution.",
			))
		elif change["kind"] == "capability_ownership_transfer":
			diagnostics.append(_diagnostic(
				"APG1104",
				"Capability ownership transfer requires review",
				"warning",
				f"{change['symbol']} moved from {change['before'].get('owner')} to {change['after'].get('owner')}.",
			))
		elif change["kind"] == "type_change":
			diagnostics.append(_diagnostic(
				"APG1106",
				"Field type change may require data conversion",
				"warning",
				f"{change['symbol']} changes type and requires explicit migration approval.",
			))
	return diagnostics


def _diagnostic(
	code: str,
	title: str,
	severity: str,
	message: str,
	fixes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
	return {
		"code": code,
		"title": title,
		"severity": severity,
		"message": message,
		"file": "",
		"range": {
			"start": {"line": 0, "character": 0},
			"end": {"line": 0, "character": 1},
		},
		"related_locations": [],
		"fixes": fixes or [],
		"docs_url": "docs/tooling.md#migration-planner",
	}


def _generate_apg_patch(old_model: dict[str, Any], new_model: dict[str, Any]) -> str:
	"""Generate an APG DSL fragment describing the changes from old_model to new_model.

	The output is a reviewable .apg snippet the developer can paste into their source.
	Only additive changes (add_table, add_field, add_capability) are emitted as DSL;
	destructive changes are annotated as comments.
	"""
	lines: list[str] = ["# APG patch — generated by apg migrate-plan --patch", ""]
	plan = build_migration_plan(old_model, new_model)
	changes = plan.get("changes", [])

	old_tables = old_model.get("tables", {})
	new_tables = new_model.get("tables", {})
	old_caps = old_model.get("capabilities", {})
	new_caps = new_model.get("capabilities", {})

	emitted_tables: set[str] = set()
	emitted_fields: set[str] = set()
	emitted_caps: set[str] = set()

	for change in changes:
		kind = change["kind"]
		symbol = change["symbol"]

		if kind == "add_table":
			# symbol is "table.<TableName>"
			table_name = symbol.split(".", 1)[1] if "." in symbol else symbol
			if table_name in emitted_tables:
				continue
			emitted_tables.add(table_name)
			table_def = new_tables.get(table_name, {})
			fields = table_def.get("fields", {})
			field_lines = [f"\t{fname}: {fdata.get('type', 'str')};" for fname, fdata in sorted(fields.items())]
			lines.append(f"table {table_name} {{")
			lines.extend(field_lines if field_lines else ["\t# (no fields)"])
			lines.append("}")
			lines.append("")

		elif kind == "add_field":
			# symbol is "field.<TableName>.<field_name>"
			parts = symbol.split(".", 2)
			if len(parts) == 3:
				_, table_name, field_name = parts
			else:
				continue
			key = f"{table_name}.{field_name}"
			if key in emitted_fields:
				continue
			emitted_fields.add(key)
			field_def = new_tables.get(table_name, {}).get("fields", {}).get(field_name, {})
			ftype = field_def.get("type", "str")
			required = " [required]" if field_def.get("required") else ""
			default = f' = {json.dumps(field_def["default"])}' if field_def.get("default") is not None else ""
			lines.append(f"# Insert into table {table_name}:")
			lines.append(f"\t{field_name}: {ftype}{required}{default};")
			lines.append("")

		elif kind == "drop_table":
			table_name = symbol.split(".", 1)[1] if "." in symbol else symbol
			lines.append(f"# DROP TABLE {table_name}  — destructive, requires approval")
			lines.append("")

		elif kind == "drop_field":
			lines.append(f"# DROP FIELD {symbol}  — destructive, requires approval")
			lines.append("")

		elif kind == "rename_table_candidate":
			before = change.get("before", {}).get("name", "")
			after = change.get("after", {}).get("name", "")
			lines.append(f"# RENAME TABLE {before} -> {after}  — confirm rename hint")
			lines.append("")

		elif kind == "rename_field_candidate":
			before = change.get("before", {}).get("name", "")
			after = change.get("after", {}).get("name", "")
			parts = symbol.split(".", 2)
			table_name = parts[1] if len(parts) >= 2 else "?"
			lines.append(f"# RENAME FIELD {table_name}.{before} -> {after}  — confirm rename hint")
			lines.append("")

	# Added capabilities
	for cap_name in sorted(set(new_caps) - set(old_caps)):
		if cap_name in emitted_caps:
			continue
		emitted_caps.add(cap_name)
		cap_def = new_caps[cap_name]
		provides = cap_def.get("provides") or [cap_name.lower()]
		if isinstance(provides, list):
			provides_str = ", ".join(str(p) for p in provides)
		else:
			provides_str = str(provides)
		lines.append(f"capability {cap_name} {{")
		lines.append("  contract: {")
		lines.append(f"    id: {cap_name.lower()},")
		lines.append(f"    provides: [{provides_str}]")
		lines.append("  };")
		lines.append("}")
		lines.append("")

	if len(lines) == 2:
		lines.append("# No additive changes detected.")
		lines.append("")

	return "\n".join(lines)


def _summary(changes: list[dict[str, Any]]) -> dict[str, int]:
	summary: dict[str, int] = {"total": len(changes)}
	for change in changes:
		kind = change["kind"]
		summary[kind] = summary.get(kind, 0) + 1
	return summary
