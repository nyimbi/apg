"""Semantic-model migration planning for APG applications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .semantic_model import build_semantic_model


MIGRATION_PLAN_FORMAT = "apg.migration-plan.v1"
SUPPORTED_MIGRATION_BACKENDS = ("postgresql", "mysql", "sqlite", "compatible")


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


def _summary(changes: list[dict[str, Any]]) -> dict[str, int]:
	summary: dict[str, int] = {"total": len(changes)}
	for change in changes:
		kind = change["kind"]
		summary[kind] = summary.get(kind, 0) + 1
	return summary
