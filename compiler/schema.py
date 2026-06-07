"""APG table-to-DDL schema generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .semantic_model import build_semantic_model

SCHEMA_REPORT_FORMAT = "apg.schema-report.v1"

_APG_TO_SQL: dict[str, str] = {
	"str": "TEXT",
	"str?": "TEXT",
	"int": "INTEGER",
	"float": "DOUBLE PRECISION",
	"decimal": "NUMERIC(18,4)",
	"bool": "BOOLEAN",
	"date": "DATE",
	"datetime": "TIMESTAMP WITH TIME ZONE",
	"time": "TIME",
	"bytes": "BYTEA",
	"List[str]": "JSONB",
	"Dict[str,str]": "JSONB",
	"Dict[str,Any]": "JSONB",
	"Any": "JSONB",
}


def _sql_type(apg_type: str) -> str:
	clean = apg_type.rstrip("?").strip()
	return _APG_TO_SQL.get(clean, "TEXT")


def _is_nullable(field: dict) -> bool:
	t = field.get("type", "str")
	return not field.get("required", True) or t.endswith("?") or "None" in t


def generate_schema(source_file: Path, dialect: str = "postgresql") -> dict[str, Any]:
	model = build_semantic_model(source_file)
	tables = model.get("tables", {})
	statements = []
	for table_name, table in sorted(tables.items()):
		cols = []
		cols.append("    id TEXT NOT NULL DEFAULT gen_random_uuid()::TEXT")
		for fname, field in sorted(table.get("fields", {}).items()):
			sql_type = _sql_type(field.get("type", "str"))
			nullable = "NULL" if _is_nullable(field) else "NOT NULL"
			cols.append(f"    {fname} {sql_type} {nullable}")
		cols.append("    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()")
		cols.append(f"    CONSTRAINT {table_name.lower()}_pkey PRIMARY KEY (id)")
		stmt = (
			f"CREATE TABLE IF NOT EXISTS {table_name.lower()} (\n"
			+ ",\n".join(cols)
			+ "\n);"
		)
		statements.append(stmt)
	ddl = "\n\n".join(statements)
	return {
		"format": SCHEMA_REPORT_FORMAT,
		"ok": bool(tables),
		"source": str(source_file),
		"dialect": dialect,
		"table_count": len(tables),
		"ddl": ddl,
		"tables": list(tables.keys()),
		"errors": [] if tables else ["No tables found in source"],
	}
