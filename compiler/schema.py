"""APG table-to-DDL schema generator."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .semantic_model import build_semantic_model

SCHEMA_REPORT_FORMAT = "apg.schema-report.v1"

# Per-dialect type mappings
_DIALECT_TYPES: dict[str, dict[str, str]] = {
	"postgresql": {
		"str": "TEXT", "int": "INTEGER", "float": "DOUBLE PRECISION",
		"decimal": "NUMERIC(18,4)", "bool": "BOOLEAN", "date": "DATE",
		"datetime": "TIMESTAMP WITH TIME ZONE", "time": "TIME",
		"bytes": "BYTEA", "List[str]": "JSONB", "Dict[str,str]": "JSONB",
		"Dict[str,Any]": "JSONB", "Any": "JSONB", "vector": "VECTOR(1536)",
	},
	"mysql": {
		"str": "VARCHAR(255)", "int": "INT", "float": "DOUBLE",
		"decimal": "DECIMAL(18,4)", "bool": "TINYINT(1)", "date": "DATE",
		"datetime": "DATETIME", "time": "TIME",
		"bytes": "BLOB", "List[str]": "JSON", "Dict[str,str]": "JSON",
		"Dict[str,Any]": "JSON", "Any": "JSON", "vector": "JSON",
	},
	"sqlite": {
		"str": "TEXT", "int": "INTEGER", "float": "REAL",
		"decimal": "TEXT", "bool": "INTEGER", "date": "TEXT",
		"datetime": "TEXT", "time": "TEXT",
		"bytes": "BLOB", "List[str]": "TEXT", "Dict[str,str]": "TEXT",
		"Dict[str,Any]": "TEXT", "Any": "TEXT", "vector": "TEXT",
	},
}

# Per-dialect UUID default and quote character
_DIALECT_UUID_DEFAULT = {
	"postgresql": "gen_random_uuid()::TEXT",
	"mysql": "(UUID())",
	"sqlite": "lower(hex(randomblob(16)))",
}
_DIALECT_QUOTE = {
	"postgresql": '"',
	"mysql": "`",
	"sqlite": '"',
}
_DIALECT_TIMESTAMP_DEFAULT = {
	"postgresql": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
	"mysql": "DATETIME DEFAULT CURRENT_TIMESTAMP",
	"sqlite": "TEXT DEFAULT (datetime('now'))",
}

_SQL_SAFE_RE = re.compile(r'^[a-z_][a-z0-9_]*$')

SUPPORTED_DIALECTS = ("postgresql", "mysql", "sqlite")


def _sql_type(apg_type: str, dialect: str) -> tuple[str, bool]:
	"""Return (sql_type, was_fallback). was_fallback=True means the type was unknown."""
	clean = apg_type.rstrip("?").strip()
	# Handle vector(N) parameterized form
	m = re.match(r'vector\((\d+)\)', clean, re.IGNORECASE)
	if m and dialect == "postgresql":
		return (f"VECTOR({m.group(1)})", False)
	sql = _DIALECT_TYPES[dialect].get(clean)
	return (sql, False) if sql else ("TEXT", True)


def _is_nullable(field: dict) -> bool:
	t = field.get("type", "str")
	return not field.get("required", True) or t.endswith("?") or "None" in t


def generate_schema(source_file: Path, dialect: str = "postgresql") -> dict[str, Any]:
	"""Generate SQL DDL from APG table declarations.

	The DDL is returned in the report dict and optionally written to a file.
	It is intended for manual review and application — this module never
	executes SQL.

	Identifier safety: APG identifiers are constrained to ^[A-Za-z_][A-Za-z0-9_]*$
	by the parser. This function validates them again as defense-in-depth.
	"""
	if dialect not in SUPPORTED_DIALECTS:
		return {
			"format": SCHEMA_REPORT_FORMAT,
			"ok": False,
			"source": str(source_file),
			"dialect": dialect,
			"errors": [f"Unsupported dialect: {dialect!r}. Choose from {SUPPORTED_DIALECTS}"],
		}

	model = build_semantic_model(source_file)
	tables = model.get("tables", {})
	q = _DIALECT_QUOTE[dialect]
	uuid_default = _DIALECT_UUID_DEFAULT[dialect]
	ts_col = _DIALECT_TIMESTAMP_DEFAULT[dialect]
	statements = []
	warnings: list[str] = []
	needs_vector_ext = False

	for table_name, table in sorted(tables.items()):
		safe_name = table_name.lower()
		if not _SQL_SAFE_RE.match(safe_name):
			warnings.append(f"Skipping table with unsafe name: {table_name!r}")
			continue

		cols = []
		cols.append(f"    {q}id{q} TEXT NOT NULL DEFAULT {uuid_default}")
		for fname, field in sorted(table.get("fields", {}).items()):
			safe_fname = fname.lower()
			if not _SQL_SAFE_RE.match(safe_fname):
				warnings.append(f"Skipping field with unsafe name: {table_name}.{fname}")
				continue
			sql_type, fallback = _sql_type(field.get("type", "str"), dialect)
			if fallback:
				warnings.append(
					f"{table_name}.{fname}: unknown APG type {field.get('type')!r}, mapped to TEXT"
				)
			if "VECTOR" in sql_type:
				needs_vector_ext = True
			nullable = "NULL" if _is_nullable(field) else "NOT NULL"
			cols.append(f"    {q}{safe_fname}{q} {sql_type} {nullable}")
		cols.append(f"    {q}created_at{q} {ts_col}")
		cols.append(f"    CONSTRAINT {safe_name}_pkey PRIMARY KEY ({q}id{q})")
		stmt = (
			f"CREATE TABLE IF NOT EXISTS {q}{safe_name}{q} (\n"
			+ ",\n".join(cols)
			+ "\n);"
		)
		statements.append(stmt)

	prefix = ""
	if needs_vector_ext and dialect == "postgresql":
		prefix = "CREATE EXTENSION IF NOT EXISTS vector;\n\n"

	ddl = prefix + "\n\n".join(statements)
	return {
		"format": SCHEMA_REPORT_FORMAT,
		"ok": bool(tables),
		"source": str(source_file),
		"dialect": dialect,
		"table_count": len(tables),
		"ddl": ddl,
		"tables": list(tables.keys()),
		"warnings": warnings,
		"errors": [] if tables else ["No tables found in source"],
	}
