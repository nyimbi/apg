#!/usr/bin/env python3
"""Migrate dict-only services to WriteThruDict / WriteThruList pattern.

Usage:
    python3 tools/migrate_services_to_store.py --dry-run   # preview changes
    python3 tools/migrate_services_to_store.py             # apply changes
    python3 tools/migrate_services_to_store.py --path capabilities/agriculture/coo/service.py
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── patterns ──────────────────────────────────────────────────────────────────

_DICT_DECL  = re.compile(r'^(\t+)(self\.(_\w+))\s*:\s*dict\[str,\s*dict\[str,\s*Any\]\]\s*=\s*\{\}', re.MULTILINE)
_LIST_DECL  = re.compile(r'^(\t+)(self\.(_\w+))\s*:\s*list\[dict\[str,\s*Any\]\]\s*=\s*\[\]', re.MULTILINE)
_INIT_SIG   = re.compile(r'(def __init__\(self,\s*tenant_id\s*:\s*str[^)]*)\)', re.MULTILINE)
_TENANT_SET = re.compile(r'^(\t+)(self\.tenant_id\s*=\s*tenant_id)', re.MULTILINE)

_DB_IMPORT = (
	"from capabilities.common.db import get_store\n"
	"from capabilities.common.db.write_thru import WriteThruDict, WriteThruList\n"
)

_INITIALIZE_TEMPLATE = '''
\tasync def initialize(self) -> None:
\t\t"""Restore persisted data from the database. Call once after __init__ in production."""
\t\tfor attr in {attrs}:
\t\t\tobj = getattr(self, attr, None)
\t\t\tif obj is not None and hasattr(obj, "reload"):
\t\t\t\tawait obj.reload()
'''


def _already_migrated(text: str) -> bool:
	return "WriteThruDict" in text or "WriteThruList" in text or "get_store" in text


def _has_dict_decls(text: str) -> bool:
	return bool(_DICT_DECL.search(text)) or bool(_LIST_DECL.search(text))


def _add_imports(text: str) -> str:
	"""Insert store imports after the last 'from __future__' or at top of imports."""
	# Don't double-add
	if "from capabilities.common.db import" in text:
		return text

	# Insert after 'from __future__ import annotations'
	future_match = re.search(r'^from __future__ import annotations\n', text, re.MULTILINE)
	if future_match:
		pos = future_match.end()
		return text[:pos] + "\n" + _DB_IMPORT + text[pos:]

	# Fallback: insert before first 'import' or 'from' line
	first_import = re.search(r'^(import |from )', text, re.MULTILINE)
	if first_import:
		pos = first_import.start()
		return text[:pos] + _DB_IMPORT + "\n" + text[pos:]

	return _DB_IMPORT + "\n" + text


def _fix_init_signature(text: str) -> str:
	"""Add db_url parameter to __init__ if not present."""
	def replacer(m: re.Match) -> str:
		sig = m.group(1)
		full = m.group(0)
		if "db_url" in full:
			return full  # already has it
		# Add db_url before closing paren
		return sig.rstrip() + ", db_url: str | None = None)"
	return _INIT_SIG.sub(replacer, text)


def _inject_store_creation(text: str) -> str:
	"""Insert `_store = get_store(db_url)` right after `self.tenant_id = tenant_id`."""
	if "_store = get_store" in text:
		return text

	def replacer(m: re.Match) -> str:
		indent = m.group(1)
		line   = m.group(2)
		return f"{indent}{line}\n{indent}_store = get_store(db_url)"

	return _TENANT_SET.sub(replacer, text, count=1)


def _replace_dict_decls(text: str) -> tuple[str, list[str]]:
	"""Replace dict[str, dict[str, Any]] = {} with WriteThruDict(...)."""
	attrs: list[str] = []

	def replacer(m: re.Match) -> str:
		indent = m.group(1)
		attr   = m.group(3)          # e.g. _coops
		col    = attr.lstrip("_")    # e.g. coops
		attrs.append(attr)
		return f"{indent}self.{attr} = WriteThruDict({col!r}, tenant_id, _store)"

	new_text = _DICT_DECL.sub(replacer, text)
	return new_text, attrs


def _replace_list_decls(text: str) -> tuple[str, list[str]]:
	"""Replace list[dict[str, Any]] = [] with WriteThruList(...)."""
	attrs: list[str] = []

	def replacer(m: re.Match) -> str:
		indent = m.group(1)
		attr   = m.group(3)
		col    = attr.lstrip("_")
		attrs.append(attr)
		return f"{indent}self.{attr} = WriteThruList({col!r}, tenant_id, _store)"

	new_text = _LIST_DECL.sub(replacer, text)
	return new_text, attrs


def _add_initialize_method(text: str, all_attrs: list[str]) -> str:
	"""Add async def initialize() method to the service class if not present."""
	if "async def initialize" in text:
		return text

	attrs_repr = repr(all_attrs)
	initialize_code = _INITIALIZE_TEMPLATE.format(attrs=attrs_repr)

	# Insert before the last line of the class (or at end of file)
	# Find the last method definition in the class
	last_method = None
	for m in re.finditer(r'^(\t)(async def |def )\w+', text, re.MULTILINE):
		last_method = m

	if last_method:
		# Find end of file and append there
		text = text.rstrip() + "\n" + initialize_code + "\n"
	else:
		text = text + "\n" + initialize_code + "\n"

	return text


def migrate_file(path: Path, dry_run: bool = False) -> bool:
	"""Migrate a single service file. Returns True if file was modified."""
	text = path.read_text(encoding="utf-8")

	if _already_migrated(text):
		return False

	if not _has_dict_decls(text):
		return False

	original = text

	# Step 1: add imports
	text = _add_imports(text)

	# Step 2: fix __init__ signature
	text = _fix_init_signature(text)

	# Step 3: inject store creation
	text = _inject_store_creation(text)

	# Step 4: replace dict/list declarations
	text, dict_attrs = _replace_dict_decls(text)
	text, list_attrs = _replace_list_decls(text)

	all_attrs = dict_attrs + list_attrs

	# Step 5: add initialize() method
	if all_attrs:
		text = _add_initialize_method(text, all_attrs)

	if text == original:
		return False

	# Validate syntax
	try:
		ast.parse(text)
	except SyntaxError as exc:
		print(f"  SYNTAX ERROR in {path}: {exc} — skipping")
		return False

	if dry_run:
		print(f"  WOULD MIGRATE: {path}")
		print(f"    dict attrs: {dict_attrs}")
		print(f"    list attrs: {list_attrs}")
	else:
		path.write_text(text, encoding="utf-8")
		print(f"  MIGRATED: {path} ({len(dict_attrs)}D {len(list_attrs)}L attrs)")

	return True


def find_target_services() -> list[Path]:
	"""Find all dict-only service files."""
	results = []
	for svc in (ROOT / "capabilities").rglob("service.py"):
		if "build/" in str(svc) or "__pycache__" in str(svc):
			continue
		text = svc.read_text(errors="ignore")
		if _already_migrated(text):
			continue
		if _has_dict_decls(text):
			results.append(svc)
	return sorted(results)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--path", help="Migrate a single file")
	args = parser.parse_args()

	if args.path:
		targets = [Path(args.path)]
	else:
		targets = find_target_services()

	print(f"Found {len(targets)} services to migrate")

	migrated = 0
	skipped  = 0
	for path in targets:
		if migrate_file(path, dry_run=args.dry_run):
			migrated += 1
		else:
			skipped += 1

	action = "Would migrate" if args.dry_run else "Migrated"
	print(f"\n{action}: {migrated}  |  Skipped (already done / no dicts): {skipped}")


if __name__ == "__main__":
	main()
