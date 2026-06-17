#!/usr/bin/env python3
"""Fix service files that have WriteThruDict/WriteThruList but are missing
`_store = get_store(db_url)` in __init__. Also adds `db_url` param if absent.

Run: python3 tools/fix_missing_store.py [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Matches a WriteThruDict or WriteThruList assignment line inside __init__
_WRITETHRU_LINE = re.compile(
	r'^(\t+)(self\.\w+\s*=\s*(?:WriteThruDict|WriteThruList)\([^\n]+)',
	re.MULTILINE,
)

# Matches __init__ signature, single-line variant ending with ):
_INIT_SINGLE = re.compile(
	r'(def __init__\([^)]+)\)',
	re.MULTILINE | re.DOTALL,
)


def _add_db_url_param(sig_text: str) -> str:
	"""Append db_url param to __init__ if not already present."""
	if "db_url" in sig_text:
		return sig_text
	# Find closing ) of the signature
	idx = sig_text.rfind(")")
	before = sig_text[:idx].rstrip()
	# Figure out indent for last param
	return before + ", db_url: str | None = None" + sig_text[idx:]


def fix_file(path: Path, dry_run: bool = False) -> bool:
	text = path.read_text(encoding="utf-8")

	if "_store = get_store" in text:
		return False  # already fixed

	if "WriteThruDict" not in text and "WriteThruList" not in text:
		return False

	original = text

	# 1. Add db_url to __init__ if missing
	if "db_url" not in text:
		m = _INIT_SINGLE.search(text)
		if m:
			new_sig = _add_db_url_param(m.group(0))
			text = text[:m.start()] + new_sig + text[m.end():]

	# 2. Insert `_store = get_store(db_url)` right before the first WriteThru call
	m = _WRITETHRU_LINE.search(text)
	if not m:
		return False

	indent = m.group(1)
	inject = f"{indent}_store = get_store(db_url)\n"
	insert_pos = m.start()
	text = text[:insert_pos] + inject + text[insert_pos:]

	if text == original:
		return False

	# Validate syntax
	try:
		ast.parse(text)
	except SyntaxError as exc:
		print(f"  SYNTAX ERROR {path}: {exc} — skipping")
		return False

	if dry_run:
		print(f"  WOULD FIX: {path}")
	else:
		path.write_text(text, encoding="utf-8")
		print(f"  FIXED: {path}")

	return True


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dry-run", action="store_true")
	args = parser.parse_args()

	targets = [
		p for p in (ROOT / "capabilities").rglob("service.py")
		if "build/" not in str(p) and "__pycache__" not in str(p)
	]

	fixed = skipped = 0
	for path in sorted(targets):
		if fix_file(path, dry_run=args.dry_run):
			fixed += 1
		else:
			skipped += 1

	action = "Would fix" if args.dry_run else "Fixed"
	print(f"\n{action}: {fixed}  |  Skipped: {skipped}")


if __name__ == "__main__":
	main()
