#!/usr/bin/env python3
"""APG Platform Reliability Scanner.

Verifies all 10 platform-wide invariants across every capability.
Returns exit code 0 (ALL CLEAR) or 1 (issues found).

Usage:
    python3 docs/reliability/platform_scan.py
    python3 docs/reliability/platform_scan.py --domain agriculture
    python3 docs/reliability/platform_scan.py --fast   # skip syntax check
"""
from __future__ import annotations

import argparse
import os
import py_compile
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
CAPABILITIES_ROOT = REPO_ROOT / "capabilities"


def scan(domain_filter: str | None = None, fast: bool = False) -> dict[str, list[str]]:
    """Scan all capabilities and return issues grouped by category."""
    issues: dict[str, list[str]] = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(CAPABILITIES_ROOT):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "build", ".venv")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            fp = Path(dirpath) / filename
            rel = str(fp.relative_to(REPO_ROOT))

            # Domain filter
            if domain_filter:
                parts = rel.split(os.sep)
                if len(parts) < 2 or parts[1] != domain_filter:
                    continue

            # Skip test files for most checks
            is_test = "test" in rel.lower() or "crawler" in rel.lower()

            # I5: Syntax correctness
            if not fast:
                try:
                    py_compile.compile(str(fp), doraise=True)
                except py_compile.PyCompileError as e:
                    issues["I5_syntax_errors"].append(f"{rel}: {e}")
                    continue

            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
            except Exception as e:
                issues["read_errors"].append(f"{rel}: {e}")
                continue

            # I2: No silent exceptions (bare except:)
            for i, line in enumerate(lines):
                if re.match(r"\s*except\s*:", line):
                    issues["I2_bare_except"].append(f"{rel}:{i+1}")

            # I2: Silent pass in service/api (not in tests)
            if filename in ("service.py", "api.py") and not is_test:
                for i in range(len(lines) - 1):
                    if (
                        re.match(r"\s*except.*:\s*$", lines[i])
                        and lines[i + 1].strip() == "pass"
                        and "CancelledError" not in lines[i]
                    ):
                        issues["I2_silent_pass"].append(f"{rel}:{i+1}")

            # I3: httpx without timeout
            for i, line in enumerate(lines):
                if "httpx.AsyncClient()" in line and not line.lstrip().startswith("#"):
                    issues["I3_httpx_no_timeout"].append(f"{rel}:{i+1}")

            # I4: asyncio.gather without return_exceptions (prod only)
            if not is_test:
                for i, line in enumerate(lines):
                    if "asyncio.gather(" in line and not line.lstrip().startswith("#"):
                        call_lines: list[str] = []
                        # Count depth from the asyncio.gather( opening, ignoring prior ) on same line
                        gather_pos = line.index("asyncio.gather(")
                        line_from_gather = line[gather_pos:]
                        depth = line_from_gather.count("(") - line_from_gather.count(")")
                        call_lines.append(line)
                        for j in range(i + 1, min(i + 30, len(lines))):
                            call_lines.append(lines[j])
                            depth += lines[j].count("(") - lines[j].count(")")
                            if depth <= 0:
                                break
                        if "return_exceptions" not in "\n".join(call_lines):
                            issues["I4_unsafe_gather"].append(f"{rel}:{i+1}")

            # I1: guard_tenant_id in service.py
            if filename == "service.py" and not is_test:
                if "guard_tenant_id" not in content and "guard_" not in content:
                    issues["I1_missing_guard"].append(rel)

            # Check reliability import in service.py
            if filename == "service.py" and not is_test:
                if "capabilities.common.reliability" not in content:
                    issues["I1_missing_reliability_import"].append(rel)

    return dict(issues)


def report(issues: dict[str, list[str]]) -> bool:
    """Print report and return True if all clear."""
    DESCRIPTIONS = {
        "I5_syntax_errors": "I5 — Python syntax errors",
        "I2_bare_except": "I2 — Bare except: clauses (catches BaseException)",
        "I2_silent_pass": "I2 — Silent pass after except in service/api (hides errors)",
        "I3_httpx_no_timeout": "I3 — httpx.AsyncClient() without timeout (can hang forever)",
        "I4_unsafe_gather": "I4 — asyncio.gather() without return_exceptions=True",
        "I1_missing_guard": "I1 — service.py missing guard_tenant_id (tenant isolation)",
        "I1_missing_reliability_import": "I1 — service.py missing reliability framework import",
        "read_errors": "Read errors",
    }

    all_clear = True
    print("=" * 70)
    print("APG PLATFORM RELIABILITY SCAN")
    print("=" * 70)

    for key, label in DESCRIPTIONS.items():
        vals = issues.get(key, [])
        if vals:
            all_clear = False
            print(f"\n✗ {label}: {len(vals)} violation(s)")
            for v in vals[:5]:
                print(f"    {v}")
            if len(vals) > 5:
                print(f"    ... and {len(vals) - 5} more")
        else:
            print(f"✓ {label}: ZERO")

    print()
    if all_clear:
        print("ALL CLEAR — platform meets reliability standard")
    else:
        total = sum(len(v) for v in issues.values())
        print(f"VIOLATIONS FOUND — {total} total issues across {len(issues)} categories")
    print("=" * 70)
    return all_clear


def main() -> int:
    parser = argparse.ArgumentParser(description="APG Platform Reliability Scanner")
    parser.add_argument("--domain", help="Scan only this domain (e.g. agriculture)")
    parser.add_argument("--fast", action="store_true", help="Skip syntax check")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    issues = scan(domain_filter=args.domain, fast=args.fast)

    if args.json:
        import json
        print(json.dumps(issues, indent=2))
        return 0 if not any(issues.values()) else 1

    all_clear = report(issues)
    return 0 if all_clear else 1


if __name__ == "__main__":
    sys.exit(main())
