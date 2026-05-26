"""Regression checks for stale demo identity literals in capability code."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
	REPO_ROOT / "capabilities" / "ckm" / "wfa" / "api_documentation.py",
	REPO_ROOT / "capabilities" / "common" / "auth" / "session_manager.py",
	REPO_ROOT / "capabilities" / "common" / "audl" / "world_class_improvements.py",
	REPO_ROOT / "capabilities" / "fin" / "cbm" / "cash_management" / "revolutionary_ux_engine.py",
)
FORBIDDEN_LITERALS = ("user_123", "tenant_123", "tenant_456", "default_tenant")


def test_stale_demo_identity_literals_are_not_used_in_cleaned_capability_surfaces():
	violations: list[str] = []

	for path in TARGETS:
		source = path.read_text(encoding="utf-8")
		for literal in FORBIDDEN_LITERALS:
			if literal in source:
				violations.append(f"{path.relative_to(REPO_ROOT)}: {literal}")

	assert violations == []
