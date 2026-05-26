"""SCM context helper fallback hygiene checks."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCM_CONTEXT_PATHS = (
	REPO_ROOT / "capabilities" / "scm" / "src" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "dpl" / "demand_planning" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "ctm" / "contract_management" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "blt" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "rep" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "req" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "edm" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "inv" / "stock_tracking_control" / "context.py",
	REPO_ROOT / "capabilities" / "scm" / "pom" / "context.py",
)


def test_scm_context_helpers_do_not_fallback_to_literal_default_tenant():
	for path in SCM_CONTEXT_PATHS:
		source = path.read_text(encoding="utf-8")
		assert 'os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")' not in source
		assert '"default_tenant"' not in source
		assert 'os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))' in source
