"""Cross-capability context fallback hygiene checks."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATHS = (
	REPO_ROOT / "capabilities" / "pde" / "pim" / "context.py",
	REPO_ROOT / "capabilities" / "hcm" / "tat" / "time_attendance" / "context.py",
	REPO_ROOT / "capabilities" / "ecd" / "esg" / "context.py",
	REPO_ROOT / "capabilities" / "bia" / "tsa" / "context.py",
	REPO_ROOT / "capabilities" / "fin" / "glr" / "general_ledger" / "context.py",
	REPO_ROOT / "capabilities" / "fin" / "rpt" / "context.py",
	REPO_ROOT / "capabilities" / "fin" / "fed" / "context.py",
	REPO_ROOT / "capabilities" / "fintech" / "gateway" / "context.py",
	REPO_ROOT / "capabilities" / "hcm" / "chr" / "employee_data_management" / "context.py",
	REPO_ROOT / "capabilities" / "fin" / "auc" / "context.py",
	REPO_ROOT / "capabilities" / "common" / "geos" / "context.py",
	REPO_ROOT / "capabilities" / "fin" / "apy" / "accounts_payable" / "context.py",
	REPO_ROOT / "capabilities" / "composition" / "gateway" / "context.py",
	REPO_ROOT / "capabilities" / "mfg" / "mro" / "context.py",
	REPO_ROOT / "capabilities" / "common" / "cvsn" / "context.py",
)
EXPECTED_FALLBACK = 'os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))'


def test_context_helpers_do_not_fallback_to_literal_default_tenant():
	for path in CONTEXT_PATHS:
		source = path.read_text(encoding="utf-8")
		assert 'os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")' not in source
		assert '"default_tenant"' not in source
		assert EXPECTED_FALLBACK in source
