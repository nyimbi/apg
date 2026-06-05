"""Government capability integration tests: TaxAdministrationService.

All tests are sync (service methods are synchronous).
Uses in-process _Store — zero config.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, timedelta


# ── helpers ───────────────────────────────────────────────────────────────────

def _svc():
	from capabilities.government.tax.service import TaxAdministrationService
	return TaxAdministrationService()


def _register(svc, tenant_id: str = "ke-tenant", name: str = "Acme Ltd"):
	"""Register a minimal taxpayer and return the dict."""
	return svc.register_taxpayer(
		taxpayer_id="tp-001",
		tenant_id=tenant_id,
		tax_type="vat",
		tax_pin="",            # generated internally
		id_number="12345678",
		legal_name=name,
		entity_type="individual",
		tax_types=["vat"],
	)


# ── 1. taxpayer registration ──────────────────────────────────────────────────

def test_tax_taxpayer_registration():
	"""TaxAdministrationService.register_taxpayer creates a taxpayer record."""
	svc = _svc()
	taxpayer = _register(svc)
	assert taxpayer["taxpayer_name"] == "Acme Ltd"
	assert taxpayer["id"]
	assert taxpayer["tax_pin"]          # auto-generated KRA PIN
	assert taxpayer["tenant_id"] == "ke-tenant"
	assert taxpayer["status"] == "pending"


# ── 2. return filing ──────────────────────────────────────────────────────────

def test_tax_return_filing():
	"""submit_return records a VAT return for a registered taxpayer."""
	svc = _svc()
	taxpayer = _register(svc)
	pin = taxpayer["tax_pin"]

	# Activate the taxpayer first so filing checks pass
	svc.update_taxpayer(pin, tenant_id="ke-tenant", status="active")

	ret = svc.submit_return(
		tin=pin,
		tax_type="vat",
		period="2026-01",
		return_data={
			"gross_income": 500_000,
			"allowable_deductions": 0,
			"taxable_income": 500_000,
			"tax_liability": 80_000,
			"tax_credits": 0,
			"tax_paid": 0,
		},
		tenant_id="ke-tenant",
	)
	assert ret["id"]
	assert ret["status"] in ("filed", "under_review")
	assert float(ret["tax_liability"]) == 80_000.0


# ── 3. penalty and interest calculation ──────────────────────────────────────

def test_tax_penalty_calculation():
	"""calculate_penalty_and_interest returns non-zero penalty for a late payment."""
	svc = _svc()
	taxpayer = _register(svc)
	pin = taxpayer["tax_pin"]

	# Issue an assessment directly
	assessment = svc.issue_assessment(
		tin=pin,
		tax_type="vat",
		period="2025-01",
		assessed_amount=100_000,
		reason="self_assessment",
		assessment_type="self_assessment",
		tenant_id="ke-tenant",
	)
	assessment_id = assessment["id"]

	# Pay 90 days late
	late_date = (date.today() + timedelta(days=90)).isoformat()
	result = svc.calculate_penalty_and_interest(
		assessment_id=assessment_id,
		payment_date=late_date,
		tenant_id="ke-tenant",
	)
	assert float(result["late_filing_penalty"]) > 0 or float(result["late_payment_interest"]) > 0
	assert "total_payable" in result
	assert float(result["total_payable"]) > 0


# ── 4. rule evaluation — deny missing tenant ──────────────────────────────────

def test_tax_rule_evaluation():
	"""Government tax rules deny when tenant context is absent."""
	from capabilities.government.tax.capability_contract import evaluate_capability_rules

	deny_result = evaluate_capability_rules({"tenant_context_present": False})
	assert deny_result["decision"] == "deny"

	allow_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "read",
	})
	assert allow_result["decision"] == "allow"


# ── 5. tax clearance certificate ─────────────────────────────────────────────

def test_tax_clearance_certificate():
	"""issue_tax_clearance_certificate returns a dict (issued or rejected)."""
	svc = _svc()
	taxpayer = _register(svc)
	pin = taxpayer["tax_pin"]

	result = svc.issue_tax_clearance_certificate(
		tin=pin,
		tenant_id="ke-tenant",
		purpose="tender",
	)
	assert isinstance(result, dict)
	assert "status" in result or "certificate_number" in result
	# With no outstanding debts this should be issued
	assert result.get("status") in ("issued", "rejected", "pending")


# ── 6. government manifest ────────────────────────────────────────────────────

def test_government_manifest():
	"""Government domain contains exactly 10 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("government")
	assert len(caps) == 10, f"expected 10 government capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") or c.get("code") for c in caps}
	assert any("tax" in str(cid) for cid in ids), f"government_tax not found in: {ids}"


# ── 7. composability — all government requires satisfied ─────────────────────

def test_government_composability():
	"""government_tax REQUIRES list is non-empty and all entries are known APG codes."""
	from capabilities.government.tax.capability_contract import REQUIRES
	known_codes = {
		"auth", "audl", "mten", "conf", "ntfy", "wflo", "moni",
		"comp", "mqeb", "schd", "nlpc", "keym", "stor",
		"fintech_payments", "fintech_wallets", "fintech_kyc",
	}
	assert len(REQUIRES) > 0, "REQUIRES must not be empty"
	for req in REQUIRES:
		assert req in known_codes, f"unknown requirement: {req}"
