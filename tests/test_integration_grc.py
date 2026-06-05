"""Governance, Risk and Compliance capability integration tests.

All tests are sync; async methods called via asyncio.run().
Uses InMemoryStore — zero config required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio


# ── helpers ───────────────────────────────────────────────────────────────────

def _rsa():
	from capabilities.grc.rsa.service import RiskAssessmentService
	from capabilities.grc.rsa.database.store import InMemoryStore
	return RiskAssessmentService(store=InMemoryStore(), tenant_id="test-tenant")


def _pol():
	from capabilities.grc.pol.service import PolicyManagementService
	return PolicyManagementService()


def _aud():
	from capabilities.grc.aud.service import AuditManagementService
	return AuditManagementService()


# ── 1. RiskAssessmentService instantiable ────────────────────────────────────

def test_risk_assessment_instantiable():
	"""RiskAssessmentService can be created with in-memory store."""
	svc = _rsa()
	assert svc is not None
	assert svc._tenant_id == "test-tenant"


# ── 2. risk_register_entry creates a risk record ─────────────────────────────

def test_risk_register_entry():
	"""risk_register_entry returns a dict with id and status='identified'."""
	svc = _rsa()
	result = asyncio.run(svc.risk_register_entry(
		entity_id="ENT-001",
		risk_name="Cyber Attack",
		category="technology",
		description="Ransomware attack on core systems",
		owner_id="ciso@datacraft.co.ke",
	))
	assert isinstance(result, dict)
	assert "id" in result
	assert result["status"] == "identified"
	assert result["risk_name"] == "Cyber Attack"
	assert result["entity_id"] == "ENT-001"


# ── 3. risk_heat_map returns dict ─────────────────────────────────────────────

def test_risk_heat_map():
	"""risk_heat_map returns a dict with entity_id and grid keys."""
	svc = _rsa()
	# Register and assess a risk first so the heat map has data
	risk = asyncio.run(svc.risk_register_entry(
		entity_id="ENT-002",
		risk_name="Data Breach",
		category="information_security",
		description="Unauthorised data exfiltration",
		owner_id="ciso@datacraft.co.ke",
	))
	asyncio.run(svc.risk_assessment(
		risk_id=risk["id"],
		likelihood_1_5=3,
		impact_1_5=4,
		velocity="high",
		assessor_id="assessor@datacraft.co.ke",
	))
	heat_map = asyncio.run(svc.risk_heat_map(
		entity_id="ENT-002",
		as_of_date="2026-06-05",
	))
	assert isinstance(heat_map, dict)
	assert heat_map["entity_id"] == "ENT-002"
	assert "grid" in heat_map
	assert heat_map["total_risks"] == 1


# ── 4. PolicyManagementService instantiable ───────────────────────────────────

def test_policy_management_instantiable():
	"""PolicyManagementService can be created."""
	svc = _pol()
	assert svc is not None


# ── 5. AuditManagementService instantiable ───────────────────────────────────

def test_audit_management_instantiable():
	"""AuditManagementService can be created."""
	svc = _aud()
	assert svc is not None


# ── 6. GRC manifest — 6 capabilities ─────────────────────────────────────────

def test_grc_manifest():
	"""GRC domain contains exactly 6 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("grc")
	assert len(caps) == 6, f"expected 6 grc capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") for c in caps}
	assert any("rsa" in str(cid) for cid in ids), f"grc_rsa not found in: {ids}"


# ── 7. GRC rule evaluation — allow on valid read context ─────────────────────

def test_grc_rule_evaluation():
	"""evaluate_rules on grc_rsa with valid context returns allow."""
	from capabilities.grc.rsa.capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "read",
	})
	assert isinstance(result, dict)
	assert "decision" in result
	assert result["decision"] == "allow"
