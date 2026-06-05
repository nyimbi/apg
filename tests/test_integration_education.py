"""Education capability integration tests.

All tests are sync; async methods called via asyncio.run().
Uses in-memory store — zero config required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio


# ── helpers ───────────────────────────────────────────────────────────────────

def _lms():
	from capabilities.education.lms.service import LmsService
	return LmsService()


def _sch():
	from capabilities.education.sch_mgmt.service import SchoolManagementService
	return SchoolManagementService()


def _ttbl():
	from capabilities.education.ttbl.service import TimetablingService
	return TimetablingService()


# ── 1. LmsService instantiable ────────────────────────────────────────────────

def test_lms_instantiable():
	"""LmsService can be created with no required args."""
	svc = _lms()
	assert svc is not None
	assert hasattr(svc, "courses")
	assert hasattr(svc, "enrolments")


# ── 2. SchoolManagementService instantiable ───────────────────────────────────

def test_school_management_instantiable():
	"""SchoolManagementService can be created."""
	svc = _sch()
	assert svc is not None


# ── 3. TimetablingService instantiable ───────────────────────────────────────

def test_timetabling_instantiable():
	"""TimetablingService can be created."""
	svc = _ttbl()
	assert svc is not None


# ── 4. Education manifest — 3 capabilities ───────────────────────────────────

def test_education_manifest():
	"""Education domain contains exactly 3 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("education")
	assert len(caps) == 3, f"expected 3 education capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") for c in caps}
	assert any("lms" in str(cid) for cid in ids), f"education_lms not found in: {ids}"


# ── 5. Education composability — all requires are known APG codes ─────────────

def test_education_composability():
	"""All education capability REQUIRES lists contain only known APG codes."""
	known_codes = {
		"auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "mqeb",
		"nlpc", "moni", "comp", "keym", "stor", "mpen",
	}
	from capabilities.education.lms.capability_contract import REQUIRES as lms_req
	from capabilities.education.sch_mgmt.capability_contract import REQUIRES as sch_req
	from capabilities.education.ttbl.capability_contract import REQUIRES as ttbl_req

	all_requires = {
		"education_lms": lms_req,
		"education_sch_mgmt": sch_req,
		"education_ttbl": ttbl_req,
	}
	for cap_id, reqs in all_requires.items():
		assert len(reqs) > 0, f"{cap_id} REQUIRES must not be empty"
		for req in reqs:
			assert req in known_codes, f"{cap_id}: unknown requirement '{req}'"


# ── 6. LMS course analytics ───────────────────────────────────────────────────

def test_lms_course_analytics():
	"""lms_analytics returns a dict with tenant_id and completion_rate_pct keys."""
	svc = _lms()

	# Create a course so there is something to analyse
	asyncio.run(svc.create_course(
		tenant_id="test-tenant",
		title="Introduction to Python",
		code="CS-101",
		course_type="self_paced",
		owner_id="instructor-01",
		created_by="admin",
		description="Beginner Python programming",
	))

	result = asyncio.run(svc.lms_analytics(
		tenant_id="test-tenant",
		period="2026-Q2",
	))
	assert isinstance(result, dict)
	assert result["tenant_id"] == "test-tenant"
	assert "completion_rate_pct" in result
	assert result["courses"] >= 1
