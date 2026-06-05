"""Project Portfolio Management capability integration tests.

All tests are sync; async/sync service methods called directly or via asyncio.run().
Uses in-memory store — zero config required.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio


# ── helpers ───────────────────────────────────────────────────────────────────

def _pps():
	from capabilities.ppm.pps.service import ProjectPlanningService
	return ProjectPlanningService(tenant_id="test-tenant", actor_id="test-actor")


def _res():
	from capabilities.ppm.res.service import ResourceManagementService
	return ResourceManagementService(tenant_id="test-tenant", actor_id="test-actor")


def _tex():
	from capabilities.ppm.tex.service import TimeExpenseService
	return TimeExpenseService(tenant_id="test-tenant", actor_id="test-actor")


# ── 1. ProjectPlanningService instantiable ────────────────────────────────────

def test_project_planning_instantiable():
	"""ProjectPlanningService can be created with no required args."""
	svc = _pps()
	assert svc is not None
	assert svc.tenant_id == "test-tenant"


# ── 2. Project analytics instantiable ────────────────────────────────────────

def test_project_analytics_instantiable():
	"""ProjectPlanningService schedule_analytics runs without error."""
	svc = _pps()
	result = asyncio.run(svc.schedule_analytics(tenant_id="test-tenant"))
	assert isinstance(result, dict)
	assert "tenant_id" in result


# ── 3. ResourceManagementService instantiable ────────────────────────────────

def test_resource_management_instantiable():
	"""ResourceManagementService can be created."""
	svc = _res()
	assert svc is not None


# ── 4. TimeExpenseService instantiable ───────────────────────────────────────

def test_time_expense_instantiable():
	"""TimeExpenseService can be created."""
	svc = _tex()
	assert svc is not None


# ── 5. PPM manifest — 6 capabilities ─────────────────────────────────────────

def test_ppm_manifest():
	"""PPM domain contains exactly 6 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("ppm")
	assert len(caps) == 6, f"expected 6 ppm capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") for c in caps}
	assert any("pps" in str(cid) for cid in ids), f"ppm_pps not found in: {ids}"


# ── 6. PPM composability — all requires are known APG codes ──────────────────

def test_ppm_composability():
	"""All ppm capability REQUIRES lists contain only known APG codes."""
	known_codes = {
		"auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "mqeb",
		"nlpc", "moni", "comp", "keym", "stor", "mpen", "mdnt",
	}
	from capabilities.ppm.pps.capability_contract import REQUIRES as pps_req
	from capabilities.ppm.res.capability_contract import REQUIRES as res_req
	from capabilities.ppm.tex.capability_contract import REQUIRES as tex_req
	from capabilities.ppm.pac.capability_contract import REQUIRES as pac_req
	from capabilities.ppm.pan.capability_contract import REQUIRES as pan_req
	from capabilities.ppm.pbl.capability_contract import REQUIRES as pbl_req

	all_requires = {
		"ppm_pps": pps_req, "ppm_res": res_req, "ppm_tex": tex_req,
		"ppm_pac": pac_req, "ppm_pan": pan_req, "ppm_pbl": pbl_req,
	}
	for cap_id, reqs in all_requires.items():
		assert len(reqs) > 0, f"{cap_id} REQUIRES must not be empty"
		for req in reqs:
			assert req in known_codes, f"{cap_id}: unknown requirement '{req}'"
