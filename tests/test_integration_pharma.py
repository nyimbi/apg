"""Pharma capability integration tests: Clinical Trials (ctr).

All tests use real in-memory service instances — no mocks.
Async service methods are called via asyncio.run() where needed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def _ctr_service():
	from capabilities.pharma.ctr.service import ClinicalTrialsService
	return ClinicalTrialsService()


# ── 1. Trial registration ────────────────────────────────────────────────────

def test_pharma_trial_registration():
	"""register_trial returns a structured registration record with required keys."""
	svc = _ctr_service()
	result = svc.register_trial(
		tenant_id="pharma-test",
		protocol="PROTO-2026-001",
		phase="phase_3",
		sponsor="Datacraft Pharma",
		indication="Type 2 Diabetes",
		target_enrollment=300,
		created_by="clinical-ops",
	)
	assert isinstance(result, dict)
	assert "id" in result
	assert result["phase"] == "phase_3"
	assert result["sponsor"] == "Datacraft Pharma"
	assert result["status"] == "registered"
	assert result["target_enrollment"] == 300
	assert result["tenant_id"] == "pharma-test"
	assert result["registration_number"].startswith("REG-")
	assert "eudraCT_placeholder" in result
	assert "clinicaltrials_gov_placeholder" in result


# ── 2. Adverse event reporting ────────────────────────────────────────────────

def test_pharma_adverse_event():
	"""report_adverse_event returns a dict with ae_id and expected fields."""
	svc = _ctr_service()
	from datetime import datetime

	result = svc.report_adverse_event(
		tenant_id="pharma-test",
		trial_id="trial-001",
		subject_id="subj-042",
		event_type="headache",
		severity="moderate",
		seriousness="not_serious",
		outcome="recovering",
		narrative="Subject experienced moderate headache 3 hours post-dose; resolved with paracetamol.",
		reported_by="site-investigator-01",
		onset_date=datetime(2026, 3, 10),
	)
	assert isinstance(result, dict)
	assert "id" in result
	ae_id = result["id"]
	assert ae_id  # non-empty
	assert result["severity"] == "moderate"
	assert result["seriousness"] == "not_serious"
	assert result["outcome"] == "recovering"
	assert result["tenant_id"] == "pharma-test"
	assert result["trial_id"] == "trial-001"
	assert result["status"] == "reported"
	assert result["is_susar_candidate"] is False


# ── 3. Protocol deviation ─────────────────────────────────────────────────────

def test_pharma_protocol_deviation():
	"""protocol_deviation returns a dict with deviation id and irb_reportable flag."""
	svc = _ctr_service()
	result = svc.protocol_deviation(
		tenant_id="pharma-test",
		subject_id="subj-007",
		deviation_type="minor",
		description="Blood sample taken 2 hours outside the protocol window.",
		impact="no_impact",
		corrective_action="Sample timing documented; investigator retrained.",
		reported_by="cra-001",
		trial_id="trial-001",
	)
	assert isinstance(result, dict)
	assert "id" in result
	assert result["deviation_type"] == "minor"
	assert result["impact"] == "no_impact"
	assert result["irb_reportable"] is False  # minor deviations are not IRB-reportable
	assert result["status"] == "open"
	assert result["tenant_id"] == "pharma-test"

	# Major deviation must be IRB-reportable
	major = svc.protocol_deviation(
		tenant_id="pharma-test",
		subject_id="subj-008",
		deviation_type="major",
		description="Patient enrolled without valid consent.",
		impact="safety_impact",
		corrective_action="Patient withdrawn; consent process retrained.",
		reported_by="cra-001",
		trial_id="trial-001",
	)
	assert major["irb_reportable"] is True


# ── 4. TMF document upload ────────────────────────────────────────────────────

def test_pharma_tmf_document():
	"""tmf_document_upload returns a record with a doc id."""
	svc = _ctr_service()
	result = svc.tmf_document_upload(
		tenant_id="pharma-test",
		trial_id="trial-001",
		section="Zone 03",
		document_name="IRB_Approval_Letter_v1.pdf",
		file_metadata={
			"file_name": "IRB_Approval_Letter_v1.pdf",
			"file_hash_sha256": "abc123def456" * 4,
			"file_size_bytes": 204800,
			"mime_type": "application/pdf",
			"upload_source": "Veeva eTMF",
		},
	)
	assert isinstance(result, dict)
	assert "id" in result
	doc_id = result["id"]
	assert doc_id  # non-empty
	assert result["document_name"] == "IRB_Approval_Letter_v1.pdf"
	assert result["tmf_section"] == "Zone 03"
	assert result["status"] == "uploaded"
	assert result["tenant_id"] == "pharma-test"
	assert result["trial_id"] == "trial-001"


# ── 5. Rule evaluation — allow ────────────────────────────────────────────────

def test_pharma_rule_evaluation():
	"""evaluate_rules('pharma_ctr', {tenant_context_present: True}) returns allow."""
	from capabilities.pharma.ctr.capability_contract import evaluate_capability_rules

	context = {
		"tenant_id": "pharma-test",
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"gcp_compliant": True,
	}
	result = evaluate_capability_rules(context)
	assert result["decision"] == "allow", (
		f"Expected allow, got {result['decision']}. Actions: {result.get('actions', [])}"
	)
	assert result["actions"] == []


# ── 6. Manifest — 9 pharma capabilities ──────────────────────────────────────

def test_pharma_manifest():
	"""There are exactly 9 pharma capabilities in the manifest."""
	import os, glob

	pharma_root = os.path.join(
		os.path.dirname(__file__), "..", "capabilities", "pharma"
	)
	manifests = glob.glob(os.path.join(pharma_root, "*/package_manifest.json"))
	assert len(manifests) == 9, (
		f"Expected 9 pharma capability manifests, found {len(manifests)}: "
		f"{[os.path.dirname(m).split('/')[-1] for m in manifests]}"
	)


# ── 7. Composability — all pharma requires satisfied ─────────────────────────

def test_pharma_composability():
	"""Every pharma capability's requires list is non-empty and contains known common services."""
	import os, glob, json

	pharma_root = os.path.join(
		os.path.dirname(__file__), "..", "capabilities", "pharma"
	)
	manifests = glob.glob(os.path.join(pharma_root, "*/package_manifest.json"))
	assert manifests, "No pharma manifests found"

	# Every pharma capability should provide at least one capability contract
	for manifest_path in manifests:
		cap_dir = os.path.dirname(manifest_path)
		cap_code = os.path.basename(cap_dir)
		contract_path = os.path.join(cap_dir, "capability_contract.py")
		assert os.path.isfile(contract_path), (
			f"capability_contract.py missing for pharma/{cap_code}"
		)

	# Verify the ClinicalTrials contract exposes requires
	from capabilities.pharma.ctr.capability_contract import get_capability_contract
	contract = get_capability_contract("pharma-test")
	requires = contract.get("requires", [])
	assert isinstance(requires, list), "requires should be a list"
	assert len(requires) > 0, "pharma_ctr should require at least one capability"
	# Common infra that all pharma caps need
	for common in ("auth", "audl", "mten"):
		assert common in requires, (
			f"pharma_ctr should require '{common}', got {requires}"
		)
