"""Regression coverage for the QUAN executable capability contract."""

import pytest

from capabilities.common.quan import register_capability
from capabilities.common.quan.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.quan.service import QuanService
from capabilities.common.quan import views


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-quan", {"jobs": {"shot_limit": 20000}})

	assert contract["capability"] == "quan"
	assert contract["configuration"]["tenant_id"] == "tenant-quan"
	assert contract["configuration"]["jobs"]["shot_limit"] == 20000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "backends", "circuits", "jobs", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "backends", "circuits", "jobs", "experiments", "results", "governance", "settings"}
	assert contract["theme"]["name"] == "quan_quantum_lab"


def test_rule_engine_enforces_quan_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_circuit", "circuit_owner_assigned": False, "sensitive_input_present": True, "encryption_applied": False, "shot_count": 20000, "job_review_recorded": False})
	backend_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_backend", "backend_approved": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "circuit_requires_owner", "sensitive_input_requires_encryption", "large_job_requires_review"}
	assert backend_result["matched_rules"] == ["backend_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "quan"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["jobs"] == "/quan/jobs"
	assert "quan:run_jobs" in registration["permissions"]


def test_quantum_lifecycle_records_backend_circuit_job_result_and_experiment():
	service = QuanService()

	backend = service.register_backend(
		backend_id="ibm-small",
		tenant_id="tenant-quan",
		name="IBM Small Quantum",
		provider="ibm",
		backend_type="qpu",
		qubit_count=16,
		approved=True,
		credentials_ref="keym://tenant-quan/ibm-small",
		actor="operator",
	)
	policy = service.attach_quota_policy(
		policy_id="quota-small",
		tenant_id="tenant-quan",
		backend_id=backend["id"],
		max_shots_per_job=4096,
		max_jobs_per_day=12,
		cost_limit=100.0,
		retry_policy="provider_retry",
	)
	circuit = service.create_circuit(
		circuit_id="bell-v1",
		tenant_id="tenant-quan",
		name="Bell Pair",
		owner="quantum-research",
		version="1.0.0",
		qubits_required=2,
		gates=["H", "CX", "Measure"],
		sensitive_input_present=True,
		encryption_applied=True,
		experiment_metadata={"purpose": "entanglement validation"},
	)
	job = service.submit_job(
		job_id="job-001",
		tenant_id="tenant-quan",
		backend_id=backend["id"],
		circuit_id=circuit["id"],
		submitted_by="researcher",
		shot_count=1024,
		job_review_recorded=False,
	)
	result = service.complete_job(
		result_id="result-001",
		tenant_id="tenant-quan",
		job_id=job["id"],
	)
	experiment = service.create_experiment(
		experiment_id="exp-001",
		tenant_id="tenant-quan",
		name="Bell Experiment",
		circuit_id=circuit["id"],
		job_ids=[job["id"]],
		hypothesis="entangled measurement distribution is stable",
	)

	assert backend["provider"] == "ibm"
	assert policy["retry_policy"] == "provider_retry"
	assert circuit["gates"] == ["h", "cx", "measure"]
	assert job["estimated_cost"] == 10.24
	assert sum(result["measurement_counts"].values()) == 1024
	assert result["confidence"] > 0
	assert experiment["job_ids"] == ["job-001"]

	summary = service.dashboard_summary("tenant-quan")
	assert summary["backend_count"] == 1
	assert summary["completed_job_count"] == 1
	assert summary["result_count"] == 1
	assert views.dashboard_model(service, "tenant-quan")["summary"]["job_count"] == 1
	assert views.backend_registry_model(service, "tenant-quan")["quota_policies"][0]["id"] == "quota-small"
	assert views.circuit_library_model(service, "tenant-quan")["circuits"][0]["id"] == "bell-v1"
	assert views.job_queue_model(service, "tenant-quan")["jobs"][0]["status"] == "completed"
	assert views.result_viewer_model(service, "tenant-quan")["results"][0]["id"] == "result-001"
	assert views.experiment_workbench_model(service, "tenant-quan")["experiments"][0]["id"] == "exp-001"
	assert views.governance_model(service, "tenant-quan")["audit_events"]


def test_quantum_guardrails_block_unsafe_operations():
	service = QuanService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_backend("backend", "", "Backend", "local", approved=True)

	with pytest.raises(PermissionError, match="backend_approval_required"):
		service.register_backend("backend", "tenant-quan", "Backend", "local", approved=False)

	with pytest.raises(PermissionError, match="provider_credentials_required"):
		service.register_backend("backend", "tenant-quan", "Backend", "ibm", approved=True)

	backend = service.register_backend(
		"backend",
		"tenant-quan",
		"Local Backend",
		"local",
		qubit_count=2,
		approved=True,
	)

	with pytest.raises(PermissionError, match="circuit_owner_required"):
		service.create_circuit("bad-owner", "tenant-quan", "Bad", "", "1.0", 1, ["h"], experiment_metadata={"purpose": "test"})

	with pytest.raises(PermissionError, match="sensitive_input_encryption_required"):
		service.create_circuit("bad-encryption", "tenant-quan", "Bad", "owner", "1.0", 1, ["h"], sensitive_input_present=True, encryption_applied=False, experiment_metadata={"purpose": "test"})

	with pytest.raises(PermissionError, match="experiment_metadata_required"):
		service.create_circuit("bad-metadata", "tenant-quan", "Bad", "owner", "1.0", 1, ["h"])

	circuit = service.create_circuit(
		"good-circuit",
		"tenant-quan",
		"Good",
		"owner",
		"1.0",
		1,
		["h"],
		experiment_metadata={"purpose": "test"},
	)

	with pytest.raises(PermissionError, match="job_quota_required"):
		service.submit_job("job-no-quota", "tenant-quan", backend["id"], circuit["id"], "operator", 8)

	service.attach_quota_policy("quota-small", "tenant-quan", backend["id"], 10, 2, 100.0)

	with pytest.raises(PermissionError, match="job_shot_quota_exceeded"):
		service.submit_job("job-quota", "tenant-quan", backend["id"], circuit["id"], "operator", 20)

	service.attach_quota_policy("quota-large", "tenant-quan", backend["id"], 30000, 2, 100.0)

	with pytest.raises(PermissionError, match="large_quantum_job_review_required"):
		service.submit_job("job-large", "tenant-quan", backend["id"], circuit["id"], "operator", 20000)

	job = service.submit_job("job-reviewed", "tenant-quan", backend["id"], circuit["id"], "operator", 20, job_review_recorded=True)

	with pytest.raises(PermissionError, match="post_quantum_review_required"):
		service.create_experiment("exp-pq", "tenant-quan", "PQ", circuit["id"], [job["id"]], "post-quantum migration assessment")


def test_compatibility_record_api_uses_quantum_backend_runtime():
	service = QuanService()

	record = service.create_record(
		record_id="local-sim",
		tenant_id="tenant-quan",
		metadata={"provider": "local", "backend_type": "simulator", "qubit_count": 8},
		status="active",
	)

	assert record["id"] == "local-sim"
	assert record["backend_type"] == "simulator"
	assert record["approved"] is True
	assert service.list_records("tenant-quan")[0]["id"] == "local-sim"
