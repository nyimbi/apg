"""Regression coverage for the QUAN executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.quan import register_capability
from capabilities.common.quan.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.quan.service import QuanService
from capabilities.common.quan import views


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-quan", {"jobs": {"shot_limit": 20000}})

	assert contract["capability"] == "quan"
	assert contract["configuration"]["tenant_id"] == "tenant-quan"
	assert contract["configuration"]["jobs"]["shot_limit"] == 20000
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"backends",
		"circuits",
		"jobs",
		"quan_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"quantum_backend_registry",
		"circuit_management",
		"quantum_job_orchestration",
		"result_analysis",
		"post_quantum_governance",
		"quan_agents",
	]
	assert contract["requires"] == ["aicr", "encr", "keym", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["quan_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "backends", "circuits", "jobs", "experiments", "results", "agents", "audit", "governance", "settings"}
	assert contract["theme"]["name"] == "quan_quantum_lab"


def test_rule_engine_enforces_quan_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_circuit",
		"circuit_owner_assigned": False,
		"circuit_version_present": False,
		"circuit_qubits_required": 0,
		"circuit_gates_present": False,
		"sensitive_input_present": True,
		"encryption_applied": False,
		"experiment_metadata_present": False,
	})
	backend_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_backend", "backend_approved": False, "external_provider": True, "credentials_ref_present": False, "backend_qubit_count": 0})
	job_result = evaluate_capability_rules({"operation": "submit_job", "quota_policy_attached": False, "job_submitter_present": False, "retry_policy_attached": False, "shot_count": 20000, "job_review_recorded": False, "event_stream": "other-stream"})
	agent_result = evaluate_capability_rules({"quan_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_quantum_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"circuit_requires_owner",
		"circuit_requires_version",
		"circuit_requires_qubits",
		"circuit_requires_gates",
		"sensitive_input_requires_encryption",
		"circuit_requires_experiment_metadata",
	}
	assert set(backend_result["matched_rules"]) == {"backend_requires_approval", "backend_requires_credentials_reference", "backend_requires_qubit_capacity"}
	assert set(job_result["matched_rules"]) == {"job_requires_quota", "job_requires_submitter", "job_requires_retry_policy", "large_job_requires_review", "job_requires_bytewax_stream"}
	assert agent_result["matched_rules"] == ["quan_agent_runtime_supported"]
	assert batch_result["matched_rules"] == ["batch_quantum_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "quan"
	assert "aicr" in registration["dependencies"]
	assert registration["ui_components"]["jobs"] == "/quan/jobs"
	assert registration["ui_components"]["agents"] == "/quan/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "quan:run_jobs" in registration["permissions"]


def test_quantum_lifecycle_records_backend_circuit_job_result_experiment_and_agent():
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
		event_stream="bytewax://quantum-jobs",
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
	agent = service.register_quan_agent(
		tenant_id="tenant-quan",
		name="Job reviewer",
		runtime="codex",
		role="job_reviewer",
		scope="review quota, retry, cost, shot-count, and stream gates",
	)

	assert backend["provider"] == "ibm"
	assert policy["retry_policy"] == "provider_retry"
	assert circuit["gates"] == ["h", "cx", "measure"]
	assert job["estimated_cost"] == 10.24
	assert sum(result["measurement_counts"].values()) == 1024
	assert result["confidence"] > 0
	assert experiment["job_ids"] == ["job-001"]
	assert agent["runtime"] == "codex"
	assert agent["role"] == "job_reviewer"

	summary = service.dashboard_summary("tenant-quan")
	assert summary["backend_count"] == 1
	assert summary["completed_job_count"] == 1
	assert summary["result_count"] == 1
	assert summary["quan_agent_count"] == 1
	assert summary["streaming"]["processor"] == "bytewax"
	assert service.validate_batch_quantum_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_quantum_mutation("other-stream")["decision"] == "deny"
	assert views.dashboard_model(service, "tenant-quan")["summary"]["job_count"] == 1
	assert views.dashboard_model(service, "tenant-quan")["streaming"]["processor"] == "bytewax"
	assert views.backend_registry_model(service, "tenant-quan")["quota_policies"][0]["id"] == "quota-small"
	assert views.circuit_library_model(service, "tenant-quan")["circuits"][0]["id"] == "bell-v1"
	assert views.job_queue_model(service, "tenant-quan")["jobs"][0]["status"] == "completed"
	assert views.result_viewer_model(service, "tenant-quan")["results"][0]["id"] == "result-001"
	assert views.experiment_workbench_model(service, "tenant-quan")["experiments"][0]["id"] == "exp-001"
	assert views.quan_agent_model(service, "tenant-quan")["quan_agents"][0]["role"] == "job_reviewer"
	assert views.audit_trail_model(service, "tenant-quan")["audit_events"]
	assert views.quantum_policy_model(service, "tenant-quan")["streaming"]["processor"] == "bytewax"


def test_quantum_guardrails_block_unsafe_operations():
	service = QuanService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_backend("backend", "", "Backend", "local", approved=True)

	with pytest.raises(PermissionError, match="backend_approval_required"):
		service.register_backend("backend", "tenant-quan", "Backend", "local", approved=False)

	with pytest.raises(PermissionError, match="provider_credentials_required"):
		service.register_backend("backend", "tenant-quan", "Backend", "ibm", approved=True)

	with pytest.raises(PermissionError, match="backend_qubit_capacity_required"):
		service.register_backend("bad-qubits", "tenant-quan", "Backend", "local", qubit_count=0, approved=True)

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

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.submit_job("job-other-stream", "tenant-quan", backend["id"], circuit["id"], "operator", 20, job_review_recorded=True, event_stream="other-stream")

	job = service.submit_job("job-reviewed", "tenant-quan", backend["id"], circuit["id"], "operator", 20, job_review_recorded=True)

	with pytest.raises(PermissionError, match="post_quantum_review_required"):
		service.create_experiment("exp-pq", "tenant-quan", "PQ", circuit["id"], [job["id"]], "post-quantum migration assessment")

	with pytest.raises(PermissionError, match="quan_agent_runtime_not_supported"):
		service.register_quan_agent("tenant-quan", "Unsupported", "unsupported", "job_reviewer", "review")

	with pytest.raises(PermissionError, match="quan_agent_scope_required"):
		service.register_quan_agent("tenant-quan", "No scope", "codex", "job_reviewer", "")


def test_lifecycle_ids_are_tenant_scoped():
	service = QuanService()

	for tenant_id, name, owner in (
		("tenant-a", "Backend A", "owner-a"),
		("tenant-b", "Backend B", "owner-b"),
	):
		service.register_backend("shared-backend", tenant_id, name, "local", qubit_count=8, approved=True)
		service.attach_quota_policy("shared-quota", tenant_id, "shared-backend", 100, 10, 10.0)
		service.create_circuit("shared-circuit", tenant_id, "Shared circuit", owner, "1.0", 1, ["h"], experiment_metadata={"tenant": tenant_id})
		service.register_quan_agent(tenant_id, "Reviewer", "codex", "job_reviewer", "review tenant quantum jobs", agent_id="shared-agent")

	assert service.list_backends("tenant-a")[0]["name"] == "Backend A"
	assert service.list_backends("tenant-b")[0]["name"] == "Backend B"
	assert service.list_circuits("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_circuits("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_quan_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_quan_agents("tenant-b")[0]["id"] == "shared-agent"


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


def test_generated_evidence_and_docs_are_current():
	app = _load_module("quan_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["quan"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["quan"]["screens"]["agents"]["route"] == "/quan/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
