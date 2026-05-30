"""Executable HCM Employee Data Management capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_employee_data", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "chr_employee_data_management"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "employee_agents" in contract["provides"]
	assert "/hcm/employees/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_compensation_review_gap():
	module = _load_module("rules_employee_data", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "employee_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "create_position",
		"compensation_band_present": True,
		"review_recorded": False,
	})["matched_rules"] == ["compensation_band_requires_review"]


def test_service_executes_employee_data_lifecycle():
	service_module = _load_module("service_employee_data", PACKAGE_DIR / "service.py")
	service = service_module.EmployeeDataManagementService()

	department = service.create_department("department-1", "tenant-test", "HR", "Human Resources", "owner-1", "HR-000")
	position = service.create_position("position-1", "tenant-test", "HRBP", "HR Business Partner", department["id"], "professional")
	employee = service.create_employee(
		"employee-1",
		"tenant-test",
		"EMP-0001",
		"Amina",
		"Otieno",
		"amina.otieno@example.com",
		department["id"],
		position["id"],
		"2026-01-01",
		"manager-1",
	)
	personal = service.record_personal_info("personal-1", "tenant-test", employee["id"], "KE", "2026-01-01", "employment_contract")
	contact = service.record_emergency_contact("contact-1", "tenant-test", employee["id"], "Sam Otieno", "Spouse", "+254700000000")
	history = service.record_employment_history("history-1", "tenant-test", employee["id"], "hire", "2026-01-01")
	skill = service.assign_skill("skill-1", "tenant-test", employee["id"], "Workforce Planning", "working")
	certification = service.assign_certification("cert-1", "tenant-test", employee["id"], "HR Analytics", "APG Academy", "2026-01-01", "2027-01-01")
	quality = service.record_data_quality_issue("quality-1", "tenant-test", "identity", "medium", "Profile needs document check")
	agent = service.register_employee_agent("tenant-test", "Profile Steward", "codex", "profile_steward", "review employee data")

	summary = service.dashboard_summary("tenant-test")
	assert position["department_id"] == department["id"]
	assert employee["full_name"] == "Amina Otieno"
	assert personal["privacy_basis"] == "employment_contract"
	assert contact["relationship"] == "Spouse"
	assert history["event_type"] == "hire"
	assert skill["level"] == "working"
	assert certification["status"] == "active"
	assert quality["severity"] == "medium"
	assert agent["role"] == "profile_steward"
	assert summary["employee_count"] == 1
	assert summary["audit_event_count"] == 10
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_employee_data", PACKAGE_DIR / "service.py")
	service = service_module.EmployeeDataManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_department("department", "", "HR", "Human Resources", "owner", "HR-000")
	with pytest.raises(PermissionError, match="department_code_required"):
		service.create_department("department", "tenant-test", "", "Human Resources", "owner", "HR-000")

	department = service.create_department("department", "tenant-test", "HR", "Human Resources", "owner", "HR-000")
	with pytest.raises(PermissionError, match="compensation_band_review_required"):
		service.create_position("position", "tenant-test", "HRBP", "HR Business Partner", department["id"], "professional", compensation_band={"min": 1})
	position = service.create_position("position", "tenant-test", "HRBP", "HR Business Partner", department["id"], "professional")
	with pytest.raises(PermissionError, match="work_email_invalid"):
		service.create_employee("employee", "tenant-test", "EMP-1", "A", "B", "bad-email", department["id"], position["id"], "2026-01-01", "manager")
	with pytest.raises(PermissionError, match="manager_required"):
		service.create_employee("employee", "tenant-test", "EMP-1", "A", "B", "a.b@example.com", department["id"], position["id"], "2026-01-01")
	employee = service.create_employee("employee", "tenant-test", "EMP-1", "A", "B", "a.b@example.com", department["id"], position["id"], "2026-01-01", "manager")
	with pytest.raises(PermissionError, match="sensitive_change_review_required"):
		service.change_employee_status(employee["id"], "tenant-test", "terminated", "role closed")
	with pytest.raises(PermissionError, match="privacy_basis_required"):
		service.record_personal_info("personal", "tenant-test", employee["id"], "KE", "2026-01-01", "")
	with pytest.raises(PermissionError, match="termination_approval_required"):
		service.record_employment_history("history", "tenant-test", employee["id"], "termination", "2026-01-01", "role closed")
	with pytest.raises(PermissionError, match="skill_evidence_required"):
		service.assign_skill("skill", "tenant-test", employee["id"], "Compensation", "expert")
	with pytest.raises(PermissionError, match="expires_on_required"):
		service.assign_certification("cert", "tenant-test", employee["id"], "HR Analytics", "APG Academy", "2026-01-01")
	with pytest.raises(PermissionError, match="quality_owner_required"):
		service.record_data_quality_issue("quality", "tenant-test", "identity", "high", "Missing document")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_employee_data", PACKAGE_DIR / "api.py")
	views = _load_module("views_employee_data", PACKAGE_DIR / "views.py")
	app = _load_module("app_employee_data", PACKAGE_DIR / "app.py")

	employee = api_module.create_record({"tenant_id": "tenant-api", "id": "employee-record"})
	agent = api_module.register_employee_agent({
		"tenant_id": "tenant-api",
		"name": "Quality Reviewer",
		"runtime": "claude_code",
		"role": "data_quality_reviewer",
	})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.employee_registry_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert employee["id"] == "employee-record"
	assert agent["role"] == "data_quality_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["full_name"] == "Employee Record"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["chr_employee_data_management"]["screens"]["agents"]["route"] == "/hcm/employees/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_employee_data", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["chr_employee_data_management"]["streaming"]["processor"] == "bytewax"
