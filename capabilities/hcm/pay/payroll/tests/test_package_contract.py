"""Executable HCM Payroll capability package tests."""

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


def _build_lifecycle(service):
	period = service.create_payroll_period("period-1", "tenant-test", "January Payroll", "monthly", "2026-01-01", "2026-01-31", "2026-02-01", "USD")
	pay_group = service.create_pay_group("group-1", "tenant-test", "US-MONTHLY", "US Monthly", "monthly", "USD", "US", "owner-1")
	profile = service.create_employee_pay_profile("profile-1", "tenant-test", "employee-1", pay_group["id"], "bank_transfer", "TAX-1", "USD", 5000, "reviewer-1")
	earning = service.create_pay_component("component-earn", "tenant-test", "BASE", "Base Pay", "earning", "USD", True)
	deduction = service.create_pay_component("component-ded", "tenant-test", "BEN", "Benefit Deduction", "deduction", "USD", False)
	service.record_time_import("time-1", "tenant-test", period["id"], profile["id"], 160, "time_attendance")
	run = service.start_payroll_run("run-1", "tenant-test", period["id"], pay_group["id"], "processor-1")
	line = service.add_line_item("line-1", "tenant-test", run["id"], profile["id"], earning["id"], 5000)
	service.add_line_item("line-2", "tenant-test", run["id"], profile["id"], deduction["id"], -250, "reviewer-2")
	tax = service.record_tax("tax-1", "tenant-test", run["id"], profile["id"], "employee", "IRS", 900)
	adjustment = service.record_adjustment("adjustment-1", "tenant-test", run["id"], profile["id"], 100, "retro pay", "approver-1")
	approved = service.approve_payroll_run(run["id"], "tenant-test", "approver-2")
	posted = service.post_payroll_run(run["id"], "tenant-test", "poster-1")
	payment = service.create_payment_batch("payment-1", "tenant-test", run["id"], "2026-02-01")
	payslip = service.publish_payslip("payslip-1", "tenant-test", run["id"], profile["id"], "employment_contract")
	filing = service.create_tax_filing("filing-1", "tenant-test", run["id"], "IRS", "2026-01", "approver-3")
	agent = service.register_payroll_agent("tenant-test", "Payroll Reviewer", "codex", "payroll_reviewer", "review payroll")
	return {"period": period, "pay_group": pay_group, "profile": profile, "line": line, "tax": tax, "adjustment": adjustment, "approved": approved, "posted": posted, "payment": payment, "payslip": payslip, "filing": filing, "agent": agent}


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_payroll", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "pay_payroll"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "payroll_agents" in contract["provides"]
	assert "/hcm/payroll/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_bank_review_gap():
	module = _load_module("rules_payroll", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "payroll_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "create_employee_pay_profile", "bank_payment": True, "review_recorded": False})["matched_rules"] == ["bank_profile_requires_review"]


def test_service_executes_payroll_lifecycle():
	service_module = _load_module("service_payroll", PACKAGE_DIR / "service.py")
	service = service_module.PayrollManagementService()
	records = _build_lifecycle(service)
	summary = service.dashboard_summary("tenant-test")

	assert records["pay_group"]["frequency"] == "monthly"
	assert records["profile"]["payment_method"] == "bank_transfer"
	assert records["line"]["amount"] == 5000
	assert records["tax"]["authority"] == "IRS"
	assert records["adjustment"]["approved_by"] == "approver-1"
	assert records["approved"]["approved_by"] == "approver-2"
	assert records["posted"]["posted_by"] == "poster-1"
	assert records["payment"]["net_pay"] == 3950
	assert records["payslip"]["privacy_basis"] == "employment_contract"
	assert records["filing"]["tax_total"] == 900
	assert records["agent"]["role"] == "payroll_reviewer"
	assert summary["run_count"] == 1
	assert summary["audit_event_count"] == 17
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_payroll", PACKAGE_DIR / "service.py")
	service = service_module.PayrollManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_payroll_period("period", "", "Payroll", "monthly", "2026-01-01", "2026-01-31", "2026-02-01")
	with pytest.raises(PermissionError, match="pay_frequency_not_supported"):
		service.create_payroll_period("period", "tenant-test", "Payroll", "daily", "2026-01-01", "2026-01-31", "2026-02-01")
	period = service.create_payroll_period("period", "tenant-test", "Payroll", "monthly", "2026-01-01", "2026-01-31", "2026-02-01")
	with pytest.raises(PermissionError, match="pay_group_owner_required"):
		service.create_pay_group("group", "tenant-test", "US", "US Payroll", "monthly", "USD", "US", "")
	pay_group = service.create_pay_group("group", "tenant-test", "US", "US Payroll", "monthly", "USD", "US", "owner")
	with pytest.raises(PermissionError, match="bank_profile_review_required"):
		service.create_employee_pay_profile("profile", "tenant-test", "employee", pay_group["id"], "bank_transfer", "TAX", "USD", 5000)
	profile = service.create_employee_pay_profile("profile", "tenant-test", "employee", pay_group["id"], "bank_transfer", "TAX", "USD", 5000, "reviewer")
	component = service.create_pay_component("component", "tenant-test", "BASE", "Base Pay", "earning", "USD", True)
	with pytest.raises(PermissionError, match="overtime_approval_required"):
		service.record_time_import("time", "tenant-test", period["id"], profile["id"], 170, "time", 10)
	run = service.start_payroll_run("run", "tenant-test", period["id"], pay_group["id"], "processor")
	with pytest.raises(PermissionError, match="negative_amount_review_required"):
		service.add_line_item("line", "tenant-test", run["id"], profile["id"], component["id"], -1)
	with pytest.raises(PermissionError, match="payroll_approver_required"):
		service.approve_payroll_run(run["id"], "tenant-test", "")
	with pytest.raises(PermissionError, match="payroll_approval_required"):
		service.post_payroll_run(run["id"], "tenant-test", "poster")
	with pytest.raises(PermissionError, match="privacy_basis_required"):
		service.publish_payslip("payslip", "tenant-test", run["id"], profile["id"], "")


def test_agents_batch_api_views_and_app_are_executable():
	api_module = _load_module("api_payroll", PACKAGE_DIR / "api.py")
	views = _load_module("views_payroll", PACKAGE_DIR / "views.py")
	app = _load_module("app_payroll", PACKAGE_DIR / "app.py")

	period = api_module.create_record({"tenant_id": "tenant-api", "id": "period-record"})
	agent = api_module.register_payroll_agent({"tenant_id": "tenant-api", "name": "Tax Reviewer", "runtime": "claude_code", "role": "tax_reviewer"})
	batch = api_module.service().validate_batch("tenant-api", 2)
	model = views.period_model(api_module.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert period["id"] == "period-record"
	assert agent["role"] == "tax_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["name"] == "Payroll Period"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["pay_payroll"]["screens"]["agents"]["route"] == "/hcm/payroll/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_payroll", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["pay_payroll"]["streaming"]["processor"] == "bytewax"
