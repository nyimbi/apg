"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 16: HR and payroll with multiple workflows and agent team in app.\n// Features: multiple workflows, agent_team in app declaration\n\nmodule hr_payroll version 1.0.0 {\n    description: "HR and payroll with multiple workflows and agent team";\n}\n\ntable Employee {\n    employee_number: str;\n    first_name: str;\n    last_name: str;\n    email: str;\n    department: str;\n    position: str;\n    employment_type: str = "permanent";\n    hire_date: str;\n    termination_date: str?;\n    salary: float;\n    currency: str = "KES";\n    bank_account: str?;\n    tax_pin: str?;\n    nssf_number: str?;\n    nhif_number: str?;\n    is_active: bool = true;\n}\n\ntable PayRun {\n    pay_run_id: str;\n    period: str;\n    pay_date: str;\n    status: str = "draft";\n    total_gross: float = 0.0;\n    total_deductions: float = 0.0;\n    total_net: float = 0.0;\n    approved_by: str?;\n    approved_at: str?;\n}\n\ntable Payslip {\n    payslip_id: str;\n    pay_run_id: str;\n    employee_id: str;\n    gross_pay: float;\n    paye_tax: float;\n    nssf_employee: float;\n    nhif_employee: float;\n    other_deductions: float = 0.0;\n    net_pay: float;\n    payment_status: str = "pending";\n}\n\nagent HRAdvisor {\n    role: "HR policy advisor";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Advise on HR policies, leave entitlements, and employment law compliance. Cite relevant sections.";\n    capabilities: [employee_records, hr_policies];\n    tools: [policy.lookup, employment_law.search, leave_balance.check];\n    memory: vector hr_memory;\n    configuration: {temperature: 0.1, max_turns: 6};\n}\n\nagent PayrollAssistant {\n    role: "payroll calculation assistant";\n    model: "openai:gpt-4.1-mini";\n    runtime: codex;\n    system: "Assist with payroll calculations, statutory deductions, and compliance. Always verify statutory rates.";\n    capabilities: [payroll_runs, statutory_deductions];\n    tools: [tax_table.lookup, statutory_rates.fetch, payslip.calculate];\n    configuration: {temperature: 0.0, max_turns: 4};\n}\n\nagent_team HRPayrollTeam {\n    agents: [HRAdvisor, PayrollAssistant];\n    flow: HRAdvisor -> PayrollAssistant [condition: payroll_query];\n    capabilities: [employee_records, payroll_runs];\n    configuration: {handoff_mode: conditional};\n}\n\ncapability HRPayroll {\n    contract: {\n        id: hr_payroll,\n        provides: [employee_records, payroll_runs, payslips, statutory_deductions, leave_management, hr_policies],\n        configuration: {tenant_id: "default", country: "KE", currency: "KES"},\n        rules: [\n            {name: "bank_account_required",   when: "bank_account missing",               action: deny},\n            {name: "tax_pin_required",         when: "tax_pin missing",                    action: require_review},\n            {name: "nssf_required",            when: "nssf_number missing",                action: require_review},\n            {name: "payslip_authorised",       when: "pay_run_status != approved",         action: deny},\n            {name: "back_date_limit",          when: "pay_date_months_ago > 3",            action: require_review}\n        ],\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Employees",   path: "/hr/employees",  component: "EmployeeList",  permission: "hr:view"},\n                {name: "Pay Runs",    path: "/hr/payroll",    component: "PayRunList",    permission: "hr:payroll"},\n                {name: "Payslips",    path: "/hr/payslips",   component: "PayslipView",   permission: "hr:payslips"}\n            ]\n        },\n        theme: {name: hr_theme, tokens: {accent: "#6A1B9A"}}\n    };\n\n    erp_modules: [hr, payroll, time_attendance];\n    approvals: {levels: 2, approvers: [hr_manager, finance_controller]};\n    master_data: {entities: [employee, department, position, pay_grade, statutory_rate]};\n}\n\nworkflow PayRunProcess {\n    steps: str = "draft -> calculated -> reviewed -> approved -> disbursed -> closed";\n    human_tasks: [reviewed, approved];\n    assignments: {reviewed: hr_manager, approved: finance_controller};\n    guards: {\n        reviewed:   "all_payslips_generated == true";\n        approved:   "review_complete == true";\n        disbursed:  "bank_file_generated == true";\n    };\n    timers: {reviewed: "PT24H", approved: "PT48H"};\n    compensation: {approved: void_bank_transfers, disbursed: initiate_recalls};\n}\n\nworkflow EmployeeOnboarding {\n    steps: str = "offer_accepted -> documents_submitted -> system_access -> orientation -> probation -> confirmed";\n    human_tasks: [documents_submitted, system_access, orientation, confirmed];\n    assignments: {\n        documents_submitted: hr_officer,\n        system_access:       it_administrator,\n        orientation:         hr_manager,\n        confirmed:           department_manager\n    };\n    guards: {\n        system_access:  "all_documents_received == true";\n        orientation:    "system_access_granted == true";\n        confirmed:      "probation_assessment_complete == true";\n    };\n    timers: {\n        documents_submitted: "P3D",\n        system_access:       "PT8H",\n        orientation:         "P1D"\n    };\n}\n\nworkflow LeaveRequest {\n    steps: str = "submitted -> manager_approval -> hr_approval -> approved -> active -> closed";\n    human_tasks: [manager_approval, hr_approval];\n    assignments: {manager_approval: line_manager, hr_approval: hr_officer};\n    guards: {\n        manager_approval: "leave_balance >= requested_days";\n        hr_approval:      "manager_approved == true";\n    };\n}\n\napp HRPayrollApp {\n    description: "Human resources and payroll management";\n    capabilities: [HRPayroll];\n    agent_teams: [HRPayrollTeam];\n    routes: ["/hr", "/hr/employees", "/hr/payroll"];\n    theme: {name: hr_app_theme, tokens: {accent: "#6A1B9A"}};\n}\n'
APG_MODULE_NAME = 'hr_payroll'
_GENERATED_TEST_ENV_KEYS = (
	'APG_API_KEY',
	'APG_AUTH_USERS',
	'APG_AUTO_MIGRATE',
	'APG_DATABASE_URL',
	'APG_DATA_FILE',
	'APG_DATA_PATH',
	'APG_DB_PATH',
	'APG_ENV',
	'APG_JWT_SECRET',
	'APG_PG_URL',
	'APG_PRODUCTION',
	'APG_SESSION_SECRET',
	'APG_SQLITE_PATH',
	'DATABASE_URL',
)


@pytest.fixture()
def generated_app_client(monkeypatch):
	for key in _GENERATED_TEST_ENV_KEYS:
		monkeypatch.delenv(key, raising=False)
	result = APGCompiler().compile_string(APG_SOURCE, APG_MODULE_NAME)
	assert result.success, result.errors
	namespace = {"__file__": "generated_app.py"}
	exec(compile(result.generated_files["app.py"], "generated_app.py", "exec"), namespace)
	app = namespace["_flask_app"]
	app.config["TESTING"] = True
	return app.test_client()
