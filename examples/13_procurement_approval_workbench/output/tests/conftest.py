"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 13: procurement with workflow coverage and db entity.\n// Features: db entity, workflow with all fields (timers, waits, retry_policy,\n//           compensation), capability with full contract\n\nmodule procurement_workbench version 1.0.0 {\n    description: "Procurement with database declaration and full workflow";\n}\n\n// Explicit database declaration\ndb ProcurementDB {\n    engine: postgresql;\n    schema: procurement;\n    migrations: alembic;\n}\n\ntable PurchaseRequest {\n    request_id: str;\n    requestor_id: str;\n    department: str;\n    description: str;\n    amount: decimal;\n    currency: str = "KES";\n    priority: str = "normal";\n    status: str = "draft";\n    justification: str?;\n    budget_code: str?;\n    created_at: datetime;\n}\n\ntable SupplierQuote {\n    quote_id: str;\n    request_id: str;\n    supplier_id: str;\n    amount: decimal;\n    validity_days: int = 30;\n    submitted_at: datetime;\n    status: str = "pending";\n}\n\ncapability ProcurementWorkbench {\n    contract: {\n        id: procurement_workbench,\n        provides: [purchase_requests, supplier_quotes, approval_plans],\n        requires: [audit_events, supplier_master, budget_control],\n        configuration: {\n            tenant_id: "default",\n            currency: "KES",\n            approval_threshold: 50000,\n            three_quote_minimum: 100000\n        },\n        rules: [\n            {name: "budget_code_required", when: "budget_code missing",                              action: deny},\n            {name: "high_value_approval",  when: "amount > approval_threshold",                      action: require_review},\n            {name: "three_quote_required", when: "amount > three_quote_minimum and quote_count < 3", action: require_review},\n            {name: "approved_supplier",    when: "supplier_status not in [approved, preferred]",     action: deny}\n        ],\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Requests",  path: "/procurement",           component: "RequestList",     permission: "proc:view"},\n                {name: "Approvals", path: "/procurement/approvals", component: "ApprovalQueue",   permission: "proc:approve"},\n                {name: "Quotes",    path: "/procurement/quotes",    component: "QuoteComparison", permission: "proc:quotes"}\n            ]\n        },\n        theme: {name: proc_theme, tokens: {accent: "#6C8EAD"}}\n    };\n\n    erp_modules: [procurement, finance, inventory];\n    approvals: {\n        levels: 3,\n        thresholds: {level1: 50000, level2: 200000, level3: 1000000},\n        approvers: [request_manager, procurement_lead, finance_controller],\n        segregation_of_duties: true\n    };\n    master_data: {entities: [supplier, item_catalog, cost_center, budget_line]};\n    screens: {\n        ApprovalQueue: {\n            route: "/procurement/approvals",\n            layout: split,\n            contains: [RequestList, ApprovalDetail],\n            binds: [purchase_requests.pending],\n            actions: [approve, reject, request_changes],\n            relationships: [RequestList -> ApprovalDetail]\n        }\n    };\n}\n\n// Full workflow: all seven field types\nworkflow ProcurementApproval {\n    steps: str = "draft -> submitted -> budget_review -> procurement_review -> finance_approval -> approved -> ordered";\n    human_tasks: [budget_review, procurement_review, finance_approval];\n    assignments: {\n        budget_review:      budget_owner,\n        procurement_review: procurement_lead,\n        finance_approval:   finance_controller\n    };\n    guards: {\n        budget_review:      "amount <= budget_limit and budget_code not missing",\n        procurement_review: "budget_review_complete == true",\n        finance_approval:   "amount > finance_threshold or is_strategic == true",\n        ordered:            "all_approvals_complete and preferred_supplier_selected"\n    };\n    timers: {\n        budget_review:      "PT24H",\n        procurement_review: "PT48H",\n        finance_approval:   "PT24H",\n        ordered:            "PT72H"\n    };\n    waits: {\n        ordered: purchase_order_issued\n    };\n    retry_policy: {\n        budget_review:      "3",\n        procurement_review: "2",\n        finance_approval:   "2"\n    };\n    compensation: {\n        budget_review: release_budget_reservation,\n        ordered:       cancel_purchase_order\n    };\n}\n\napp ProcurementApp {\n    description: "Procurement approval workbench";\n    capabilities: [ProcurementWorkbench];\n    routes: ["/procurement"];\n}\n'
APG_MODULE_NAME = 'procurement_workbench'
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
