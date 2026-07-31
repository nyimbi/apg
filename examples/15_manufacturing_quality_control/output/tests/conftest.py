"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 15: manufacturing quality control with environment variable references.\n// Features: env() function, $VAR_NAME syntax for configuration secrets\n\nmodule manufacturing_qc version 1.0.0 {\n    description: "Manufacturing quality control with env-var configuration";\n}\n\ntable WorkOrder {\n    work_order_id: str;\n    product_code: str;\n    quantity_planned: int;\n    quantity_produced: int = 0;\n    quantity_rejected: int = 0;\n    status: str = "planned";\n    started_at: datetime?;\n    completed_at: datetime?;\n    production_line: str;\n}\n\ntable QualityInspection {\n    inspection_id: str;\n    work_order_id: str;\n    inspector_id: str;\n    inspection_type: str;     // in_process | final | audit\n    result: str?;             // pass | fail | conditional\n    defect_count: int = 0;\n    defect_rate: float = 0.0;\n    inspected_at: datetime;\n    notes: str?;\n}\n\ncapability ManufacturingQC {\n    contract: {\n        id: manufacturing_qc,\n        provides: [quality_inspections, defect_tracking, production_yield],\n        requires: [audit_events, work_orders],\n        configuration: {\n            // Environment variable references — secrets stay out of source\n            tenant_id:           env("APG_TENANT_ID"),\n            db_url:              env("MANUFACTURING_DB_URL"),\n            erp_api_key:         $ERP_API_KEY,\n            erp_endpoint:        env("ERP_API_ENDPOINT"),\n            // Literal configuration\n            max_defect_rate:     0.02,\n            inspection_interval: 50,\n            auto_fail_threshold: 0.05,\n            alert_email:         env("QC_ALERT_EMAIL")\n        },\n        configuration_schema: {\n            required: ["tenant_id", "db_url", "erp_api_key"]\n        },\n        rules: [\n            {name: "defect_rate_fail",   when: "defect_rate >= auto_fail_threshold",  action: deny},\n            {name: "defect_review",      when: "defect_rate > max_defect_rate",        action: require_review},\n            {name: "inspector_required", when: "inspector_id missing",                 action: deny},\n            {name: "valid_result",       when: "result in [pass, fail, conditional]",  action: allow}\n        ],\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Inspections", path: "/qc",         component: "InspectionQueue", permission: "qc:view"},\n                {name: "Defects",     path: "/qc/defects", component: "DefectTracker",   permission: "qc:defects"},\n                {name: "Yield",       path: "/qc/yield",   component: "YieldReport",     permission: "qc:reports"}\n            ]\n        },\n        theme: {name: manufacturing_theme, tokens: {accent: "#F57F17"}}\n    };\n\n    erp_modules: [manufacturing, quality, production_planning];\n    master_data: {entities: [product_spec, quality_standard, defect_category, production_line]};\n}\n\nworkflow QualityInspectionFlow {\n    steps: str = "initiated -> in_process_check -> final_inspection -> disposition -> closed";\n    human_tasks: [in_process_check, final_inspection, disposition];\n    assignments: {\n        in_process_check: line_inspector,\n        final_inspection: quality_manager,\n        disposition:      production_manager\n    };\n    guards: {\n        final_inspection: "in_process_check_passed == true",\n        disposition:      "final_inspection_complete == true",\n        closed:           "disposition_recorded == true",\n    };\n    timers: {\n        in_process_check: "PT2H",\n        final_inspection: "PT4H"\n    };\n}\n\napp ManufacturingQCApp {\n    description: "Manufacturing quality control";\n    capabilities: [ManufacturingQC];\n    routes: ["/qc", "/production"];\n}\n'
APG_MODULE_NAME = 'manufacturing_qc'
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
