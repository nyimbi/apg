"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 18: operations dashboard exercising all layout types and duration literals.\n// Features: dashboard/grid/tabs/split/stack/wizard/form layouts in one program,\n//           streaming window with duration literal\n\nmodule operations_dashboard version 1.0.0 {\n    description: "Operations dashboard with all layout types and streaming";\n}\n\ncapability OperationsDashboard {\n    contract: {\n        id: operations_dashboard,\n        provides: [operations_ui, kpi_views, approval_interfaces],\n        configuration: {tenant_id: "default"},\n        ui: {shell: python, routes: [\n            {name: "Home",      path: "/ops",           component: "OpsDashboard",  permission: "ops:view"},\n            {name: "Approvals", path: "/ops/approvals", component: "Approvals",     permission: "ops:approve"}\n        ]},\n        theme: {name: ops_theme, tokens: {accent: "#1565C0"}}\n    };\n\n    screens: {\n        MainDashboard: {\n            route: "/ops",\n            title: "Operations Centre",\n            layout: dashboard,\n            contains: [RevenueKpi, CostKpi, MarginKpi, AlertBanner],\n            composes: [LedgerSummary],\n            binds: [kpis.daily, alerts.active, ledger.summary],\n            actions: [refresh, export, drill_down],\n            events: [{on: "click", do: "navigate", target: LedgerSummary}],\n            relationships: [RevenueKpi -> LedgerSummary, AlertBanner -> LedgerSummary]\n        },\n\n        KpiGrid: {\n            route: "/ops/kpis",\n            title: "KPI Grid",\n            layout: grid,\n            contains: [SalesKpi, InventoryKpi, PayrollKpi, ComplianceKpi],\n            binds: [kpis.all],\n            actions: [compare, export, date_range]\n        },\n\n        ReportsTabs: {\n            route: "/ops/reports",\n            title: "Reports",\n            layout: tabs,\n            contains: [FinancialReports, OperationalReports, ComplianceReports],\n            binds: [reports.all],\n            actions: [download, schedule, share]\n        },\n\n        ApprovalSplit: {\n            route: "/ops/approvals",\n            title: "Approvals",\n            layout: split,\n            contains: [ApprovalList, ApprovalDetail],\n            binds: [approvals.pending],\n            actions: [approve, reject, request_changes],\n            relationships: [ApprovalList -> ApprovalDetail]\n        },\n\n        ActivityStack: {\n            route: "/ops/activity",\n            title: "Activity Feed",\n            layout: stack,\n            contains: [ActivityItem],\n            binds: [activity.recent],\n            actions: [load_more, filter, mark_read]\n        },\n\n        PeriodCloseWizard: {\n            route: "/ops/period-close",\n            title: "Period Close",\n            layout: wizard,\n            contains: [ReconciliationStep, ReviewStep, ApprovalStep, CloseStep],\n            binds: [period.status],\n            actions: [next, previous, save_progress]\n        },\n\n        ConfigurationForm: {\n            route: "/ops/config",\n            title: "Configuration",\n            layout: form,\n            contains: [GeneralSettings, NotificationSettings, IntegrationSettings],\n            binds: [config.current],\n            actions: [save, reset, test_connection]\n        }\n    };\n\n    streaming: {\n        processor: bytewax,\n        input:  operations_event_bus,\n        output: operations_alerts,\n        state:  ops_state,\n        window: 10min\n    };\n}\n\napp OperationsDashboardApp {\n    description: "Operations centre with all layout types";\n    capabilities: [OperationsDashboard];\n    routes: ["/ops"];\n}\n'
APG_MODULE_NAME = 'operations_dashboard'
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
