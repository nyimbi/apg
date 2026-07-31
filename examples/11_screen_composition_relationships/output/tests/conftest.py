"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 11: comprehensive screen composition.\n// Features: all layouts, contains/composes/binds/actions/events/relationships,\n//           both relationship syntaxes (arrow and object form)\n\nmodule operations_workbench version 1.0.0 {\n    description: "Operations workbench with full screen composition coverage";\n}\n\ncapability OperationsWorkbench {\n    contract: {\n        id: operations_workbench,\n        provides: [operations_ui, approval_queue, kpi_dashboard],\n        configuration: {tenant_id: "default"},\n        ui: {\n            shell: python,\n            routes: [\n                {name: "Dashboard",   path: "/ops",            component: "OpsDashboard",    permission: "ops:view"},\n                {name: "Approvals",   path: "/ops/approvals",  component: "ApprovalQueue",   permission: "ops:approve"},\n                {name: "Reports",     path: "/ops/reports",    component: "OpsReports",      permission: "ops:reports"}\n            ]\n        },\n        theme: {name: ops_theme, tokens: {accent: "#345995", "border.radius": "4px"}}\n    };\n\n    screens: {\n        // dashboard layout — all fields exercised\n        MainDashboard: {\n            route: "/ops",\n            title: "Operations Dashboard",\n            layout: dashboard,\n            contains: [KpiStrip, AlertBanner, ActivityFeed],\n            composes: [LedgerSummary, ApprovalWidget],\n            binds: [ledger.entries, approvals.pending, alerts.active],\n            actions: [refresh, export, configure],\n            events: [\n                {on: "select",  do: "filter",    target: LedgerSummary},\n                {on: "click",   do: "navigate",  target: ApprovalWidget},\n                {on: "hover",   do: "tooltip",   target: KpiStrip}\n            ],\n            relationships: [\n                // Arrow syntax (shorthand)\n                KpiStrip -> LedgerSummary,\n                AlertBanner -> ApprovalWidget,\n                // Object syntax with metadata\n                {from: KpiStrip,       to: ActivityFeed,    via: filters,   type: filter},\n                {from: ApprovalWidget, to: LedgerSummary,   via: selection, type: drill_down}\n            ]\n        },\n\n        // split layout\n        ApprovalQueue: {\n            route: "/ops/approvals",\n            title: "Approval Queue",\n            layout: split,\n            contains: [RequestList, ApprovalDetail],\n            binds: [approvals.pending, approvals.history],\n            actions: [approve, reject, request_changes, escalate],\n            events: [\n                {on: "select", do: "load_detail", target: ApprovalDetail}\n            ],\n            relationships: [\n                RequestList -> ApprovalDetail\n            ]\n        },\n\n        // tabs layout\n        ReportingHub: {\n            route: "/ops/reports",\n            title: "Reporting Hub",\n            layout: tabs,\n            contains: [SummaryTab, DetailTab, TrendTab],\n            binds: [reports.summary, reports.detail, reports.trends],\n            actions: [download, schedule, share]\n        },\n\n        // grid layout\n        KpiGrid: {\n            route: "/ops/kpis",\n            title: "KPI Grid",\n            layout: grid,\n            contains: [RevenueKpi, CostKpi, MarginKpi, VolumeKpi],\n            binds: [kpis.daily],\n            actions: [drill_down, compare]\n        },\n\n        // wizard layout\n        OnboardingWizard: {\n            route: "/ops/onboard",\n            title: "Onboarding Wizard",\n            layout: wizard,\n            contains: [Step1, Step2, Step3, Step4],\n            binds: [onboarding.state],\n            actions: [next, back, save_progress, finish]\n        },\n\n        // stack layout\n        ActivityStack: {\n            route: "/ops/activity",\n            title: "Activity Feed",\n            layout: stack,\n            contains: [ActivityItem],\n            binds: [activity.events],\n            actions: [load_more, filter]\n        },\n\n        // form layout\n        ConfigForm: {\n            route: "/ops/config",\n            title: "Configuration",\n            layout: form,\n            contains: [ConfigFields],\n            binds: [config.current],\n            actions: [save, reset, preview]\n        }\n    };\n}\n\napp OperationsWorkbenchApp {\n    description: "Operations management workbench";\n    capabilities: [OperationsWorkbench];\n    routes: ["/ops"];\n}\n'
APG_MODULE_NAME = 'operations_workbench'
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
