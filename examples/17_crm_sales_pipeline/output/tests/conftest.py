"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 17: CRM sales pipeline with imports, components, and app-level screens.\n// Features: import statement, app.components, app.screens, multiple capabilities\n\nmodule crm_sales version 1.0.0 {\n    description: "CRM sales pipeline with multi-capability composition";\n}\n\nfrom common.contacts import BaseContact;\nimport common.enumerations;\n\ntable Lead {\n    lead_id: str;\n    first_name: str;\n    last_name: str;\n    email: str;\n    phone: str?;\n    company: str?;\n    source: str = "website";\n    status: str = "new";\n    score: float = 0.0;\n    owner_id: str;\n    created_at: str;\n    notes: str?;\n}\n\ntable Opportunity {\n    opportunity_id: str;\n    lead_id: str?;\n    account_id: str;\n    name: str;\n    stage: str = "prospecting";\n    amount: float;\n    probability: float = 0.0;\n    expected_close: str;\n    owner_id: str;\n    competitor: str?;\n    loss_reason: str?;\n}\n\ncapability LeadManagement {\n    contract: {\n        id: lead_management,\n        provides: [lead_capture, lead_scoring, lead_conversion],\n        configuration: {tenant_id: "default", lead_timeout_days: 30},\n        rules: [\n            {name: "duplicate_check",  when: "email_exists == true",  action: require_review},\n            {name: "score_threshold",  when: "score >= 75",           action: warn},\n            {name: "stale_lead",       when: "days_since_contact > 30", action: warn}\n        ],\n        ui: {shell: python, routes: [\n            {name: "Leads", path: "/crm/leads", component: "LeadList", permission: "crm:leads"}\n        ]},\n        theme: {name: lead_theme, tokens: {accent: "#00838F"}}\n    };\n}\n\ncapability OpportunityPipeline {\n    contract: {\n        id: opportunity_pipeline,\n        provides: [deal_tracking, pipeline_analytics, forecast],\n        requires: [lead_capture],\n        configuration: {\n            tenant_id: "default",\n            max_discount_pct: 30\n        },\n        rules: [\n            {name: "large_deal_review",  when: "amount > 500000",        action: require_review},\n            {name: "discount_limit",     when: "discount_pct > 30",      action: deny},\n            {name: "stage_progression",  when: "next_stage missing",     action: warn}\n        ],\n        ui: {shell: python, routes: [\n            {name: "Pipeline",   path: "/crm/pipeline",   component: "PipelineView",   permission: "crm:pipeline"},\n            {name: "Forecast",   path: "/crm/forecast",   component: "ForecastView",   permission: "crm:forecast"},\n            {name: "Analytics",  path: "/crm/analytics",  component: "SalesAnalytics", permission: "crm:analytics"}\n        ]},\n        theme: {name: pipeline_theme, tokens: {accent: "#E65100"}}\n    };\n\n    screens: {\n        PipelineKanban: {\n            route: "/crm/kanban",\n            title: "Pipeline Board",\n            layout: grid,\n            contains: [StageColumn, DealCard],\n            binds: [opportunities.by_stage],\n            actions: [move_stage, create_deal, filter]\n        }\n    };\n}\n\napp CRMSalesPipeline {\n    description: "CRM sales pipeline with lead and opportunity management";\n    capabilities: [LeadManagement, OpportunityPipeline];\n    routes: ["/crm", "/crm/leads", "/crm/pipeline", "/crm/forecast"];\n\n    components: {\n        lead_desk:        {capability: lead_capture,       route: "/crm/leads"},\n        deal_pipeline:    {capability: deal_tracking,      route: "/crm/pipeline"},\n        forecast_console: {capability: forecast,           route: "/crm/forecast"},\n        analytics_hub:    {capability: pipeline_analytics, route: "/crm/analytics"}\n    };\n\n    screens: {\n        SalesDashboard: {\n            route: "/crm",\n            title: "Sales Dashboard",\n            layout: dashboard,\n            contains: [PipelineSummary, LeadFunnel, ForecastWidget, RecentActivity],\n            binds: [pipeline.summary, leads.recent, forecast.current],\n            actions: [create_lead, refresh, export]\n        }\n    };\n\n    theme: {name: crm_theme, tokens: {accent: "#FF6D00", "border.radius": "4px"}};\n    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};\n}\n'
APG_MODULE_NAME = 'crm_sales'
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
