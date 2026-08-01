"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// ============================================================================\n// CRM Platform — APG Capability Composition Example\n//\n// Composability pattern: Hub-and-Spoke\n//   auth, audl, ntfy, wflo underpin CRMCore\n//   SalesAssistant AI agent augments the pipeline\n//\n// Copyright (c) 2025 Datacraft — Nyimbi Odero\n// ============================================================================\n\nmodule crm_platform version 1.0.0 {\n    description: "Composable CRM platform";\n    author: "Datacraft";\n}\n\ntable Contact {\n    contact_number: str;\n    first_name: str;\n    last_name: str;\n    email: str;\n    phone: str;\n    company: str;\n    status: str;\n    owner_id: str;\n}\n\ntable Account {\n    account_number: str;\n    legal_name: str;\n    industry: str;\n    tier: str;\n    health_score: float;\n    owner_id: str;\n}\n\ntable Opportunity {\n    opportunity_number: str;\n    account_id: str;\n    name: str;\n    stage: str;\n    amount: decimal;\n    probability: float;\n    expected_close: datetime;\n    owner_id: str;\n}\n\ncapability CRMCore {\n    contract: {\n        id: crm_platform_core,\n        provides: [contact_lifecycle, account_management, opportunity_pipeline, sales_analytics],\n        requires: [auth, audl, ntfy, wflo],\n        configuration: {\n            tenant_id: "default",\n            pipeline_stages: ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]\n        },\n        rules: [\n            {name: "large_deal_requires_approval", when: "amount > 50000", action: require_review},\n            {name: "discount_cap", when: "discount_pct > 25", action: deny},\n            {name: "cross_tenant_denied", when: "contact_tenant != actor_tenant", action: deny}\n        ],\n        ui: {shell: python, routes: [\n            {name: "Dashboard",   path: "/crm",          component: "CRMDashboard",   permission: "crm:view"},\n            {name: "Contacts",    path: "/crm/contacts", component: "ContactList",     permission: "crm:contacts"},\n            {name: "Pipeline",    path: "/crm/pipeline", component: "Pipeline",        permission: "crm:pipeline"},\n            {name: "Analytics",   path: "/crm/analytics",component: "SalesAnalytics",  permission: "crm:analytics"}\n        ]},\n        theme: {name: crm_theme, tokens: {\n            "color.primary": "#1565C0",\n            "color.accent":  "#FF6D00",\n            "border.radius": "6px"\n        }, components: {opportunities: {icon: "target", status_indicator: "stage-chip"}}}\n    };\n    streaming: {processor: bytewax, state: crm_event_state};\n}\n\nagent SalesAssistant {\n    role: "sales assistant";\n    model: "openai:gpt-4.1-mini";\n    system: "Analyse CRM context and suggest next best actions, talking points, and risk factors.";\n    capabilities: [contact_lifecycle, opportunity_pipeline];\n    tools: [contact_search, deal_analysis];\n    memory: vector sales_memory;\n    configuration: {temperature: 0.2, max_turns: 6};\n}\n\nworkflow LeadQualification {\n    steps: str = "new_lead -> researched -> contacted -> qualified -> opportunity_created";\n    human_tasks: [contacted, qualified];\n    assignments: {contacted: sales_rep, qualified: sales_manager};\n    guards: {qualified: "budget_confirmed and timeline_defined"};\n}\n\nworkflow DealApproval {\n    steps: str = "submitted -> manager_review -> finance_review -> approved";\n    human_tasks: [manager_review, finance_review];\n    assignments: {manager_review: sales_manager, finance_review: finance_controller};\n    guards: {finance_review: "amount > 100000"};\n}\n\napp CRMPlatform {\n    description: "Enterprise CRM composed from APG capabilities";\n    capabilities: [CRMCore];\n    agents: [SalesAssistant];\n    routes: ["/crm", "/crm/contacts", "/crm/accounts", "/crm/pipeline"];\n    theme: {name: crm_platform_theme, tokens: {"accent": "#FF6D00", "border.radius": "6px"}};\n    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};\n    deployments: {default: local, container: docker};\n}\n'
APG_MODULE_NAME = 'crm_platform'
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
