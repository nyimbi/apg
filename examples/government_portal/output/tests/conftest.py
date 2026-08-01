"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// ============================================================================\n// Government Digital Portal — APG Capability Composition Example\n//\n// Composability pattern: Hub-and-Spoke + Pipeline\n//   CitizenServices is the citizen-facing hub\n//   Tax, Permits, CaseManagement, Electoral feed into the hub\n//   Notifications and Audit underpin all operations\n//\n// Copyright (c) 2025 Datacraft — Nyimbi Odero\n// ============================================================================\n\nmodule government_portal version 1.0.0 {\n    description: "Digital government portal — citizen services, tax, permits, case management";\n    author: "Datacraft";\n}\n\ntable Citizen {\n    national_id: str;\n    full_name: str;\n    date_of_birth: datetime;\n    phone: str;\n    email: str;\n    county: str;\n    registration_status: str;  // pending | verified | active\n}\n\ntable ServiceApplication {\n    application_number: str;\n    citizen_id: str;\n    service_type: str;   // tax_registration | business_permit | id_renewal | birth_certificate\n    status: str;         // submitted | under_review | approved | rejected | completed\n    fee_amount: decimal;\n    paid: bool;\n    submitted_at: datetime;\n}\n\ncapability CitizenServicesHub {\n    contract: {\n        id: government_portal_core,\n        provides: [\n            citizen_registration,\n            service_application_lifecycle,\n            payment_processing,\n            status_tracking,\n            digital_notification,\n            multi_channel_access\n        ],\n        requires: [auth, audl, ntfy, wflo, schd],\n        configuration: {\n            tenant_id: "default",\n            supported_channels: ["web", "mobile", "ussd", "sms"],\n            require_id_verification: true,\n            self_service_enabled: true,\n            payment_providers: ["mpesa", "airtel_money", "bank_transfer"],\n            sla_days_service_delivery: 5\n        },\n        rule_engine: {\n            type: deterministic,\n            default_decision: allow,\n            rules: [\n                {name: "id_verification_required",\n                 when: "service_requires_verification == true and id_verified != true",\n                 action: deny},\n                {name: "fee_payment_required",\n                 when: "fee_amount > 0 and payment_confirmed != true",\n                 action: deny},\n                {name: "duplicate_application_blocked",\n                 when: "duplicate_application_exists == true and grace_period_expired == true",\n                 action: deny},\n                {name: "cross_county_service_escalation",\n                 when: "applicant_county != service_county",\n                 action: require_review}\n            ]\n        },\n        ui: {shell: python, routes: [\n            {name: "Citizen Portal",    path: "/gov",              component: "CitizenHome",    permission: "gov:public"},\n            {name: "My Applications",   path: "/gov/applications", component: "ApplicationList",permission: "gov:citizen"},\n            {name: "Apply for Service", path: "/gov/apply",        component: "ServiceApply",   permission: "gov:citizen"},\n            {name: "Pay Fees",          path: "/gov/pay",          component: "FeePayment",     permission: "gov:citizen"},\n            {name: "Track Status",      path: "/gov/track",        component: "StatusTracker",  permission: "gov:citizen"},\n            {name: "Tax Portal",        path: "/gov/tax",          component: "TaxPortal",      permission: "gov:citizen"},\n            {name: "Permits",           path: "/gov/permits",      component: "PermitPortal",   permission: "gov:citizen"},\n            {name: "Officer Dashboard", path: "/gov/officer",      component: "OfficerDesk",    permission: "gov:officer"}\n        ]},\n        theme: {name: government_theme, tokens: {\n            "color.primary":   "#006600",\n            "color.accent":    "#CC0000",\n            "color.success":   "#004D00",\n            "color.warning":   "#FF6600",\n            "color.danger":    "#990000",\n            "surface.canvas":  "#F0F5F0",\n            "surface.panel":   "#FFFFFF",\n            "text.primary":    "#1A2B1A",\n            "border.radius":   "4px",\n            "density":         "comfortable"\n        }, components: {\n            applications: {icon: "file-text",      status_indicator: "application-status-chip"},\n            payments:     {icon: "credit-card",    status_indicator: "payment-status-chip"},\n            services:     {icon: "building-2",     status_indicator: "service-type-chip"}\n        }}\n    };\n    streaming: {processor: bytewax, state: citizen_services_event_state};\n}\n\nagent CitizenAssistant {\n    role: "digital government assistant";\n    model: "openai:gpt-4.1-mini";\n    system: "You assist citizens with government services in Kenya. Speak in simple, clear language. Support Swahili and English. Guide citizens through service applications, explain requirements, and provide status updates.";\n    capabilities: [citizen_registration, service_application_lifecycle];\n    tools: [service_catalogue_search, application_status_query, fee_calculator, document_checklist];\n    memory: vector citizen_help_memory;\n    configuration: {temperature: 0.3, max_turns: 8};\n    rules: [\n        {name: "no_personal_data_collection", when: "requests_personal_data", action: deny}\n    ];\n}\n\nworkflow CitizenServiceApplication {\n    steps: str = "submitted -> id_verified -> fee_calculated -> fee_paid -> under_review -> decision_made -> completed";\n    human_tasks: [id_verified, under_review, decision_made];\n    assignments: {id_verified: verification_officer, under_review: service_officer, decision_made: service_manager};\n    guards: {\n        fee_paid: "fee_amount == 0 or payment_confirmed",\n        decision_made: "supporting_documents_complete and review_notes_provided"\n    };\n    waits: {completed: service_delivery_confirmation};\n    retry_policy: {under_review: {attempts: 3, interval: 24hour}};\n}\n\nworkflow TaxRegistrationPipeline {\n    steps: str = "application -> kra_pin_check -> biometric_capture -> registration -> pin_issued";\n    human_tasks: [kra_pin_check, biometric_capture];\n    assignments: {kra_pin_check: tax_officer, biometric_capture: biometrics_officer};\n    guards: {\n        kra_pin_check: "national_id_valid and not_blacklisted",\n        pin_issued: "biometric_captured and tax_type_selected"\n    };\n}\n\napp GovernmentPortal {\n    description: "Digital government services portal composed from APG capabilities";\n    capabilities: [CitizenServicesHub];\n    agents: [CitizenAssistant];\n    routes: ["/gov", "/gov/applications", "/gov/tax", "/gov/permits"];\n    theme: {name: gov_portal_theme, tokens: {"accent": "#CC0000", "border.radius": "4px"}};\n    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};\n    deployments: {default: local, container: docker};\n}\n'
APG_MODULE_NAME = 'government_portal'
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
