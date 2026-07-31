"""Pytest fixtures for the generated APG application."""

from __future__ import annotations

import pytest

from compiler.compiler import APGCompiler


APG_SOURCE = '// Example 07: agents with multiple runtimes including ollama and pi.\n// Features: codex, claude_code, ollama, opencode, and pi runtimes;\n//           conditional team flow; local open-weight model\n\nmodule multi_runtime_team version 1.0.0 {\n    description: "Diverse agent runtimes with model diversity and local fallback";\n}\n\n// Cloud-primary agent for deep technical analysis\nagent CloudAnalyst {\n    role: "cloud analyst";\n    model: "openai:gpt-4.1";\n    runtime: codex;\n    system: "Perform deep technical analysis. Use the highest-capability model available.";\n    tools: [data.query, reports.generate];\n    configuration: {temperature: 0.1, max_turns: 10};\n}\n\n// Privacy-first agent running on local Ollama\nagent LocalPrivacyAgent {\n    role: "privacy-safe local analyst";\n    model: "ollama:llama3.3";\n    runtime: ollama;\n    system: "Analyse data locally. No data leaves this environment. Use Llama for privacy-sensitive workloads.";\n    tools: [local_data.query, local_reports.generate];\n    memory: vector local_memory;\n    configuration: {temperature: 0.2, max_turns: 6};\n}\n\n// Conversational agent using Pi runtime\nagent ConversationalAgent {\n    role: "customer conversation handler";\n    model: "openai:gpt-4.1-mini";\n    runtime: pi;\n    system: "Handle warm, empathetic customer conversations. Keep responses concise and human.";\n    tools: [customer.profile, conversation.history];\n    configuration: {temperature: 0.7, max_turns: 20};\n}\n\n// Open-source agent via opencode runtime\nagent OpenSourceCoder {\n    role: "open source code generator";\n    model: "ollama:mistral";\n    runtime: opencode;\n    system: "Generate code using open-weight models. Prefer Apache/MIT licensed outputs.";\n    tools: [code.search, docs.lookup];\n    configuration: {temperature: 0.0, max_turns: 8};\n}\n\nagent_team AnalyticsCrew {\n    agents: [CloudAnalyst, LocalPrivacyAgent];\n    flow: CloudAnalyst -> LocalPrivacyAgent [condition: requires_privacy];\n    capabilities: [analytics_reports];\n    configuration: {handoff_mode: conditional};\n    rules: [\n        {name: "privacy_routing", when: "data_sensitivity == high", action: require_review}\n    ];\n}\n\napp MultiRuntimeApp {\n    description: "Multi-runtime agent platform with model diversity";\n    agents: [ConversationalAgent, OpenSourceCoder];\n    agent_teams: [AnalyticsCrew];\n    routes: ["/analytics", "/conversations", "/codegen"];\n}\n'
APG_MODULE_NAME = 'multi_runtime_team'
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
