"""Focused compiler baseline regressions for documented APG invocation."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from click.testing import CliRunner

from cli.main import cli
from compiler.code_generator import CodeGenerator
from compiler.compiler import APGCompiler, compile_apg_string
from compiler.ide_integration import audit_vscode_extension
from compiler.semantic_analyzer import SemanticError
from compiler.studio import build_studio_edit_plan, build_studio_snapshot
from language_server.semantic_service import (
	build_code_action_report,
	build_rename_plan,
	build_language_service_snapshot,
	code_actions,
	definition,
	hover,
	references,
	workspace_symbols,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL_AGENT_SOURCE = """
module baseline version 1.0.0 {
	description: "Compiler baseline";
}

agent Planner {
	role: "planner";
	model: "openai:gpt-4.1-mini";
	runtime: codex;
	system: "Plan concrete work.";
}
"""

DATA_APP_SOURCE = """
module customer_ops version 1.0.0 {
	description: "Customer operations";
}

table Customer {
	name: str;
	email: str;
}
"""

TYPED_DATA_APP_SOURCE = """
module inventory_ops version 1.0.0 {
	description: "Inventory operations";
}

table InventoryItem {
	name: str;
	quantity: int;
	active: bool;
}
"""

RELATIONSHIP_APP_SOURCE = """
module sales_ops version 1.0.0 {
	description: "Sales operations";
}

table Customer {
	name: str;
}

table SalesOrder {
	customer_id: int;
	customer: Customer;
	total: float;
}
"""

MIGRATION_PREVIOUS_SOURCE = """
module migration_previous version 1.0.0 {
	description: "Migration previous";
}

table Customer {
	name: str;
	age: int;
	legacy_code: bool;
}

capability CustomerMaster {
	contract: {
		id: customer_master,
		provides: [customer_master],
		configuration: {owned_tables: [Customer]}
	};
}
"""

MIGRATION_CURRENT_SOURCE = """
module migration_current version 1.0.0 {
	description: "Migration current";
}

table Customer {
	name: str;
	email: str;
	age: float;
}

table Invoice {
	customer_id: int;
	total: float;
}

capability Billing {
	contract: {
		id: billing,
		provides: [invoice_generation],
		configuration: {owned_tables: [Customer, Invoice]}
	};
}
"""

CAPABILITY_CATALOG_SOURCE = """
module capability_basics version 1.0.0 {
	description: "Small capability contract with a generated runtime manifest";
}

capability AuditLog {
	contract: {
		id: audit_log,
		provides: [audit_events],
		configuration: {retention_days: 365}
	};
	master_data: {entities: [audit_event]};
}
"""

DATABASE_BACKED_FORM_SOURCE = """
module database_backed_form version 1.0.0 {
	description: "Database-backed form";
}

table Customer {
	name: str;
	email: str;
}

form CustomerForm {
	name: str;
	email: str;
}
"""

INVALID_DATABASE_BACKED_FORM_SOURCE = """
module invalid_database_backed_form version 1.0.0 {
	description: "Database-backed form with bad binding";
}

table Customer {
	name: str;
}

form CustomerForm {
	name: str;
	phone: str;
}
"""


def write_capability_catalog(root: Path, capability_id: str = "customer_master") -> Path:
	catalog_dir = root / "capabilities" / capability_id
	catalog_dir.mkdir(parents=True)
	(catalog_dir / "capability_contract.py").write_text(
		f'''
def get_capability_contract(tenant_id="default"):
	return {{
		"capability": "{capability_id}",
		"display_name": "{capability_id.replace("_", " ").title()}",
		"configuration": {{"tenant_id": tenant_id, "ui": {{}}, "theme": {{}}}},
		"configuration_schema": {{
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {{"tenant_id": {{"type": "string"}}, "ui": {{"type": "object"}}, "theme": {{"type": "object"}}}},
		}},
		"rule_engine": {{
			"type": "deterministic",
			"rules": [
				{{"name": "allow_default", "condition": {{"tenant_context_present": True}}, "effect": {{"decision": "allow"}}}},
			],
		}},
		"ui": {{
			"requires_theme": True,
			"shell": "apg_python",
			"template_roots": ["templates/"],
			"routes": [
				{{"name": "dashboard", "path": "/{capability_id}/dashboard", "component": "Dashboard", "permission": "{capability_id}:view"}},
			],
		}},
		"theme": {{
			"name": "{capability_id}_theme",
			"tokens": {{"border.radius": "8px"}},
			"components": {{"dashboard": {{"density": "compact"}}}},
		}},
	}}
''',
		encoding="utf-8",
	)
	return root


def write_local_capability_catalog(path: Path, capability_id: str = "audit_log") -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				"format": "apg.capability-catalog.v1",
				"capabilities": {
					capability_id: {
						"capability": capability_id,
						"package": capability_id,
						"version": "1.0.0",
						"profile": "python",
						"package_dir": f"capabilities/common/{capability_id}",
						"provides": ["audit_events"] if capability_id == "audit_log" else [capability_id],
						"requires": [],
						"configuration": {"tenant_scoped": True},
						"rules": [
							{"name": "allow_default", "when": "tenant_context_present", "action": "allow"}
						],
						"rule_engine": {"type": "deterministic"},
						"ui": {"routes": [{"name": "dashboard", "path": f"/{capability_id}"}]},
						"screens": {},
						"theme": {"name": f"{capability_id}_theme"},
						"streaming": {},
						"release_evidence": "release_report.json",
						"manifest": "package_manifest.json",
						"entrypoint": "app.py",
					}
				},
			},
			indent=2,
			sort_keys=True,
		)
		+ "\n",
		encoding="utf-8",
	)
	return path


def test_documented_python_target_generates_executable_application_files():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)

	assert result.success is True
	assert result.target_language == "python"
	assert "app.py" in result.generated_files
	assert "ai_agents.py" in result.generated_files
	assert "semantic_model.json" in result.generated_files
	assert ".dockerignore" in result.generated_files
	assert ".env.example" in result.generated_files
	assert "Dockerfile" in result.generated_files
	assert "README.md" in result.generated_files
	assert "smoke_test.py" in result.generated_files
	app = result.generated_files["app.py"]
	dockerfile = result.generated_files["Dockerfile"]
	env_example = result.generated_files[".env.example"]
	readme = result.generated_files["README.md"]
	smoke_test = result.generated_files["smoke_test.py"]
	semantic_artifact = json.loads(result.generated_files["semantic_model.json"])
	assert "APG Python Application" in app
	assert "Flask-AppBuilder" not in app
	assert "flask_appbuilder" not in app
	assert "django" not in app.lower()
	assert "HTTPServer" in app
	assert "run_server" in app
	assert "openapi_document" in app
	assert "python:3.11-slim" in dockerfile
	assert "python app.py --self-test" in dockerfile
	assert "APG_PORT=8080" in env_example
	assert "python app.py --self-test" in readme
	assert "python smoke_test.py" in readme
	assert "GET /component.json" in readme
	assert "GET /semantic-model.json" in readme
	assert "## Browser UI" in readme
	assert "create, edit, delete, and validation-error flows" in readme
	assert "_revision` checks" in readme
	assert "docker build -t apg-generated-app ." in readme
	assert "POST /agents/Planner/invoke" in readme
	assert "component_manifest" in smoke_test
	assert "openapi_contract" in smoke_test
	assert "route_dispatch" in smoke_test
	assert "capability_health" in smoke_test
	assert semantic_artifact["format"] == "apg.semantic-model.v1"
	assert semantic_artifact["agents"]["Planner"]["runtime"] == "codex"
	compile(app, "app.py", "exec")
	compile(smoke_test, "smoke_test.py", "exec")


def test_generated_capability_runtime_has_no_bare_pass_stubs():
	result = compile_apg_string(CAPABILITY_CATALOG_SOURCE)

	assert result.success is True
	capability_runtime = result.generated_files["apg_capabilities.py"]
	assert not [
		line_number
		for line_number, line in enumerate(capability_runtime.splitlines(), start=1)
		if line.strip() == "pass"
	]
	compile(capability_runtime, "apg_capabilities.py", "exec")


def test_generated_python_package_is_importable_with_runtime_manifests(tmp_path):
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	package_dir = tmp_path / "generated_pkg"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	spec = importlib.util.spec_from_file_location(
		"generated_pkg",
		package_dir / "__init__.py",
		submodule_search_locations=[str(package_dir)],
	)
	module = importlib.util.module_from_spec(spec)
	sys.modules["generated_pkg"] = module
	try:
		spec.loader.exec_module(module)
		manifest = module.describe_application()
		component = module.component_manifest()
		semantic_model = module.semantic_model()
		invocation = module.invoke_agent("Planner", {"input": {"task": "plan"}})
		self_test = module.self_test()
		openapi_contract = module.validate_openapi_contract()
		component_contract = module.validate_component_manifest_contract()
		route_contract = module.validate_route_dispatch_contract()
	finally:
		sys.modules.pop("generated_pkg", None)
		for name in list(sys.modules):
			if name.startswith("generated_pkg."):
				sys.modules.pop(name, None)

	assert module.__version__ == "1.0.0"
	assert module.list_entities() == [
		{"name": "Planner", "type": "ai_agent", "properties": [], "fields": [], "methods": []}
	]
	assert module.list_events() == []
	assert module.list_records("Planner") == []
	assert module.list_agents() == ["Planner"]
	assert manifest["ai_agents"] == ["Planner"]
	assert component["kind"] == "apg.application"
	assert component["composable"] is True
	assert "/component.json" in component["interfaces"]["http"]["paths"]
	assert "/semantic-model.json" in component["interfaces"]["http"]["paths"]
	assert component["interfaces"]["semantic_model"] == "/semantic-model.json"
	assert "component_manifest" in component["interfaces"]["python"]["exports"]
	assert "semantic_model" in component["interfaces"]["python"]["exports"]
	assert "auth_status" in component["interfaces"]["python"]["exports"]
	assert "create_record" in component["interfaces"]["python"]["exports"]
	assert "database_status" in component["interfaces"]["python"]["exports"]
	assert "list_entities" in component["interfaces"]["python"]["exports"]
	assert "list_agents" in component["interfaces"]["python"]["exports"]
	assert "invoke_agent" in component["interfaces"]["python"]["exports"]
	assert "query_records" in component["interfaces"]["python"]["exports"]
	assert "runtime_adapter_command_candidates" in component["interfaces"]["python"]["exports"]
	assert "runtime_adapter_environment_keys" in component["interfaces"]["python"]["exports"]
	assert "update_record" in component["interfaces"]["python"]["exports"]
	assert "delete_record" in component["interfaces"]["python"]["exports"]
	assert "validate_agent_runtimes" in component["interfaces"]["python"]["exports"]
	assert "validate_component_manifest_contract" in component["interfaces"]["python"]["exports"]
	assert "validate_route_dispatch_contract" in component["interfaces"]["python"]["exports"]
	assert invocation["agent"] == "Planner"
	assert invocation["runtime"] == "codex"
	assert invocation["status"] == "adapter_required"
	assert invocation["mode"] == "adapter_missing"
	assert "adapter command" in invocation["output"]["message"]
	assert invocation["output"]["adapter_command_candidates"] == [["apg-agent-codex"]]
	assert invocation["input"] == {"task": "plan"}
	assert self_test["passed"] is True
	assert "/self-test" in self_test["routes"]
	assert self_test["checks"]["entity_count"] == 1
	assert self_test["checks"]["validation"]["checks"]["openapi_contract"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["component_manifest"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["route_dispatch"]["errors"] == []
	assert semantic_model["format"] == "apg.semantic-model.v1"
	assert semantic_model["agents"]["Planner"]["runtime"] == "codex"
	assert openapi_contract["errors"] == []
	assert "AgentInvocationRequest" in openapi_contract["referenced_schemas"]
	assert component_contract["errors"] == []
	assert component_contract["artifact_count"] >= 9
	assert "validate_component_manifest_contract" in component_contract["python_exports"]
	assert "semantic_model" in component_contract["python_exports"]
	assert route_contract["errors"] == []
	assert route_contract["routes"]["/semantic-model.json"][0]["target"] == "_route_payload"
	assert route_contract["routes"]["/agents/Planner/invoke"][0]["target"] == "_agent_invocation_payload"
	assert {"method": "GET", "target": "_records_payload_with_query"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert {"method": "PUT", "target": "_update_record_payload"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert {"method": "DELETE", "target": "_delete_record_payload"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert module.auth_status()["mode"] == "open"
	assert module.metrics_snapshot()["entity_count"] == 1
	openapi = module.openapi_document()
	assert openapi["openapi"] == "3.1.0"
	assert "SemanticModel" in openapi["components"]["schemas"]
	assert openapi["components"]["schemas"]["SelfTestReport"]["properties"]["checks"] == {
		"$ref": "#/components/schemas/SelfTestChecks"
	}
	assert openapi["components"]["schemas"]["SelfTestChecks"]["properties"]["validation"] == {
		"$ref": "#/components/schemas/ValidationReport"
	}
	assert openapi["components"]["schemas"]["SelfTestChecks"]["properties"]["metrics"] == {
		"$ref": "#/components/schemas/MetricsSnapshot"
	}
	assert openapi["components"]["schemas"]["SelfTestChecks"]["properties"]["capability_health"] == {
		"$ref": "#/components/schemas/CapabilityHealthReport"
	}
	assert module.relationship_graph()["nodes"][0]["id"] == "Planner"
	assert module.storage_status()["mode"] == "memory"
	assert module.coerce_record_types("Planner", {"x": "1"}) == {"x": "1"}
	assert module.validate_record("Planner", {})["valid"] is True
	assert module.validate_application()["valid"] is True
	assert "auth_status" in module.__all__
	assert "coerce_record_types" in module.__all__
	assert "component_manifest" in module.__all__
	assert "create_record" in module.__all__
	assert "delete_record" in module.__all__
	assert "describe_application" in module.__all__
	assert "get_record" in module.__all__
	assert "list_events" in module.__all__
	assert "invoke_agent" in module.__all__
	assert "metrics_snapshot" in module.__all__
	assert "openapi_document" in module.__all__
	assert "query_records" in module.__all__
	assert "relationship_graph" in module.__all__
	assert "self_test" in module.__all__
	assert "semantic_model" in module.__all__
	assert "storage_status" in module.__all__
	assert "update_record" in module.__all__
	assert "validate_application" in module.__all__
	assert "validate_component_manifest_contract" in module.__all__
	assert "validate_openapi_contract" in module.__all__
	assert "validate_route_dispatch_contract" in module.__all__
	assert "validate_record" in module.__all__
	assert "list_records" in module.__all__
	assert "list_agents" in module.__all__
	assert "runtime_adapter_command_candidates" in module.__all__
	assert "runtime_adapter_environment_keys" in module.__all__


def test_generated_openapi_contract_rejects_required_fields_missing_from_schema():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	assert result.success is True

	namespace: dict[str, object] = {"__name__": "generated_app"}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	original_openapi_document = namespace["openapi_document"]

	def broken_openapi_document():
		document = copy.deepcopy(original_openapi_document())
		document["components"]["schemas"]["SelfTestChecks"]["required"].append("missing_field")
		return document

	namespace["openapi_document"] = broken_openapi_document
	contract = namespace["validate_openapi_contract"]()

	assert contract["errors"] == [
		"OpenAPI schema SelfTestChecks requires missing property missing_field"
	]


def test_generated_component_manifest_contract_rejects_invalid_deployment_commands():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	assert result.success is True

	namespace: dict[str, object] = {"__name__": "generated_app"}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	original_component_manifest = namespace["component_manifest"]

	def broken_component_manifest():
		manifest = copy.deepcopy(original_component_manifest())
		manifest["deployment"]["commands"]["self_test"] = "python app.py"
		manifest["deployment"]["environment"].remove("APG_DEBUG")
		return manifest

	namespace["component_manifest"] = broken_component_manifest
	contract = namespace["validate_component_manifest_contract"]()

	assert "component manifest deployment command self_test must be 'python app.py --self-test'" in contract["errors"]
	assert "component manifest deployment environment does not match generated runtime variables" in contract["errors"]


def test_generated_component_manifest_contract_rejects_missing_artifact_files(tmp_path):
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	assert result.success is True

	package_dir = tmp_path / "generated_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")
	(package_dir / "README.md").unlink()

	spec = importlib.util.spec_from_file_location("generated_app", package_dir / "app.py")
	module = importlib.util.module_from_spec(spec)
	try:
		spec.loader.exec_module(module)
		contract = module.validate_component_manifest_contract()
	finally:
		sys.modules.pop("generated_app", None)

	assert "component manifest deployment artifact README.md does not exist" in contract["errors"]


def test_generated_component_manifest_contract_rejects_unexpected_artifacts():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	assert result.success is True

	namespace: dict[str, object] = {"__name__": "generated_app"}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)
	original_component_manifest = namespace["component_manifest"]

	def broken_component_manifest():
		manifest = copy.deepcopy(original_component_manifest())
		manifest["deployment"]["artifacts"].extend(["legacy_views.py", 42])
		return manifest

	namespace["component_manifest"] = broken_component_manifest
	contract = namespace["validate_component_manifest_contract"]()

	assert "component manifest deployment has unexpected artifact legacy_views.py" in contract["errors"]
	assert "component manifest deployment artifacts must be strings" in contract["errors"]


def test_generated_python_app_serves_http_endpoints(tmp_path):
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)
	package_dir = tmp_path / "generated_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		port = sock.getsockname()[1]

	process = subprocess.Popen(
		[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
		cwd=package_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	try:
		base_url = f"http://127.0.0.1:{port}"
		for _attempt in range(30):
			try:
				with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
					health = json.loads(response.read().decode("utf-8"))
				break
			except OSError:
				if process.poll() is not None:
					stdout, stderr = process.communicate(timeout=1)
					raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
				time.sleep(0.05)
		else:
			raise AssertionError("generated app did not answer /health")

		with urllib.request.urlopen(f"{base_url}/manifest", timeout=1) as response:
			manifest = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/component.json", timeout=1) as response:
			component = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/semantic-model.json", timeout=1) as response:
			semantic_model = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/agents", timeout=1) as response:
			agents = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/validate", timeout=1) as response:
			validation = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=1) as response:
			openapi = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/self-test", timeout=1) as response:
			self_test = json.loads(response.read().decode("utf-8"))
		request = urllib.request.Request(
			f"{base_url}/agents/Planner/invoke",
			data=json.dumps({"input": {"ticket": "reset password"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			invocation = json.loads(response.read().decode("utf-8"))
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert health["status"] == "ok"
	assert health["auth"]["mode"] == "open"
	assert manifest["name"] == "baseline"
	assert component["kind"] == "apg.application"
	assert component["deployment"]["commands"]["self_test"] == "python app.py --self-test"
	assert component["deployment"]["commands"]["semantic_model"] == "python app.py --semantic-model"
	assert "/agents/Planner/invoke" in component["interfaces"]["http"]["paths"]
	assert "/semantic-model.json" in component["interfaces"]["http"]["paths"]
	assert "/openapi.json" in component["interfaces"]["http"]["paths"]
	assert semantic_model["format"] == "apg.semantic-model.v1"
	assert semantic_model["agents"]["Planner"]["runtime"] == "codex"
	assert agents["agents"]["Planner"]["runtime"] == "codex"
	assert invocation["agent"] == "Planner"
	assert invocation["status"] == "adapter_required"
	assert invocation["mode"] == "adapter_missing"
	assert invocation["input"] == {"ticket": "reset password"}
	assert "/agents/Planner/invoke" in openapi["paths"]
	assert "/component.json" in openapi["paths"]
	assert "/semantic-model.json" in openapi["paths"]
	assert "/self-test" in openapi["paths"]
	assert self_test["passed"] is True
	assert self_test["checks"]["validation"]["checks"]["openapi_contract"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["component_manifest"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["route_dispatch"]["errors"] == []
	assert self_test["checks"]["route_count"] >= 1
	assert validation["valid"] is True


def test_generated_python_app_serves_entity_record_endpoints(tmp_path):
	result = compile_apg_string(DATA_APP_SOURCE)
	package_dir = tmp_path / "generated_data_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		port = sock.getsockname()[1]

	process = subprocess.Popen(
		[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
		cwd=package_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	try:
		base_url = f"http://127.0.0.1:{port}"
		for _attempt in range(30):
			try:
				with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
					json.loads(response.read().decode("utf-8"))
				break
			except OSError:
				if process.poll() is not None:
					stdout, stderr = process.communicate(timeout=1)
					raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
				time.sleep(0.05)
		else:
			raise AssertionError("generated data app did not answer /health")

		request = urllib.request.Request(
			f"{base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Asha", "email": "asha@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			created = json.loads(response.read().decode("utf-8"))

		with urllib.request.urlopen(f"{base_url}/entities/Customer/records", timeout=1) as response:
			listed = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/entities/Customer/records/1", timeout=1) as response:
			fetched = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/records", timeout=1) as response:
			all_records = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/ui", timeout=1) as response:
			ui_content_type = response.headers["Content-Type"]
			ui_index = response.read().decode("utf-8")
		with urllib.request.urlopen(f"{base_url}/ui/entities/Customer", timeout=1) as response:
			entity_ui = response.read().decode("utf-8")
		with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=1) as response:
			openapi_content_type = response.headers["Content-Type"]
			openapi = json.loads(response.read().decode("utf-8"))
		update_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records/1",
			data=json.dumps({"record": {"email": "asha@new.example", "status": "active"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="PUT",
		)
		with urllib.request.urlopen(update_request, timeout=1) as response:
			updated = json.loads(response.read().decode("utf-8"))
		delete_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records/1",
			method="DELETE",
		)
		with urllib.request.urlopen(delete_request, timeout=1) as response:
			deleted = json.loads(response.read().decode("utf-8"))
		try:
			urllib.request.urlopen(f"{base_url}/entities/Customer/records/1", timeout=1)
		except urllib.error.HTTPError as error:
			missing_status = error.code
			missing_payload = json.loads(error.read().decode("utf-8"))
		else:
			raise AssertionError("deleted generated app record was still fetchable")
		with urllib.request.urlopen(f"{base_url}/entities/Customer/records", timeout=1) as response:
			empty_list = json.loads(response.read().decode("utf-8"))
		form_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records",
			data=b"name=Kofi&email=kofi%40example.com",
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			method="POST",
		)
		with urllib.request.urlopen(form_request, timeout=1) as response:
			form_created = json.loads(response.read().decode("utf-8"))
		request = urllib.request.Request(
			f"{base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Amara", "email": "amara@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			third_created = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(
			f"{base_url}/entities/Customer/records?filter.name=Kofi",
			timeout=1,
		) as response:
			filtered = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(
			f"{base_url}/entities/Customer/records?sort=name&order=desc&limit=2&offset=0",
			timeout=1,
		) as response:
			sorted_page = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/entities/Customer/records/export", timeout=1) as response:
			exported = json.loads(response.read().decode("utf-8"))
		import_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records/import",
			data=json.dumps({
				"records": [
					{"name": "Zuri", "email": "zuri@example.com"},
					{"name": "Broken"},
				]
			}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(import_request, timeout=1) as response:
			imported = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/events", timeout=1) as response:
			events = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/metrics", timeout=1) as response:
			metrics = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/theme.css", timeout=1) as response:
			theme_content_type = response.headers["Content-Type"]
			theme_css = response.read().decode("utf-8")
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert created["entity"] == "Customer"
	assert created["record"] == {"id": 1, "_revision": 1, "name": "Asha", "email": "asha@example.com"}
	assert created["count"] == 1
	assert listed["records"] == [created["record"]]
	assert fetched["record"] == created["record"]
	assert all_records["records"]["Customer"] == [created["record"]]
	assert created["event"]["action"] == "create"
	assert created["event"]["record_id"] == 1
	assert ui_content_type.startswith("text/html")
	assert "/ui/entities/Customer" in ui_index
	assert 'href="/theme.css"' in ui_index
	assert "/component.json" in ui_index
	assert "/events" in ui_index
	assert "/metrics" in ui_index
	assert "/self-test" in ui_index
	assert "/openapi.json" in ui_index
	assert 'action="/ui/entities/Customer/records"' in entity_ui
	assert "<pre>" in entity_ui
	assert "asha@example.com" in entity_ui
	assert openapi_content_type.startswith("application/json")
	assert openapi["openapi"] == "3.1.0"
	assert openapi["info"]["title"] == "customer_ops"
	assert "/auth" in openapi["paths"]
	assert "/component.json" in openapi["paths"]
	assert "/events" in openapi["paths"]
	assert "/metrics" in openapi["paths"]
	assert "/self-test" in openapi["paths"]
	assert "/theme.css" in openapi["paths"]
	assert openapi["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/HealthReport"
	}
	assert openapi["paths"]["/validate"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/ValidationReport"
	}
	assert openapi["paths"]["/metrics"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/MetricsSnapshot"
	}
	assert openapi["paths"]["/relationships"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/RelationshipGraph"
	}
	assert openapi["paths"]["/storage"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/StorageStatus"
	}
	assert openapi["paths"]["/records"]["get"]["responses"]["200"]["content"]["application/json"]["schema"] == {
		"$ref": "#/components/schemas/RecordsByEntity"
	}
	assert openapi["components"]["schemas"]["HealthReport"]["properties"]["storage"] == {
		"$ref": "#/components/schemas/StorageStatus"
	}
	assert openapi["components"]["schemas"]["MetricsSnapshot"]["properties"]["database_status"] == {
		"$ref": "#/components/schemas/DatabaseStatus"
	}
	assert theme_content_type.startswith("text/css")
	assert "--apg-accent" in theme_css
	assert "ApiKeyAuth" in openapi["components"]["securitySchemes"]
	assert "/entities/Customer/records" in openapi["paths"]
	assert "/entities/Customer/records/{id}" in openapi["paths"]
	customer_schema = openapi["components"]["schemas"]["CustomerRecord"]
	assert customer_schema["required"] == ["name", "email"]
	assert customer_schema["properties"]["name"]["type"] == "string"
	assert updated["record"] == {
		"id": 1,
		"_revision": 2,
		"name": "Asha",
		"email": "asha@new.example",
		"status": "active",
	}
	assert updated["event"]["action"] == "update"
	assert updated["event"]["before"] == created["record"]
	assert updated["event"]["after"] == updated["record"]
	assert deleted["deleted"] == updated["record"]
	assert deleted["event"]["action"] == "delete"
	assert deleted["event"]["before"] == updated["record"]
	assert deleted["count"] == 0
	assert missing_status == 404
	assert missing_payload["error"] == "record_not_found"
	assert empty_list["records"] == []
	assert form_created["record"] == {"id": 2, "_revision": 1, "name": "Kofi", "email": "kofi@example.com"}
	assert third_created["record"] == {"id": 3, "_revision": 1, "name": "Amara", "email": "amara@example.com"}
	assert filtered["records"] == [form_created["record"]]
	assert filtered["filters"] == {"name": "Kofi"}
	assert filtered["total"] == 1
	assert sorted_page["records"] == [form_created["record"], third_created["record"]]
	assert sorted_page["count"] == 2
	assert sorted_page["total"] == 2
	assert sorted_page["limit"] == 2
	assert sorted_page["sort"] == "name"
	assert sorted_page["order"] == "desc"
	assert exported["records"] == [form_created["record"], third_created["record"]]
	assert exported["count"] == 2
	assert imported["count"] == 1
	assert imported["failed"] == 1
	assert imported["imported"][0] == {"id": 4, "_revision": 1, "name": "Zuri", "email": "zuri@example.com"}
	assert imported["errors"] == [{"index": 1, "errors": ["email is required"]}]
	assert imported["events"][0]["action"] == "import"
	parameters = openapi["paths"]["/entities/Customer/records"]["get"]["parameters"]
	assert {parameter["name"] for parameter in parameters} >= {"filter.<field>", "sort", "order", "limit", "offset"}
	assert "/entities/Customer/records/export" in openapi["paths"]
	assert "/entities/Customer/records/import" in openapi["paths"]
	assert [event["action"] for event in events["events"]] == ["create", "update", "delete", "create", "create", "import"]
	assert events["events"][-1]["after"] == imported["imported"][0]
	assert metrics["entity_count"] == 1
	assert metrics["record_counts"] == {"Customer": 3}
	assert metrics["total_records"] == 3
	assert metrics["event_count"] == 6
	assert metrics["event_counts"] == {"create": 3, "delete": 1, "import": 1, "update": 1}
	assert metrics["auth"]["mode"] == "open"


def test_generated_python_app_exposes_relationship_graph_from_fields(tmp_path):
	result = compile_apg_string(RELATIONSHIP_APP_SOURCE)
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	graph = namespace["relationship_graph"]()
	assert {"id": "Customer", "name": "Customer", "type": "entity"} in graph["nodes"]
	assert {"id": "SalesOrder", "name": "SalesOrder", "type": "entity"} in graph["nodes"]
	assert {
		"from": "SalesOrder",
		"to": "Customer",
		"field": "customer_id",
		"relationship": "references",
	} in graph["edges"]
	assert {
		"from": "SalesOrder",
		"to": "Customer",
		"field": "customer",
		"relationship": "typed_as",
	} in graph["edges"]
	assert "/relationships" in namespace["openapi_document"]()["paths"]
	assert namespace["_route_payload"]("/relationships") == (200, graph)


def test_generated_python_app_validates_records_from_entity_fields():
	result = compile_apg_string(TYPED_DATA_APP_SOURCE)
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	entity = namespace["list_entities"]()[0]
	assert entity["fields"] == [
		{"name": "name", "type": "str", "required": True},
		{"name": "quantity", "type": "int", "required": True},
		{"name": "active", "type": "bool", "required": True},
	]
	assert namespace["openapi_document"]()["components"]["schemas"]["InventoryItemRecord"] == {
		"type": "object",
		"additionalProperties": True,
		"properties": {
			"id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
			"_revision": {"type": "integer"},
			"name": {"type": "string"},
			"quantity": {"type": "integer"},
			"active": {"type": "boolean"},
		},
		"required": ["name", "quantity", "active"],
	}
	openapi = namespace["openapi_document"]()
	assert openapi["components"]["schemas"]["InventoryItemRecordPatch"] == {
		"type": "object",
		"additionalProperties": True,
		"properties": {
			"id": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
			"_revision": {"type": "integer"},
			"name": {"type": "string"},
			"quantity": {"type": "integer"},
			"active": {"type": "boolean"},
		},
	}
	record_collection = openapi["paths"]["/entities/InventoryItem/records"]
	assert record_collection["get"]["responses"]["200"]["content"]["application/json"]["schema"]["properties"]["records"] == {
		"type": "array",
		"items": {"$ref": "#/components/schemas/InventoryItemRecord"},
	}
	assert record_collection["post"]["requestBody"]["content"]["application/json"]["schema"] == {
		"type": "object",
		"additionalProperties": False,
		"properties": {"record": {"$ref": "#/components/schemas/InventoryItemRecord"}},
		"required": ["record"],
	}
	assert record_collection["post"]["responses"]["201"]["content"]["application/json"]["schema"]["properties"]["record"] == {
		"$ref": "#/components/schemas/InventoryItemRecord"
	}
	record_item = openapi["paths"]["/entities/InventoryItem/records/{id}"]
	assert record_item["put"]["requestBody"]["content"]["application/json"]["schema"] == {
		"type": "object",
		"additionalProperties": False,
		"properties": {"record": {"$ref": "#/components/schemas/InventoryItemRecordPatch"}},
		"required": ["record"],
	}
	assert record_item["delete"]["responses"]["200"]["content"]["application/json"]["schema"]["properties"]["deleted"] == {
		"$ref": "#/components/schemas/InventoryItemRecord"
	}
	assert openapi["paths"]["/entities/InventoryItem/records/import"]["post"]["requestBody"]["content"]["application/json"]["schema"] == {
		"type": "object",
		"additionalProperties": False,
		"properties": {
			"records": {
				"type": "array",
				"items": {"$ref": "#/components/schemas/InventoryItemRecord"},
			}
		},
		"required": ["records"],
	}
	assert namespace["coerce_record_types"](
		"InventoryItem",
		{"name": "Widget", "quantity": "5", "active": "true"},
	) == {"name": "Widget", "quantity": 5, "active": True}

	status, invalid = namespace["_post_payload"](
		"/entities/InventoryItem/records",
		{"record": {"name": "Widget", "quantity": "many", "active": True}},
	)
	assert status == 422
	assert invalid["error"] == "record_validation_failed"
	assert invalid["errors"] == ["quantity must be integer"]

	status, missing = namespace["_post_payload"](
		"/entities/InventoryItem/records",
		{"record": {"name": "Widget", "quantity": 3}},
	)
	assert status == 422
	assert missing["errors"] == ["active is required"]

	status, created = namespace["_post_payload"](
		"/entities/InventoryItem/records",
		{"record": {"name": "Widget", "quantity": 3, "active": True}},
	)
	assert status == 201
	assert created["record"] == {"id": 1, "_revision": 1, "name": "Widget", "quantity": 3, "active": True}

	status, invalid_update = namespace["_put_payload"](
		"/entities/InventoryItem/records/1",
		{"record": {"active": "maybe"}},
	)
	assert status == 422
	assert invalid_update["errors"] == ["active must be boolean"]


def test_generated_python_app_exposes_programmatic_record_mutations():
	result = compile_apg_string(TYPED_DATA_APP_SOURCE)
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, created = namespace["create_record"](
		"InventoryItem",
		{"name": "Widget", "quantity": "3", "active": "true"},
	)
	assert status == 201
	assert created["record"] == {"id": 1, "_revision": 1, "name": "Widget", "quantity": 3, "active": True}

	status, fetched = namespace["get_record"]("InventoryItem", 1)
	assert status == 200
	assert fetched["record"] == created["record"]

	queried = namespace["query_records"]("InventoryItem", {"filter.name": ["Widget"]})
	assert queried["records"] == [created["record"]]

	status, updated = namespace["update_record"](
		"InventoryItem",
		1,
		{"quantity": "4", "active": "false"},
		expected_revision=1,
	)
	assert status == 200
	assert updated["record"] == {"id": 1, "_revision": 2, "name": "Widget", "quantity": 4, "active": False}

	status, stale = namespace["update_record"]("InventoryItem", 1, {"quantity": 5}, expected_revision=1)
	assert status == 409
	assert stale["error"] == "revision_conflict"

	status, deleted = namespace["delete_record"]("InventoryItem", 1, expected_revision=2)
	assert status == 200
	assert deleted["deleted"] == updated["record"]


def test_generated_python_app_coerces_typed_form_records(tmp_path):
	result = compile_apg_string(TYPED_DATA_APP_SOURCE)
	package_dir = tmp_path / "generated_typed_form_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		port = sock.getsockname()[1]

	process = subprocess.Popen(
		[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
		cwd=package_dir,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	try:
		base_url = f"http://127.0.0.1:{port}"
		for _attempt in range(30):
			try:
				with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
					json.loads(response.read().decode("utf-8"))
				break
			except OSError:
				if process.poll() is not None:
					stdout, stderr = process.communicate(timeout=1)
					raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
				time.sleep(0.05)
		else:
			raise AssertionError("generated typed form app did not answer /health")

		with urllib.request.urlopen(f"{base_url}/ui/entities/InventoryItem", timeout=1) as response:
			entity_ui = response.read().decode("utf-8")
		invalid_form_data = urllib.parse.urlencode(
			{"name": "Widget", "quantity": "many", "active": "true"}
		).encode("utf-8")
		invalid_request = urllib.request.Request(
			f"{base_url}/ui/entities/InventoryItem/records",
			data=invalid_form_data,
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			method="POST",
		)
		try:
			urllib.request.urlopen(invalid_request, timeout=1)
		except urllib.error.HTTPError as error:
			invalid_status = error.code
			invalid_content_type = error.headers["Content-Type"]
			invalid_ui = error.read().decode("utf-8")
		else:
			raise AssertionError("invalid generated UI form submission unexpectedly succeeded")
		form_data = urllib.parse.urlencode(
			{"name": "Widget", "quantity": "7", "active": "true"}
		).encode("utf-8")
		request = urllib.request.Request(
			f"{base_url}/ui/entities/InventoryItem/records",
			data=form_data,
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			created_ui = response.read().decode("utf-8")
		with urllib.request.urlopen(f"{base_url}/entities/InventoryItem/records/1", timeout=1) as response:
			created = json.loads(response.read().decode("utf-8"))
		update_data = urllib.parse.urlencode(
			{
				"expected_revision": "1",
				"name": "Widget Pro",
				"quantity": "8",
				"active": "false",
			}
		).encode("utf-8")
		update_request = urllib.request.Request(
			f"{base_url}/ui/entities/InventoryItem/records/1",
			data=update_data,
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			method="POST",
		)
		with urllib.request.urlopen(update_request, timeout=1) as response:
			updated_ui = response.read().decode("utf-8")
		with urllib.request.urlopen(f"{base_url}/entities/InventoryItem/records/1", timeout=1) as response:
			updated = json.loads(response.read().decode("utf-8"))
		second_request = urllib.request.Request(
			f"{base_url}/entities/InventoryItem/records",
			data=json.dumps({"record": {"name": "Widget Spare", "quantity": 1, "active": True}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(second_request, timeout=1) as response:
			second_created = json.loads(response.read().decode("utf-8"))
		query_url = (
			f"{base_url}/ui/entities/InventoryItem?"
			+ urllib.parse.urlencode({"filter.name": "Widget Pro", "sort": "quantity", "order": "desc", "limit": "1"})
		)
		with urllib.request.urlopen(query_url, timeout=1) as response:
			queried_content_type = response.headers["Content-Type"]
			queried_ui = response.read().decode("utf-8")
		delete_data = urllib.parse.urlencode({"expected_revision": "2"}).encode("utf-8")
		delete_request = urllib.request.Request(
			f"{base_url}/ui/entities/InventoryItem/records/1/delete",
			data=delete_data,
			headers={"Content-Type": "application/x-www-form-urlencoded"},
			method="POST",
		)
		with urllib.request.urlopen(delete_request, timeout=1) as response:
			deleted_ui = response.read().decode("utf-8")
		with urllib.request.urlopen(f"{base_url}/entities/InventoryItem/records", timeout=1) as response:
			records_after_delete = json.loads(response.read().decode("utf-8"))
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert 'name="quantity" type="number" step="1"' in entity_ui
	assert 'type="hidden" name="active" value="false"' in entity_ui
	assert 'type="checkbox" name="active" value="true"' in entity_ui
	assert 'action="/ui/entities/InventoryItem/records"' in entity_ui
	assert 'name="filter.name"' in entity_ui
	assert "Query records" in entity_ui
	assert invalid_status == 422
	assert invalid_content_type.startswith("text/html")
	assert 'role="alert"' in invalid_ui
	assert "quantity must be integer" in invalid_ui
	assert 'action="/ui/entities/InventoryItem/records"' in invalid_ui
	assert "<table>" in created_ui
	assert "Widget" in created_ui
	assert 'action="/ui/entities/InventoryItem/records/1"' in created_ui
	assert 'action="/ui/entities/InventoryItem/records/1/delete"' in created_ui
	assert created["record"] == {"id": 1, "_revision": 1, "name": "Widget", "quantity": 7, "active": True}
	assert "Widget Pro" in updated_ui
	assert updated["record"] == {"id": 1, "_revision": 2, "name": "Widget Pro", "quantity": 8, "active": False}
	assert queried_content_type.startswith("text/html")
	assert 'value="Widget Pro"' in queried_ui
	assert "Showing 1 of 1 matching records." in queried_ui
	assert "Widget Pro" in queried_ui
	assert "Widget Spare" not in queried_ui
	assert "Widget Pro" not in deleted_ui
	assert "Widget Spare" in deleted_ui
	assert records_after_delete["records"] == [second_created["record"]]


def test_generated_python_app_supports_optimistic_record_revisions():
	result = compile_apg_string(TYPED_DATA_APP_SOURCE)
	namespace: dict[str, object] = {}
	exec(compile(result.generated_files["app.py"], "app.py", "exec"), namespace)

	status, created = namespace["_post_payload"](
		"/entities/InventoryItem/records",
		{"record": {"name": "Widget", "quantity": 3, "active": True}},
	)
	assert status == 201
	assert created["record"]["_revision"] == 1

	status, stale_update = namespace["_put_payload"](
		"/entities/InventoryItem/records/1",
		{"expected_revision": 99, "record": {"quantity": 4}},
	)
	assert status == 409
	assert stale_update["error"] == "revision_conflict"
	assert stale_update["expected_revision"] == 99
	assert stale_update["current_revision"] == 1

	status, updated = namespace["_put_payload"](
		"/entities/InventoryItem/records/1",
		{"expected_revision": 1, "record": {"quantity": 4}},
	)
	assert status == 200
	assert updated["record"]["quantity"] == 4
	assert updated["record"]["_revision"] == 2

	status, stale_delete = namespace["_delete_record_payload"](
		"/entities/InventoryItem/records/1?expected_revision=1"
	)
	assert status == 409
	assert stale_delete["current_revision"] == 2

	status, deleted = namespace["_delete_record_payload"](
		"/entities/InventoryItem/records/1?expected_revision=2"
	)
	assert status == 200
	assert deleted["deleted"]["_revision"] == 2


def test_generated_python_app_persists_records_with_data_file(tmp_path):
	result = compile_apg_string(DATA_APP_SOURCE)
	package_dir = tmp_path / "generated_persistent_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	data_file = tmp_path / "records.json"
	env = dict(os.environ, APG_DATA_FILE=str(data_file))

	def start_app() -> tuple[subprocess.Popen, str]:
		with socket.socket() as sock:
			sock.bind(("127.0.0.1", 0))
			port = sock.getsockname()[1]
		process = subprocess.Popen(
			[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
			cwd=package_dir,
			env=env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)
		base_url = f"http://127.0.0.1:{port}"
		for _attempt in range(30):
			try:
				with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
					health = json.loads(response.read().decode("utf-8"))
				assert health["storage"]["mode"] == "file"
				assert health["storage"]["path"] == str(data_file)
				return process, base_url
			except OSError:
				if process.poll() is not None:
					stdout, stderr = process.communicate(timeout=1)
					raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
				time.sleep(0.05)
		raise AssertionError("generated persistent app did not answer /health")

	def stop_app(process: subprocess.Popen) -> None:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	first_process, first_base_url = start_app()
	try:
		request = urllib.request.Request(
			f"{first_base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Asha", "email": "asha@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			created = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{first_base_url}/storage", timeout=1) as response:
			storage = json.loads(response.read().decode("utf-8"))
	finally:
		stop_app(first_process)

	assert created["record"]["id"] == 1
	assert created["event"]["id"] == 1
	assert storage["records"]["Customer"] == [created["record"]]
	assert storage["events"] == [created["event"]]
	persisted = json.loads(data_file.read_text(encoding="utf-8"))
	assert persisted["records"]["Customer"] == [created["record"]]
	assert persisted["events"] == [created["event"]]

	second_process, second_base_url = start_app()
	try:
		with urllib.request.urlopen(f"{second_base_url}/entities/Customer/records/1", timeout=1) as response:
			reloaded = json.loads(response.read().decode("utf-8"))
		request = urllib.request.Request(
			f"{second_base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Kofi", "email": "kofi@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(request, timeout=1) as response:
			second_created = json.loads(response.read().decode("utf-8"))
	finally:
		stop_app(second_process)

	assert reloaded["record"] == created["record"]
	assert second_created["record"]["id"] == 2
	assert second_created["event"]["id"] == 2
	persisted = json.loads(data_file.read_text(encoding="utf-8"))
	assert persisted["records"]["Customer"] == [created["record"], second_created["record"]]
	assert [event["id"] for event in persisted["events"]] == [1, 2]


def test_generated_python_app_can_require_api_key_for_mutations(tmp_path):
	result = compile_apg_string(DATA_APP_SOURCE)
	package_dir = tmp_path / "generated_secured_app"
	package_dir.mkdir()
	for filename, content in result.generated_files.items():
		(package_dir / filename).write_text(content, encoding="utf-8")

	with socket.socket() as sock:
		sock.bind(("127.0.0.1", 0))
		port = sock.getsockname()[1]

	env = dict(os.environ, APG_API_KEY="secret-key")
	process = subprocess.Popen(
		[sys.executable, "app.py", "--host", "127.0.0.1", "--port", str(port)],
		cwd=package_dir,
		env=env,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)
	try:
		base_url = f"http://127.0.0.1:{port}"
		for _attempt in range(30):
			try:
				with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
					health = json.loads(response.read().decode("utf-8"))
				break
			except OSError:
				if process.poll() is not None:
					stdout, stderr = process.communicate(timeout=1)
					raise AssertionError(f"generated app exited early\nstdout={stdout}\nstderr={stderr}")
				time.sleep(0.05)
		else:
			raise AssertionError("generated secured app did not answer /health")

		with urllib.request.urlopen(f"{base_url}/auth", timeout=1) as response:
			auth = json.loads(response.read().decode("utf-8"))
		unauthorized_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Asha", "email": "asha@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			urllib.request.urlopen(unauthorized_request, timeout=1)
		except urllib.error.HTTPError as error:
			unauthorized_status = error.code
			unauthorized = json.loads(error.read().decode("utf-8"))
		else:
			raise AssertionError("generated secured app accepted mutation without an API key")
		authorized_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records",
			data=json.dumps({"record": {"name": "Asha", "email": "asha@example.com"}}).encode("utf-8"),
			headers={"Content-Type": "application/json", "Authorization": "Bearer secret-key"},
			method="POST",
		)
		with urllib.request.urlopen(authorized_request, timeout=1) as response:
			created = json.loads(response.read().decode("utf-8"))
		delete_request = urllib.request.Request(
			f"{base_url}/entities/Customer/records/1",
			headers={"X-APG-API-Key": "secret-key"},
			method="DELETE",
		)
		with urllib.request.urlopen(delete_request, timeout=1) as response:
			deleted = json.loads(response.read().decode("utf-8"))
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert health["auth"]["mode"] == "api_key"
	assert auth["mode"] == "api_key"
	assert unauthorized_status == 401
	assert unauthorized["error"] == "unauthorized"
	assert created["record"] == {"id": 1, "_revision": 1, "name": "Asha", "email": "asha@example.com"}
	assert deleted["deleted"] == created["record"]


def test_cli_compile_default_target_writes_generated_application(tmp_path):
	source = tmp_path / "baseline.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["compile", str(source), "--output", str(output), "--verbose", "--verify"])

	assert result.exit_code == 0, result.output
	assert "Compilation successful" in result.output
	assert f"python {output}/app.py" in result.output
	assert f"python {output}/app.py --describe" in result.output
	assert f"python {output}/app.py --self-test" in result.output
	assert f"apg compile {source} --output {output} --verify" in result.output
	assert "standard-library HTTP server" in result.output
	assert "Generated verification passed" in result.output
	assert (output / "app.py").exists()
	assert (output / "ai_agents.py").exists()
	assert (output / "Dockerfile").exists()
	assert (output / ".dockerignore").exists()
	assert (output / ".env.example").exists()
	assert (output / "README.md").exists()
	assert (output / "smoke_test.py").exists()
	app = (output / "app.py").read_text(encoding="utf-8")
	dockerfile = (output / "Dockerfile").read_text(encoding="utf-8")
	env_example = (output / ".env.example").read_text(encoding="utf-8")
	readme = (output / "README.md").read_text(encoding="utf-8")
	requirements = (output / "requirements.txt").read_text(encoding="utf-8")
	smoke_test = (output / "smoke_test.py").read_text(encoding="utf-8")
	assert "APG Python Application" in app
	assert "HTTPServer" in app
	assert "HEALTHCHECK" in dockerfile
	assert "APG_HOST=127.0.0.1" in env_example
	assert "python app.py --self-test" in readme
	assert "python smoke_test.py" in readme
	assert "GET /component.json" in readme
	assert "Dockerfile" in readme
	assert "GET /openapi.json" in readme
	assert "Typed APG fields render as matching HTML controls" in readme
	assert "Flask-AppBuilder" not in app
	assert "flask_appbuilder" not in requirements
	assert "standard library" in requirements
	assert "openapi_contract" in smoke_test
	assert "component_manifest" in smoke_test
	assert "route_dispatch" in smoke_test


def test_cli_compile_accepts_local_capability_catalog_preflight(tmp_path):
	source = tmp_path / "capability.apg"
	output = tmp_path / "generated"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, [
		"compile",
		str(source),
		"--catalog",
		str(catalog),
		"--output",
		str(output),
		"--verify",
	])

	assert result.exit_code == 0, result.output
	assert "Capability catalog preflight OK" in result.output
	assert "Generated verification passed" in result.output
	assert (output / "app.py").exists()
	assert (output / "apg_capabilities.py").exists()


def test_cli_compile_blocks_unresolved_catalog_capability_before_writing(tmp_path):
	source = tmp_path / "capability.apg"
	output = tmp_path / "generated"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, [
		"compile",
		str(source),
		"--catalog",
		str(catalog),
		"--output",
		str(output),
	])

	assert result.exit_code == 1, result.output
	assert "Compilation preflight validation failed" in result.output
	assert "APG0901" in result.output
	assert not output.exists()


def test_cli_lint_json_reports_valid_apg_file_without_generation(tmp_path):
	source = tmp_path / "baseline.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-report.v1"
	assert report["ok"] is True
	assert report["source_mode"] == "file"
	assert report["files"] == [str(source)]
	assert report["severity_counts"] == {"error": 0, "warning": 0, "info": 0, "hint": 0}
	assert report["diagnostics"] == []
	assert report["semantic_model_available"] is True
	assert not output.exists()


def test_cli_lint_treats_declarative_data_fields_as_contract_surface(tmp_path):
	source = tmp_path / "data_app.apg"
	source.write_text(DATA_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-report.v1"
	assert report["ok"] is True
	assert report["severity_counts"] == {"error": 0, "warning": 0, "info": 0, "hint": 0}
	assert report["diagnostics"] == []
	assert report["semantic_model_available"] is True


def test_cli_lint_catalog_uses_shared_semantic_model_for_capability_resolution(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_capability_catalog(tmp_path / "catalog", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-report.v1"
	assert report["ok"] is True
	assert report["semantic_model_available"] is True
	assert report["capability_catalog"]["checked"] is True
	assert report["capability_catalog"]["ok"] is True
	assert report["capability_catalog"]["contract_count"] == 1
	assert report["capability_catalog"]["declared_capabilities"] == ["AuditLog"]
	assert report["capability_catalog"]["matched_capabilities"] == [
		{"name": "AuditLog", "matched_key": "audit_log"}
	]
	assert report["diagnostics"] == []


def test_cli_lint_accepts_local_capability_catalog_json(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-report.v1"
	assert report["ok"] is True
	assert report["capability_catalog"]["checked"] is True
	assert report["capability_catalog"]["ok"] is True
	assert report["capability_catalog"]["catalog_kind"] == "local_catalog"
	assert report["capability_catalog"]["contract_count"] == 1
	assert report["capability_catalog"]["declared_capabilities"] == ["AuditLog"]
	assert report["capability_catalog"]["matched_capabilities"] == [
		{"name": "AuditLog", "matched_key": "audit_log"}
	]
	assert report["diagnostics"] == []


def test_cli_lint_catalog_reports_unknown_declared_capability(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_capability_catalog(tmp_path / "catalog", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["ok"] is False
	assert report["capability_catalog"]["checked"] is True
	assert report["capability_catalog"]["ok"] is False
	assert report["capability_catalog"]["missing_capabilities"] == [
		{"name": "AuditLog", "candidates": ["AuditLog", "audit_log", "audit_events"]}
	]
	assert report["severity_counts"]["error"] == 1
	diagnostic_codes = [diagnostic["code"] for diagnostic in report["diagnostics"]]
	assert diagnostic_codes == ["APG0901"]
	assert "does not resolve in catalog" in report["diagnostics"][0]["message"]


def test_cli_lint_local_capability_catalog_reports_unknown_declared_capability(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["ok"] is False
	assert report["capability_catalog"]["checked"] is True
	assert report["capability_catalog"]["ok"] is False
	assert report["capability_catalog"]["catalog_kind"] == "local_catalog"
	assert report["capability_catalog"]["missing_capabilities"] == [
		{"name": "AuditLog", "candidates": ["AuditLog", "audit_log", "audit_events"]}
	]
	assert report["severity_counts"]["error"] == 1
	diagnostic_codes = [diagnostic["code"] for diagnostic in report["diagnostics"]]
	assert diagnostic_codes == ["APG0901"]
	assert "does not resolve in catalog" in report["diagnostics"][0]["message"]


def test_cli_lint_directory_json_aggregates_apg_files_deterministically(tmp_path):
	first = tmp_path / "01_valid.apg"
	second = tmp_path / "02_invalid.apg"
	first.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")
	second.write_text("invalid_entity Broken {", encoding="utf-8")
	(tmp_path / "ignored.txt").write_text("invalid_entity Ignored {", encoding="utf-8")

	result = CliRunner().invoke(cli, ["lint", str(tmp_path), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-report.v1"
	assert report["ok"] is False
	assert report["source_mode"] == "directory"
	assert report["files"] == [str(first), str(second)]
	assert report["severity_counts"]["error"] >= 1
	assert [file_report["file"] for file_report in report["file_reports"]] == [str(first), str(second)]
	assert report["diagnostics"][0]["code"] == "APG0001"
	assert report["diagnostics"][0]["file"] == str(second)


def test_cli_lint_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["lint", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.lint-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 6
	assert report["summary"]["failing_fixture_count"] == 0
	assert {"valid", "syntax", "strict"}.issubset(report["tags_covered"])
	assert {"semantic_model", "capability_catalog", "database_backed_forms"}.issubset(report["tags_covered"])
	assert {fixture["id"] for fixture in report["fixtures"]} >= {
		"valid",
		"syntax_error",
		"strict_warning",
		"catalog_match",
		"catalog_missing",
		"invalid_database_backed_form",
	}
	assert all(fixture["format"] == "apg.lint-report.v1" for fixture in report["fixtures"])


def test_cli_validate_json_reports_generator_readiness_without_generation(tmp_path):
	source = tmp_path / "baseline.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["validate", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.validate-report.v1"
	assert report["ok"] is True
	assert report["generator_ready"] is True
	assert report["target_compatibility"] == {
		"requested": "python",
		"supported": ["python"],
		"ok": True,
	}
	assert report["lint"]["format"] == "apg.lint-report.v1"
	assert report["files"] == [str(source)]
	assert not output.exists()


def test_cli_validate_accepts_local_capability_catalog_for_generator_readiness(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["validate", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.validate-report.v1"
	assert report["ok"] is True
	assert report["generator_ready"] is True
	assert report["lint"]["capability_catalog"]["catalog_kind"] == "local_catalog"
	assert report["lint"]["capability_catalog"]["matched_capabilities"] == [
		{"name": "AuditLog", "matched_key": "audit_log"}
	]


def test_cli_validate_blocks_generator_readiness_for_unknown_catalog_capability(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["validate", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.validate-report.v1"
	assert report["ok"] is False
	assert report["generator_ready"] is False
	assert report["lint"]["capability_catalog"]["catalog_kind"] == "local_catalog"
	assert report["diagnostics"][0]["code"] == "APG0901"


def test_cli_validate_rejects_non_python_target_with_apg0802(tmp_path):
	source = tmp_path / "baseline.apg"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["validate", str(source), "--target", "django", "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.validate-report.v1"
	assert report["ok"] is False
	assert report["generator_ready"] is False
	assert report["target_compatibility"]["ok"] is False
	assert report["diagnostics"][-1]["code"] == "APG0802"
	assert "must be 'python'" in report["diagnostics"][-1]["message"]


def test_cli_model_json_emits_semantic_model_without_generation(tmp_path):
	source = tmp_path / "sales.apg"
	output = tmp_path / "generated"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["model", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.semantic-model.v1"
	assert report["ok"] is True
	assert report["source_files"] == [str(source)]
	assert report["app"]["name"] == "sales_ops"
	assert "module.sales_ops" in report["symbols"]
	assert "table.Customer" in report["symbols"]
	assert "field.SalesOrder.customer_id" in report["symbols"]
	assert report["tables"]["SalesOrder"]["fields"]["customer"]["relationship"] == {
		"target_table": "Customer",
		"target_field": "id",
		"cardinality": "many-to-one",
	}
	assert report["tables"]["SalesOrder"]["fields"]["customer_id"]["relationship"] == {
		"target_table": "Customer",
		"target_field": "id",
		"cardinality": "many-to-one",
		"alias": "customer",
	}
	assert report["graphs"]["er"]["edges"] >= 1
	assert report["deployment"] == {"target": "python", "source": str(source)}
	assert not output.exists()


def test_cli_model_json_validates_database_backed_form_bindings(tmp_path):
	source = tmp_path / "customer_form.apg"
	source.write_text(DATABASE_BACKED_FORM_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["model", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.semantic-model.v1"
	assert report["ok"] is True
	assert report["views"]["CustomerForm"]["type"] == "form"
	assert report["views"]["CustomerForm"]["bindings"] == ["name", "email"]
	assert all(diagnostic["severity"] != "error" for diagnostic in report["diagnostics"])


def test_cli_model_json_rejects_unknown_database_backed_form_field(tmp_path):
	source = tmp_path / "customer_form.apg"
	source.write_text(INVALID_DATABASE_BACKED_FORM_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["model", str(source), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.semantic-model.v1"
	assert report["ok"] is False
	errors = [diagnostic for diagnostic in report["diagnostics"] if diagnostic["severity"] == "error"]
	assert [diagnostic["code"] for diagnostic in errors] == ["APG0402"]
	assert "CustomerForm" in errors[0]["message"]
	assert "phone" in errors[0]["message"]


def test_cli_model_text_summarizes_agent_semantics(tmp_path):
	source = tmp_path / "agent.apg"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["model", str(source)])

	assert result.exit_code == 0, result.output
	assert "APG model OK" in result.output
	assert "agent(s)" in result.output
	assert "1 agent(s)" in result.output


def test_cli_model_audits_semantic_model_fixture_catalog():
	result = CliRunner().invoke(cli, ["model", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.semantic-model-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 4
	assert report["summary"]["failing_fixture_count"] == 0
	assert {"symbols", "relationships", "graphs"}.issubset(report["tags_covered"])
	assert {"capabilities", "database_backed_forms", "diagnostics"}.issubset(report["tags_covered"])
	assert {fixture["id"] for fixture in report["fixtures"]} >= {
		"relationships",
		"capability_contract",
		"database_backed_form",
		"invalid_database_backed_form",
	}
	assert all(fixture["format"] == "apg.semantic-model.v1" for fixture in report["fixtures"])


def test_cli_release_json_emits_generated_application_evidence_without_output(tmp_path):
	source = tmp_path / "agent.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["release", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-report.v1"
	assert report["ok"] is True
	assert report["source"] == str(source)
	assert report["target"] == "python"
	assert report["generated"]["semantic_model_artifact"] is True
	assert "semantic_model.json" in report["generated"]["files"]
	assert report["evidence"]["self_test"]["passed"] is True
	assert report["evidence"]["self_test"]["status"] == "ok"
	assert report["evidence"]["self_test"]["route_count"] >= 1
	assert report["evidence"]["semantic_model"]["format"] == "apg.semantic-model.v1"
	assert report["evidence"]["semantic_model"]["agent_count"] == 1
	assert report["evidence"]["component_manifest"]["semantic_model"] == "/semantic-model.json"
	assert report["evidence"]["contracts"]["component_manifest"]["errors"] == []
	assert report["evidence"]["contracts"]["openapi"]["errors"] == []
	assert report["evidence"]["contracts"]["route_dispatch"]["errors"] == []
	assert not output.exists()


def test_cli_release_text_summarizes_evidence(tmp_path):
	source = tmp_path / "agent.apg"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["release", str(source)])

	assert result.exit_code == 0, result.output
	assert "APG release OK" in result.output
	assert "artifact(s)" in result.output
	assert "self-test=ok" in result.output


def test_cli_release_accepts_local_capability_catalog_preflight(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["release", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-report.v1"
	assert report["ok"] is True
	assert report["preflight"]["checked"] is True
	assert report["preflight"]["catalog_kind"] == "local_catalog"
	assert report["preflight"]["matched_capabilities"] == [
		{"name": "AuditLog", "matched_key": "audit_log"}
	]
	assert report["generated"]["file_count"] > 0


def test_cli_release_blocks_unresolved_catalog_capability_before_generation(tmp_path):
	source = tmp_path / "capability.apg"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["release", str(source), "--catalog", str(catalog), "--json"])

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-report.v1"
	assert report["ok"] is False
	assert report["preflight"]["checked"] is True
	assert report["preflight"]["catalog_kind"] == "local_catalog"
	assert report["generated"] == {}
	assert any("APG0901" in error for error in report["errors"])


def test_cli_baseline_json_audits_numbered_examples():
	result = CliRunner().invoke(cli, ["baseline", str(REPO_ROOT / "examples"), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.compiler-baseline-report.v1"
	assert report["ok"] is True
	assert report["example_count"] == 20
	assert report["checks"] == {
		"numbered_examples_present": True,
		"domain_coverage_ok": True,
		"all_examples_ok": True,
		"python_target_only": True,
	}
	assert report["domains"]["workflows"]["ok"] is True
	assert report["domains"]["bytewax_streaming"]["ok"] is True
	assert report["summary"]["failed_examples"] == 0
	assert report["summary"]["checked_generated_python_files"] >= 20
	assert report["summary"]["generated_source_hygiene_violations"] == 0
	assert all(example["checks"]["compile_verify_ok"] for example in report["examples"])
	assert all(example["checks"]["generated_source_hygiene_ok"] for example in report["examples"])
	assert all(example["compile_verify"]["source_hygiene"]["ok"] for example in report["examples"])
	assert all(example["checks"]["release_ok"] for example in report["examples"])


def test_cli_parser_golden_json_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["parser-golden", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.parser-golden-audit.v1"
	assert report["ok"] is True
	assert report["missing_constructs"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 8
	assert report["summary"]["invalid_fixture_count"] >= 4
	assert "agents" in report["constructs_covered"]
	assert "capabilities" in report["constructs_covered"]
	assert "deployment_units" in report["constructs_covered"]
	assert any(
		fixture["expected_valid"] is False and fixture["actual_valid"] is False
		for fixture in report["fixtures"]
	)


def test_cli_diagnostics_audits_registry_fixture_catalog():
	result = CliRunner().invoke(cli, ["diagnostics", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.diagnostic-audit.v1"
	assert report["ok"] is True
	assert report["summary"]["registry_count"] >= 35
	assert report["summary"]["fixture_count"] == report["summary"]["registry_count"]
	assert report["missing_fixture_codes"] == []
	assert report["unknown_fixture_codes"] == []
	assert report["severity_mismatches"] == []
	assert report["registry"]["APG0001"]["title"] == "Syntax error"
	assert report["registry"]["APG1201"]["area"] == "natural-language"
	assert "APG1104" in report["fixture_codes"]


def test_cli_drift_json_compares_compiler_generated_artifact_and_runtime_surfaces(tmp_path):
	source = tmp_path / "agent.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["drift", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.drift-report.v1"
	assert report["ok"] is True
	assert set(report["surfaces"]) == {"compiler", "generated_artifact", "generated_runtime"}
	assert report["surfaces"]["compiler"]["format"] == "apg.semantic-model.v1"
	assert report["surfaces"]["generated_artifact"]["agent_count"] == 1
	assert report["surfaces"]["generated_runtime"]["agent_count"] == 1
	assert [comparison["ok"] for comparison in report["comparisons"]] == [True, True, True]
	assert report["summary"]["drift_count"] == 0
	assert not output.exists()


def test_cli_drift_audit_fixtures_json_runs_checked_in_catalog():
	result = CliRunner().invoke(cli, ["drift", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.drift-audit.v1"
	assert report["ok"] is True
	assert report["summary"]["fixture_count"] >= 2
	assert report["summary"]["failed"] == 0
	assert all(item["actual_ok"] is True for item in report["reports"])


def test_cli_capabilities_contracts_json_uses_click_surface():
	result = CliRunner().invoke(cli, ["capabilities", "contracts", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.capability-contracts.v1"
	assert report["ok"] is True
	assert report["contract_count"] >= 100
	assert any(record["capability"] == "composition_events" for record in report["records"])
	assert any(record["capability"] == "fintech_gateway" for record in report["records"])
	assert all(record["routes"] > 0 for record in report["records"])
	assert all(record["rules"] > 0 for record in report["records"])


def test_cli_capabilities_validate_contracts_json_uses_click_surface():
	result = CliRunner().invoke(cli, ["capabilities", "validate-contracts", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.capability-contract-validation.v1"
	assert report["ok"] is True
	assert report["valid"] is True
	assert report["contract_count"] >= 100
	assert report["error_count"] == 0


def test_cli_explain_json_covers_symbols_diagnostics_and_handlers():
	source = REPO_ROOT / "examples" / "20_enterprise_erp_platform" / "main.apg"
	diagnostic_source = REPO_ROOT / "tests" / "fixtures" / "lint" / "relationship_warning.apg"

	symbol_result = CliRunner().invoke(
		cli,
		["explain", str(source), "--symbol", "capability.EnterpriseFinance", "--json"],
	)
	assert symbol_result.exit_code == 0, symbol_result.output
	symbol_report = json.loads(symbol_result.output)
	assert symbol_report["format"] == "apg.explain-report.v1"
	assert symbol_report["ok"] is True
	assert symbol_report["explanations"][0]["symbol"]["id"] == "capability.EnterpriseFinance"
	assert symbol_report["explanations"][0]["related"]["provides"] == [
		"journal_entries",
		"invoice_generation",
		"payment_allocation",
	]

	diagnostic_result = CliRunner().invoke(
		cli,
		["explain", str(diagnostic_source), "--diagnostic", "APG0100", "--json"],
	)
	assert diagnostic_result.exit_code == 0, diagnostic_result.output
	diagnostic_report = json.loads(diagnostic_result.output)
	assert diagnostic_report["ok"] is True
	assert diagnostic_report["explanations"][0]["code"] == "APG0100"
	assert diagnostic_report["explanations"][0]["match_count"] >= 1
	assert diagnostic_report["explanations"][0]["registry"]["title"] == "Semantic warning"

	handler_result = CliRunner().invoke(
		cli,
		["explain", str(source), "--handler", "OperationsDashboard.select", "--json"],
	)
	assert handler_result.exit_code == 0, handler_result.output
	handler_report = json.loads(handler_result.output)
	assert handler_report["ok"] is True
	assert handler_report["explanations"][0]["handler"]["screen"] == "OperationsDashboard"
	assert handler_report["explanations"][0]["handler"]["handler"]["target"] == "FulfillmentTable"


def test_cli_package_json_writes_executable_profile(tmp_path):
	source = REPO_ROOT / "examples" / "05_single_support_agent" / "main.apg"
	out_dir = tmp_path / "dist"

	result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "desktop", "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	package_dir = Path(report["output_dir"])
	assert report["format"] == "apg.package-report.v1"
	assert report["ok"] is True
	assert report["target"] == "desktop"
	assert package_dir == out_dir / "support_agent-desktop"
	assert report["manifest"]["format"] == "apg.package-manifest.v1"
	assert report["manifest"]["base_target"] == "python"
	assert report["manifest"]["entrypoints"]["run"] == "python run_desktop.py"
	assert report["release"]["ok"] is True
	assert report["checks"]["release_evidence_ok"] is True
	assert "run_desktop.py" in report["files"]
	assert "package_manifest.json" in report["files"]
	assert "release_report.json" in report["files"]
	assert "semantic_model.json" in report["files"]
	assert (package_dir / "package_manifest.json").exists()
	assert (package_dir / "release_report.json").exists()

	self_test = subprocess.run(
		[sys.executable, "app.py", "--self-test"],
		cwd=package_dir,
		text=True,
		capture_output=True,
		check=False,
	)
	assert self_test.returncode == 0, self_test.stderr
	assert json.loads(self_test.stdout)["passed"] is True


def test_cli_package_accepts_local_capability_catalog_preflight(tmp_path):
	source = tmp_path / "capability.apg"
	out_dir = tmp_path / "dist"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "web", "--catalog", str(catalog), "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	package_dir = Path(report["output_dir"])
	assert report["format"] == "apg.package-report.v1"
	assert report["ok"] is True
	assert report["catalog"] == str(catalog)
	assert report["release"]["preflight"]["catalog_kind"] == "local_catalog"
	assert (package_dir / "package_manifest.json").exists()
	assert (package_dir / "release_report.json").exists()


def test_cli_package_blocks_unresolved_catalog_capability_before_writing(tmp_path):
	source = tmp_path / "capability.apg"
	out_dir = tmp_path / "dist"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "web", "--catalog", str(catalog), "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.package-report.v1"
	assert report["ok"] is False
	assert report["output_dir"] == ""
	assert not out_dir.exists()
	assert any("APG0901" in error for error in report["errors"])


def test_cli_capabilities_publish_plan_validates_package_without_writing_catalog(tmp_path):
	source = REPO_ROOT / "examples" / "08_basic_capability_contract" / "main.apg"
	out_dir = tmp_path / "dist"

	package_result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "web", "--out", str(out_dir), "--json"],
	)
	assert package_result.exit_code == 0, package_result.output
	package_report = json.loads(package_result.output)
	package_dir = Path(package_report["output_dir"])

	publish_result = CliRunner().invoke(
		cli,
		["capabilities", "publish-plan", str(package_dir), "--json"],
	)

	assert publish_result.exit_code == 0, publish_result.output
	report = json.loads(publish_result.output)
	assert report["format"] == "apg.capability-publish-report.v1"
	assert report["ok"] is True
	assert report["side_effect_free"] is True
	assert report["manifest"]["format"] == "apg.package-manifest.v1"
	assert report["release_evidence"]["ok"] is True
	assert report["runtime_evidence"]["loaded"] is True
	assert report["runtime_evidence"]["self_test"]["passed"] is True
	assert [record["capability"] for record in report["capabilities"]] == ["AuditLog"]
	assert report["catalog_patch"] == [
		{
			"op": "add_or_replace",
			"path": "/capabilities/AuditLog",
			"value": report["capabilities"][0],
		}
	]
	assert not (package_dir / "capability_catalog.json").exists()


def test_cli_deployment_verify_reports_package_evidence(tmp_path):
	source = REPO_ROOT / "examples" / "10_themed_i18n_streaming_capability" / "main.apg"
	out_dir = tmp_path / "dist"

	package_result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "container", "--out", str(out_dir), "--json"],
	)
	assert package_result.exit_code == 0, package_result.output
	package_report = json.loads(package_result.output)
	package_dir = Path(package_report["output_dir"])

	result = CliRunner().invoke(
		cli,
		["deployment", "verify", str(package_dir), "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.deployment-verification-report.v1"
	assert report["ok"] is True
	assert report["kind"] == "package"
	assert report["manifest"]["profile"] == "container"
	assert report["checks"] == {
		"units_declared": True,
		"health_checks_declared": True,
		"environment_variables_named": True,
		"secret_values_absent": True,
		"resource_hints_present": True,
		"topology_graph_connected": True,
	}
	assert report["runtime"]["self_test"]["passed"] is True
	assert report["runtime"]["semantic_model"]["format"] == "apg.semantic-model.v1"
	assert report["topology"]["connected"] is True
	assert report["resource_hints"]["healthcheck"] is True


def test_cli_package_verify_reports_mobile_profile_evidence(tmp_path):
	source = REPO_ROOT / "examples" / "16_hr_payroll_operations" / "main.apg"
	out_dir = tmp_path / "dist"

	package_result = CliRunner().invoke(
		cli,
		["package", str(source), "--target", "mobile", "--out", str(out_dir), "--json"],
	)
	assert package_result.exit_code == 0, package_result.output
	package_report = json.loads(package_result.output)
	package_dir = Path(package_report["output_dir"])

	result = CliRunner().invoke(cli, ["package-verify", str(package_dir), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.package-verification-report.v1"
	assert report["ok"] is True
	assert report["profile"] == "mobile"
	assert report["manifest"]["base_target"] == "python"
	assert report["release_evidence"]["ok"] is True
	assert report["runtime"]["loaded"] is True
	assert report["runtime"]["self_test"]["passed"] is True
	assert report["runtime"]["contracts"]["openapi"]["errors"] == []
	assert report["runtime"]["contracts"]["route_dispatch"]["errors"] == []
	assert report["profile_checks"] == {
		"metadata_exists": True,
		"signing_posture_declared": True,
		"offline_policy_declared": True,
		"permissions_explained": True,
		"screens_fit_target_density": True,
		"smoke_launch_path_exists": True,
		"smoke_test_runs": True,
	}


def test_cli_evidence_json_builds_release_bundle(tmp_path):
	source = REPO_ROOT / "examples" / "08_basic_capability_contract" / "main.apg"
	out_dir = tmp_path / "evidence"

	result = CliRunner().invoke(
		cli,
		["evidence", str(source), "--target", "web", "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-evidence-bundle.v1"
	assert report["ok"] is True
	assert report["checks"] == {
		"release_ok": True,
		"package_ok": True,
		"package_verification_ok": True,
		"deployment_verification_ok": True,
		"capability_publish_plan_ok": True,
	}
	assert report["release"]["format"] == "apg.release-report.v1"
	assert report["package"]["format"] == "apg.package-report.v1"
	assert report["package_verification"]["format"] == "apg.package-verification-report.v1"
	assert report["deployment_verification"]["format"] == "apg.deployment-verification-report.v1"
	assert report["capability_publish"]["format"] == "apg.capability-publish-report.v1"
	assert report["capability_publish"]["side_effect_free"] is True
	assert report["capability_publish"]["capabilities"] == ["AuditLog"]
	assert Path(report["package"]["output_dir"]).exists()


def test_cli_evidence_accepts_local_capability_catalog_preflight(tmp_path):
	source = tmp_path / "capability.apg"
	out_dir = tmp_path / "evidence"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "audit_log")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["evidence", str(source), "--target", "web", "--catalog", str(catalog), "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-evidence-bundle.v1"
	assert report["ok"] is True
	assert report["catalog"] == str(catalog)
	assert report["release"]["preflight"]["catalog_kind"] == "local_catalog"
	assert report["package"]["format"] == "apg.package-report.v1"
	assert Path(report["package"]["output_dir"]).exists()


def test_cli_evidence_blocks_unresolved_catalog_capability_before_package(tmp_path):
	source = tmp_path / "capability.apg"
	out_dir = tmp_path / "evidence"
	catalog = write_local_capability_catalog(tmp_path / "catalog" / "capabilities.json", "customer_master")
	source.write_text(CAPABILITY_CATALOG_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["evidence", str(source), "--target", "web", "--catalog", str(catalog), "--out", str(out_dir), "--json"],
	)

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-evidence-bundle.v1"
	assert report["ok"] is False
	assert report["catalog"] == str(catalog)
	assert report["release"]["preflight"]["catalog_kind"] == "local_catalog"
	assert report["package"] == {}
	assert not out_dir.exists()
	assert any("APG0901" in error for error in report["errors"])


def test_cli_evidence_audits_release_verifier_fixture_catalog():
	result = CliRunner().invoke(cli, ["evidence", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.release-evidence-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_targets"] == []
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["targets_covered"] == ["container", "desktop", "mobile", "web"]
	assert report["summary"]["fixture_count"] >= 1
	assert report["summary"]["target_run_count"] >= 4
	assert {"release_evidence", "package_verification", "deployment_verification"}.issubset(report["tags_covered"])
	fixture = report["fixtures"][0]
	assert fixture["ok"] is True
	assert {target["target"] for target in fixture["target_reports"]} == {"web", "desktop", "mobile", "container"}
	assert all(target["ok"] for target in fixture["target_reports"])
	assert all(target["capability_publish_side_effect_free"] is True for target in fixture["target_reports"])


def test_cli_nl_plan_json_proposes_valid_credit_memo_dsl_diff_without_writing(tmp_path):
	source = tmp_path / "finance.apg"
	output = tmp_path / "generated"
	original = DATA_APP_SOURCE
	source.write_text(original, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["nl-plan", str(source), "--prompt", "Add credit memos to accounts receivable", "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.nl-plan.v1"
	assert report["ok"] is True
	assert report["intent"] == "domain_feature"
	assert report["confidence"] == "high"
	assert "CreditMemo" in report["append_text"]
	assert "CreditMemoManagement" in report["dsl_patch"]
	assert report["affected_symbols"] == ["table.CreditMemo", "capability.CreditMemoManagement"]
	assert report["lint"]["format"] == "apg.lint-report.v1"
	assert report["lint"]["ok"] is True
	assert report["migration_preview"]["format"] == "apg.migration-preview.v1"
	assert report["migration_preview"]["destructive"] is False
	assert {change["kind"] for change in report["migration_preview"]["changes"]} == {
		"add_capability",
		"add_table",
	}
	assert [step["phase"] for step in report["test_plan"]] == ["lint", "validate", "compile", "release"]
	assert source.read_text(encoding="utf-8") == original
	assert not output.exists()


def test_cli_nl_plan_rejects_unrepresentable_prompt(tmp_path):
	source = tmp_path / "finance.apg"
	source.write_text(DATA_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["nl-plan", str(source), "--prompt", "make it delightful and scalable", "--json"],
	)

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.nl-plan.v1"
	assert report["ok"] is False
	assert report["intent"] == "unrepresentable"
	assert report["dsl_patch"] == ""
	assert report["lint"]["diagnostics"][0]["code"] == "APG1201"
	assert report["migration_preview"]["changes"] == []


def test_cli_nl_plan_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["nl-plan", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.nl-plan-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 5
	assert report["summary"]["failing_fixture_count"] == 0
	assert {"domain_feature", "add_table", "add_capability", "add_agent", "unrepresentable"}.issubset(report["tags_covered"])
	assert {"lint_integration", "migration_preview", "source_immutability"}.issubset(report["tags_covered"])
	intents = {fixture["intent"] for fixture in report["fixtures"]}
	assert {"domain_feature", "add_table", "add_capability", "add_agent", "unrepresentable"}.issubset(intents)
	rejected = next(fixture for fixture in report["fixtures"] if fixture["id"] == "reject_unbounded_style_prompt")
	assert rejected["diagnostic_codes"] == ["APG1201"]


def test_cli_migrate_plan_json_detects_destructive_schema_and_ownership_changes(tmp_path):
	previous = tmp_path / "previous.apg"
	current = tmp_path / "current.apg"
	previous.write_text(MIGRATION_PREVIOUS_SOURCE, encoding="utf-8")
	current.write_text(MIGRATION_CURRENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["migrate-plan", str(previous), str(current), "--backend", "postgresql", "--json"],
	)

	assert result.exit_code == 1, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.migration-plan.v1"
	assert report["ok"] is False
	assert report["backend"] == "postgresql"
	assert report["destructive"] is True
	assert report["requires_approval"] is True
	changes = {(change["kind"], change["symbol"]): change for change in report["changes"]}
	assert ("add_table", "table.Invoice") in changes
	assert ("add_field", "field.Customer.email") in changes
	assert changes[("add_field", "field.Customer.email")]["requires_backfill"] is True
	assert changes[("drop_field", "field.Customer.legacy_code")]["destructive"] is True
	assert changes[("type_change", "field.Customer.age")]["destructive"] is True
	assert changes[("capability_ownership_transfer", "table.Customer")]["before"] == {
		"owner": "CustomerMaster"
	}
	assert changes[("capability_ownership_transfer", "table.Customer")]["after"] == {
		"owner": "Billing"
	}
	assert {diagnostic["code"] for diagnostic in report["diagnostics"]} >= {
		"APG1101",
		"APG1103",
		"APG1104",
		"APG1106",
	}


def test_cli_migrate_plan_json_allows_additive_table_changes(tmp_path):
	previous = tmp_path / "previous.apg"
	current = tmp_path / "current.apg"
	previous.write_text(DATA_APP_SOURCE, encoding="utf-8")
	current.write_text(DATA_APP_SOURCE + "\ntable Note {\n\tbody: str;\n}\n", encoding="utf-8")

	result = CliRunner().invoke(
		cli,
		["migrate-plan", str(previous), str(current), "--backend", "mysql", "--json"],
	)

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.migration-plan.v1"
	assert report["ok"] is True
	assert report["backend"] == "mysql"
	assert report["destructive"] is False
	assert report["requires_approval"] is False
	assert report["changes"][0]["kind"] == "add_table"
	assert report["changes"][0]["symbol"] == "table.Note"


def test_cli_migrate_plan_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["migrate-plan", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.migration-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 2
	assert report["summary"]["failing_fixture_count"] == 0
	assert {"add_table", "drop_field", "type_change", "ownership_transfer"}.issubset(report["tags_covered"])
	destructive = next(fixture for fixture in report["fixtures"] if fixture["id"] == "customer_destructive_review")
	assert {"APG1101", "APG1103", "APG1104", "APG1106"}.issubset(destructive["diagnostic_codes"])
	additive = next(fixture for fixture in report["fixtures"] if fixture["id"] == "additive_note_table")
	assert additive["change_kinds"] == ["add_table"]


def test_checked_in_example_outputs_match_current_compiler():
	examples = sorted((REPO_ROOT / "examples").glob("[0-9][0-9]_*/main.apg"))
	assert len(examples) == 20
	mismatches: list[str] = []

	for source_file in examples:
		compiler = APGCompiler()
		result = compiler.compile_file(source_file, target_language="python")
		assert result.success is True, f"{source_file}: {result.errors}"
		output_dir = source_file.parent / "output"
		assert output_dir.exists(), f"{source_file.parent.name}: missing output directory"

		expected_files = set(result.generated_files)
		actual_files = {
			str(path.relative_to(output_dir))
			for path in output_dir.rglob("*")
			if path.is_file()
			and "__pycache__" not in path.relative_to(output_dir).parts
			and not path.name.endswith(".pyc")
		}
		for missing in sorted(expected_files - actual_files):
			mismatches.append(f"{source_file.parent.name}: missing {missing}")
		for extra in sorted(actual_files - expected_files):
			mismatches.append(f"{source_file.parent.name}: extra {extra}")
		for file_name in sorted(expected_files & actual_files):
			actual = (output_dir / file_name).read_text(encoding="utf-8")
			if actual != result.generated_files[file_name]:
				mismatches.append(f"{source_file.parent.name}: changed {file_name}")

	assert mismatches == []


def test_cli_format_json_formats_source_without_writing(tmp_path):
	source = tmp_path / "messy.apg"
	source.write_text(
		'module messy version 1.0.0 {\n\tdescription: "Messy";   \n}\n'
		'agent Planner {\n\trole: "planner";\n}\n',
		encoding="utf-8",
	)

	result = CliRunner().invoke(cli, ["format", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.format-result.v1"
	assert report["changed"] is True
	assert report["idempotent"] is True
	assert report["written"] is False
	assert report["text"] == (
		'module messy version 1.0.0 {\n'
		'  description: "Messy";\n'
		'}\n'
		'\n'
		'agent Planner {\n'
		'  role: "planner";\n'
		'}\n'
	)
	assert "\t" in source.read_text(encoding="utf-8")


def test_cli_format_check_and_write_are_idempotent(tmp_path):
	source = tmp_path / "messy.apg"
	source.write_text('module messy version 1.0.0 {\n\tname: str;\n}\n', encoding="utf-8")

	check_result = CliRunner().invoke(cli, ["format", str(source), "--check"])
	assert check_result.exit_code == 1
	assert "would change" in check_result.output

	write_result = CliRunner().invoke(cli, ["format", str(source), "--write"])
	assert write_result.exit_code == 0, write_result.output
	assert "formatted" in write_result.output
	assert source.read_text(encoding="utf-8") == (
		"module messy version 1.0.0 {\n"
		"  name: str;\n"
		"}\n"
	)

	second_check = CliRunner().invoke(cli, ["format", str(source), "--check"])
	assert second_check.exit_code == 0, second_check.output
	assert "already formatted" in second_check.output


def test_cli_format_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["format", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.formatter-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 3
	assert report["summary"]["changed_fixture_count"] >= 2
	assert {"file_comment", "declaration_comment", "inline_comment"}.issubset(report["tags_covered"])
	assert {"modifier_order", "relationship_modifier", "idempotency"}.issubset(report["tags_covered"])
	assert all(fixture["idempotent"] for fixture in report["fixtures"])


def test_cli_graph_json_emits_entity_relationship_graph(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["graph", str(source), "--kind", "er", "--format", "json"])

	assert result.exit_code == 0, result.output
	graph = json.loads(result.output)
	assert graph["format"] == "apg.graph.v1"
	assert graph["kind"] == "er"
	node_ids = {node["id"] for node in graph["nodes"]}
	edge_keys = {(edge["source"], edge["target"], edge["kind"]) for edge in graph["edges"]}
	assert "entity:Customer" in node_ids
	assert "entity:SalesOrder" in node_ids
	assert "field:SalesOrder.customer" in node_ids
	assert ("field:SalesOrder.customer", "entity:Customer", "references") in edge_keys


def test_cli_graph_infers_conventional_foreign_key_edges(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(
		"""
module sales version 1.0.0 {
	description: "Sales";
}

table Customer {
	name: str;
}

table SalesOrder {
	customer_id: int;
}
""",
		encoding="utf-8",
	)

	result = CliRunner().invoke(cli, ["graph", str(source), "--kind", "er", "--format", "json"])

	assert result.exit_code == 0, result.output
	graph = json.loads(result.output)
	edge_keys = {(edge["source"], edge["target"], edge["kind"]) for edge in graph["edges"]}
	assert ("field:SalesOrder.customer_id", "entity:Customer", "inferred_reference") in edge_keys


def test_cli_graph_mermaid_and_dot_outputs_are_renderable(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	mermaid = CliRunner().invoke(cli, ["graph", str(source), "--kind", "er", "--format", "mermaid"])
	dot = CliRunner().invoke(cli, ["graph", str(source), "--kind", "er", "--format", "dot"])

	assert mermaid.exit_code == 0, mermaid.output
	assert mermaid.output.startswith("graph TD")
	assert "SalesOrder.customer: Customer" in mermaid.output
	assert dot.exit_code == 0, dot.output
	assert dot.output.startswith("digraph er")
	assert '"field:SalesOrder.customer" -> "entity:Customer"' in dot.output


def test_cli_graph_suite_json_emits_all_supported_renderings(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["graph-suite", str(source), "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.graph-suite-report.v1"
	assert report["ok"] is True
	assert report["graph_kinds"] == [
		"er",
		"lookup",
		"workflow",
		"handler",
		"capability",
		"security",
		"agent",
		"deployment",
		"package",
	]
	assert report["graphs"]["er"]["json"]["format"] == "apg.graph.v1"
	assert report["graphs"]["er"]["mermaid"].startswith("graph TD")
	assert report["graphs"]["er"]["dot"].startswith("digraph er")
	assert report["summary"]["er"]["nodes"] >= 3
	assert report["summary"]["er"]["edges"] >= 1


def test_cli_graph_suite_text_summarizes_graph_counts(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["graph-suite", str(source)])

	assert result.exit_code == 0, result.output
	assert "APG graph-suite OK" in result.output
	assert "er:" in result.output
	assert "agent:" in result.output


def test_cli_graph_suite_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["graph-suite", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.graph-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_graph_kinds"] == []
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 3
	assert report["summary"]["failing_fixture_count"] == 0
	assert "er" in report["graph_kinds_observed"]
	assert "agent" in report["graph_kinds_observed"]
	assert "capability" in report["graph_kinds_observed"]
	assert {"er_relationships", "agent_team", "capability_dependencies"}.issubset(report["tags_covered"])
	assert all(fixture["graph_kinds"] == report["graph_kinds_required"] for fixture in report["fixtures"])


def test_cli_init_describes_python_artifact_flow():
	runner = CliRunner()
	with runner.isolated_filesystem():
		result = runner.invoke(cli, ["init"])

	assert result.exit_code == 0, result.output
	assert "generate Python artifacts" in result.output
	assert "python generated/app.py" in result.output
	assert "python generated/app.py --self-test" in result.output
	assert "Flask-AppBuilder" not in result.output


def test_cli_create_basic_project_scaffolds_python_target(tmp_path):
	output = tmp_path / "demo"
	result = CliRunner().invoke(cli, [
		"create",
		"project",
		"--name",
		"demo",
		"--description",
		"Demo project",
		"--template",
		"basic_agent",
		"--output",
		str(output),
		"--no-interactive",
	])

	assert result.exit_code == 0, result.output
	assert "python generated/app.py" in result.output
	assert "Flask-AppBuilder" not in result.output
	assert "default Flask-AppBuilder credentials" not in result.output

	readme = (output / "README.md").read_text(encoding="utf-8")
	requirements = (output / "requirements.txt").read_text(encoding="utf-8")
	config = (output / "config.py").read_text(encoding="utf-8")
	agent_tests = (output / "tests" / "test_agents.py").read_text(encoding="utf-8")
	apg_config = json.loads((output / "apg.json").read_text(encoding="utf-8"))

	assert "python generated/app.py" in readme
	assert "Python Manifest" in readme
	assert "standard library" in requirements
	assert "flask_appbuilder" not in config
	assert "Flask-AppBuilder" not in readme
	assert "def describe_application()" in agent_tests
	assert "set_value_api" not in agent_tests
	assert apg_config["target_language"] == "python"
	assert apg_config["target_framework"] == "python"


def test_cli_doctor_recognizes_spec_parser_artifacts():
	result = CliRunner().invoke(cli, ["doctor"])

	assert result.exit_code == 0, result.output
	assert "Generated parser found" in result.output
	assert "flask-appbuilder" not in result.output
	assert "django" not in result.output


def test_cli_doctor_json_emits_serviceability_contract():
	result = CliRunner().invoke(cli, ["doctor", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.doctor-report.v1"
	assert report["ok"] is True
	assert report["target_language"] == "Python"
	checks = {check["name"]: check for check in report["checks"]}
	assert checks["python_version"]["ok"] is True
	assert checks["parser_artifacts"]["message"] == "Generated parser found"
	assert checks["capability_contract_registry"]["contract_count"] >= 100
	assert report["summary"]["blocking_failure_count"] == 0


def test_cli_version_advertises_python_target_not_framework_target():
	result = CliRunner().invoke(cli, ["version"])

	assert result.exit_code == 0, result.output
	assert "Target Language: Python" in result.output
	assert "Executable Python application artifacts" in result.output
	assert "Flask-AppBuilder" not in result.output
	assert "Django" not in result.output


def test_compiler_error_rendering_handles_internal_node_less_errors():
	error = SemanticError("Unsupported target language: bad-target", None, "internal")

	assert str(error) == "unknown:0:0: internal error: Unsupported target language: bad-target"


def test_python_is_the_only_advertised_compiler_target():
	help_result = CliRunner().invoke(cli, ["compile", "--help"])

	assert help_result.exit_code == 0, help_result.output
	assert "[python]" in help_result.output
	assert "flask-appbuilder" not in help_result.output
	assert "django" not in help_result.output
	assert "fastapi" not in help_result.output
	assert APGCompiler().get_supported_targets() == ["python"]
	assert CodeGenerator.normalize_target("python") == "python"


def test_language_service_uses_shared_semantic_model_for_editor_features(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")
	snapshot = build_language_service_snapshot(source.read_text(encoding="utf-8"), source)
	model = snapshot["semantic_model"]

	assert snapshot["format"] == "apg.language-service-snapshot.v1"
	assert model["format"] == "apg.semantic-model.v1"
	assert snapshot["ok"] is True
	assert "table.Customer" in model["symbols"]
	assert "table.SalesOrder" in model["symbols"]
	assert {item["label"] for item in snapshot["completions"]} >= {"table", "Customer", "SalesOrder"}
	assert {symbol["name"] for symbol in snapshot["document_symbols"]} >= {"Customer", "SalesOrder"}
	assert "Customer" in hover(model, "Customer")["value"]
	assert definition(model, "Customer")["file"] == str(source)
	assert len(references(source.read_text(encoding="utf-8"), "Customer", source)) >= 2


def test_language_service_proposes_code_actions_for_invalid_source(tmp_path):
	source = tmp_path / "bad.apg"
	source.write_text("this is not apg", encoding="utf-8")
	snapshot = build_language_service_snapshot(source.read_text(encoding="utf-8"), source)
	actions = code_actions(snapshot["diagnostics"])

	assert snapshot["ok"] is False
	assert snapshot["diagnostics"][0]["code"] == "APG0001"
	assert actions[0]["command"] == "apg.format.check"


def test_cli_language_server_check_json_uses_shared_semantic_model(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["language-server", str(source), "--check", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.language-server-check.v1"
	assert report["ok"] is True
	assert report["semantic_model_format"] == "apg.semantic-model.v1"
	assert report["completion_count"] >= 10
	assert report["document_symbol_count"] >= 3
	assert "textDocument/completion" in report["capabilities"]
	assert "textDocument/codeAction" in report["capabilities"]


def test_language_service_rename_plan_updates_symbol_references_without_writing(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")
	snapshot = build_language_service_snapshot(source.read_text(encoding="utf-8"), source)

	plan = build_rename_plan(source.read_text(encoding="utf-8"), "Customer", "Account", source)

	assert plan["format"] == "apg.language-server-rename.v1"
	assert plan["ok"] is True
	assert plan["changed"] is True
	assert plan["replacement_count"] == 2
	assert plan["requires_review"] is True
	assert "table renames can affect database migrations" in plan["review_reasons"][0]
	assert "table Account" in plan["new_source"]
	assert "customer: Account;" in plan["new_source"]
	assert "--- " in plan["diff"]
	assert source.read_text(encoding="utf-8") == RELATIONSHIP_APP_SOURCE
	assert {symbol["name"] for symbol in workspace_symbols(snapshot["semantic_model"], "customer")} >= {
		"Customer",
		"SalesOrder.customer",
	}


def test_language_service_rename_blocks_ambiguous_field_symbols(tmp_path):
	source = tmp_path / "ambiguous.apg"
	source.write_text(
		"""
module ambiguous version 1.0.0 {
	description: "Ambiguous rename";
}

table Customer {
	name: str;
}

table Product {
	name: str;
}
""",
		encoding="utf-8",
	)

	plan = build_rename_plan(source.read_text(encoding="utf-8"), "name", "label", source)

	assert plan["ok"] is False
	assert plan["changed"] is False
	assert "ambiguous" in plan["errors"][0].lower()
	assert {candidate["id"] for candidate in plan["candidates"]} == {
		"field.Customer.name",
		"field.Product.name",
	}


def test_cli_language_server_rename_json_dry_runs_without_writing(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, [
		"language-server",
		str(source),
		"--rename",
		"Customer",
		"--to",
		"Account",
		"--json",
	])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.language-server-rename.v1"
	assert report["ok"] is True
	assert report["written"] is False
	assert report["replacement_count"] == 2
	assert "new_source" not in report
	assert source.read_text(encoding="utf-8") == RELATIONSHIP_APP_SOURCE


def test_language_service_code_actions_create_missing_table_patch(tmp_path):
	source = tmp_path / "missing_table.apg"
	source.write_text(
		"""
module missing_table version 1.0.0 {
	description: "Missing table";
}

table SalesOrder {
	customer: Customer;
}
""",
		encoding="utf-8",
	)

	report = build_code_action_report(source.read_text(encoding="utf-8"), source)
	action = next(item for item in report["actions"] if item["id"] == "create-table-Customer")

	assert report["format"] == "apg.language-server-code-actions.v1"
	assert report["ok"] is True
	assert action["title"] == "Create missing table Customer"
	assert action["requires_review"] is True
	assert action["metadata"]["created_symbol"] == "table.Customer"
	assert "table Customer" in action["new_source"]
	assert "new table Customer affects schema and migrations" in action["review_reasons"]


def test_cli_language_server_code_actions_json_dry_runs_without_writing(tmp_path):
	source = tmp_path / "missing_table.apg"
	source.write_text(
		"""
module missing_table version 1.0.0 {
	description: "Missing table";
}

table SalesOrder {
	customer: Customer;
}
""",
		encoding="utf-8",
	)

	result = CliRunner().invoke(cli, ["language-server", str(source), "--code-actions", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.language-server-code-actions.v1"
	assert report["action_count"] >= 1
	assert "create-table-Customer" in {action["id"] for action in report["actions"]}
	assert all("new_source" not in action for action in report["actions"])
	assert "table Customer" not in source.read_text(encoding="utf-8")


def test_cli_language_server_apply_code_action_writes_explicitly(tmp_path):
	source = tmp_path / "missing_table.apg"
	source.write_text(
		"""
module missing_table version 1.0.0 {
	description: "Missing table";
}

table SalesOrder {
	customer: Customer;
}
""",
		encoding="utf-8",
	)

	result = CliRunner().invoke(cli, [
		"language-server",
		str(source),
		"--code-actions",
		"--apply-action",
		"create-table-Customer",
		"--write",
		"--json",
	])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["written"] is True
	assert report["selected_action"]["id"] == "create-table-Customer"
	assert "table Customer" in source.read_text(encoding="utf-8")


def test_cli_language_server_audits_fixture_catalog():
	result = CliRunner().invoke(cli, ["language-server", "--audit-fixtures", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.language-server-fixture-audit.v1"
	assert report["ok"] is True
	assert report["missing_tags"] == []
	assert report["blocking_gaps"] == []
	assert report["summary"]["fixture_count"] >= 4
	assert report["summary"]["failing_fixture_count"] == 0
	assert {"check", "completion", "document_symbols", "formatting"}.issubset(report["tags_covered"])
	assert {"rename", "code_actions", "diagnostics", "source_immutability"}.issubset(report["tags_covered"])
	assert {fixture["operation"] for fixture in report["fixtures"]} >= {"check", "rename", "code_actions"}
	assert all(fixture["ok"] for fixture in report["fixtures"])


def test_cli_tooling_audit_json_runs_all_fixture_catalogs():
	result = CliRunner().invoke(cli, ["tooling", "audit", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.tooling-fixture-audit.v1"
	assert report["ok"] is True
	assert report["surface_count"] == 14
	assert report["summary"]["failing_surface_count"] == 0
	assert report["summary"]["blocking_gap_count"] == 0
	assert report["summary"]["error_count"] == 0
	surfaces = {surface["name"]: surface for surface in report["surfaces"]}
	assert set(surfaces) == {
		"parser_golden",
		"diagnostics",
		"lint",
		"formatter",
		"drift",
		"semantic_model",
		"graph",
		"language_server",
		"nl_plan",
		"migration",
		"release_evidence",
		"cli_surface",
		"ide_integration",
		"studio_designer",
	}
	assert all(surface["ok"] and surface["format_ok"] for surface in surfaces.values())
	assert surfaces["parser_golden"]["format"] == "apg.parser-golden-audit.v1"
	assert surfaces["lint"]["format"] == "apg.lint-fixture-audit.v1"
	assert surfaces["semantic_model"]["format"] == "apg.semantic-model-fixture-audit.v1"
	assert surfaces["release_evidence"]["format"] == "apg.release-evidence-fixture-audit.v1"


def test_vscode_extension_audit_tracks_current_cli_contracts():
	report = audit_vscode_extension(REPO_ROOT)

	assert report["format"] == "apg.ide-audit.v1"
	assert report["ok"] is True
	assert report["summary"]["failing"] == 0
	checks = {check["name"]: check for check in report["checks"]}
	assert checks["required-commands"]["details"]["missing"] == []
	assert checks["python-target-config"]["details"] == {"default": "python", "enum": ["python"]}
	assert checks["contributed-files-exist"]["details"]["missing"] == []
	assert checks["no-framework-target-drift"]["details"]["forbidden_fragments"] == []


def test_cli_ide_audit_json_reports_vscode_contracts():
	result = CliRunner().invoke(cli, ["ide", "audit", "--json"])

	assert result.exit_code == 0, result.output
	report = json.loads(result.output)
	assert report["format"] == "apg.ide-audit.v1"
	assert report["ok"] is True
	assert report["surface"] == "vscode"
	assert report["summary"]["check_count"] >= 6


def test_studio_snapshot_projects_dsl_into_designer_panels(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	report = build_studio_snapshot(source)

	assert report["format"] == "apg.studio-snapshot.v1"
	assert report["ok"] is True
	assert report["semantic_model_format"] == "apg.semantic-model.v1"
	assert report["round_trip"]["invalid_visual_edits_are_rejected"] is True
	assert {table["name"] for table in report["panels"]["database_designer"]["tables"]} == {
		"Customer",
		"SalesOrder",
	}
	assert {form["name"] for form in report["panels"]["form_designer"]["forms"]} >= {
		"CustomerForm",
		"SalesOrderForm",
	}
	assert "add_field" in report["round_trip"]["supported_edit_operations"]


def test_studio_visual_edit_plan_adds_field_without_writing(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	report = build_studio_edit_plan(
		source,
		{"operation": "add_field", "table": "Customer", "name": "phone", "type": "str"},
	)

	assert report["format"] == "apg.studio-edit-plan.v1"
	assert report["ok"] is True
	assert report["changed"] is True
	assert report["written"] is False
	assert "phone: str;" in report["new_source"]
	assert "form/database designer edit changes schema" in report["review_reasons"][0]
	assert "phone: str;" not in source.read_text(encoding="utf-8")


def test_studio_visual_edit_rejects_unknown_table(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	report = build_studio_edit_plan(
		source,
		{"operation": "add_field", "table": "Missing", "name": "phone", "type": "str"},
	)

	assert report["ok"] is False
	assert report["changed"] is False
	assert report["errors"] == ["Cannot add field to unknown table: Missing"]


def test_cli_studio_snapshot_and_plan_edit_json(tmp_path):
	source = tmp_path / "sales.apg"
	source.write_text(RELATIONSHIP_APP_SOURCE, encoding="utf-8")

	snapshot_result = CliRunner().invoke(cli, ["studio", "snapshot", str(source), "--json"])
	edit_result = CliRunner().invoke(cli, [
		"studio",
		"plan-edit",
		str(source),
		"--edit-json",
		json.dumps({"operation": "add_field", "table": "Customer", "name": "phone", "type": "str"}),
		"--json",
	])

	assert snapshot_result.exit_code == 0, snapshot_result.output
	snapshot = json.loads(snapshot_result.output)
	assert snapshot["format"] == "apg.studio-snapshot.v1"
	assert snapshot["panels"]["database_designer"]["tables"]
	assert edit_result.exit_code == 0, edit_result.output
	edit = json.loads(edit_result.output)
	assert edit["format"] == "apg.studio-edit-plan.v1"
	assert edit["ok"] is True
	assert edit["written"] is False
	assert "new_source" not in edit
	assert "phone: str;" not in source.read_text(encoding="utf-8")


def test_framework_names_are_not_silent_compiler_target_aliases():
	result = CliRunner().invoke(cli, [
		"compile",
		"baseline.apg",
		"--target",
		"flask-appbuilder",
	])

	assert result.exit_code != 0
	assert "Invalid value for '--target'" in result.output
