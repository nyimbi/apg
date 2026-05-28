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

from click.testing import CliRunner

from cli.main import cli
from compiler.code_generator import CodeGenerator
from compiler.compiler import APGCompiler, compile_apg_string
from compiler.semantic_analyzer import SemanticError


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


def test_documented_python_target_generates_executable_application_files():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)

	assert result.success is True
	assert result.target_language == "python"
	assert "app.py" in result.generated_files
	assert "ai_agents.py" in result.generated_files
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
	assert "## Browser UI" in readme
	assert "create, edit, delete, and validation-error flows" in readme
	assert "_revision` checks" in readme
	assert "docker build -t apg-generated-app ." in readme
	assert "POST /agents/Planner/invoke" in readme
	assert "component_manifest" in smoke_test
	assert "openapi_contract" in smoke_test
	assert "route_dispatch" in smoke_test
	assert "capability_health" in smoke_test
	compile(app, "app.py", "exec")
	compile(smoke_test, "smoke_test.py", "exec")


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
	assert "component_manifest" in component["interfaces"]["python"]["exports"]
	assert "auth_status" in component["interfaces"]["python"]["exports"]
	assert "create_record" in component["interfaces"]["python"]["exports"]
	assert "database_status" in component["interfaces"]["python"]["exports"]
	assert "list_entities" in component["interfaces"]["python"]["exports"]
	assert "list_agents" in component["interfaces"]["python"]["exports"]
	assert "invoke_agent" in component["interfaces"]["python"]["exports"]
	assert "query_records" in component["interfaces"]["python"]["exports"]
	assert "runtime_adapter_environment_keys" in component["interfaces"]["python"]["exports"]
	assert "update_record" in component["interfaces"]["python"]["exports"]
	assert "delete_record" in component["interfaces"]["python"]["exports"]
	assert "validate_agent_runtimes" in component["interfaces"]["python"]["exports"]
	assert "validate_component_manifest_contract" in component["interfaces"]["python"]["exports"]
	assert "validate_route_dispatch_contract" in component["interfaces"]["python"]["exports"]
	assert invocation["agent"] == "Planner"
	assert invocation["runtime"] == "codex"
	assert invocation["status"] == "adapter_required"
	assert invocation["input"] == {"task": "plan"}
	assert self_test["passed"] is True
	assert "/self-test" in self_test["routes"]
	assert self_test["checks"]["entity_count"] == 1
	assert self_test["checks"]["validation"]["checks"]["openapi_contract"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["component_manifest"]["errors"] == []
	assert self_test["checks"]["validation"]["checks"]["route_dispatch"]["errors"] == []
	assert openapi_contract["errors"] == []
	assert "AgentInvocationRequest" in openapi_contract["referenced_schemas"]
	assert component_contract["errors"] == []
	assert component_contract["artifact_count"] >= 8
	assert "validate_component_manifest_contract" in component_contract["python_exports"]
	assert route_contract["errors"] == []
	assert route_contract["routes"]["/agents/Planner/invoke"][0]["target"] == "_agent_invocation_payload"
	assert {"method": "GET", "target": "_records_payload_with_query"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert {"method": "PUT", "target": "_update_record_payload"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert {"method": "DELETE", "target": "_delete_record_payload"} in route_contract["routes"]["/entities/Planner/records/{id}"]
	assert module.auth_status()["mode"] == "open"
	assert module.metrics_snapshot()["entity_count"] == 1
	openapi = module.openapi_document()
	assert openapi["openapi"] == "3.1.0"
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
	assert "storage_status" in module.__all__
	assert "update_record" in module.__all__
	assert "validate_application" in module.__all__
	assert "validate_component_manifest_contract" in module.__all__
	assert "validate_openapi_contract" in module.__all__
	assert "validate_route_dispatch_contract" in module.__all__
	assert "validate_record" in module.__all__
	assert "list_records" in module.__all__
	assert "list_agents" in module.__all__
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
	assert "/agents/Planner/invoke" in component["interfaces"]["http"]["paths"]
	assert "/openapi.json" in component["interfaces"]["http"]["paths"]
	assert agents["agents"]["Planner"]["runtime"] == "codex"
	assert invocation["agent"] == "Planner"
	assert invocation["status"] == "adapter_required"
	assert invocation["input"] == {"ticket": "reset password"}
	assert "/agents/Planner/invoke" in openapi["paths"]
	assert "/component.json" in openapi["paths"]
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


def test_framework_names_are_not_silent_compiler_target_aliases():
	result = CliRunner().invoke(cli, [
		"compile",
		"baseline.apg",
		"--target",
		"flask-appbuilder",
	])

	assert result.exit_code != 0
	assert "Invalid value for '--target'" in result.output
