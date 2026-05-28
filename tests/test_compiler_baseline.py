"""Focused compiler baseline regressions for documented APG invocation."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import time
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


def test_documented_python_target_generates_executable_application_files():
	result = compile_apg_string(MINIMAL_AGENT_SOURCE)

	assert result.success is True
	assert result.target_language == "python"
	assert "app.py" in result.generated_files
	assert "ai_agents.py" in result.generated_files
	app = result.generated_files["app.py"]
	assert "APG Python Application" in app
	assert "Flask-AppBuilder" not in app
	assert "flask_appbuilder" not in app
	assert "django" not in app.lower()
	assert "HTTPServer" in app
	assert "run_server" in app
	compile(app, "app.py", "exec")


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
	finally:
		sys.modules.pop("generated_pkg", None)
		for name in list(sys.modules):
			if name.startswith("generated_pkg."):
				sys.modules.pop(name, None)

	assert module.__version__ == "1.0.0"
	assert module.list_entities() == [
		{"name": "Planner", "type": "ai_agent", "properties": [], "methods": []}
	]
	assert module.list_records("Planner") == []
	assert module.list_agents() == ["Planner"]
	assert manifest["ai_agents"] == ["Planner"]
	assert module.validate_application()["valid"] is True
	assert "describe_application" in module.__all__
	assert "validate_application" in module.__all__
	assert "list_records" in module.__all__
	assert "list_agents" in module.__all__


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
		with urllib.request.urlopen(f"{base_url}/agents", timeout=1) as response:
			agents = json.loads(response.read().decode("utf-8"))
		with urllib.request.urlopen(f"{base_url}/validate", timeout=1) as response:
			validation = json.loads(response.read().decode("utf-8"))
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert health["status"] == "ok"
	assert manifest["name"] == "baseline"
	assert agents["agents"]["Planner"]["runtime"] == "codex"
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
	finally:
		process.terminate()
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			process.kill()
			process.wait(timeout=2)

	assert created["entity"] == "Customer"
	assert created["record"] == {"id": 1, "name": "Asha", "email": "asha@example.com"}
	assert created["count"] == 1
	assert listed["records"] == [created["record"]]
	assert fetched["record"] == created["record"]
	assert all_records["records"]["Customer"] == [created["record"]]


def test_cli_compile_default_target_writes_generated_application(tmp_path):
	source = tmp_path / "baseline.apg"
	output = tmp_path / "generated"
	source.write_text(MINIMAL_AGENT_SOURCE, encoding="utf-8")

	result = CliRunner().invoke(cli, ["compile", str(source), "--output", str(output), "--verbose"])

	assert result.exit_code == 0, result.output
	assert "Compilation successful" in result.output
	assert f"python {output}/app.py" in result.output
	assert f"python {output}/app.py --describe" in result.output
	assert "standard-library HTTP server" in result.output
	assert (output / "app.py").exists()
	assert (output / "ai_agents.py").exists()
	app = (output / "app.py").read_text(encoding="utf-8")
	requirements = (output / "requirements.txt").read_text(encoding="utf-8")
	assert "APG Python Application" in app
	assert "HTTPServer" in app
	assert "Flask-AppBuilder" not in app
	assert "flask_appbuilder" not in requirements
	assert "standard library" in requirements


def test_cli_init_describes_python_artifact_flow():
	runner = CliRunner()
	with runner.isolated_filesystem():
		result = runner.invoke(cli, ["init"])

	assert result.exit_code == 0, result.output
	assert "generate Python artifacts" in result.output
	assert "python generated/app.py" in result.output
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
