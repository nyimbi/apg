"""Repository layout hygiene checks for APG."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_MARKDOWN = {"README.md"}
FORBIDDEN_STREAMING_RUNTIME_TERMS = (
	"kafka",
	"confluent",
	"redpanda",
	"bootstrap.servers",
	"bootstrap_servers",
	"bytewax_brokers",
	"bytewax broker",
	"bytewax brokers",
	"broker connection string",
)
STREAMING_TERM_EXCLUDED_PREFIXES = (
	"tmp/",
	"uploads/",
)
STREAMING_TERM_EXCLUDED_PATHS = {
	"docs/progress_log.md",
	"tests/test_repository_hygiene.py",
}
PYTHON_TEMPLATE_FORBIDDEN_TERMS = (
	"Flask-AppBuilder",
	"flask_appbuilder",
	"FastAPI",
	"fastapi",
	"Django",
	"django",
	"python app.py",
	"http://localhost:8080",
	"Flask>=2.3.0",
	"SQLAlchemy>=2.0.0",
)
COMPOSABLE_CAPABILITY_DOC_FORBIDDEN_TERMS = (
	"Flask-AppBuilder",
	"flask_appbuilder",
	"FastAPI",
	"fastapi",
	"Flask-SocketIO",
	"http://localhost:8080",
	"python app.py",
)
PYTHON_FIRST_PUBLIC_DOCS = {
	"README.md",
	"docs/README.md",
	"docs/architecture.md",
	"docs/capabilities/README.md",
	"docs/language_reference.md",
	"docs/proposed_capability_architecture.md",
}
PYTHON_FIRST_REPORTS = {
	"docs/reports/system_capabilities_report.md",
	"docs/reports/final_system_report.md",
	"docs/reports/final_system_summary.md",
	"docs/reports/marketplace_completion_report.md",
}


def _tracked_files() -> list[str]:
	result = subprocess.run(
		["git", "ls-files"],
		cwd=REPO_ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.splitlines()


def _tracked_index_entries() -> list[str]:
	result = subprocess.run(
		["git", "ls-files", "--stage"],
		cwd=REPO_ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.splitlines()


def test_generated_cache_artifacts_are_not_tracked():
	forbidden = [
		path for path in _tracked_files()
		if path == ".DS_Store"
		or path.endswith("/.DS_Store")
		or "/__pycache__/" in path
		or path.endswith((".pyc", ".pyo"))
	]

	assert forbidden == []


def test_root_tests_and_docs_stay_in_expected_directories():
	misplaced = [
		path for path in _tracked_files()
		if "/" not in path and (
			path.startswith("test_")
			or path.endswith("_test.py")
			or (path.endswith(".md") and path not in ALLOWED_ROOT_MARKDOWN)
		)
	]

	assert misplaced == []


def test_root_dependency_files_stay_python_first():
	forbidden = {
		"requirements_flask_appbuilder.txt",
		"requirements_django.txt",
		"requirements_fastapi.txt",
	}

	assert sorted(forbidden.intersection(_tracked_files())) == []


def test_legacy_framework_submodules_are_not_tracked():
	tracked = set(_tracked_files())
	assert "fab" not in tracked
	assert ".gitmodules" not in tracked

	gitlinks = [
		entry.rsplit("\t", 1)[-1] for entry in _tracked_index_entries()
		if entry.startswith("160000 ")
	]

	assert "fab" not in gitlinks


def test_package_metadata_does_not_install_framework_targets_by_default():
	setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
	forbidden_terms = (
		"Flask>=",
		"Flask-AppBuilder",
		"Flask-SQLAlchemy",
		"fastapi>=",
		"uvicorn>=",
		"SQLAlchemy>=",
		"flask-appbuilder",
		"Web Environment",
		"WWW/HTTP",
	)

	for term in forbidden_terms:
		assert term not in setup_source


def test_top_level_generated_and_capability_tests_stay_out_of_source_roots():
	forbidden_prefixes = ("capabilities/", "gen/")
	misplaced = [
		path for path in _tracked_files()
		if path.startswith(forbidden_prefixes)
		and Path(path).parent.as_posix() in {"capabilities", "gen"}
		and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
	]

	assert misplaced == []


def test_project_templates_describe_python_artifact_flow():
	violations: list[str] = []
	for path in _tracked_files():
		if not (
			path.startswith("templates/templates/")
			or path.startswith("templates/application_templates/")
			or path == "templates/application_template_manager.py"
			or path == "templates/template_manager.py"
			or path == "scripts/template_generation/create_template_structure.py"
			or path == "cli/compile_command.py"
			or path in PYTHON_FIRST_PUBLIC_DOCS
		):
			continue
		if not path.endswith((".md.template", ".txt.template", ".py.template", ".json", ".py")):
			continue
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in PYTHON_TEMPLATE_FORBIDDEN_TERMS:
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_composable_capability_docs_do_not_advertise_framework_runtime():
	violations: list[str] = []
	for path in _tracked_files():
		if path.startswith("templates/composable/capabilities/"):
			if not path.endswith(("README.md", "API.md", "requirements.txt", "capability.json", ".py")):
				continue
		elif path.startswith("templates/composable/bases/"):
			if not path.endswith((
				"README.md.template",
				"requirements.txt.template",
				"__init__.py.template",
				"app.py.template",
				"config.py.template",
				"base.json",
			)):
				continue
		else:
			continue
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in COMPOSABLE_CAPABILITY_DOC_FORBIDDEN_TERMS:
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_composable_integration_templates_are_framework_neutral():
	violations: list[str] = []
	for path in _tracked_files():
		if not (
			path.startswith("templates/composable/capabilities/")
			and path.endswith("/integration.py.template")
		):
			continue
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in (
			"from flask import Blueprint",
			"flask_appbuilder",
			"Flask-AppBuilder",
			"appbuilder",
			"SQLALCHEMY_DATABASE_URI",
		):
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_composable_engine_glue_is_framework_neutral():
	violations: list[str] = []
	paths = [
		"templates/composable/composition_engine.py",
		"templates/composable/capabilities/auth/role_based_access_control/capability.json",
	]

	for path in paths:
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in (
			"appbuilder",
			"flask-principal",
			"Flask-AppBuilder",
			"flask_appbuilder",
		):
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_composable_model_and_view_templates_are_framework_neutral():
	violations: list[str] = []
	for path in _tracked_files():
		if not (
			path.startswith("templates/composable/capabilities/")
			and (
				path.endswith("/models/__init__.py.template")
				or path.endswith("/views/__init__.py.template")
			)
		):
			continue
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in (
			"flask_appbuilder",
			"Flask-AppBuilder",
			"SQLAInterface",
			"AuditMixin",
			"from sqlalchemy",
			"sqlalchemy",
			"Column(",
			"relationship(",
			"has_access",
		):
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_composable_base_names_are_python_first():
	violations: list[str] = []
	for path in _tracked_files():
		if not path.startswith("templates/composable/"):
			continue
		candidate = REPO_ROOT / path
		if not candidate.is_file():
			continue
		content = candidate.read_text(encoding="utf-8", errors="ignore")
		for term in ("flask_webapp", "FLASK_WEBAPP"):
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_status_reports_describe_python_first_platform_defaults():
	violations: list[str] = []
	for path in PYTHON_FIRST_REPORTS:
		content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
		for term in (
			"Flask-AppBuilder",
			"FastAPI Integration",
			"Dynamic Flask integration",
			"Flask, SQLAlchemy",
			"Flask Web Application",
			"python app.py",
		):
			if term in content:
				violations.append(f"{path}: {term}")

	assert violations == []


def test_apg_streaming_runtime_stays_bytewax_native():
	violations: list[str] = []

	for path in _tracked_files():
		if path in STREAMING_TERM_EXCLUDED_PATHS:
			continue
		if path.startswith(STREAMING_TERM_EXCLUDED_PREFIXES):
			continue

		candidate = REPO_ROOT / path
		if not candidate.is_file():
			continue

		content = candidate.read_text(encoding="utf-8", errors="ignore")
		lowered = content.lower()
		for term in FORBIDDEN_STREAMING_RUNTIME_TERMS:
			if term in lowered:
				violations.append(f"{path}: {term}")

	assert violations == []
