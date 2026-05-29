"""Repository layout hygiene audit for APG."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable


REPOSITORY_HYGIENE_AUDIT_FORMAT = "apg.repository-hygiene-audit.v1"

ALLOWED_ROOT_TRACKED_FILES = {
	".gitignore",
	"LICENSE",
	"README.md",
	"cli.py",
	"pytest.ini",
	"setup.py",
	"uuid_extensions.py",
}
ALLOWED_ROOT_MARKDOWN = {"README.md"}
SOURCE_ROOT_OPERATIONAL_DOC_DIRECTORIES = {"capabilities", "gen", "mobile_apps"}
SOURCE_ROOT_OPERATIONAL_DOC_SUFFIXES = (
	"_COMPLETE.md",
	"_COMPLETION.md",
	"_PLAN.md",
	"_SUMMARY.md",
	"_STATUS.md",
)
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
	"compiler/repository_hygiene.py",
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
ROOT_RUNTIME_OUTPUT_PATHS = {
	".pytest_cache",
	"__pycache__",
	"apg_demo_output",
	"apg_language.egg-info",
	"audit_logs",
	"singer_state",
	"tmp",
	"uploads",
}

CheckFunction = Callable[[Path, list[str], list[str]], list[str]]


def audit_repository_hygiene(repo_root: Path | None = None) -> dict[str, object]:
	"""Return APG repository hygiene evidence for tracked project files."""
	root = repo_root or Path(__file__).resolve().parents[1]
	tracked_files = _tracked_files(root)
	tracked_index_entries = _tracked_index_entries(root)
	checks = [
		_run_check("generated_cache_artifacts_not_tracked", root, tracked_files, tracked_index_entries, _generated_cache_artifacts_not_tracked),
		_run_check("root_runtime_output_directories_not_tracked", root, tracked_files, tracked_index_entries, _root_runtime_output_directories_not_tracked),
		_run_check("root_tracked_files_intentional_and_minimal", root, tracked_files, tracked_index_entries, _root_tracked_files_intentional_and_minimal),
		_run_check("root_tests_and_docs_expected_directories", root, tracked_files, tracked_index_entries, _root_tests_and_docs_expected_directories),
		_run_check("operational_markdown_lives_under_docs_archive", root, tracked_files, tracked_index_entries, _operational_markdown_lives_under_docs_archive),
		_run_check("root_dependency_files_python_first", root, tracked_files, tracked_index_entries, _root_dependency_files_python_first),
		_run_check("legacy_framework_submodules_not_tracked", root, tracked_files, tracked_index_entries, _legacy_framework_submodules_not_tracked),
		_run_check("package_metadata_does_not_install_framework_targets", root, tracked_files, tracked_index_entries, _package_metadata_does_not_install_framework_targets),
		_run_check("top_level_generated_and_capability_tests_out_of_source_roots", root, tracked_files, tracked_index_entries, _top_level_generated_and_capability_tests_out_of_source_roots),
		_run_check("project_templates_describe_python_artifact_flow", root, tracked_files, tracked_index_entries, _project_templates_describe_python_artifact_flow),
		_run_check("composable_capability_docs_framework_neutral", root, tracked_files, tracked_index_entries, _composable_capability_docs_framework_neutral),
		_run_check("composable_integration_templates_framework_neutral", root, tracked_files, tracked_index_entries, _composable_integration_templates_framework_neutral),
		_run_check("composable_engine_glue_framework_neutral", root, tracked_files, tracked_index_entries, _composable_engine_glue_framework_neutral),
		_run_check("composable_model_and_view_templates_framework_neutral", root, tracked_files, tracked_index_entries, _composable_model_and_view_templates_framework_neutral),
		_run_check("composable_base_names_python_first", root, tracked_files, tracked_index_entries, _composable_base_names_python_first),
		_run_check("status_reports_describe_python_first_platform_defaults", root, tracked_files, tracked_index_entries, _status_reports_describe_python_first_platform_defaults),
		_run_check("apg_streaming_runtime_bytewax_native", root, tracked_files, tracked_index_entries, _apg_streaming_runtime_bytewax_native),
	]
	violations = [
		{"check": check["name"], "violation": violation}
		for check in checks
		for violation in check["violations"]
	]
	passing_check_count = sum(1 for check in checks if check["ok"])
	return {
		"format": REPOSITORY_HYGIENE_AUDIT_FORMAT,
		"ok": not violations,
		"repo_root": str(root),
		"scope": "tracked_files",
		"tracked_file_count": len(tracked_files),
		"checks": checks,
		"summary": {
			"check_count": len(checks),
			"passing_check_count": passing_check_count,
			"failing_check_count": len(checks) - passing_check_count,
			"violation_count": len(violations),
		},
		"violations": violations,
		"blocking_gaps": violations,
	}


def _run_check(
	name: str,
	root: Path,
	tracked_files: list[str],
	tracked_index_entries: list[str],
	check: CheckFunction,
) -> dict[str, object]:
	violations = check(root, tracked_files, tracked_index_entries)
	return {
		"name": name,
		"ok": not violations,
		"violation_count": len(violations),
		"violations": violations,
	}


def _tracked_files(root: Path) -> list[str]:
	result = subprocess.run(
		["git", "ls-files"],
		cwd=root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.splitlines()


def _tracked_index_entries(root: Path) -> list[str]:
	result = subprocess.run(
		["git", "ls-files", "--stage"],
		cwd=root,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.splitlines()


def _generated_cache_artifacts_not_tracked(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	return [
		path for path in tracked_files
		if path == ".DS_Store"
		or path.endswith("/.DS_Store")
		or "/__pycache__/" in path
		or path.endswith((".pyc", ".pyo"))
	]


def _root_runtime_output_directories_not_tracked(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	return [
		path for path in tracked_files
		if path.split("/", 1)[0] in ROOT_RUNTIME_OUTPUT_PATHS
	]


def _root_tracked_files_intentional_and_minimal(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	root_files = sorted(path for path in tracked_files if "/" not in path)
	expected = sorted(ALLOWED_ROOT_TRACKED_FILES)
	if root_files == expected:
		return []
	return [
		f"unexpected root tracked files: {', '.join(path for path in root_files if path not in ALLOWED_ROOT_TRACKED_FILES)}",
		f"missing allowed root tracked files: {', '.join(path for path in expected if path not in root_files)}",
	]


def _root_tests_and_docs_expected_directories(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	return [
		path for path in tracked_files
		if "/" not in path and (
			path.startswith("test_")
			or path.endswith("_test.py")
			or (path.endswith(".md") and path not in ALLOWED_ROOT_MARKDOWN)
		)
	]


def _operational_markdown_lives_under_docs_archive(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	return [
		path for path in tracked_files
		if Path(path).parent.as_posix() in SOURCE_ROOT_OPERATIONAL_DOC_DIRECTORIES
		and Path(path).suffix == ".md"
		and Path(path).name != "README.md"
		and Path(path).name.endswith(SOURCE_ROOT_OPERATIONAL_DOC_SUFFIXES)
	]


def _root_dependency_files_python_first(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	forbidden = {
		"requirements_flask_appbuilder.txt",
		"requirements_django.txt",
		"requirements_fastapi.txt",
	}
	return sorted(forbidden.intersection(tracked_files))


def _legacy_framework_submodules_not_tracked(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	tracked = set(tracked_files)
	if "fab" in tracked:
		violations.append("fab")
	if ".gitmodules" in tracked:
		violations.append(".gitmodules")
	gitlinks = [
		entry.rsplit("\t", 1)[-1] for entry in tracked_index_entries
		if entry.startswith("160000 ")
	]
	if "fab" in gitlinks:
		violations.append("gitlink:fab")
	return violations


def _package_metadata_does_not_install_framework_targets(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	setup_source = (root / "setup.py").read_text(encoding="utf-8")
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
	return [
		f"setup.py: {term}"
		for term in forbidden_terms
		if term in setup_source
	]


def _top_level_generated_and_capability_tests_out_of_source_roots(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	forbidden_prefixes = ("capabilities/", "gen/")
	return [
		path for path in tracked_files
		if path.startswith(forbidden_prefixes)
		and Path(path).parent.as_posix() in {"capabilities", "gen"}
		and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
	]


def _project_templates_describe_python_artifact_flow(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
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
		violations.extend(_term_violations(root, path, PYTHON_TEMPLATE_FORBIDDEN_TERMS))
	return violations


def _composable_capability_docs_framework_neutral(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
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
		violations.extend(_term_violations(root, path, COMPOSABLE_CAPABILITY_DOC_FORBIDDEN_TERMS))
	return violations


def _composable_integration_templates_framework_neutral(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
		if not (
			path.startswith("templates/composable/capabilities/")
			and path.endswith("/integration.py.template")
		):
			continue
		violations.extend(_term_violations(root, path, (
			"from flask import Blueprint",
			"flask_appbuilder",
			"Flask-AppBuilder",
			"appbuilder",
			"SQLALCHEMY_DATABASE_URI",
		)))
	return violations


def _composable_engine_glue_framework_neutral(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in [
		"templates/composable/composition_engine.py",
		"templates/composable/capabilities/auth/role_based_access_control/capability.json",
	]:
		violations.extend(_term_violations(root, path, (
			"appbuilder",
			"flask-principal",
			"Flask-AppBuilder",
			"flask_appbuilder",
		)))
	return violations


def _composable_model_and_view_templates_framework_neutral(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
		if not (
			path.startswith("templates/composable/capabilities/")
			and (
				path.endswith("/models/__init__.py.template")
				or path.endswith("/views/__init__.py.template")
			)
		):
			continue
		violations.extend(_term_violations(root, path, (
			"flask_appbuilder",
			"Flask-AppBuilder",
			"SQLAInterface",
			"AuditMixin",
			"from sqlalchemy",
			"sqlalchemy",
			"Column(",
			"relationship(",
			"has_access",
		)))
	return violations


def _composable_base_names_python_first(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
		if not path.startswith("templates/composable/"):
			continue
		candidate = root / path
		if not candidate.is_file():
			continue
		violations.extend(_term_violations(root, path, ("flask_webapp", "FLASK_WEBAPP")))
	return violations


def _status_reports_describe_python_first_platform_defaults(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in PYTHON_FIRST_REPORTS:
		violations.extend(_term_violations(root, path, (
			"Flask-AppBuilder",
			"FastAPI Integration",
			"Dynamic Flask integration",
			"Flask, SQLAlchemy",
			"Flask Web Application",
			"python app.py",
		)))
	return violations


def _apg_streaming_runtime_bytewax_native(root: Path, tracked_files: list[str], tracked_index_entries: list[str]) -> list[str]:
	violations: list[str] = []
	for path in tracked_files:
		if path in STREAMING_TERM_EXCLUDED_PATHS:
			continue
		if path.startswith(STREAMING_TERM_EXCLUDED_PREFIXES):
			continue
		candidate = root / path
		if not candidate.is_file():
			continue
		content = candidate.read_text(encoding="utf-8", errors="ignore").lower()
		for term in FORBIDDEN_STREAMING_RUNTIME_TERMS:
			if term in content:
				violations.append(f"{path}: {term}")
	return violations


def _term_violations(root: Path, path: str, terms: tuple[str, ...]) -> list[str]:
	content = (root / path).read_text(encoding="utf-8", errors="ignore")
	return [
		f"{path}: {term}"
		for term in terms
		if term in content
	]
