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
	"fab/flask_appbuilder/static/appbuilder/js/swagger-ui-bundle.js",
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
PYTHON_FIRST_PUBLIC_DOCS = {
	"README.md",
	"docs/README.md",
	"docs/architecture.md",
	"docs/capabilities/README.md",
	"docs/language_reference.md",
	"docs/proposed_capability_architecture.md",
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
