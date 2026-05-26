"""Repository layout hygiene checks for APG."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_MARKDOWN = {"README.md"}


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
