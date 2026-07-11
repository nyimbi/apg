from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
	pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
	match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', pyproject)
	assert match is not None
	return match.group(1)


def test_package_importable():
	import apg

	assert apg is not None


def test_cli_entry_point():
	result = subprocess.run(
		["apg", "--help"],
		capture_output=True,
		text=True,
		check=False,
	)

	assert result.returncode == 0, result.stderr or result.stdout
	for command in ("init", "doctor", "compile", "serve", "export"):
		assert command in result.stdout


def test_version_consistent():
	assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == _pyproject_version()


def test_changelog_exists():
	changelog = ROOT / "CHANGELOG.md"

	assert changelog.exists()
	assert "Wave A" in changelog.read_text(encoding="utf-8")
