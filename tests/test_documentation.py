"""
Wave W — Documentation tests.

Validates that the MkDocs site config is well-formed and that every page
referenced in the nav exists and has meaningful content.
"""

from pathlib import Path
import yaml

DOCS_ROOT = Path(__file__).parent.parent / "docs" / "site"
MKDOCS_YML = DOCS_ROOT / "mkdocs.yml"
DOCS_DIR = DOCS_ROOT / "docs"


def _nav_paths(nav_node, paths=None):
	"""Recursively collect all relative paths declared in the nav."""
	if paths is None:
		paths = []
	if isinstance(nav_node, dict):
		for v in nav_node.values():
			_nav_paths(v, paths)
	elif isinstance(nav_node, list):
		for item in nav_node:
			_nav_paths(item, paths)
	elif isinstance(nav_node, str):
		paths.append(nav_node)
	return paths


def test_mkdocs_yml_valid():
	"""mkdocs.yml parses as valid YAML without errors."""
	assert MKDOCS_YML.exists(), f"mkdocs.yml not found at {MKDOCS_YML}"
	with MKDOCS_YML.open() as fh:
		config = yaml.safe_load(fh)
	assert isinstance(config, dict), "mkdocs.yml must be a YAML mapping"
	assert "site_name" in config, "mkdocs.yml must have a site_name"
	assert "nav" in config, "mkdocs.yml must have a nav"


def test_all_nav_pages_exist():
	"""Every path listed in the nav must correspond to an existing file."""
	with MKDOCS_YML.open() as fh:
		config = yaml.safe_load(fh)
	nav = config.get("nav", [])
	paths = _nav_paths(nav)
	assert paths, "nav must not be empty"
	missing = []
	for rel_path in paths:
		full = DOCS_DIR / rel_path
		if not full.exists():
			missing.append(rel_path)
	assert not missing, f"Nav references missing files: {missing}"


def test_config_page_covers_secrets():
	"""configuration.md must mention the three critical env vars."""
	config_page = DOCS_DIR / "generated" / "configuration.md"
	assert config_page.exists(), "generated/configuration.md does not exist"
	text = config_page.read_text()
	for var in ("APG_SECRET_KEY", "APG_AUTH_USERS", "APG_PRODUCTION"):
		assert var in text, f"configuration.md must mention {var}"


def test_quickstart_has_code_blocks():
	"""quickstart.md must have at least two fenced code blocks."""
	qs = DOCS_DIR / "getting-started" / "quickstart.md"
	assert qs.exists(), "getting-started/quickstart.md does not exist"
	text = qs.read_text()
	fence_count = text.count("```")
	# Each code block uses two ``` markers (open + close)
	block_count = fence_count // 2
	assert block_count >= 2, (
		f"quickstart.md must have at least 2 code blocks, found {block_count}"
	)
