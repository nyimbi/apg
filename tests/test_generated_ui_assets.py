"""Generated UI asset pipeline regressions."""

from __future__ import annotations

import re
from pathlib import Path

from compiler.compiler import compile_apg_file


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "compiler" / "templates"
GENERATOR_SOURCE = REPO_ROOT / "compiler" / "code_generator.py"
APG_CSS = REPO_ROOT / "compiler" / "assets" / "apg.css"
EXAMPLE_20 = REPO_ROOT / "examples" / "20_enterprise_erp_platform" / "main.apg"

EXTERNAL_MARKERS = (
	"cdn.",
	"unpkg.com",
	"jsdelivr.net",
	"googleapis.com",
	"http://",
	"https://",
)

CONTROL_TOKENS = {
	"if",
	"elif",
	"else",
	"endif",
	"in",
	"lower",
	"page",
	"string",
	"field_val",
	"fv",
	"stars",
	"status_val",
}


def _class_tokens_from_markup(markup: str) -> set[str]:
	tokens: set[str] = set()
	for match in re.finditer(r"class\s*=\s*([\"'])(.*?)\1", markup, re.DOTALL):
		for token in re.split(r"\s+", match.group(2).strip()):
			if _is_static_class_token(token):
				tokens.add(token)
	return tokens


def _class_tokens_from_generator(source: str) -> set[str]:
	tokens: set[str] = set()
	for match in re.finditer(r"class(?:Name)?=[\"']([^\"']+)[\"']", source):
		for token in re.split(r"\s+", match.group(1).strip()):
			if _is_static_class_token(token):
				tokens.add(token)
	return tokens


def _is_static_class_token(token: str) -> bool:
	if not token:
		return False
	if token in CONTROL_TOKENS:
		return False
	if any(part in token for part in ("{{", "}}", "{%", "%}", "|", "==", "<", ">", "=", "'", '"')):
		return False
	if "." in token and not re.search(r"\d\.\d", token):
		return False
	return re.match(r"^-?[A-Za-z0-9_:/\.\[\]+-]+$", token) is not None


def _css_contains_class(css: str, class_name: str) -> bool:
	escaped = (
		class_name
		.replace("\\", "\\\\")
		.replace(":", "\\:")
		.replace("/", "\\/")
		.replace(".", "\\.")
		.replace("[", "\\[")
		.replace("]", "\\]")
	)
	return f".{escaped}" in css or f".{class_name}" in css


def test_apg_css_covers_all_template_classes():
	css = APG_CSS.read_text(encoding="utf-8")
	classes: set[str] = set()
	for template in TEMPLATE_DIR.rglob("*.j2"):
		classes.update(_class_tokens_from_markup(template.read_text(encoding="utf-8")))
	classes.update(_class_tokens_from_generator(GENERATOR_SOURCE.read_text(encoding="utf-8")))

	ignored_prefixes = ("dark:",)
	ignored_classes = {
		"apg-row-cb",
		"apg-select-all",
		"group",
		"group/card",
		"group/field",
		"group/row",
	}
	missing = sorted(
		class_name
		for class_name in classes
		if class_name not in ignored_classes
		and not class_name.startswith(ignored_prefixes)
		and not class_name.startswith("group-hover")
		and not _css_contains_class(css, class_name)
	)

	assert missing == []


def test_no_external_urls_in_generated_output():
	result = compile_apg_file(str(EXAMPLE_20))

	assert result.success, result.errors
	for required in ("static/apg.css", "static/htmx.min.js", "static/sortable.min.js"):
		assert required in result.generated_files
		assert result.generated_files[required].strip()

	ui_files = {
		path: content
		for path, content in result.generated_files.items()
		if path == "app.py" or path.startswith("static/")
	}
	violations = {
		path: [marker for marker in EXTERNAL_MARKERS if marker in content]
		for path, content in ui_files.items()
	}
	violations = {path: markers for path, markers in violations.items() if markers}

	assert violations == {}
