"""Renderer tests: the HTML emitter and the stylesheet it promises to use.

``compiler/openui_lang/render.py`` claims every class it emits already ships in
``compiler/assets/apg.css``, which is what keeps a generated app free of a build
step. That claim is the gate here: the test reads the stylesheet and refuses a
class the renderer invents. The check runs in both directions -- a class in the
HTML must be defined, and every class the renderer's own tables name must be
defined -- because a one-directional sweep fails open on the side nobody wrote.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.openui_lang import Library, Program
from compiler.openui_library import build_library
from compiler.openui_lang.render import (
	class_table,
	component_classes,
	render_document,
	render_html,
	supported_components,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures" / "openui" / "1.0"
STYLESHEET = ROOT.parent / "compiler" / "assets" / "apg.css"

# Layout-only classes the browser streamer adds at runtime; the server emitter
# does not paint them but the stylesheet still has to carry them.
RUNTIME_CLASSES = ("apg-openui-stream",)


def _library() -> Library:
	"""APG's own generated library: what a real app hands the renderer."""
	return Library.from_spec(build_library().to_spec())


def _render(source: str) -> str:
	result = Program.parse(source).evaluate(_library())
	assert result.root is not None, f"program did not render: {source!r}"
	return render_html(result.root)


def _classes_in_html(html: str) -> set[str]:
	classes: set[str] = set()
	for match in re.finditer(r'class="([^"]*)"', html):
		classes.update(match.group(1).split())
	return classes


def _defined_in_stylesheet() -> set[str]:
	"""Every class selector the stylesheet defines, as bare names."""
	text = STYLESHEET.read_text()
	names: set[str] = set()
	for match in re.finditer(r"\.([A-Za-z_][\w-]*)", text):
		names.add(match.group(1))
	return names


# A program exercising every component APG's base library defines.
EVERY_COMPONENT = """
root = Page([head, m, b, l, link, table, chart, form], "Dashboard")
head = Card("Overview", [t])
m = Metric("Revenue", 1250, "KES", 12.5)
b = Badge("Live", "success")
t = Text("All good", "body")
l = Text("Details", "heading")
link = Link("Docs", "/docs")
table = Table([c1, c2], "Latest")
c1 = Column("Name", ["a", "b"])
c2 = Column("Amount", [1, 2])
chart = Chart("line", [p1, p2], "Trend")
p1 = {label: "Jan", value: 3}
p2 = {label: "Feb", value: 5}
form = Form("search", [i, s, btn])
i = Input("q", "Search", "text")
s = Select("kind", [o1, o2], "Kind", $kind)
o1 = SelectItem("all", "All")
o2 = SelectItem("open", "Open")
btn = Button("Go", Action([@Set($kind, "all")]), "primary")
"""


def test_every_component_renders() -> None:
	html = _render(EVERY_COMPONENT)
	for component in ("apg-openui-page", "apg-card", "apg-stat", "apg-badge", "apg-table",
					"apg-chart", "apg-field", "apg-btn", "apg-openui-link"):
		assert component in html, f"{component} missing from rendered output"


def test_emitted_classes_are_defined() -> None:
	"""Nothing the renderer paints is missing from the stylesheet."""
	defined = _defined_in_stylesheet()
	html = _render(EVERY_COMPONENT)
	missing = sorted(_classes_in_html(html) - defined)
	assert not missing, f"classes emitted but not defined in apg.css: {missing}"


def test_renderer_tables_are_defined() -> None:
	"""Every class the renderer can name is defined, not only the ones a sample hit.

	The previous test only sees classes the sample program reached; a variant no
	sample uses (a danger button, a warning badge) would pass unseen.
	"""
	defined = _defined_in_stylesheet()
	named: set[str] = set(RUNTIME_CLASSES)
	for category in class_table().values():
		named |= set(category.values())
	named.discard("")
	missing = sorted(named - defined)
	assert not missing, f"classes named by the renderer but not defined in apg.css: {missing}"


def test_component_class_table_is_complete() -> None:
	"""Every renderable component appears in the shared class table."""
	missing = sorted(set(supported_components()) - set(component_classes()))
	assert not missing, f"components without a class entry: {missing}"


@pytest.mark.parametrize(
	"source, needle",
	[
		('root = Text("<script>alert(1)</script>")', "&lt;script&gt;"),
		('root = Text("a & b")', "a &amp; b"),
		('root = Button("\\" onmouseover=\\"x")', "&quot; onmouseover=&quot;"),
	],
)
def test_text_is_escaped(source: str, needle: str) -> None:
	assert needle in _render(source)


def test_attribute_values_are_escaped() -> None:
	html = _render('root = Link("click", "\\" onclick=\\"alert(1)")')
	assert 'onclick="alert(1)"' not in html
	assert "&quot; onclick=&quot;" in html


def test_binding_lands_in_a_data_attribute() -> None:
	html = _render("$q = \"\"\nroot = Input(\"search\", \"Find\", \"text\", {}, $q)")
	assert 'data-apg-bind="$q"' in html
	assert 'value=""' not in html, "a bound field must not also carry a literal value"


def test_action_payload_is_json() -> None:
	html = _render('root = Button("Go", Action([@Set($n, 3)]))')
	match = re.search(r'data-apg-action="([^"]*)"', html)
	assert match is not None
	payload = json.loads(match.group(1).replace("&quot;", '"'))
	assert payload == {"actions": [{"action": "Set", "args": ["$n", 3]}]}


def test_unknown_component_renders_nothing() -> None:
	"""An unknown component is dropped with an error and paints no markup."""
	library = _library()
	result = Program.parse('root = Stack([Text("ok"), Mystery("x")])').evaluate(library)
	html = render_html(result.root)
	assert "ok" in html
	assert "Mystery" not in html
	assert "unknown-component" in [error.code for error in result.errors]


def test_null_renders_nothing() -> None:
	assert render_html(None) == ""


def test_bare_scalar_renders_as_text() -> None:
	assert "apg-openui-text" in render_html("hello")


def test_render_document_is_self_contained() -> None:
	html = render_document({"type": "Text", "props": {"text": "hi"}}, title="Preview")
	assert html.startswith("<!doctype html>")
	assert 'href="/static/apg.css"' in html
	assert "http://" not in html and "https://" not in html, "no external asset may be referenced"


def test_browser_renderer_matches_the_fixtures() -> None:
	"""The browser port is held to the same normative suite as the parser.

	Patch merging (spec 7) is applied server-side and has no browser counterpart,
	so the JavaScript runner skips those five cases; everything else must agree.
	"""
	script = Path(__file__).resolve().parent / "openui_render_js.test.js"
	node = shutil.which("node")
	if node is None:
		pytest.skip("node is not installed")
		return
	assert node is not None
	completed = subprocess.run(
		[node, str(script)], capture_output=True, text=True, cwd=str(ROOT), check=False
	)
	assert completed.returncode == 0, completed.stdout + completed.stderr
	assert "0 failed" in completed.stdout
