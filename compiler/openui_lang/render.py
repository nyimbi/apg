"""Render an evaluated OpenUI tree to HTML using APG's existing classes.

Two renderers, one mapping:

* :func:`render_html` emits a complete fragment server-side. It is what a
  generated app uses for a non-interactive screen or an initial paint.
* ``compiler/assets/apg-openui.js`` re-renders in the browser as chunks arrive.
  Both read the same component-to-class table, so a streamed screen and a
  server-painted one look the same.

No component here invents CSS: every class already ships in ``apg.css``, which
is what keeps the generated app free of a build step and a CDN.
"""

from __future__ import annotations

from html import escape
from typing import Any

from .evaluator import NOTHING, tree_summary

# Classes the generated stylesheet already provides. A component absent from
# this table renders as a plain container, which is the spec's recovery for an
# unknown component (it renders nothing) applied to the DOM.
_COMPONENT_CLASS = {
	"Page": "apg-openui-page",
	"Card": "apg-card",
	"Stack": "apg-openui-stack",
	"Text": "apg-openui-text",
	"Badge": "apg-badge",
	"Tag": "apg-badge",
	"Metric": "apg-stat",
	"Table": "apg-table-wrap",
	"Column": "",
	"Chart": "apg-chart",
	"Form": "apg-openui-form",
	"Input": "apg-field",
	"Select": "apg-field",
	"SelectItem": "",
	"Button": "apg-btn",
	"Link": "apg-openui-link",
}

_TEXT_CLASS = {
	"heading": "apg-openui-heading",
	"body": "apg-openui-body",
	"muted": "apg-openui-muted",
}

_BADGE_CLASS = {
	"neutral": "apg-badge-neutral",
	"success": "apg-badge-success",
	"warning": "apg-badge-warning",
	"danger": "apg-badge-danger",
	"info": "apg-badge-info",
}

_BUTTON_CLASS = {
	"primary": "",
	"secondary": "apg-btn-secondary",
	"danger": "apg-btn-danger",
}


def render_html(tree: Any) -> str:
	"""Render a component instance (or a list of them) as an HTML fragment."""
	return _render_node(tree)


def _render_node(node: Any) -> str:
	if node is None or node is NOTHING:
		# The spec's rule for a null in a component position: nothing renders.
		return ""
	if isinstance(node, list):
		return "".join(_render_node(item) for item in node)
	if not isinstance(node, dict):
		# Strings, numbers, and booleans in a component position render as text.
		return f'<span class="apg-openui-text">{escape(_text_of(node))}</span>'
	if "type" not in node or "props" not in node:
		return ""
	renderer = _RENDERERS.get(str(node["type"]))
	if renderer is None:
		# An unknown component renders nothing; the error list carries the report.
		return ""
	return renderer(node["props"])


def _text_of(value: Any) -> str:
	if value is True:
		return "true"
	if value is False:
		return "false"
	if isinstance(value, float) and value.is_integer():
		return str(int(value))
	return str(value)


def _attr(name: str, value: Any) -> str:
	return f' {name}="{escape(str(value))}"'


def _render_page(props: dict[str, Any]) -> str:
	title = props.get("title")
	header = f'<h1 class="apg-openui-heading">{escape(_text_of(title))}</h1>' if title else ""
	body = _render_node(props.get("children"))
	return f'<div class="apg-openui-page">{header}{body}</div>'


def _render_card(props: dict[str, Any]) -> str:
	title = props.get("title")
	head = (
		f'<header class="apg-card-header"><h2>{escape(_text_of(title))}</h2></header>'
		if title else ""
	)
	body = _render_node(props.get("children"))
	return f'<section class="apg-card">{head}{body}</section>'


def _render_stack(props: dict[str, Any]) -> str:
	direction = str(props.get("direction") or "column")
	body = _render_node(props.get("children"))
	return f'<div class="apg-openui-stack" data-direction="{escape(direction)}">{body}</div>'


def _render_text(props: dict[str, Any]) -> str:
	variant = str(props.get("variant") or "body")
	text = escape(_text_of(props.get("text", "")))
	if variant == "heading":
		return f'<h2 class="apg-openui-heading">{text}</h2>'
	if variant == "muted":
		return f'<p class="apg-openui-muted">{text}</p>'
	return f'<p class="apg-openui-body">{text}</p>'


def _render_badge(props: dict[str, Any]) -> str:
	tone = str(props.get("tone") or props.get("color") or "neutral")
	modifier = _BADGE_CLASS.get(tone, "apg-badge-neutral")
	return f'<span class="apg-badge {modifier}">{escape(_text_of(props.get("text", "")))}</span>'


def _render_metric(props: dict[str, Any]) -> str:
	label = escape(_text_of(props.get("label", "")))
	value = props.get("value")
	unit = props.get("unit")
	shown = escape(_text_of(value)) + (escape(_text_of(unit)) if unit else "")
	delta = props.get("delta")
	delta_html = ""
	if isinstance(delta, (int, float)) and not isinstance(delta, bool):
		direction = "up" if delta >= 0 else "down"
		sign = "+" if delta >= 0 else ""
		delta_html = f'<span class="apg-stat-delta {direction}">{sign}{escape(_text_of(delta))}</span>'
	return (
		f'<article class="apg-stat"><span class="apg-stat-label">{label}</span>'
		f'<strong class="apg-stat-value">{shown}</strong>{delta_html}</article>'
	)


def _render_table(props: dict[str, Any]) -> str:
	columns = props.get("columns") or []
	if not isinstance(columns, list):
		return ""
	labels: list[str] = []
	cells: list[list[Any]] = []
	for column in columns:
		if not isinstance(column, dict) or column.get("type") != "Column":
			continue
		column_props = column.get("props") or {}
		labels.append(escape(_text_of(column_props.get("label", ""))))
		data = column_props.get("data")
		cells.append(list(data) if isinstance(data, list) else [])
	rows = max((len(column) for column in cells), default=0)
	head = "".join(f"<th>{label}</th>" for label in labels)
	body_rows: list[str] = []
	for index in range(rows):
		row_cells = []
		for column in cells:
			value = column[index] if index < len(column) else None
			row_cells.append(f"<td>{escape(_text_of(value)) if value is not None else ''}</td>")
		body_rows.append(f"<tr>{''.join(row_cells)}</tr>")
	caption = props.get("caption")
	caption_html = f"<caption>{escape(_text_of(caption))}</caption>" if caption else ""
	return (
		f'<div class="apg-table-wrap"><table class="apg-table">{caption_html}'
		f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
	)


def _render_chart(props: dict[str, Any]) -> str:
	kind = str(props.get("kind") or "line")
	points = props.get("points")
	if not isinstance(points, list):
		points = []
	title = props.get("title")
	header = (
		f'<header class="apg-card-header"><h2>{escape(_text_of(title))}</h2></header>'
		if title else ""
	)
	# The chart body is left to apg-charts.js, which reads the data attribute.
	payload = escape(_json_attr(points))
	return (
		f'<figure class="apg-chart" data-apg-chart="{escape(kind)}" data-apg-chart-data="{payload}">'
		f"{header}<figcaption class=\"apg-openui-muted\">{len(points)} points</figcaption></figure>"
	)


def _render_form(props: dict[str, Any]) -> str:
	name = props.get("name")
	body = _render_node(props.get("children"))
	return f'<form class="apg-openui-form" data-apg-openui-form="{escape(_text_of(name))}">{body}</form>'


def _render_input(props: dict[str, Any]) -> str:
	name = _text_of(props.get("name", ""))
	label = _text_of(props.get("label") or props.get("placeholder") or name)
	field_type = str(props.get("type") or "text")
	value = props.get("value")
	binding = ""
	value_attr = ""
	if isinstance(value, dict) and "$binding" in value:
		binding = _attr("data-apg-bind", value["$binding"])
	elif value is not None:
		value_attr = _attr("value", _text_of(value))
	rules = props.get("rules")
	rules_attr = _attr("data-apg-rules", _json_attr(rules)) if isinstance(rules, dict) else ""
	return (
		f'<label class="apg-field"><span>{escape(label)}</span>'
		f'<input type="{escape(field_type)}" name="{escape(name)}"{value_attr}{binding}{rules_attr}></label>'
	)


def _render_select(props: dict[str, Any]) -> str:
	name = _text_of(props.get("name", ""))
	label = _text_of(props.get("label") or name)
	value = props.get("value")
	binding = ""
	if isinstance(value, dict) and "$binding" in value:
		binding = _attr("data-apg-bind", value["$binding"])
	options: list[str] = []
	items = props.get("items")
	if isinstance(items, list):
		for item in items:
			if not isinstance(item, dict) or item.get("type") != "SelectItem":
				continue
			item_props = item.get("props") or {}
			option_value = _text_of(item_props.get("value", ""))
			option_label = _text_of(item_props.get("label", option_value))
			options.append(f'<option value="{escape(option_value)}">{escape(option_label)}</option>')
	return (
		f'<label class="apg-field"><span>{escape(label)}</span>'
		f'<select name="{escape(name)}"{binding}>{"".join(options)}</select></label>'
	)


def _render_button(props: dict[str, Any]) -> str:
	label = _text_of(props.get("label", ""))
	variant = str(props.get("variant") or "primary")
	modifier = _BUTTON_CLASS.get(variant, "")
	classes = " ".join(part for part in ("apg-btn", modifier) if part)
	action = props.get("action")
	payload = _attr("data-apg-action", _json_attr(action)) if action is not None else ""
	return f'<button type="button" class="{classes}"{payload}>{escape(label)}</button>'


def _render_link(props: dict[str, Any]) -> str:
	label = _text_of(props.get("label", ""))
	href = _text_of(props.get("href", "#"))
	return f'<a class="apg-openui-link" href="{escape(href)}">{escape(label)}</a>'


_RENDERERS = {
	"Page": _render_page,
	"Card": _render_card,
	"Stack": _render_stack,
	"Text": _render_text,
	"Badge": _render_badge,
	"Tag": _render_badge,
	"Metric": _render_metric,
	"Table": _render_table,
	"Chart": _render_chart,
	"Form": _render_form,
	"Input": _render_input,
	"Select": _render_select,
	"Button": _render_button,
	"Link": _render_link,
}


def _json_attr(value: Any) -> str:
	import json

	return json.dumps(tree_summary(value), separators=(",", ":"), ensure_ascii=False)


def supported_components() -> list[str]:
	"""Component names this renderer paints; the rest render nothing."""
	return sorted(_RENDERERS)


def component_classes() -> dict[str, str]:
	"""The component-to-class table, shared with the browser renderer."""
	return dict(_COMPONENT_CLASS)


def render_document(tree: Any, *, title: str = "") -> str:
	"""Wrap a rendered fragment in a self-contained page.

	Used for previews and tests; a generated app renders the fragment inside its
	own shell template.
	"""
	body = render_html(tree)
	return (
		"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
		'<meta name="viewport" content="width=device-width, initial-scale=1">'
		f"<title>{escape(title)}</title>"
		'<link rel="stylesheet" href="/static/apg.css">'
		"</head><body><main class=\"apg-shell-content\">"
		f"{body}</main></body></html>"
	)


def class_table() -> dict[str, dict[str, str]]:
	"""Every class the renderer can emit, by category.

	The stylesheet-coverage test reads this rather than a sample render, so a
	variant no sample happens to use (a danger button, a warning badge) is
	still checked against ``apg.css``.
	"""
	return {
		"component": dict(_COMPONENT_CLASS),
		"badge": dict(_BADGE_CLASS),
		"button": dict(_BUTTON_CLASS),
		"text": dict(_TEXT_CLASS),
	}
