"""Human and machine-readable APG semantic explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .diagnostics import explain_diagnostic
from .semantic_model import build_semantic_model


def build_explain_report(
	source_file: Path,
	*,
	symbol: str | None = None,
	diagnostic: str | None = None,
	handler: str | None = None,
) -> dict[str, Any]:
	"""Build an ``apg.explain-report.v1`` payload."""
	requested = {
		"symbol": symbol,
		"diagnostic": diagnostic,
		"handler": handler,
	}
	active = [key for key, value in requested.items() if value]
	report: dict[str, Any] = {
		"format": "apg.explain-report.v1",
		"ok": False,
		"source": str(source_file),
		"query": requested,
		"model": {},
		"explanations": [],
		"errors": [],
		"warnings": [],
	}

	if len(active) != 1:
		report["errors"].append("Specify exactly one of --symbol, --diagnostic, or --handler")
		return report

	model = build_semantic_model(source_file)
	report["model"] = {
		"format": model.get("format"),
		"ok": model.get("ok"),
		"symbol_count": len(model.get("symbols", {})),
		"diagnostic_count": len(model.get("diagnostics", [])),
		"table_count": len(model.get("tables", {})),
		"agent_count": len(model.get("agents", {})),
		"capability_count": len(model.get("capabilities", {})),
	}

	if symbol:
		report["explanations"] = _explain_symbol(model, symbol)
	elif diagnostic:
		report["explanations"] = _explain_diagnostic(model, diagnostic)
	elif handler:
		report["explanations"] = _explain_handler(model, handler)

	if not report["explanations"]:
		report["errors"].append(f"No explanation found for {active[0]} {requested[active[0]]!r}")
	report["ok"] = not report["errors"]
	return report


def _explain_symbol(model: dict[str, Any], query: str) -> list[dict[str, Any]]:
	symbols = model.get("symbols", {})
	matches = [
		symbol
		for symbol in symbols.values()
		if query in {symbol.get("id"), symbol.get("name")}
		or str(symbol.get("id", "")).endswith(f".{query}")
		or str(symbol.get("name", "")).endswith(f".{query}")
	]
	return [_symbol_explanation(model, symbol) for symbol in matches]


def _symbol_explanation(model: dict[str, Any], symbol: dict[str, Any]) -> dict[str, Any]:
	kind = str(symbol.get("kind", "symbol"))
	name = str(symbol.get("name", ""))
	detail: dict[str, Any] = {}
	if kind == "table":
		detail = model.get("tables", {}).get(name, {})
	elif kind == "field" and "." in name:
		table_name, field_name = name.split(".", 1)
		detail = model.get("tables", {}).get(table_name, {}).get("fields", {}).get(field_name, {})
	elif kind == "agent":
		detail = model.get("agents", {}).get(name, {})
	elif kind == "capability":
		detail = model.get("capabilities", {}).get(name, {})
	elif kind == "app":
		detail = model.get("composition", {}).get("applications", {}).get(name, {})
	elif kind == "composition":
		detail = model.get("composition", {}).get("agent_teams", {}).get(name, {})

	return {
		"kind": "symbol",
		"summary": f"{symbol.get('id')} is a {kind} declaration named {name}.",
		"symbol": symbol,
		"detail": detail,
		"related": _related_symbol_context(model, kind, name, detail),
	}


def _related_symbol_context(model: dict[str, Any], kind: str, name: str, detail: dict[str, Any]) -> dict[str, Any]:
	if kind == "capability":
		dependencies = model.get("composition", {}).get("capability_dependencies", {})
		return {
			"requires": dependencies.get(name, []),
			"provides": detail.get("provides", []),
			"routes": detail.get("ui", {}).get("routes", []),
		}
	if kind == "table":
		return {
			"field_count": len(detail.get("fields", {})),
			"lookup_paths": sorted(detail.get("lookup_paths", {})),
		}
	if kind == "field":
		return {"relationship": detail.get("relationship")}
	return {}


def _explain_diagnostic(model: dict[str, Any], code: str) -> list[dict[str, Any]]:
	normalized = code.upper()
	matching = [
		diagnostic
		for diagnostic in model.get("diagnostics", [])
		if str(diagnostic.get("code", "")).upper() == normalized
	]
	registry = explain_diagnostic(normalized)
	return [{
		"kind": "diagnostic",
		"summary": f"{normalized}: {registry['title']}",
		"code": normalized,
		"registry": registry,
		"matches": matching,
		"match_count": len(matching),
	}]


def _explain_handler(model: dict[str, Any], query: str) -> list[dict[str, Any]]:
	matches: list[dict[str, Any]] = []
	needle = query.lower()
	for view_name, view in model.get("views", {}).items():
		for handler in view.get("handlers", []):
			haystacks = [
				view_name,
				str(handler.get("event", "")),
				str(handler.get("target", "")),
				f"{view_name}.{handler.get('event', '')}",
			]
			if any(needle == haystack.lower() or needle in haystack.lower() for haystack in haystacks):
				matches.append({
					"surface": "view",
					"name": view_name,
					"handler": handler,
				})

	for capability_name, capability in model.get("capabilities", {}).items():
		for screen_name, screen in capability.get("screens", {}).items():
			for event in screen.get("events", []):
				haystacks = [
					capability_name,
					screen_name,
					str(event.get("on", "")),
					str(event.get("target", "")),
					f"{screen_name}.{event.get('on', '')}",
				]
				if any(needle == haystack.lower() or needle in haystack.lower() for haystack in haystacks):
					matches.append({
						"surface": "capability_screen",
						"capability": capability_name,
						"screen": screen_name,
						"handler": event,
						"relationships": screen.get("relationships", []),
					})

	return [
		{
			"kind": "handler",
			"summary": _handler_summary(match),
			"handler": match,
		}
		for match in matches
	]


def _handler_summary(match: dict[str, Any]) -> str:
	if match.get("surface") == "capability_screen":
		handler = match.get("handler", {})
		return (
			f"{match.get('capability')}.{match.get('screen')} handles "
			f"{handler.get('on')} by {handler.get('do')} targeting {handler.get('target')}."
		)
	handler = match.get("handler", {})
	return f"{match.get('name')} handles {handler.get('event')} by targeting {handler.get('target')}."
