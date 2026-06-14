"""Widget assignment rules — auto-detect which UI widgets apply to an entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


def _has_both(fields: list[dict], set_a: set[str], set_b: set[str]) -> bool:
	names = {str(f.get("name", "")).lower() for f in fields}
	return bool(set_a & names) and bool(set_b & names)


@dataclass
class WidgetRule:
	name: str
	field_patterns: set[str]
	widget: str
	condition: Callable[[list[dict]], bool] | None = None
	priority: int = 0


WIDGET_RULES: list[WidgetRule] = [
	WidgetRule(
		"map",
		{"lat", "lng", "latitude", "longitude"},
		"map_widget",
		condition=lambda fs: _has_both(fs, {"lat", "latitude"}, {"lng", "longitude"}),
		priority=20,
	),
	WidgetRule(
		"kanban",
		{"status", "state", "stage", "phase"},
		"kanban_toggle",
		priority=10,
	),
	WidgetRule(
		"image",
		{"avatar", "photo", "image", "thumbnail", "picture", "logo"},
		"image_preview",
		priority=8,
	),
	WidgetRule(
		"chart",
		{"amount", "total", "balance", "revenue", "cost", "price", "quantity", "count", "salary"},
		"trend_chart",
		priority=5,
	),
	WidgetRule(
		"progress",
		{"progress", "percent", "completion", "done_count"},
		"progress_bar",
		priority=5,
	),
	WidgetRule(
		"timeline",
		{"start_date", "end_date", "due_date", "deadline", "scheduled_at"},
		"timeline_bar",
		priority=5,
	),
	WidgetRule(
		"rating",
		{"rating", "score", "stars", "grade"},
		"star_rating",
		priority=5,
	),
	WidgetRule(
		"currency",
		{"amount", "price", "cost", "fee", "salary", "balance", "revenue", "total"},
		"currency_display",
		priority=4,
	),
	WidgetRule(
		"phone",
		{"phone", "mobile", "tel"},
		"click_to_call",
		priority=3,
	),
	WidgetRule(
		"email",
		{"email"},
		"mailto_link",
		priority=3,
	),
	WidgetRule(
		"url",
		{"url", "website", "link", "href"},
		"external_link",
		priority=3,
	),
	WidgetRule(
		"color",
		{"color", "colour", "hex"},
		"color_swatch",
		priority=3,
	),
	WidgetRule(
		"json_viewer",
		{"config", "metadata", "settings", "payload", "data", "extra"},
		"json_viewer",
		priority=2,
	),
	WidgetRule(
		"heatmap",
		{"created_at", "updated_at", "occurred_at", "timestamp"},
		"activity_heatmap",
		priority=1,
	),
]

_priority_map: dict[str, int] = {r.widget: r.priority for r in WIDGET_RULES}


def detect_widgets(fields: list[dict]) -> list[str]:
	"""Return widget names applicable to this entity, ordered by priority descending."""
	field_names = {str(f.get("name", "")).lower() for f in fields}
	active: list[str] = []
	for rule in WIDGET_RULES:
		if rule.field_patterns & field_names:
			if rule.condition is None or rule.condition(fields):
				active.append(rule.widget)
	return sorted(active, key=lambda w: _priority_map.get(w, 0), reverse=True)


def detect_semantic_type(field_name: str, field_type: str) -> str:
	"""Infer a semantic type tag for a field based on its name and declared type."""
	name = field_name.lower()
	ft = field_type.lower()

	if any(x in name for x in ("email",)):
		return "email"
	if any(x in name for x in ("phone", "mobile", "tel")):
		return "phone"
	if any(x in name for x in ("url", "website", "link", "href")):
		return "url"
	if any(x in name for x in ("avatar", "photo", "image", "thumbnail", "picture", "logo")):
		return "image_url"
	if any(x in name for x in ("amount", "price", "cost", "fee", "salary", "balance", "revenue", "total")):
		return "currency"
	if any(x in name for x in ("percent", "progress", "completion")):
		return "percent"
	if any(x in name for x in ("lat", "latitude")):
		return "geo_lat"
	if any(x in name for x in ("lng", "longitude")):
		return "geo_lng"
	if any(x in name for x in ("status", "state", "stage", "phase")):
		return "status"
	if any(x in name for x in ("rating", "score", "stars", "grade")):
		return "rating"
	if any(x in name for x in ("color", "colour", "hex")):
		return "color"
	if any(x in name for x in ("config", "metadata", "settings", "payload", "extra")) or ft in ("json", "jsonb", "object"):
		return "json"
	if ft in ("bool", "boolean"):
		return "boolean"
	if any(x in name for x in ("_at", "date", "time", "timestamp", "deadline", "due")):
		return "datetime"
	if ft in ("text", "markdown", "longtext"):
		return "text_long"
	return "text"
