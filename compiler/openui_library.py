"""APG component library for OpenUI Lang.

Derives an OpenUI ``LibrarySpec`` from APG's own design system: the class
vocabulary in ``compiler/assets/apg.css``, the templates in
``compiler/templates/``, and — when a semantic model is supplied — one typed
composite per entity so the model fills named, typed fields instead of
re-deriving them from generic primitives.

The spec this conforms to is archived at
``docs/research/openui-generative-ui/sources/openui-lang-1.0-prompt.md``.
Positional argument order is the key order of ``properties`` in ``$defs``
(section 2.3), so prop tuples are ordered required-first and never reordered.

Usage::

	spec = build_library()
	spec = build_library(semantic_model=model, entities=["Patient", "Appointment"])
	errors = validate_library(spec)          # [] when conformant
	payload = spec.to_spec()                 # JSON-ready dict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

LIBRARY_ID = "apg"
LIBRARY_VERSION = "1.0.0"
ROOT_COMPONENT = "Page"

# Names OpenUI reserves for statement forms, built-ins, and actions (spec 1.3).
RESERVED_NAMES = frozenset({
	"Query", "Mutation", "Action",
	"Count", "First", "Last", "Sum", "Avg", "Min", "Max", "Filter", "Sort",
	"Round", "Abs", "Floor", "Ceil", "Each",
	"Set", "Reset", "Run", "ToAssistant", "OpenUrl",
	"true", "false", "null",
})

# APG DSL scalar type -> JSON Schema primitive.
_TYPE_MAP = {
	"str": "string",
	"text": "string",
	"int": "number",
	"float": "number",
	"decimal": "number",
	"bool": "boolean",
	"boolean": "boolean",
	"date": "string",
	"datetime": "string",
	"time": "string",
	"uuid": "string",
	"bytes": "string",
	"json": "any",
	"file": "string",
}

# Fields APG generates on every table; carrying them in every row inflates the
# prompt without giving the model a choice worth making.
_IMPLICIT_FIELDS = frozenset({"id", "created_at", "updated_at", "deleted_at"})

_STATUS_FIELDS = ("status", "state", "stage", "phase")


# Sentinel ref for a slot that accepts any component. Resolved to an `anyOf`
# over every component name when the spec is serialized, so the prompt and the
# schema agree on what may be nested where.
COMPONENT_REF = "Component"


@dataclass(frozen=True)
class Prop:
	"""One component prop, in positional order."""

	name: str
	type: str = "any"
	enum: tuple[str, ...] = ()
	ref: str = ""
	items: "Prop | None" = None
	object_fields: tuple["Prop", ...] = ()
	any_of_refs: tuple[str, ...] = ()
	required: bool = False
	description: str = ""
	default: Any = None
	has_default: bool = False
	binding: bool = False
	action_slot: bool = False

	def schema(self) -> dict[str, Any]:
		if self.action_slot:
			node: dict[str, Any] = {"$ref": "#/$defs/ActionExpression"}
		elif self.ref:
			node = {"$ref": f"#/$defs/{self.ref}"}
		elif self.any_of_refs:
			node = {"anyOf": [{"$ref": f"#/$defs/{name}"} for name in self.any_of_refs]}
		elif self.enum:
			node = {"type": "string", "enum": list(self.enum)}
		elif self.items is not None:
			node = {"type": "array", "items": self.items.schema()}
		elif self.object_fields:
			node = {
				"type": "object",
				"properties": {item.name: item.schema() for item in self.object_fields},
				"required": [item.name for item in self.object_fields if item.required],
			}
		elif self.type == "any":
			# `any` has no JSON Schema spelling; an empty schema accepts everything.
			node = {}
		else:
			node = {"type": self.type}
		if self.description:
			node["description"] = self.description
		if self.has_default:
			node["default"] = self.default
		return node

	def signature(self) -> str:
		"""Render the prop as it appears in the prompt (spec 5.1)."""
		return f"{self.name}{'' if self.required else '?'}: {self.type_signature()}"

	def type_signature(self) -> str:
		if self.action_slot:
			rendered = "ActionExpression"
		elif self.ref == COMPONENT_REF:
			# Any component may nest here; the schema carries the full anyOf.
			rendered = "any"
		elif self.ref:
			rendered = self.ref
		elif self.any_of_refs:
			rendered = " | ".join(f"@{name}" for name in self.any_of_refs)
		elif self.enum:
			rendered = " | ".join(f'"{value}"' for value in self.enum)
		elif self.items is not None:
			rendered = f"{self.items.type_signature()}[]"
		elif self.object_fields:
			inner = ", ".join(f"{item.name}: {item.type_signature()}" for item in self.object_fields)
			rendered = "{" + inner + "}"
		else:
			rendered = self.type
		if self.binding:
			return f"$binding<{rendered}>"
		return rendered


@dataclass(frozen=True)
class Component:
	name: str
	description: str
	props: tuple[Prop, ...] = ()
	group: str = ""

	def signature(self) -> str:
		return f"{self.name}({', '.join(prop.signature() for prop in self.props)})"

	def schema(self) -> dict[str, Any]:
		required = [prop.name for prop in self.props if prop.required]
		node: dict[str, Any] = {
			"type": "object",
			"description": self.description,
			"properties": {prop.name: prop.schema() for prop in self.props},
		}
		if required:
			node["required"] = required
		return node


@dataclass(frozen=True)
class FunctionSpec:
	name: str
	description: str
	params: tuple[Prop, ...] = ()
	returns: str = "any"

	def signature(self) -> str:
		args = ", ".join(f"{p.name}: {p.type_signature()}" for p in self.params)
		return f"{self.name}({args}) → {self.returns}"

	def schema(self) -> dict[str, Any]:
		return {
			"description": self.description,
			"params": {
				"type": "object",
				"properties": {p.name: p.schema() for p in self.params},
				"required": [p.name for p in self.params if p.required],
			},
			"returns": {"type": self.returns},
		}


@dataclass(frozen=True)
class LibrarySpec:
	"""An OpenUI LibrarySpec: the single source of truth for a renderable screen."""

	id: str = LIBRARY_ID
	version: str = LIBRARY_VERSION
	root: str = ROOT_COMPONENT
	components: tuple[Component, ...] = ()
	functions: tuple[FunctionSpec, ...] = ()
	actions: tuple[FunctionSpec, ...] = ()
	groups: tuple[tuple[str, tuple[str, ...], str], ...] = ()

	def component(self, name: str) -> Component:
		for item in self.components:
			if item.name == name:
				return item
		raise KeyError(name)

	def to_spec(self) -> dict[str, Any]:
		"""Render the JSON document OpenUI's prompt generator consumes."""
		names = [component.name for component in self.components]
		properties: dict[str, Any] = {}
		defs: dict[str, Any] = {}
		for component in self.components:
			properties[component.name] = {"$ref": f"#/$defs/{component.name}"}
			defs[component.name] = _resolve_component_refs(component.schema(), names)
		defs["ActionExpression"] = _action_expression_schema()

		schema: dict[str, Any] = {"properties": properties, "$defs": defs}
		if self.functions:
			schema["functions"] = {f.name: f.schema() for f in self.functions}
		if self.actions:
			schema["actions"] = {a.name: a.schema() for a in self.actions}

		document: dict[str, Any] = {
			"id": self.id,
			"version": self.version,
			"root": self.root,
			"schema": schema,
			"components": {
				c.name: {"signature": c.signature(), "description": c.description}
				for c in self.components
			},
		}
		if self.functions:
			document["functions"] = {
				f.name: {"signature": f.signature(), "description": f.description}
				for f in self.functions
			}
		if self.actions:
			document["actions"] = {
				a.name: {"signature": a.signature(), "description": a.description}
				for a in self.actions
			}
		if self.groups:
			document["componentGroups"] = [
				{"name": name, "components": list(members), "notes": notes}
				for name, members, notes in self.groups
			]
		return document


def _resolve_component_refs(node: Any, names: Sequence[str]) -> Any:
	"""Expand the ``Component`` sentinel into an ``anyOf`` over every component.

	The spec requires a slot to name its permitted types (section 2.3); a bare
	``$ref`` to a name that is not in ``$defs`` is not resolvable by a client.
	"""
	if isinstance(node, dict):
		if node.get("$ref") == f"#/$defs/{COMPONENT_REF}":
			return {"anyOf": [{"$ref": f"#/$defs/{name}"} for name in names]}
		return {key: _resolve_component_refs(value, names) for key, value in node.items()}
	if isinstance(node, list):
		return [_resolve_component_refs(item, names) for item in node]
	return node


def _action_expression_schema() -> dict[str, Any]:
	"""The action slot type: one action object, or a list wrapped by Action([...])."""
	return {
		"anyOf": [
			{
				"type": "object",
				"properties": {"type": {"const": "continue_conversation"}, "context": {"type": "any"}},
				"required": ["type"],
			},
			{
				"type": "object",
				"properties": {"type": {"const": "open_url"}, "url": {"type": "string"}},
				"required": ["type", "url"],
			},
			{
				"type": "object",
				"properties": {"type": {"type": "string"}, "params": {"type": "object"}},
				"required": ["type"],
			},
		]
	}


# ── base component set ──────────────────────────────────────────────────────
# Each maps to a class or template already shipped by the generated UI, so a
# renderer needs no new CSS.

BASE_COMPONENTS: tuple[Component, ...] = (
	Component(
		"Page",
		"Top-level screen. One per program.",
		(
			Prop("children", items=Prop("child", ref="Component"), required=True),
			Prop("title", type="string", description="Screen title, shown in the page header"),
		),
		group="Layout",
	),
	Component(
		"Card",
		"A titled panel (.apg-card).",
		(
			Prop("title", type="string", required=True),
			Prop("children", items=Prop("child", ref="Component"), required=True),
		),
		group="Layout",
	),
	Component(
		"Stack",
		"Lays children out in a row or a column.",
		(
			Prop("children", items=Prop("child", ref="Component"), required=True),
			Prop("direction", enum=("column", "row")),
		),
		group="Layout",
	),
	Component(
		"Text",
		"A run of text.",
		(
			Prop("text", type="string", required=True),
			Prop("variant", enum=("body", "heading", "muted")),
		),
		group="Content",
	),
	Component(
		"Badge",
		"A small status label (.apg-badge).",
		(
			Prop("text", type="string", required=True),
			Prop("tone", enum=("neutral", "success", "warning", "danger", "info")),
		),
		group="Content",
	),
	Component(
		"Metric",
		"One headline number (.apg-stat).",
		(
			Prop("label", type="string", required=True),
			Prop("value", type="number", required=True),
			Prop("unit", type="string", description="Currency code, %, or other suffix"),
			Prop("delta", type="number", description="Change since the previous period"),
		),
		group="Content",
	),
	Component(
		"Table",
		"A data table (.apg-table).",
		(
			Prop("columns", items=Prop("column", ref="Column"), required=True),
			Prop("caption", type="string"),
		),
		group="Data",
	),
	Component(
		"Column",
		"One table column: a header and its cell values.",
		(
			Prop("label", type="string", required=True),
			Prop("data", items=Prop("cell", type="any"), required=True),
			Prop("align", enum=("left", "right")),
		),
		group="Data",
	),
	Component(
		"Chart",
		"A chart rendered by apg-charts.js.",
		(
			Prop("kind", enum=("line", "area", "donut", "progress"), required=True),
			Prop("points", items=Prop("point", object_fields=(
				Prop("label", type="string", required=True),
				Prop("value", type="number", required=True),
			)), required=True),
			Prop("title", type="string"),
		),
		group="Data",
	),
	Component(
		"Form",
		"Groups its inputs under a form name.",
		(
			Prop("name", type="string", required=True),
			Prop("children", items=Prop("child", ref="Component"), required=True),
		),
		group="Forms",
	),
	Component(
		"Input",
		"A text field.",
		(
			Prop("name", type="string", required=True),
			Prop("label", type="string", required=True),
			Prop("type", enum=("text", "number", "date", "email")),
			Prop("rules", type="object", description="required, email, url, min, max, minLength, maxLength, pattern"),
			Prop("value", type="string", binding=True),
		),
		group="Forms",
	),
	Component(
		"Select",
		"A dropdown.",
		(
			Prop("name", type="string", required=True),
			Prop("label", type="string", required=True),
			Prop("items", items=Prop("item", ref="SelectItem"), required=True),
			Prop("value", type="string", binding=True),
		),
		group="Forms",
	),
	Component(
		"SelectItem",
		"One dropdown option.",
		(
			Prop("value", type="string", required=True),
			Prop("label", type="string", required=True),
		),
		group="Forms",
	),
	Component(
		"Button",
		"A button (.apg-btn).",
		(
			Prop("label", type="string", required=True),
			Prop("action", action_slot=True),
			Prop("variant", enum=("primary", "secondary", "danger")),
		),
		group="Actions",
	),
	Component(
		"Link",
		"A navigation link to a generated route.",
		(
			Prop("label", type="string", required=True),
			Prop("href", type="string", required=True),
		),
		group="Actions",
	),
)


# ── entity composites ───────────────────────────────────────────────────────


def _field_type(raw: Mapping[str, Any]) -> str:
	return _TYPE_MAP.get(str(raw.get("type", "str")).lower(), "string")


def _row_props(fields: Mapping[str, Any], limit: int = 12) -> tuple[Prop, ...]:
	props: list[Prop] = []
	for name, meta in fields.items():
		if name in _IMPLICIT_FIELDS:
			continue
		if len(props) >= limit:
			break
		props.append(Prop(name, type=_field_type(meta), required=bool(meta.get("required", False))))
	return tuple(props)


def _status_field(fields: Mapping[str, Any]) -> str:
	for name in _STATUS_FIELDS:
		if name in fields:
			return name
	return ""


def _entity_components(name: str, table: Mapping[str, Any]) -> list[Component]:
	"""Two to five typed components per entity, so the model never invents a field."""
	fields = table.get("fields") or {}
	if not isinstance(fields, Mapping) or not fields:
		return []
	row = _row_props(fields)
	if not row:
		return []

	components = [
		Component(
			f"{name}Row",
			f"One {name} record.",
			row,
			group=f"{name}",
		),
		Component(
			f"{name}Table",
			f"A table of {name} records, pre-typed from the schema.",
			(
				Prop("rows", items=Prop("row", ref=f"{name}Row"), required=True),
				Prop("title", type="string"),
			),
			group=f"{name}",
		),
	]

	status = _status_field(fields)
	if status:
		components.extend([
			Component(
				f"{name}Card",
				f"One {name} card for a board column.",
				(
					Prop("title", type="string", required=True),
					Prop("subtitle", type="string"),
					Prop("badge", type="string", description=f"Usually the {status} value"),
				),
				group=f"{name}",
			),
			Component(
				f"{name}Board",
				f"A kanban board of {name} records grouped by {status}.",
				(
					Prop("columns", items=Prop("column", ref=f"{name}Column"), required=True),
					Prop("title", type="string"),
				),
				group=f"{name}",
			),
			Component(
				f"{name}Column",
				f"One {status} column on the {name} board.",
				(
					Prop("title", type="string", required=True),
					Prop("cards", items=Prop("card", ref=f"{name}Card"), required=True),
				),
				group=f"{name}",
			),
		])
	return components


def build_library(
	semantic_model: Mapping[str, Any] | None = None,
	entities: Sequence[str] | None = None,
	max_entities: int = 12,
) -> LibrarySpec:
	"""Build the APG OpenUI library, optionally specialised per entity.

	*semantic_model* is an ``apg.semantic-model.v1`` document. *entities*
	selects a subset by table name; the default takes the first *max_entities*
	non-system tables, because every entity adds prompt tokens on every call.
	"""
	components: list[Component] = list(BASE_COMPONENTS)
	groups: list[tuple[str, tuple[str, ...], str]] = []

	base_groups: dict[str, list[str]] = {}
	for component in BASE_COMPONENTS:
		base_groups.setdefault(component.group, []).append(component.name)
	for group_name, members in base_groups.items():
		groups.append((group_name, tuple(members), ""))

	if semantic_model:
		tables = semantic_model.get("tables") or {}
		if isinstance(tables, Mapping):
			names = [name for name in tables if name != "security"]
			if entities is not None:
				names = [name for name in names if name in set(entities)]
			else:
				names = names[:max_entities]
			for name in names:
				table = tables.get(name)
				if not isinstance(table, Mapping):
					continue
				added = _entity_components(name, table)
				if not added:
					continue
				components.extend(added)
				groups.append((
					name,
					tuple(component.name for component in added),
					f"Typed components generated from the {name} schema. Field names and types are fixed.",
				))

	return LibrarySpec(
		components=tuple(components),
		functions=(
			FunctionSpec(
				"Pct",
				"Format part/total as a whole-number percentage.",
				(Prop("part", type="number", required=True), Prop("total", type="number", required=True)),
				returns="string",
			),
			FunctionSpec(
				"Money",
				"Format an amount with thousands separators and two decimals.",
				(Prop("amount", type="number", required=True), Prop("currency", type="string")),
				returns="string",
			),
		),
		actions=(
			FunctionSpec(
				"Navigate",
				"Navigate the generated app to a route.",
				(Prop("href", type="string", required=True),),
			),
		),
		groups=tuple(groups),
	)


# ── conformance (spec 3.1) ──────────────────────────────────────────────────


def validate_library(spec: LibrarySpec) -> list[str]:
	"""Return conformance errors; an empty list means the library is valid."""
	errors: list[str] = []
	seen: set[str] = set()

	for component in spec.components:
		if not component.name[:1].isupper() or not component.name.isidentifier():
			errors.append(f"component {component.name!r}: name must start uppercase and be an identifier")
		if component.name in RESERVED_NAMES:
			errors.append(f"component {component.name!r}: reserved name")
		if component.name in seen:
			errors.append(f"component {component.name!r}: duplicate")
		seen.add(component.name)

		optional_seen = False
		for prop in component.props:
			effective_required = prop.required and not prop.has_default
			if effective_required and optional_seen:
				errors.append(
					f"component {component.name!r}: required prop {prop.name!r} follows an optional one"
				)
			if not effective_required:
				optional_seen = True

	for function in (*spec.functions, *spec.actions):
		if not function.name[:1].isupper():
			errors.append(f"function {function.name!r}: name must start uppercase")
		if function.name in RESERVED_NAMES:
			errors.append(f"function {function.name!r}: reserved name")

	if spec.root and spec.root not in seen:
		errors.append(f"root {spec.root!r} is not a component")

	for group in spec.groups:
		group_name, members = group[0], group[1]
		for member in members:
			if member not in seen:
				errors.append(f"group {group_name!r}: unknown component {member!r}")

	return errors


def render_signatures(spec: LibrarySpec) -> list[str]:
	"""One prompt line per component: signature, em dash, description (spec 5)."""
	return [f"{c.signature()} — {c.description}" for c in spec.components]
