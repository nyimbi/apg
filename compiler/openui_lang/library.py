"""Library schema: the renderable contract a program is checked against.

Reads an OpenUI ``LibrarySpec`` (prompt.md section 2) and answers the questions
validation asks: what components exist, what their props are in positional
order, which are required, what types they accept, and which slots take
components, actions, or a two-way binding.

Two facts live only in the human-readable ``signature`` strings: which props are
bindable (``$binding<type>``) and, in effect, nothing else -- the schema is
otherwise authoritative (prompt.md section 5.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

_BINDING_RE = re.compile(r"\$binding<\s*([^>]*)\s*>")


@dataclass(frozen=True)
class PropSpec:
	name: str
	schema: Mapping[str, Any]
	required: bool
	order: int
	# Names a prop may be pointed at: component defs for slots, action defs for
	# action positions. Empty when the prop takes plain data.
	component_defs: frozenset[str] = frozenset()
	action_defs: frozenset[str] = frozenset()
	binding: bool = False

	@property
	def refs(self) -> tuple[str, ...]:
		"""Defs this prop points at, looked for in `items` for a list prop."""
		if self.schema.get("type") == "array":
			return _refs_of(self.item_schema)
		return _refs_of(self.schema)

	@property
	def is_component_slot(self) -> bool:
		"""Every option is a component (spec 8.4); `anyOf` counts as every."""
		refs = self.refs
		return bool(refs) and all(target in self.component_defs for target in refs)

	@property
	def is_action_slot(self) -> bool:
		return any(target in self.action_defs for target in self.refs)

	@property
	def item_schema(self) -> Mapping[str, Any]:
		items = self.schema.get("items")
		return items if isinstance(items, Mapping) else {}

	@property
	def is_list(self) -> bool:
		return self.schema.get("type") == "array"

	@property
	def enum(self) -> tuple[str, ...]:
		values = self.schema.get("enum")
		if isinstance(values, list):
			return tuple(str(value) for value in values)
		return ()

	@property
	def type_name(self) -> str:
		value = self.schema.get("type")
		return value if isinstance(value, str) else ""

	@property
	def default(self) -> Any:
		return self.schema.get("default")

	@property
	def has_default(self) -> bool:
		return "default" in self.schema


@dataclass(frozen=True)
class ComponentSpec:
	name: str
	props: tuple[PropSpec, ...]
	required: tuple[str, ...]
	description: str = ""
	signature: str = ""

	def prop(self, name: str) -> PropSpec | None:
		for item in self.props:
			if item.name == name:
				return item
		return None


@dataclass(frozen=True)
class FunctionSpec:
	name: str
	params: tuple[PropSpec, ...]
	required: tuple[str, ...]
	returns: Mapping[str, Any] = field(default_factory=dict)

	def param(self, name: str) -> PropSpec | None:
		for item in self.params:
			if item.name == name:
				return item
		return None


@dataclass
class Library:
	root: str = ""
	components: dict[str, ComponentSpec] = field(default_factory=dict)
	functions: dict[str, FunctionSpec] = field(default_factory=dict)
	actions: dict[str, FunctionSpec] = field(default_factory=dict)

	def component(self, name: str) -> ComponentSpec | None:
		return self.components.get(name)

	def component_names(self) -> list[str]:
		return list(self.components)

	def signature_hint(self, component_name: str) -> str:
		"""Prop names with required starred (spec 8.3, prompt.md 5.3)."""
		component = self.components.get(component_name)
		if component is None:
			return f"Available components: {', '.join(self.component_names())}"
		if component.signature:
			return f"Signature: {component.signature}. * marks required"
		parts = [f"{prop.name}*" if prop.required else prop.name for prop in component.props]
		return f"Signature: {component_name}({', '.join(parts)}). * marks required"

	@classmethod
	def from_spec(cls, document: Mapping[str, Any]) -> "Library":
		"""Build from an OpenUI LibrarySpec JSON document."""
		schema = document.get("schema") or {}
		defs = schema.get("$defs") or {}
		properties = schema.get("properties") or {}
		component_defs = frozenset(str(name) for name in properties)
		action_defs = frozenset(str(name) for name in (schema.get("actions") or {})) | {"ActionExpression"}
		signatures = {
			str(name): str(entry.get("signature") or "")
			for name, entry in (document.get("components") or {}).items()
			if isinstance(entry, Mapping)
		}

		components: dict[str, ComponentSpec] = {}
		for name in properties:
			definition = defs.get(name)
			if not isinstance(definition, Mapping):
				continue
			signature = signatures.get(str(name), "")
			components[str(name)] = _component_from(
				str(name), definition, component_defs, action_defs, signature
			)

		functions = _callables_from(schema.get("functions"), component_defs, action_defs)
		actions = _callables_from(schema.get("actions"), component_defs, action_defs)
		return cls(
			root=str(document.get("root") or ""),
			components=components,
			functions=functions,
			actions=actions,
		)


def _component_from(
	name: str,
	definition: Mapping[str, Any],
	component_defs: frozenset[str],
	action_defs: frozenset[str],
	signature: str,
) -> ComponentSpec:
	raw_props = definition.get("properties") or {}
	required = tuple(definition.get("required") or ())
	bindable = _bindable_props(signature)
	props = tuple(
		PropSpec(
			name=str(prop_name),
			schema=prop_schema,
			required=str(prop_name) in required,
			order=index,
			component_defs=component_defs,
			action_defs=action_defs,
			binding=str(prop_name) in bindable,
		)
		for index, (prop_name, prop_schema) in enumerate(raw_props.items())
	)
	return ComponentSpec(
		name=name,
		props=props,
		required=required,
		description=str(definition.get("description") or ""),
		signature=signature,
	)


def _callables_from(
	raw: Any,
	component_defs: frozenset[str],
	action_defs: frozenset[str],
) -> dict[str, FunctionSpec]:
	result: dict[str, FunctionSpec] = {}
	if not isinstance(raw, Mapping):
		return result
	for name, definition in raw.items():
		if not isinstance(definition, Mapping):
			continue
		params_schema = definition.get("params") or {}
		raw_params = params_schema.get("properties") or {}
		required = tuple(params_schema.get("required") or ())
		params = tuple(
			PropSpec(
				name=str(param),
				schema=param_schema,
				required=str(param) in required,
				order=index,
				component_defs=component_defs,
				action_defs=action_defs,
			)
			for index, (param, param_schema) in enumerate(raw_params.items())
		)
		returns = definition.get("returns")
		result[str(name)] = FunctionSpec(
			name=str(name),
			params=params,
			required=required,
			returns=returns if isinstance(returns, Mapping) else {},
		)
	return result


def _bindable_props(signature: str) -> frozenset[str]:
	"""Props whose signature prints `$binding<type>` (prompt.md section 2.3)."""
	if "$binding<" not in signature:
		return frozenset()
	open_index = signature.find("(")
	close_index = signature.rfind(")")
	if open_index < 0 or close_index < open_index:
		return frozenset()
	names: set[str] = set()
	for raw in _split_params(signature[open_index + 1:close_index]):
		param = raw.strip()
		if not param:
			continue
		head, _, tail = param.partition(":")
		if _BINDING_RE.search(tail):
			names.add(head.strip().rstrip("?").strip())
	return frozenset(names)


def _split_params(text: str) -> list[str]:
	parts: list[str] = []
	depth = 0
	current: list[str] = []
	for char in text:
		if char in "<{[( ":
			depth += 1 if char != " " else 0
		elif char in ">}]) ":
			depth -= 1 if char != " " else 0
		if char == "," and depth <= 0:
			parts.append("".join(current))
			current = []
			continue
		current.append(char)
	parts.append("".join(current))
	return parts


def _refs_of(schema: Mapping[str, Any]) -> tuple[str, ...]:
	refs: list[str] = []
	direct = schema.get("$ref")
	if isinstance(direct, str):
		refs.append(ref_target(direct))
	for option in schema.get("anyOf") or ():
		if isinstance(option, Mapping) and isinstance(option.get("$ref"), str):
			refs.append(ref_target(str(option["$ref"])))
	return tuple(refs)


def ref_target(ref: str) -> str:
	"""`#/$defs/Card` -> `Card`."""
	return ref.rsplit("/", 1)[-1]
