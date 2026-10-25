"""Program evaluation and tree validation (spec 2, 3, 6, 8).

Takes parsed statements plus a library and produces the renderable tree, the
error list, and the unresolved/orphan sets that the conformance fixtures
compare against.

Two rules shape the design:

* **Attribute errors to the statement being evaluated.** ``unknown-component``
  in ``chart = Chart([1, 2])`` belongs to ``chart``, and the same call written
  inline in ``root`` belongs to ``root``. A current-statement field is the whole
  mechanism.
* **A mistake removes the smallest possible unit** (spec 8.1). An invalid
  argument degrades a prop, an invalid component becomes ``NOTHING`` so the
  enclosing array leaves it out, and an invalid statement is dropped upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, Iterable

from . import values
from .library import ComponentSpec, FunctionSpec, Library, PropSpec
from .parser import (
	ActionCall, ArrayLiteral, Binary, BuiltinCall, Call, Index, Literal, Member,
	Node, ObjectLiteral, ParsedStatement, Reference, StateRef, Ternary, Unary,
)

BUILTINS = frozenset({
	"Count", "First", "Last", "Sum", "Avg", "Min", "Max", "Filter", "Sort",
	"Round", "Abs", "Floor", "Ceil", "Each",
})
ACTION_NAMES = frozenset({"Set", "Reset", "Run", "ToAssistant", "OpenUrl"})


class _Nothing:
	"""A value that renders nothing and is left out of any enclosing array."""

	__slots__ = ()

	def __repr__(self) -> str:  # pragma: no cover - debugging aid
		return "NOTHING"


NOTHING = _Nothing()


@dataclass
class Diagnostic:
	code: str
	statement_id: str = ""
	component: str = ""
	path: str = ""
	message: str = ""
	hint: str = ""
	source: str = "parser"
	severity: str = ""
	tool_name: str = ""

	def brief(self) -> dict[str, Any]:
		"""The two-key form `expected.json` compares (Appendix B)."""
		data: dict[str, Any] = {"code": self.code}
		if self.statement_id:
			data["statementId"] = self.statement_id
		return data

	def shaped(self) -> dict[str, Any]:
		"""The full error object (spec 8.3); only listed keys are compared."""
		data: dict[str, Any] = {
			"source": self.source,
			"code": self.code,
			"statementId": self.statement_id,
			"component": self.component,
			"path": self.path,
			"message": self.message,
			"hint": self.hint,
		}
		if self.tool_name:
			data["toolName"] = self.tool_name
		if self.severity:
			data["severity"] = self.severity
		return data


@dataclass
class EvaluationResult:
	root: Any
	errors: list[Diagnostic] = field(default_factory=list)
	unresolved: set[str] = field(default_factory=set)
	orphans: list[str] = field(default_factory=list)
	state: dict[str, Any] = field(default_factory=dict)
	reachable: set[str] = field(default_factory=set)

	def error_codes(self) -> list[str]:
		return [error.code for error in self.errors]


class Evaluator:
	"""Evaluate one parsed program against one library."""

	def __init__(
		self,
		library: Library,
		statements: Iterable[ParsedStatement],
		state: dict[str, Any] | None = None,
		functions: dict[str, Any] | None = None,
	) -> None:
		self.library = library
		self.functions = functions or {}
		self.by_name: dict[str, ParsedStatement] = {}
		for statement in statements:
			# The later statement wins, whatever their kinds (spec 2.4).
			self.by_name[statement.name] = statement
		self.declared: dict[str, Any] = {}
		for statement in self.by_name.values():
			if statement.kind == "state":
				self.declared[statement.name] = self.evaluate_quietly(statement.expression)
		self.state: dict[str, Any] = dict(self.declared)
		if state:
			self.state.update(state)
		self.errors: list[Diagnostic] = []
		self.unresolved: set[str] = set()
		self.reachable: set[str] = set()
		self._cache: dict[str, Any] = {}
		self._resolving: set[str] = set()
		self._current = ""
		self._entry_name = ""
		self._each_bindings: dict[str, Any] = {}
		self._no_entry = False

	# ── public ──────────────────────────────────────────────────────────────

	def run(self) -> EvaluationResult:
		self.reachable.clear()
		self._cache.clear()
		root = self._entry()
		self._reach_structurally()
		orphans = [] if self._no_entry else [
			statement.name
			for statement in self.by_name.values()
			if statement.kind == "value" and statement.name not in self.reachable
		]
		return EvaluationResult(
			root=None if root is NOTHING else root,
			errors=self.errors,
			unresolved=self.unresolved,
			orphans=orphans,
			state=self.state_snapshot(),
			reachable=set(self.reachable),
		)

	def state_snapshot(self) -> dict[str, Any]:
		"""Declared and auto-declared keys, in first-seen order."""
		return dict(self.state)

	def evaluate_quietly(self, node: Node) -> Any:
		"""Evaluate without recording errors, for state defaults."""
		saved_errors = self.errors if hasattr(self, "errors") else []
		self.errors = []
		self._current = ""
		try:
			return self.evaluate(node)
		finally:
			self.errors = saved_errors

	# ── entry ───────────────────────────────────────────────────────────────

	def _entry(self) -> Any:
		statement = self.by_name.get("root")
		if statement is not None:
			self._entry_name = "root"
			value = self._statement_value(statement)
			if _is_component_instance(value):
				return value
			self._error("no-root", statement_id="")
			return NOTHING
		if self.library.root:
			for candidate in self.by_name.values():
				if candidate.kind != "value":
					continue
				value = self._statement_value(candidate)
				if _is_component_instance(value) and value.get("type") == self.library.root:
					self._entry_name = candidate.name
					self._error("no-root", statement_id="", severity="warning")
					return value
		# No entry at all: nothing is reachable, and no statement is an orphan.
		self.reachable.clear()
		self._no_entry = True
		self._error("no-root", statement_id="")
		return NOTHING

	def _statement_value(self, statement: ParsedStatement) -> Any:
		previous = self._current
		self._current = statement.name
		try:
			value = self._reference(statement.name)
		finally:
			self._current = previous
		return value

	# ── expressions ─────────────────────────────────────────────────────────

	def evaluate(self, node: Node) -> Any:
		if isinstance(node, Literal):
			return node.value
		if isinstance(node, StateRef):
			key = f"${node.name}"
			if key not in self.state:
				# An undeclared `$name` is declared null (spec 5.1).
				self.state[key] = None
			return self.state[key]
		if isinstance(node, Reference):
			return self._reference(node.name)
		if isinstance(node, Call):
			return self._call(node)
		if isinstance(node, BuiltinCall):
			if node.name in self.library.actions:
				return {"actions": [self._custom_action(node.name, node.args)]}
			return self._builtin(node)
		if isinstance(node, ActionCall):
			return {"actions": self._action_items(node)}
		if isinstance(node, ArrayLiteral):
			return self._array(node)
		if isinstance(node, ObjectLiteral):
			return {key: self.evaluate(value) for key, value in node.entries}
		if isinstance(node, Member):
			return self._member(node)
		if isinstance(node, Index):
			return self._index(node)
		if isinstance(node, Unary):
			value = self.evaluate(node.operand)
			return (not values.to_boolean(value)) if node.op == "!" else values.negate(value)
		if isinstance(node, Binary):
			return self._binary(node)
		if isinstance(node, Ternary):
			branch = node.consequent if values.to_boolean(self.evaluate(node.condition)) else node.alternate
			return self.evaluate(branch)
		return None

	def _reference(self, name: str) -> Any:
		if name in self._each_bindings:
			# The loop variable shadows a statement of the same name (spec 3.5).
			return self._each_bindings[name]
		statement = self.by_name.get(name)
		if statement is None:
			self.unresolved.add(name)
			return None
		self.reachable.add(name)
		if name in self._cache:
			return self._cache[name]
		if name in self._resolving:
			# A reference that closes a cycle is unresolved at that point (2.3).
			self.unresolved.add(name)
			return None
		self._resolving.add(name)
		previous = self._current
		self._current = name
		try:
			value = self._evaluate_statement(statement)
		finally:
			self._current = previous
			self._resolving.discard(name)
		self._cache[name] = value
		return value

	def _evaluate_statement(self, statement: ParsedStatement) -> Any:
		if statement.kind == "state":
			return self.state.get(statement.name)
		if statement.kind == "query":
			return self._query_result(statement)
		if statement.kind == "mutation":
			return self._mutation_result()
		return self.evaluate(statement.expression)

	def _query_result(self, statement: ParsedStatement) -> Any:
		"""A query reference reads its defaults until a fetch lands (6.1)."""
		expression = statement.expression
		if isinstance(expression, Call) and len(expression.args) >= 3:
			return self.evaluate(expression.args[2])
		return None

	def _mutation_result(self) -> Any:
		return {"status": "idle", "data": None, "error": None}

	def _array(self, node: ArrayLiteral) -> list[Any]:
		items: list[Any] = []
		for item in node.items:
			if _is_reserved_call(item):
				# A direct array element is a value position (spec 2.1).
				self._error("inline-reserved")
				continue
			if isinstance(item, Reference):
				if item.name not in self.by_name:
					self.unresolved.add(item.name)
					continue
				if item.name in self._resolving:
					# A cycle closes here; the element renders nothing (2.3).
					self.unresolved.add(item.name)
					continue
			value = self.evaluate(item)
			# Unresolved references and inline-reserved calls render nothing and
			# are left out of an array (spec 2.3, 8.4).
			if value is NOTHING:
				continue
			items.append(value)
		return items

	def _member(self, node: Member) -> Any:
		target = self.evaluate(node.target)
		return member_of(target, node.field)

	def _index(self, node: Index) -> Any:
		target = self.evaluate(node.target)
		index = self.evaluate(node.index)
		if isinstance(target, list):
			number = values.to_number(index)
			position = int(number) if isinstance(number, float) and number.is_integer() else number
			if isinstance(position, int) and 0 <= position < len(target):
				return target[position]
			return None
		if isinstance(target, dict):
			return target.get(values.to_string(index))
		return None

	def _binary(self, node: Binary) -> Any:
		if node.op == "&&":
			left = self.evaluate(node.left)
			return self.evaluate(node.right) if values.to_boolean(left) else left
		if node.op == "||":
			left = self.evaluate(node.left)
			return left if values.to_boolean(left) else self.evaluate(node.right)
		left = self.evaluate(node.left)
		right = self.evaluate(node.right)
		if node.op == "+":
			return values.add(left, right)
		if node.op == "-":
			return values.subtract(left, right)
		if node.op == "*":
			return values.multiply(left, right)
		if node.op == "/":
			return values.divide(left, right)
		if node.op == "%":
			return values.modulo(left, right)
		if node.op == "==":
			return values.loose_equal(left, right)
		if node.op == "!=":
			return not values.loose_equal(left, right)
		return values.compare(left, right, node.op)

	# ── component calls ─────────────────────────────────────────────────────

	def _call(self, node: Call) -> Any:
		if node.name in ("Query", "Mutation"):
			# In a computed expression these are null with no error; only a value
			# position reports `inline-reserved` (spec 2.1), which the two value
			# positions check before they evaluate.
			return None
		if node.name == "Action":
			return {"actions": self._action_items(node)}
		if not node.is_component:
			# `Count(x)` parses as the reference `Count`; nothing to evaluate.
			return self._reference(node.name)
		return self._component(node)

	def _component(self, node: Call) -> Any:
		spec = self.library.component(node.name)
		if spec is None:
			self._error("unknown-component", component=node.name)
			return NOTHING
		props: dict[str, Any] = {}
		instance: dict[str, Any] = {"type": node.name, "props": props}
		dropped = self._assign_props(spec, node.args, props)
		if dropped:
			return NOTHING
		return instance

	def _assign_props(self, spec: ComponentSpec, args: list[Node], props: dict[str, Any]) -> bool:
		"""Map arguments to props by position; return True if the component drops."""
		if len(args) > len(spec.props):
			self._error(
				"excess-args", component=spec.name,
				message=f"too many arguments for {spec.name}",
				hint=self.library.signature_hint(spec.name),
			)
		for position, arg in enumerate(args):
			if position >= len(spec.props):
				break
			prop = spec.props[position]
			outcome = self._assign_prop(spec, prop, arg, props)
			if outcome == "drop":
				return True
		for prop in spec.props:
			if prop.name in props:
				continue
			if prop.has_default:
				props[prop.name] = prop.default
				continue
			if prop.required:
				self._error(
					"missing-required", component=spec.name, path=f"/{prop.name}",
					message=f'missing required field "/{prop.name}"',
					hint=self.library.signature_hint(spec.name),
				)
				return True
		return False

	def _assign_prop(
		self, spec: ComponentSpec, prop: PropSpec, arg: Node, props: dict[str, Any],
	) -> str:
		if prop.binding and isinstance(arg, StateRef):
			# A `$variable` in a bindable prop is a two-way binding (spec 5.2).
			props[prop.name] = {"$binding": f"${arg.name}"}
			return "ok"

		if _is_reserved_call(arg):
			# A direct argument of a component call is a value position (2.1).
			self._error("inline-reserved")
			return "drop" if prop.required and not prop.has_default else "ok"

		value = self.evaluate(arg)

		if value is NOTHING:
			# inline-reserved or a dropped component: the position stays unfilled.
			return "drop" if prop.required and not prop.has_default else "ok"

		if value is None:
			if prop.required and not prop.has_default:
				self._error(
					"null-required", component=spec.name, path=f"/{prop.name}",
					message=f'null required field "/{prop.name}"',
					hint=self.library.signature_hint(spec.name),
				)
				return "drop"
			return "ok"

		if prop.is_action_slot:
			# A restricted action slot accepts any action or a plain list (6.3).
			props[prop.name] = value
			return "ok"

		if prop.is_component_slot:
			if prop.is_list:
				kept = self._component_list(spec, prop, value)
				if kept is None:
					# A bare value where a list belongs is a `type-mismatch` that
					# removes the prop; the component still renders (8.2, 8.4).
					return "ok"
				props[prop.name] = kept
				return "ok"
			if not _is_component_instance(value):
				self._error(
					"type-mismatch", component=spec.name, path=f"/{prop.name}",
					message=f'type mismatch on "/{prop.name}"',
					hint=self.library.signature_hint(spec.name),
				)
				return "drop" if prop.required else "ok"
			props[prop.name] = value
			return "ok"

		if not self._type_ok(prop, value):
			self._error(
				"type-mismatch", component=spec.name, path=f"/{prop.name}",
				message=f'type mismatch on "/{prop.name}"',
				hint=self.library.signature_hint(spec.name),
			)
			if prop.has_default:
				props[prop.name] = prop.default
				return "ok"
			return "drop" if prop.required else "ok"
		props[prop.name] = value
		return "ok"

	def _component_list(
		self, spec: ComponentSpec, prop: PropSpec, value: Any,
	) -> list[Any] | None:
		"""A list slot needs `[...]`; data where a component belongs is left out."""
		if not isinstance(value, list):
			self._error(
				"type-mismatch", component=spec.name, path=f"/{prop.name}",
				message=f'a list is required for "/{prop.name}"',
				hint=self.library.signature_hint(spec.name),
			)
			return None
		kept: list[Any] = []
		for item in value:
			if item is None or _is_component_instance(item):
				kept.append(item)
				continue
			self._error(
				"type-mismatch", component=spec.name, path=f"/{prop.name}",
				message=f'data in component slot "/{prop.name}"',
				hint=self.library.signature_hint(spec.name),
			)
		return kept

	def _type_ok(self, prop: PropSpec, value: Any) -> bool:
		if prop.is_list:
			return isinstance(value, list)
		if prop.enum:
			return isinstance(value, str) and value in prop.enum
		type_name = prop.type_name
		if type_name == "string":
			return isinstance(value, str)
		if type_name == "number":
			return values.is_number(value)
		if type_name == "boolean":
			return isinstance(value, bool)
		if type_name == "object":
			return isinstance(value, dict)
		return True

	# ── built-ins and custom functions ──────────────────────────────────────

	def _builtin(self, node: BuiltinCall) -> Any:
		if node.name in self.functions:
			return self._custom_function(node)
		if node.name not in BUILTINS:
			self._error("unknown-function", component=node.name)
			return None
		if node.name == "Each":
			return self._each(node)
		args = [self.evaluate(arg) for arg in node.args]
		first = args[0] if args else None
		if node.name == "Filter":
			return _filter(first, args[1:])
		if node.name == "Sort":
			return _sort(first, args[1:])
		if node.name == "Count":
			return len(first) if isinstance(first, list) else 0
		if node.name == "First":
			return first[0] if isinstance(first, list) and first else None
		if node.name == "Last":
			return first[-1] if isinstance(first, list) and first else None
		if node.name == "Sum":
			return sum(values.to_number(item) for item in first) if isinstance(first, list) else 0
		if node.name == "Avg":
			if not isinstance(first, list) or not first:
				return 0
			return sum(values.to_number(item) for item in first) / len(first)
		if node.name in ("Min", "Max"):
			if not isinstance(first, list) or not first:
				return 0
			numbers = [values.to_number(item) for item in first]
			return min(numbers) if node.name == "Min" else max(numbers)
		if node.name == "Round":
			decimals = values.to_number(args[1]) if len(args) > 1 else 0
			return _js_round(values.to_number(first), int(decimals))
		if node.name == "Abs":
			return abs(values.to_number(first))
		if node.name == "Floor":
			return int(values.to_number(first) // 1)
		if node.name == "Ceil":
			number = values.to_number(first)
			floor = int(number // 1)
			return floor if floor == number else floor + 1
		return None

	def _custom_function(self, node: BuiltinCall) -> Any:
		spec = self.library.functions[node.name]
		implementation = self.functions.get(node.name)
		bound = self._bind_call_args(spec, node.args, node.name)
		if bound is None:
			return None
		if implementation is None:
			return None
		try:
			result = implementation(**bound)
		except Exception:
			self._error("runtime-error", component=node.name)
			return None
		if not _matches_return(spec, result):
			self._error("runtime-error", component=node.name)
			return None
		return result

	def _bind_call_args(
		self, spec: FunctionSpec, args: list[Node], call_name: str,
	) -> dict[str, Any] | None:
		"""Positional binding with the component-prop recovery rules (spec 3.6)."""
		if len(args) > len(spec.params):
			self._error(
				"excess-args", component=call_name,
				message=f"too many arguments for @{call_name}",
			)
		bound: dict[str, Any] = {}
		failed = False
		attempted: set[str] = set()
		for position, arg in enumerate(args[: len(spec.params)]):
			param = spec.params[position]
			attempted.add(param.name)
			value = self.evaluate(arg)
			if value is None:
				if param.required and not param.has_default:
					# A required argument that is null when the call runs makes the
					# call null with no error (spec 3.6).
					failed = True
				elif param.has_default:
					bound[param.name] = param.default
				continue
			if not self._type_ok(param, value):
				self._error(
					"type-mismatch", component=call_name,
					path=f"/{param.name}",
					message=f'type mismatch on "/{param.name}"',
				)
				if param.has_default:
					bound[param.name] = param.default
				elif param.required:
					failed = True
				continue
			bound[param.name] = value
		for param in spec.params:
			if param.name in bound or param.name in attempted:
				continue
			if param.has_default:
				bound[param.name] = param.default
			elif param.required:
				self._error(
					"missing-required", component=call_name, path=f"/{param.name}",
					message=f'missing required argument "/{param.name}"',
				)
				failed = True
		return None if failed else bound

	def _each(self, node: BuiltinCall) -> Any:
		"""`@Each(array, varName, template)` (spec 3.5)."""
		if len(node.args) < 3:
			return []
		items = self.evaluate(node.args[0])
		if not isinstance(items, list):
			return []
		var_name = values.to_string(self.evaluate(node.args[1]))
		template = node.args[2]
		results: list[Any] = []
		saved = self._each_bindings.copy()
		try:
			for item in items:
				self._each_bindings[var_name] = item
				value = self.evaluate(template)
				if value is not NOTHING:
					results.append(value)
		finally:
			self._each_bindings = saved
		return results

	# ── actions ─────────────────────────────────────────────────────────────

	def _action_items(self, node: Node) -> list[dict[str, Any]]:
		"""Flatten an action expression into the `{action, args}` entries."""
		if isinstance(node, ArrayLiteral):
			items: list[dict[str, Any]] = []
			for element in node.items:
				items.extend(self._action_items(element))
			return items
		if isinstance(node, Call | ActionCall) and node.name == "Action":
			items: list[dict[str, Any]] = []
			for element in node.args:
				items.extend(self._action_items(element))
			return items
		if isinstance(node, ActionCall):
			if node.name in self.library.actions:
				return [self._custom_action(node.name, node.args)]
			if node.name in ACTION_NAMES:
				return [self._builtin_action(node)]
			self._error("unknown-function", component=node.name)
			return []
		if isinstance(node, BuiltinCall):
			if node.name in self.library.actions:
				return [self._custom_action(node.name, node.args)]
			if node.name in ACTION_NAMES:
				return [self._builtin_action(ActionCall(node.name, node.args))]
			self._error("unknown-function", component=node.name)
			return []
		value = self.evaluate(node)
		if isinstance(value, dict) and isinstance(value.get("actions"), list):
			return list(value["actions"])
		return []

	def _builtin_action(self, node: ActionCall) -> dict[str, Any]:
		if node.name == "Set":
			target = _target_name(node.args[0]) if node.args else ""
			value = self.evaluate(node.args[1]) if len(node.args) > 1 else None
			return {"action": "Set", "args": [target, value]}
		if node.name == "Reset":
			return {"action": "Reset", "args": [_target_name(arg) for arg in node.args]}
		if node.name == "Run":
			return {"action": "Run", "args": [_target_name(node.args[0])] if node.args else []}
		if node.name == "ToAssistant":
			args: list[Any] = [self.evaluate(node.args[0]) if node.args else ""]
			if len(node.args) > 1:
				args.append(self.evaluate(node.args[1]))
			return {"action": "ToAssistant", "args": args}
		url = self.evaluate(node.args[0]) if node.args else ""
		return {"action": "OpenUrl", "args": [url]}

	def _custom_action(self, name: str, args: list[Node]) -> dict[str, Any]:
		spec = self.library.actions[name]
		bound = self._bind_call_args(spec, args, name)
		if bound is None:
			# Any invalid argument makes the action do nothing (spec 6.3).
			return {"action": name, "args": []}
		return {"action": name, "args": [bound[param.name] for param in spec.params if param.name in bound]}

	# ── reachability and errors ─────────────────────────────────────────────

	def _reach_structurally(self) -> None:
		"""Reachability follows references in the tree, not the values chosen.

		Both arms of a ternary are reachable: the state that picks one can change
		without the program being re-parsed (entry/006). Queries and mutations
		are roots of their own, since they run whether or not the entry reaches
		them (spec 2.3).
		"""
		# Evaluation already recorded which statements it actually visited; the
		# walk below re-derives reachability from the tree, so start clean.
		self.reachable.clear()
		if self._no_entry:
			return
		pending = [self._entry_name] if self._entry_name else []
		pending.extend(
			name for name, statement in self.by_name.items()
			if statement.kind in ("query", "mutation")
		)
		while pending:
			name = pending.pop()
			if name in self.reachable:
				continue
			statement = self.by_name.get(name)
			if statement is None:
				continue
			self.reachable.add(name)
			for node in _walk(statement.expression):
				if isinstance(node, Reference) and node.name in self.by_name:
					pending.append(node.name)
				elif isinstance(node, StateRef):
					continue

	def _error(self, code: str, *, statement_id: str | None = None, **kwargs: Any) -> None:
		self.errors.append(
			Diagnostic(code=code, statement_id=self._current if statement_id is None else statement_id, **kwargs)
		)


def member_of(target: Any, field: str) -> Any:
	"""`a.b` including array pluck and `.length` (spec 3.3)."""
	if isinstance(target, list):
		if field == "length":
			return len(target)
		return [values.pluck_field(element, field) for element in target]
	if isinstance(target, dict):
		return values.pluck_field(target, field)
	return None


def _filter(items: Any, rest: list[Any]) -> Any:
	if not isinstance(items, list):
		return []
	field = values.to_string(rest[0]) if rest and rest[0] is not None else ""
	op = values.to_string(rest[1]) if len(rest) > 1 and rest[1] is not None else "=="
	target = rest[2] if len(rest) > 2 else None
	kept = []
	for item in items:
		candidate = values.pluck_field(item, field) if field else item
		if _matches(candidate, op, target):
			kept.append(item)
	return kept


def _matches(candidate: Any, op: str, target: Any) -> bool:
	if op == "contains":
		return values.to_string(target) in values.to_string(candidate)
	if op == "==":
		return values.loose_equal(candidate, target)
	if op == "!=":
		return not values.loose_equal(candidate, target)
	if op in (">", "<", ">=", "<="):
		return values.compare(candidate, target, op)
	return False


def _sort(items: Any, rest: list[Any]) -> Any:
	if not isinstance(items, list):
		return items
	field = values.to_string(rest[0]) if rest and rest[0] is not None else ""
	descending = len(rest) > 1 and values.to_string(rest[1]) == "desc"

	def key(item: Any) -> Any:
		return values.pluck_field(item, field) if field else item

	def compare(left: Any, right: Any) -> int:
		left_text = "" if left is None else values.to_string(left)
		right_text = "" if right is None else values.to_string(right)
		left_number = _as_number(left, left_text)
		right_number = _as_number(right, right_text)
		if left_number is not None and right_number is not None:
			return (left_number > right_number) - (left_number < right_number)
		left_key = values.collation_key(left)
		right_key = values.collation_key(right)
		return (left_key > right_key) - (left_key < right_key)

	ordered = sorted(items, key=cmp_to_key(lambda a, b: compare(key(a), key(b))))
	if descending:
		ordered.reverse()
	return ordered


def _as_number(value: Any, text: str) -> float | int | None:
	if values.is_number(value):
		return value
	if text == "":
		return None
	stripped = text.strip()
	try:
		return float(stripped)
	except ValueError:
		return None


def _js_round(number: float | int, decimals: int) -> float | int:
	factor = 10 ** decimals
	scaled = values.to_number(number) * factor
	rounded = int(scaled + 0.5) if scaled >= 0 else -int(-scaled + 0.5)
	result = rounded / factor
	return int(result) if decimals <= 0 else result


def _target_name(node: Node) -> str:
	"""`@Run`/`@Set`/`@Reset` name their targets instead of evaluating them."""
	if isinstance(node, StateRef):
		return f"${node.name}"
	if isinstance(node, Reference):
		return node.name
	if isinstance(node, Literal):
		return values.to_string(node.value)
	return ""


def _matches_return(spec: FunctionSpec, value: Any) -> bool:
	returns = spec.returns
	if not returns:
		return True
	type_name = returns.get("type")
	if type_name == "string":
		return isinstance(value, str)
	if type_name == "number":
		return values.is_number(value)
	if type_name == "boolean":
		return isinstance(value, bool)
	return True


def _is_reserved_call(node: Node) -> bool:
	"""`Query(...)`/`Mutation(...)` in a value position (spec 2.1)."""
	return isinstance(node, Call) and node.name in ("Query", "Mutation")


def _is_component_instance(value: Any) -> bool:
	return isinstance(value, dict) and "type" in value and "props" in value


def _walk(node: Node):
	yield node
	for child in _children(node):
		yield from _walk(child)


def _children(node: Node) -> list[Node]:
	if isinstance(node, (Call, BuiltinCall, ActionCall)):
		return list(node.args)
	if isinstance(node, ArrayLiteral):
		return list(node.items)
	if isinstance(node, ObjectLiteral):
		return [entry[1] for entry in node.entries]
	if isinstance(node, Member):
		return [node.target]
	if isinstance(node, Index):
		return [node.target, node.index]
	if isinstance(node, Unary):
		return [node.operand]
	if isinstance(node, Binary):
		return [node.left, node.right]
	if isinstance(node, Ternary):
		return [node.condition, node.consequent, node.alternate]
	return []


def tree_summary(value: Any) -> Any:
	"""Render a component instance in the fixtures' `{type, props}` shape."""
	if isinstance(value, dict) and "type" in value and "props" in value:
		props: dict[str, Any] = {}
		for key, item in value["props"].items():
			if item is None or item is NOTHING:
				continue
			props[key] = tree_summary(item)
		return {"type": value["type"], "props": props}
	if isinstance(value, list):
		return [tree_summary(item) for item in value]
	if isinstance(value, dict):
		return {key: tree_summary(item) for key, item in value.items()}
	return value
