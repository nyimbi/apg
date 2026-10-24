"""Deterministic system-prompt generation for the APG OpenUI library.

The prompt is a pure function of the ``LibrarySpec`` and its options: same
inputs, byte-identical output on every run and platform (spec 4.1). Section
order follows spec 4.4 — syntax rules, component list, built-ins and tools,
custom functions, hoisting and streaming guidance, then the optional sections.

Usage::

	from compiler.openui_library import build_library
	from compiler.openui_prompt import generate_system_prompt

	prompt = generate_system_prompt(build_library())
	prompt = generate_system_prompt(
		spec, tools=["list_patients", "create_appointment"], preamble="You build clinic screens."
	)
"""

from __future__ import annotations

from typing import Sequence, TypedDict

from .openui_library import LibrarySpec, render_signatures

PROMPT_FORMAT = "apg.openui-prompt.v1"


class PromptReport(TypedDict):
	format: str
	library: str
	library_version: str
	components: int
	functions: int
	actions: int
	prompt_chars: int
	prompt_lines: int
	missing_signatures: list[str]
	ok: bool


def _syntax_rules(spec: LibrarySpec) -> str:
	return f"""## Syntax

Write one statement per line: `name = expression`. Order does not matter; names
are hoisted. A statement may reference any other statement by name.

The program's entry is the statement named `root`. Write it first. Its value
must be a `{spec.root}`.

Arguments are positional, in the order shown in each signature. Omit optional
arguments from the end; never leave a gap in the middle. Commas between
arguments are optional.

Strings are double-quoted. Numbers, `true`, `false`, and `null` are literals.
Operators, lowest to highest precedence: `? :`, `||`, `&&`, `==` `!=`,
`>` `<` `>=` `<=`, `+` `-`, `*` `/` `%`, unary `!` `-`.

Component names start uppercase. References start lowercase. `$name` is state.
`@Name(...)` calls a built-in function, an action, or a library function.

One action needs no wrapper: `Button("Save", @ToAssistant("Save"))`. Several go
in `Action([...])`.

Example:

```openui-lang
root = {spec.root}([header, table])
header = Text("Open appointments", "heading")
table = Table([Column("Patient", names), Column("Date", dates)])
names = ["Achieng", "Wanjiru"]
dates = ["2026-10-24", "2026-10-25"]
```"""


def _component_list(spec: LibrarySpec) -> str:
	lines = ["## Components", ""]
	lines.append("One line each: signature, then what the type does not say.")
	lines.append("")

	grouped = {group[0]: group[1] for group in spec.groups}
	notes = {group[0]: group[2] for group in spec.groups}
	placed: set[str] = set()

	for group_name, members in grouped.items():
		present = [m for m in members if any(c.name == m for c in spec.components)]
		if not present:
			continue
		lines.append(f"### {group_name}")
		lines.append("")
		if notes.get(group_name):
			lines.append(notes[group_name])
			lines.append("")
		for name in present:
			component = spec.component(name)
			lines.append(f"`{component.signature()}` — {component.description}")
			placed.add(name)
		lines.append("")

	rest = [c for c in spec.components if c.name not in placed]
	if rest:
		lines.append("### Other")
		lines.append("")
		for component in rest:
			lines.append(f"`{component.signature()}` — {component.description}")
		lines.append("")

	return "\n".join(lines).rstrip()


def _builtins(*, tool_calls: bool, bindings: bool) -> str:
	sections: list[str] = []

	sections.append("""### Built-in functions

`@Count(a)` element count · `@First(a)` `@Last(a)` · `@Sum(a)` `@Avg(a)`
`@Min(a)` `@Max(a)` · `@Filter(a, field, op, value)` · `@Sort(a, field, dir?)` ·
`@Round(n, decimals?)` · `@Abs(n)` `@Floor(n)` `@Ceil(n)` ·
`@Each(a, "v", template)` iterates, binding `v` inside the template.

`a.b` reads a field; on an array it plucks `b` from every element. `.length` is
the element count.""")

	if bindings:
		sections.append("""### State variables

`$name` is a state variable; declare it with a default: `$status = "open"`.
Pass a `$variable` to a `$binding<T>` prop for two-way binding. Write state
with `@Set($var, value)` and restore defaults with `@Reset($var)`.""")

	if tool_calls:
		sections.append("""### Queries and mutations

`name = Query(tool, args, defaults)` fetches when streaming ends and again when
a `$variable` written in its `args` changes. `defaults` is what renders before
the first result arrives — always supply it.

`name = Mutation(tool, args)` runs only through `@Run(name)`, never on load.

`@ToAssistant(message, context?)` sends the conversation onward.
`@OpenUrl(url)` opens a URL. Actions run in order; a mutation run is awaited.""")

	return "\n\n".join(sections)


def _custom(spec: LibrarySpec) -> str:
	lines: list[str] = []
	if spec.functions:
		lines.append("### Library functions")
		lines.append("")
		for function in spec.functions:
			lines.append(f"`{function.signature()}` — {function.description}")
		lines.append("")
	if spec.actions:
		lines.append("### Library actions")
		lines.append("")
		for action in spec.actions:
			lines.append(f"`{action.signature()}` — {action.description}")
		lines.append("")
	return "\n".join(lines).rstrip()


_STREAMING_HEAD = """## Output rules

- Emit only the program. No prose, no explanation, no code fences unless asked
  for prose.
- Every line is one statement. A newline ends a statement.
- Put `root` first so the screen renders while the rest streams.
- Reference a statement by name; never inline a subtree twice.
"""

_STREAMING_TAIL = """
- If a value is unknown, use `null` or a sensible default from the `Query`
  defaults. Never write a placeholder like "TBD".
- Do not use a component that is not listed above."""


def _entity_hint(spec: LibrarySpec) -> str:
	"""Name the entity composites actually present, so the hint never lies."""
	rows = [c.name for c in spec.components if c.name.endswith("Row")]
	boards = [c.name for c in spec.components if c.name.endswith("Board")]
	if not rows and not boards:
		return ""
	named = ", ".join(f"`{name}`" for name in (rows[:2] + boards[:1]))
	return (
		f"- Prefer the typed entity components ({named}) over the generic ones. "
		"Their field names and types come from the schema; do not invent fields."
	)


def _streaming_rules(spec: LibrarySpec) -> str:
	hint = _entity_hint(spec)
	middle = f"\n{hint}\n" if hint else ""
	return _STREAMING_HEAD + middle + _STREAMING_TAIL


def generate_system_prompt(
	spec: LibrarySpec,
	*,
	tools: Sequence[str] | None = None,
	preamble: str = "",
	examples: Sequence[str] = (),
	additional_rules: Sequence[str] = (),
	tool_calls: bool | None = None,
	bindings: bool | None = None,
) -> str:
	"""Render the OpenUI system prompt for *spec*.

	*tool_calls* and *bindings* default to on when *tools* is supplied, matching
	spec 4.2. Pass them explicitly to override.
	"""
	if tool_calls is None:
		tool_calls = bool(tools)
	if bindings is None:
		bindings = tool_calls

	parts: list[str] = []
	if preamble.strip():
		parts.append(preamble.strip())

	parts.append(
		f"You build interface screens in OpenUI Lang for the `{spec.id}` component "
		f"library (version {spec.version}). Each reply is one program that a client "
		f"parses and renders incrementally as it streams."
	)
	parts.append(_syntax_rules(spec))
	parts.append(_component_list(spec))

	builtins = _builtins(tool_calls=tool_calls, bindings=bindings)
	if builtins:
		parts.append("## Functions and data")
		parts.append(builtins)

	custom = _custom(spec)
	if custom:
		parts.append(custom)

	if tools:
		lines = ["## Available tools", "", "Pass the tool name as the first argument to `Query` or `Mutation`.", ""]
		for tool in tools:
			lines.append(f"- `{tool}`")
		parts.append("\n".join(lines))

	if examples:
		parts.append("## Examples\n\n" + "\n\n".join(example.strip() for example in examples))

	parts.append(_streaming_rules(spec))

	if additional_rules:
		parts.append("## Additional rules\n\n" + "\n".join(f"- {rule}" for rule in additional_rules))

	return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def prompt_report(spec: LibrarySpec, prompt: str) -> PromptReport:
	"""Machine-checkable facts about a generated prompt, for snapshot tests."""
	signatures = render_signatures(spec)
	missing = [line for line in signatures if line.split(" — ")[0] not in prompt]
	return {
		"format": PROMPT_FORMAT,
		"library": spec.id,
		"library_version": spec.version,
		"components": len(spec.components),
		"functions": len(spec.functions),
		"actions": len(spec.actions),
		"prompt_chars": len(prompt),
		"prompt_lines": prompt.count("\n") + 1,
		"missing_signatures": missing,
		"ok": not missing,
	}
