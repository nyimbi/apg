# APG Grammar Guide

This guide explains how to read and extend `spec/apg.g4`. It is written for
people modifying the grammar and compiler, not only for APG application
authors.

## Grammar Architecture

The APG grammar is an ANTLR4 grammar that targets Python:

```antlr
grammar apg;

options {
    language = Python3;
}
```

The top-level parser rule is:

```antlr
program
    : module_declaration? (import_statement | include_statement | export_statement | entity)* EOF
    ;
```

This means an APG file has:

1. Optional `module` declaration.
2. Zero or more imports, includes, exports, and entities.
3. End of file.

Most language growth should happen inside the universal `entity` body rather
than by adding unrelated top-level constructs.

## Universal Entity Rule

The central grammar rule is:

```antlr
entity
    : decorator* entity_type IDENTIFIER inheritance? version_tag? '{' entity_body '}' ';'?
    ;
```

Entity examples:

```apg
table Customer { name: str; }
capability Billing { contract: {id: billing, provides: [invoice_generation]}; }
agent Planner { role: "planner"; model: "openai:gpt-4.1-mini"; runtime: codex; }
```

The entity rule should remain stable. Add new domain nouns to `entity_type` only
when the noun is a durable first-class concept. For one-off capability settings,
prefer contract members or configuration objects.

## Entity Types

`entity_type` is a keyword union. It currently includes:

- core app types: `app`, `application`, `capability`, `composition`,
  `contract`, `policy`, `guardrail`
- data and UI types: `db`, `table`, `form`, `screen`, `view`, `ui`,
  `component`, `widget`
- workflow and business types: `flow`, `rule`, `rule_set`, `process`, `stream`
- ERP types: `erp_module`, `ledger`, `finance`, `procurement`, `inventory`,
  `warehouse`, `manufacturing`, `sales`, `crm`, `hr`, `payroll`
- AI agent types: `agent`, `team`, `agent_team`, `swarm`, `agent_runtime`,
  `agent_tool`, `agent_memory`, `agent_handoff`, `prompt`, `model`, `tool`,
  `memory_store`, `handoff`
- platform and infrastructure types: `config`, `env`, `settings`, `secrets`,
  `vault`, `registry`, `gateway`, `proxy`, `cache`, `session`, `store`,
  `repository`
- specialized domain types for digital twins, robotics, OSINT, analytics,
  industrial monitoring, testing, observability, notifications, and
  transformation

Compiler mapping happens in `compiler/ast_builder.py`. If a new entity type
needs generated behavior, update both the grammar and the AST builder mapping.

## Entity Body Members

An entity body is a list of `entity_member` rules:

```antlr
entity_member
    : capability_contract_block
    | erp_component_block
    | agent_composition_block
    | rule_engine_block
    | screen_contract_block
    | ui_contract_block
    | theme_contract_block
    | stream_runtime_block
    | i18n_contract_block
    | config_item
    | behavior_item
    | annotation
    | method_def
    | nested_entity
    | class_def
    | exception_def
    | variable_declaration
    | database_schema
    ;
```

This order matters when alternatives overlap. Put more specific constructs
before generic `config_item` or generic expressions.

## Configuration Items

Configuration items support three forms:

```antlr
config_item
    : IDENTIFIER ':' type_annotation? '=' value_expr ';'
    | IDENTIFIER ':' value_expr ';'
    | IDENTIFIER ':' type_annotation ';'
    ;
```

Examples:

```apg
name: str;
active: bool = true;
description: "Human-readable text";
runtime: {target: python};
```

This is one reason APG remains terse: a property can be a typed field, a
configuration value, or a typed field with default.

## Contract Objects

Most APG structured metadata flows through these reusable rules:

```antlr
contract_object
    : '{' contract_member* '}'
    ;

contract_member
    : (IDENTIFIER | STRING) ':' contract_value contract_separator?
    ;

contract_separator
    : ';' | ','
    ;
```

This allows both of these forms:

```apg
configuration: {currency: "KES", fiscal_calendar: "monthly"};
configuration: {
    currency: "KES";
    fiscal_calendar: "monthly";
};
```

Grammar extensions should reuse `contract_object`, `contract_array`,
`contract_value`, `reference_list`, and `contract_scalar` unless a construct
requires stricter syntax.

## Capability Grammar

Capability contracts are parsed by:

```antlr
capability_contract_block
    : 'contract' ':' capability_contract ';'?
    | 'capability_contract' ':' capability_contract ';'?
    ;
```

Recognized capability contract members include:

- `id`
- `name`
- `version`
- `provides`
- `requires`
- `configuration`
- `configuration_schema`
- `rule_engine`
- `rules`
- `ui`
- `theme`
- `runtime`
- custom identifiers

The compiler semantic analyzer currently requires a capability declaration to
have a contract and at least one provided service. It also rejects duplicate
`provides` and `requires` entries and unnamed rules.

## Agent Grammar

Agent composition uses:

```antlr
agent_composition_block
    : 'agents' ':' agent_set contract_separator?
    | 'runtimes' ':' agent_runtime_set contract_separator?
    | 'tools' ':' agent_tool_set contract_separator?
    | 'handoffs' ':' handoff_graph contract_separator?
    | 'memory' ':' agent_memory_contract contract_separator?
    ;
```

Runtime references include known names:

```antlr
agent_runtime_ref
    : 'local' | 'codex' | 'codex_cli' | 'claude' | 'claude_code'
    | 'opencode' | 'open_code' | 'pi' | 'openai' | 'ollama'
    | IDENTIFIER | STRING
    ;
```

Keep this list conservative. New fast-moving agent tools should usually be
integrated as adapter names, environment variables, or runtime contract objects
instead of hard grammar forks.

## Handoff Graphs

Handoffs use edge syntax:

```apg
handoffs: SupportPlanner -> QualityReviewer [condition: drafted];
```

The rule is:

```antlr
handoff_edge
    : IDENTIFIER '->' IDENTIFIER handoff_modifier*
    ;
```

Semantic analysis verifies that handoff endpoints reference known agents in the
same module.

## Rule Grammar

Rules can be an array or object-like set:

```antlr
rule_list
    : '[' (rule_contract (',' rule_contract)*)? ']'
    | '{' rule_contract* '}'
    ;
```

Rule members include `name`, `when`, `condition`, `then`, `action`, `effect`,
`decision`, `priority`, `applies_to`, effective dates, exceptions, approvals,
and audit metadata.

Use the grammar to accept the contract. Put executable rule semantics in the
compiler/runtime. Do not make the parser responsible for deciding whether a
rule condition is meaningful business logic.

## Screen Grammar

Screens are first-class contracts:

```antlr
screen_contract_member
    : 'route' ':' contract_scalar contract_separator?
    | 'title' ':' contract_scalar contract_separator?
    | 'layout' ':' screen_layout contract_separator?
    | 'contains' ':' screen_element_list contract_separator?
    | 'composes' ':' screen_element_list contract_separator?
    | 'binds' ':' reference_list contract_separator?
    | 'actions' ':' reference_list contract_separator?
    | 'events' ':' screen_event_list contract_separator?
    | 'relationships' ':' screen_relationship_list contract_separator?
    | ...
    ;
```

Relationships can be object-based or edge-based:

```apg
relationships: [
    InventoryKpis -> ExceptionQueue,
    {from: OrderQueue, to: FulfillmentTable, via: selection}
];
```

Use object form when metadata is richer. Use edge form when readability matters.

## Theme, UI, I18n, And Streaming

These contracts are deliberately parallel:

```apg
ui: {shell: python, routes: [{name: "Finance", path: "/finance", component: "FinanceWorkbench"}]};
theme: {name: finance_theme, tokens: {accent: "#126E82"}};
i18n: {supported_languages: [en, sw, ha], default_language: en, fallback_language: en};
streaming: {processor: bytewax, input: events, output: alerts, state: event_state};
```

When extending one of these areas, prefer adding a member to the specific
contract before adding a new top-level entity keyword.

## Type Grammar

Type annotations are unions of primary types:

```antlr
type_annotation
    : union_type
    ;

union_type
    : primary_type ('|' primary_type)*
    ;
```

Primary types include primitive types, database types, generics, lists, and
dictionaries. Optional types use `?`.

Examples:

```apg
email: str;
external_id: str?;
status: str | None;
tags: List[str];
metadata: Dict[str, str];
```

When adding new type syntax, update parser rules, AST builder conversion, and
semantic type validation together.

## Expression Grammar

APG expression grammar is Python-like. It includes:

- lambda expressions
- conditional expressions
- boolean `or`, `and`, `not`
- comparisons
- bitwise operators
- shifts
- arithmetic
- power
- calls, indexing, and dotted access
- list, dict, and comprehension syntax
- `await`
- `yield`
- pattern matching support in statements

Value expressions are a more declarative subset used in contracts and config:

- simple values
- lists and dictionaries
- model chains with `->`
- memory contracts
- references
- combinations with `+`
- URL and regex patterns
- time expressions
- async expressions

Keep executable Python-like statements separate from declarative contract
values. This prevents capability contracts from becoming arbitrary code.

## Lexer Rules

Important lexer rules:

- `IDENTIFIER`: `[a-zA-Z_][a-zA-Z0-9_]*`
- `NUMBER`: integer, binary, octal, hex, float, or complex
- `STRING`: single, double, or triple quoted
- `REGEX`: slash-delimited regex
- `URL`: `http`, `https`, `ftp`, or `ftps`
- `TIME_LITERAL`: `HH:MM` or `HH:MM:SS`
- `DURATION`: numeric value plus unit such as `s`, `min`, `hour`, `day`
- `SEMVER`: `major.minor.patch` with optional prerelease/build suffix
- `BOOLEAN`: true/false plus yes/no/on/off
- comments: `//`, `#`, and `/* ... */`

ANTLR lexer rule order matters. Adding broad keyword tokens can create token
overlap conflicts. Prefer parser-level string literals for keywords unless a
separate token is genuinely required.

## How To Extend The Grammar

Use this sequence:

1. Add or adjust grammar rules in `spec/apg.g4`.
2. Regenerate parser artifacts if the project workflow requires checked-in
   generated parser files.
3. Update `compiler/ast_builder.py` so parsed syntax becomes AST data.
4. Update `compiler/semantic_analyzer.py` for validation.
5. Update `compiler/code_generator.py` if the construct should generate
   executable Python behavior.
6. Add or update fixtures under `tests/fixtures/`.
7. Add focused parser, semantic, codegen, and generated runtime tests.
8. Compile at least one example with `apg compile --verify`.
9. Regenerate checked-in example outputs when compiler output changes.
10. Document the authoring contract.

## Practical Extension Rules

- Preserve the universal entity pattern.
- Prefer reusable contract objects over bespoke syntax.
- Add grammar keywords only for durable first-class concepts.
- Keep AI provider churn out of grammar. Use adapter configuration.
- Keep streaming implementation vocabulary centered on Bytewax.
- Make syntax terse, but choose names that reveal business intent.
- Keep parser acceptance broader than runtime execution, but document the
  executable subset.
- Never claim a new syntax is serviceable until it parses, builds AST data,
  passes semantic checks, generates code where expected, and appears in focused
  tests.

## Common Pitfalls

- Adding a parser rule but not wiring the AST builder.
- Adding an entity keyword without generated runtime behavior.
- Putting broad lexer tokens before identifiers and causing keyword conflicts.
- Letting `config_item` capture syntax that needed a more specific rule.
- Hiding external service behavior inside rule strings.
- Adding agent runtime keywords for every new tool instead of using adapters.
- Forgetting to update examples and docs after compiler output changes.
