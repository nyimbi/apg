# APG DSL Tooling Specification

APG needs a complete DSL tooling stack, not just a grammar. The platform
must make authoring, reviewing, generating, packaging, and evolving applications
safe across the CLI, IDE, language server, visual designers, generated apps,
and agent workflows.

This document is the implementation specification for that tooling. It keeps the
language surface generic while making the surrounding tooling strong enough for
enterprise application generation, capability composition, database-backed form design,
workflow design, agentic-system design, and natural-language evolution.

## Current Executable Baseline

APG currently has an executable compiler path:

- source files use the `.apg` extension;
- the installed command is `apg`;
- the primary generation command is `apg compile <file> --output <dir>`;
- `apg lint <file-or-directory> --json` emits `apg.lint-report.v1` without
  writing generated code;
- `apg validate <file> --target python --json` emits `apg.validate-report.v1`
  with lint results and generator-readiness metadata;
- `apg model <file> --json` emits `apg.semantic-model.v1` with normalized
  symbols, tables, agents, capabilities, composition metadata, diagnostics,
  deployment metadata, and graph summaries without generating application files;
- `apg format <file> --check|--write|--json` emits `apg.format-result.v1`
  and applies deterministic APG whitespace formatting;
- `apg graph <file> --kind er --format json|mermaid|dot` emits
  `apg.graph.v1` data or renderable graph text;
- `apg graph-suite <file> --json` emits `apg.graph-suite-report.v1` with
  every supported graph rendered as JSON, Mermaid, and DOT;
- the only advertised compiler target is `python`;
- generated applications are dependency-light Python artifacts with `app.py`,
  package exports, OpenAPI metadata, component manifests, smoke tests, and
  `semantic_model.json`, plus optional `ai_agents.py`, `apg_capabilities.py`,
  and `apg_application.py` sidecars;
- generated apps expose `--self-test`, `--describe`, `--validate`, HTTP
  routes, OpenAPI JSON, component manifest JSON, and semantic-model JSON;
- capability contracts can be inspected with
  `python cli.py capabilities contracts --json` and validated with
  `python cli.py capabilities validate-contracts`.

The sections below define where the tooling needs to go. When a command is not
implemented yet, the command name is the desired stable contract, not a claim
that it is already available.

## Goals

The tooling stack must provide:

- one shared parser and semantic model used by every surface;
- deterministic linting, formatting, and validation;
- IDE feedback while the user types;
- safe refactors across tables, fields, views, flows, operations, capability contracts,
  packages, and deployment units;
- generator-ready normalized application metadata;
- machine-readable diagnostics and fix suggestions;
- graph and explain output for human review;
- migration planning for PostgreSQL and MySQL-compatible backends;
- natural-language change planning that produces DSL diffs before code changes;
- release evidence for generated apps and capabilities.

## Non-Goals

The tooling must not:

- hard-code specific capability names into the grammar;
- let visual designers create database-backed fields that do not resolve to
  columns, calculated fields, or valid lookup paths;
- treat generated code as the source of truth when the DSL can express the
  intent;
- expose arbitrary framework targets that multiply the generator matrix;
- accept direct secret literals in DSL files;
- let agents bypass linting, semantic validation, or release evidence.

## Core Architecture

All tooling should consume the same pipeline.

```text
source files
  -> parse tree
  -> AST
  -> symbol table
  -> normalized semantic model
  -> diagnostics
  -> graphs and indexes
  -> generator, CLI, LSP, IDE, tests, natural-language planner
```

The parser is necessary but not sufficient. The semantic model is the contract
that prevents drift between the linter, IDE, generator, language server, visual
designers, and natural-language tools.

### Proposed Modules

| Module | Responsibility |
| --- | --- |
| `compiler.parser` | Existing ANTLR-backed parser wrapper over `spec/apg.g4`, `spec/apgLexer.py`, `spec/apgParser.py`, and `spec/apgVisitor.py`. |
| `compiler.ast_builder` | Existing parse-tree-to-AST conversion layer. Extend it until every first-class APG construct has a stable AST node or normalized metadata record. |
| `compiler.semantic_analyzer` | Existing semantic analyzer. Extend it into the shared semantic model producer so CLI, LSP, generator, tests, and agents do not re-implement APG meaning. |
| `compiler.code_generator` | Existing Python artifact generator. It remains the only compile target and should consume the normalized semantic model rather than ad hoc parse fragments. |
| `compiler.diagnostics` | New diagnostic registry for APG code ranges, severities, related locations, and fix IDs. |
| `compiler.formatter` | New deterministic formatter for `.apg` source. |
| `compiler.graphs` | New graph builders for ER, lookup, workflow, handler, capability, security, agent, package, and deployment graphs. |
| `compiler.migrations` | New semantic-model diff planner for database and capability ownership changes. |
| `compiler.nl_plan` | New constrained natural-language-to-APG-patch planner. |
| `compiler.release` | New release evidence and drift checks for generated applications and capability packages. |
| `language_server.server` | Existing LSP entry point. It should move from direct parser/analyzer calls to the shared semantic model. |
| `cli.*` | Existing Click CLI entry points. Add stable JSON/text contracts for new lint, format, graph, explain, package, and planning commands. |

The exact file layout can evolve, but these boundaries should remain visible.
The parser, semantic model, diagnostics, and formatter must be usable without
starting a web app or generator.

## Semantic Model Contract

The semantic model should be serializable to JSON and stable enough for CLI,
IDE, tests, and agents.

Required top-level fields:

```json
{
  "format": "apg.semantic-model.v1",
  "source_files": [],
  "app": {},
  "symbols": {},
  "tables": {},
  "views": {},
  "flows": {},
  "operations": {},
  "rules": {},
  "roles": {},
  "security": {},
  "agents": {},
  "llms": {},
  "capabilities": {},
  "composition": {},
  "contracts": {},
  "deployment": {},
  "packages": {},
  "graphs": {},
  "diagnostics": []
}
```

### Symbol Table

Every named declaration must produce a symbol:

```json
{
  "id": "table.Invoice",
  "kind": "table",
  "name": "Invoice",
  "file": "finance.apg",
  "range": {"start": {"line": 4, "character": 0}, "end": {"line": 15, "character": 1}},
  "references": ["view.InvoiceForm", "rule.InvoicePolicy"]
}
```

Symbol kinds include:

- `app`
- `table`
- `field`
- `group`
- `enum`
- `enum_value`
- `view`
- `component_binding`
- `handler`
- `flow`
- `flow_state`
- `operation`
- `role`
- `permission`
- `rule`
- `llm`
- `agent`
- `agent_skill`
- `capability`
- `composition`
- `api`
- `event`
- `job`
- `report`
- `menu`
- `component`
- `package`
- `deployment_unit`
- `audit`
- `version`
- `security`

### Table Model

Each table should normalize fields, directives, relationships, calculated
fields, lookup aliases, indexes, uniqueness, checks, and backend constraints.

```json
{
  "name": "Invoice",
  "fields": {
    "customer_id": {
      "type": "int",
      "required": true,
      "relationship": {
        "target_table": "Customer",
        "target_field": "id",
        "cardinality": "many-to-one",
        "alias": "customer"
      }
    },
    "total": {
      "type": "decimal",
      "calculated": true,
      "expression": "subtotal + tax"
    }
  },
  "lookup_paths": {
    "customer.name": {
      "chain": ["Invoice.customer_id", "Customer.name"],
      "valid": true
    }
  }
}
```

### View Model

Views must bind only to real fields, calculated fields, or valid lookup paths.
The model should distinguish section fields from component placements and event
handlers.

```json
{
  "name": "InvoiceForm",
  "table": "Invoice",
  "sections": [{"name": "Header", "fields": ["invoice_number", "customer.name"]}],
  "components": [
    {"binding": "customer.name", "component": "Lookup", "x": 4, "y": 0, "w": 4, "h": 1}
  ],
  "handlers": [{"event": "Save", "target": "SubmitInvoice"}]
}
```

### Workflow Model

Flows should normalize states, transitions, directives, participants, human
tasks, timers, compensation, and handler targets.

```json
{
  "name": "SubmitInvoice",
  "states": ["draft", "reviewed", "approved", "posted"],
  "transitions": [
    {"from": "draft", "to": "reviewed"},
    {"from": "approved", "to": "posted"}
  ],
  "human_tasks": [
    {"name": "FinanceReview", "assignee": "Accountant", "to": "approved"}
  ],
  "timers": [{"state": "reviewed", "duration": "P2D", "to": "escalated"}],
  "compensations": [{"state": "posted", "operation": "ReverseInvoice"}]
}
```

### AI Agent Model

AI agents are first-class APG citizens, not opaque generator annotations. The
semantic model must normalize each agent, its runtime adapter, permissions,
skills, rule bindings, UI bindings, theme tokens, and team membership.

```json
{
  "name": "SupportPlanner",
  "runtime": "codex",
  "model": "openai:gpt-4.1-mini",
  "role": "support planner",
  "system": "Plan customer support follow-up.",
  "configuration": {
    "temperature": 0.2,
    "max_steps": 8
  },
  "rules": ["SupportEscalationPolicy"],
  "skills": [
    {"name": "create_ticket", "target": "CreateTicket", "write": true}
  ],
  "permissions": ["ticket:create"],
  "ui": {"screen": "SupportWorkbench", "panel": "agent_assist"},
  "theme": {"accent": "support"}
}
```

Runtime names must resolve through the APG agent integration registry. The
tooling must support fast-moving runtimes such as Codex, Claude Code, OpenCode,
Inflection Pi, local/offline adapters, and future providers without changing
the grammar for every provider.

### Capability And Composition Model

The grammar remains generic. Concrete capability names are resolved through the
registered APG capability catalog and package manifests. A capability can carry
specific configuration, a rule engine contract, UI routes/screens, and theme
tokens; the semantic model must keep those pieces together so capabilities can
be composed into larger applications.

```json
{
  "composition": "FinanceSuite",
  "includes": [
    {"capability": "gl_core", "version": "1.0.0", "catalog_resolved": true},
    {"capability": "ap_automation", "version": "1.0.0", "catalog_resolved": true}
  ],
  "connections": [
    {
      "from_capability": "ap_automation",
      "from_kind": "event",
      "from_contract": "InvoiceApproved",
      "to_capability": "gl_core",
      "to_kind": "command",
      "to_contract": "PostJournal"
    }
  ]
}
```

Capability records should also normalize the executable contract shape:

```json
{
  "name": "GeneralLedger",
  "configuration_schema": "GeneralLedgerConfiguration",
  "rules": ["PostingPeriodOpen", "BalancedJournal"],
  "ui_routes": ["/ledger", "/ledger/journals"],
  "themes": ["default", "high_contrast"],
  "exports": ["post_journal", "trial_balance"],
  "dependencies": ["TaxConfiguration"]
}
```

Streaming contracts must use ByteWax-oriented semantics. `bytewax` and
`bytewax_streams` are the built-in stream processors. External brokers may be
modeled only as capability endpoints; they are never APG's internal streaming
runtime.

## Diagnostic Specification

Diagnostics must be machine-readable, stable, and documented. Every diagnostic
has:

- `code`
- `title`
- `severity`: `error`, `warning`, `info`, or `hint`
- `message`
- `range`
- `related_locations`
- `fixes`
- `docs_url`

### Diagnostic Code Ranges

| Range | Area |
| --- | --- |
| `APG0000-APG0099` | Syntax and parser errors. |
| `APG0100-APG0199` | Naming, duplicates, reserved words, and style. |
| `APG0200-APG0299` | Tables, fields, types, defaults, calculated fields, and directives. |
| `APG0300-APG0399` | Relationships, foreign keys, lookup paths, and multi-hop traversal. |
| `APG0400-APG0499` | Views, visual components, handlers, menus, and UI binding. |
| `APG0500-APG0599` | Rules, expressions, required checks, and policy actions. |
| `APG0600-APG0699` | Flows, workflow states, timers, human tasks, and compensation. |
| `APG0700-APG0799` | Roles, permissions, security, tenancy, and secrets. |
| `APG0800-APG0899` | APIs, events, jobs, reports, packages, deployment, audit, and versioning. |
| `APG0900-APG0999` | Capability catalog, composition, cross-capability contracts, and package manifests. |
| `APG1000-APG1099` | LLMs, agents, skills, tools, and model/provider configuration. |
| `APG1100-APG1199` | Migration planning and destructive-change detection. |
| `APG1200-APG1299` | Natural-language change plans and agent safety. |
| `APG9000-APG9999` | Internal tooling errors and unsupported parser states. |

### Required Diagnostics

| Code | Severity | Trigger | Example Fix |
| --- | --- | --- | --- |
| `APG0001` | error | Source cannot be parsed. | Show syntax location and nearest valid construct. |
| `APG0101` | error | Duplicate top-level declaration in the same namespace. | Rename one symbol. |
| `APG0201` | error | Field references unknown type where no custom type is allowed. | Create enum/table/type or choose known scalar. |
| `APG0202` | error | Calculated field references unknown field. | Create field or fix expression. |
| `APG0301` | error | Relationship target table does not exist. | Create table or correct target. |
| `APG0302` | error | Relationship target field does not exist. | Create field or correct target. |
| `APG0303` | error | Lookup path cannot be resolved. | Add relationship or change binding. |
| `APG0304` | error | Multi-hop lookup chain breaks at an intermediate segment. | Add missing relationship. |
| `APG0401` | error | View subject table does not exist. | Create table or correct `for` target. |
| `APG0402` | error | Database-backed view binding is not a field, calculated field, or lookup path. | Replace binding or create valid field/path. |
| `APG0403` | error | Handler target does not resolve. | Create operation/flow/agent/contract target. |
| `APG0404` | warning | Component is unknown to the registered component catalog. | Use known component or register one. |
| `APG0501` | error | Rule expression uses single `=` instead of `==`. | Rewrite equality operator. |
| `APG0502` | error | Rule references unknown field. | Correct field or lookup path. |
| `APG0601` | error | Flow transition references undeclared or unreachable state where strict mode is enabled. | Add transition or state directive. |
| `APG0602` | warning | Human task has no assignee/participant. | Add participant or assignment. |
| `APG0701` | error | Permission references unknown resource. | Create resource or correct permission subject. |
| `APG0702` | error | Secret literal appears in source. | Replace with env/secret binding. |
| `APG0801` | error | Deployment unit target is unknown. | Use supported unit kind. |
| `APG0802` | error | Package target does not match app targets. | Add app target or change package target. |
| `APG0901` | error | Composition includes unknown capability key. | Register capability or correct key. |
| `APG0902` | error | Cross-capability connection references unknown event/API/command. | Declare contract or correct reference. |
| `APG0903` | error | Capability attempts shared private-table access. | Use API/event/projection contract. |
| `APG1001` | error | Agent skill target does not resolve. | Create operation/flow/contract target. |
| `APG1002` | error | Agent has write-capable skill with no permission. | Add permission or remove skill. |
| `APG1101` | warning | Migration plan contains destructive drop. | Require explicit migration approval. |
| `APG1201` | error | Natural-language plan cannot be represented as DSL diff. | Ask for narrower DSL-scoped change. |

### Diagnostic Example

Bad DSL:

```apg
view InvoiceForm for Invoice {
  Main: customer.display_name
}
```

If `Invoice.customer_id -> Customer.id` exists but `Customer.display_name` does
not, the linter should return:

```json
{
  "code": "APG0303",
  "severity": "error",
  "title": "Unresolved lookup path",
  "message": "customer.display_name does not resolve from table Invoice.",
  "range": {"start": {"line": 2, "character": 8}, "end": {"line": 2, "character": 29}},
  "fixes": [
    {"id": "replace_with_customer.name", "title": "Use customer.name"},
    {"id": "create_customer_display_name", "title": "Create calculated field Customer.display_name"}
  ]
}
```

## Linter Specification

The linter must run in three stages:

1. **Syntax stage**: parser errors, invalid tokens, unterminated strings,
   malformed blocks.
2. **Semantic stage**: references, lookup paths, handler targets, workflows,
   capability catalog bindings, permissions, deployment/package compatibility.
3. **Policy stage**: enterprise safety, style, secrets, catalog rules,
   release-readiness checks.

### Linter Inputs

- One `.apg` file.
- A directory containing multiple `.apg` files.
- Optional registered capability catalog path.
- Optional component catalog path.
- Optional generator target profile.
- Optional previous semantic-model JSON for migration comparison.

Directory input is an executable contract, not just a planned mode:
`apg lint path/to/apg --json` recursively discovers `*.apg` files,
sorts them for deterministic output, runs the same single-file lint contract for
each file, aggregates diagnostics with a `file` field, and returns one
`apg.lint-report.v1` payload with `source_mode: "directory"` and nested
`file_reports`.

### Linter Outputs

Text mode is for humans. JSON mode is for CI, IDEs, agents, and generated apps.

```json
{
  "format": "apg.lint-report.v1",
  "ok": false,
  "files": ["finance.apg"],
  "severity_counts": {"error": 1, "warning": 0, "info": 0, "hint": 0},
  "diagnostics": [],
  "fixes_available": true,
  "semantic_model_available": false
}
```

### Linter Rules By Domain

Tables:

- primary key exists or can be generated;
- field names are unique inside a table;
- defaults match field type where known;
- calculated expressions reference existing fields;
- relationship targets resolve;
- table directives reference existing fields/calculated fields/lookup paths.

Views and forms:

- `for` table exists;
- section fields resolve;
- component bindings resolve;
- placement rectangles are non-negative and non-zero;
- unknown components are warnings unless strict component mode is enabled;
- handlers target declared operations, flows, agents, APIs, events, jobs, or
  supported navigation targets.

Workflows:

- transitions form a valid graph;
- directives are captured in the workflow model;
- human tasks have assignable participants in strict mode;
- timers use recognizable duration literals;
- compensation targets resolve to operations or flows.

Capabilities and composition:

- included capabilities are declared locally or registered in the catalog;
- versions are present for catalog capability includes;
- cross-capability connections reference exposed APIs, events, or commands;
- private tables are not referenced across capability boundaries;
- datastore backend is one of the allowed backends.

Agents:

- LLM provider references resolve;
- agent skill targets resolve;
- write-capable skills require permissions;
- API keys use environment variable references;
- local model endpoints are configuration, not secrets.

Deployment and packages:

- package targets match app targets;
- deployment units have supported kinds;
- health checks target declared units;
- environment bindings name variables without literal secret values;
- mobile and desktop packages declare signing posture before release.

## Formatter Specification

The formatter must be deterministic and idempotent. Running it twice must
produce byte-identical output.

### Formatting Rules

- Two-space indentation.
- One declaration statement per line inside blocks.
- One blank line between top-level declarations.
- Keep comments attached to the nearest following node when possible.
- Preserve file-level comments at the top of the file.
- Normalize optional semicolons away by default, except in compact one-line
  examples if preserve mode is enabled.
- Place field modifiers in this order: `pk`, `required`, `unique`, `hidden`,
  `search`, `default`, relationship arrow.
- Keep calculated expression immediately after the type: `total: decimal = subtotal + tax`.
- Order table fields as identity, business keys, relationship fields, editable
  scalar fields, calculated fields, audit fields, directives when `--organize`
  is enabled.
- Do not reorder top-level declarations by default; users often keep domain
  context by proximity.

### Formatter Output

```json
{
  "format": "apg.format-result.v1",
  "changed": true,
  "idempotent": true,
  "diagnostics": [],
  "text": "app FinanceOps { ... }"
}
```

The executable formatter contract also proves comment preservation for
file-level, declaration-adjacent, and inline comments, plus canonical field
modifier ordering for `pk`, `required`, `unique`, `hidden`, `search`, `default`,
and relationship arrows.

## CLI Contracts

The installed command should keep the current executable `apg compile` path
stable while adding tooling subcommands around the same parser and semantic
model. Existing helper commands may remain as aliases if they call the same
contracts.

### Current Commands

These commands are executable today and should remain compatible:

```console
apg compile app.apg --output generated/app --verify
apg compile app.apg --target python --output generated/app
apg lint app.apg --json
apg lint src/apg --strict --json
apg model app.apg --json
apg format app.apg --check
apg format app.apg --write
apg graph app.apg --kind er --format json
apg graph app.apg --kind agent --format mermaid
apg graph-suite app.apg --json
apg validate
apg run
apg doctor
apg language-server
python cli.py capabilities contracts --json
python cli.py capabilities validate-contracts
```

`python` is the only compiler target. Desktop, mobile, web, and deployment
packaging are profiles layered on generated Python application artifacts; they
are not separate compile targets.

### `apg lint`

```console
apg lint app.apg
apg lint app.apg --json
apg lint src/apg --strict --catalog docs/capability-catalog.json
```

Exit codes:

- `0`: no errors;
- `1`: lint errors;
- `2`: CLI usage/configuration error;
- `3`: internal tool error.

The executable CLI contract tests cover JSON schemas, default text summaries,
success and failure exit codes, and argparse usage failures for invalid choices
or missing required options.

### `apg format`

```console
apg format app.apg --check
apg format app.apg --write
apg format app.apg --json
```

`--check` exits `1` when formatting changes are needed.

### `apg validate`

```console
apg validate app.apg --target python --json
```

Runs lint plus generator-readiness checks without writing generated code.
Validation fails with `APG0802` when a requested target is not `python`, and
the `apg.validate-report.v1` payload includes the requested target, declared
application packaging profiles, the nested `apg.lint-report.v1`, and
generator-readiness checks.

### `apg compile`

```console
apg compile app.apg --output generated/app
apg compile app.apg --target python --output generated/app --verify
```

Compilation must fail if lint or semantic validation has errors. `--verify`
must run the generated `app.py --self-test` and generated `smoke_test.py`
without requiring third-party runtime services.

### `apg graph`

```console
apg graph app.apg --kind er --format mermaid
apg graph app.apg --kind workflow --format json
apg graph app.apg --kind capability --format dot
```

### `apg graph-suite`

```console
apg graph-suite app.apg --json
```

Emits `apg.graph-suite-report.v1` release evidence for every required graph
kind and renders each graph as JSON, Mermaid, and DOT. This command is the
preferred CI and IDE health check because it proves that graph previews,
documentation diagrams, release evidence, and downstream graph tooling all use
the same semantic model.

Supported graph kinds:

- `er`
- `lookup`
- `workflow`
- `handler`
- `capability`
- `security`
- `agent`
- `deployment`
- `package`

### `apg explain`

```console
apg explain app.apg --symbol Invoice.customer_id
apg explain app.apg --diagnostic APG0303
apg explain app.apg --handler InvoiceForm.Save
```

Explain output should be human-readable by default and JSON with `--json`.

### `apg doctor`

Checks parser generation, Python package imports, catalog availability, template
paths, generator backends, and optional IDE/LSP dependencies.

### `apg package`

```console
apg package app.apg --target desktop --out dist
apg package app.apg --target mobile --out dist
```

Runs package validation, signing posture checks, release evidence generation,
and target-specific smoke checks.

### `apg capabilities`

```console
apg capabilities contracts --json
apg capabilities validate-contracts
apg capabilities list
```

Capability commands operate on executable contracts and package directories,
not grammar changes. A future publish command should emit
`apg.capability-publish-report.v1`; it must load the package entrypoint,
validate the manifest, prove the manifest is publishable, return the catalog
patch, attach release-evidence verification, and record that the publish plan
is side-effect-free. Catalog writes are an explicit downstream step, not an
implicit CLI side effect.

### `apg nl-plan`

```console
apg nl-plan app.apg --prompt "Add credit memos to accounts receivable" --json
```

Produces a proposed DSL diff, lint report, migration preview, and test plan. It
must not write generated code unless the DSL diff validates.

## Language Server Specification

The language server should use the same parser, semantic model, diagnostics,
formatter, and graph builders as the CLI.

### Capabilities

| LSP Feature | Required Behavior |
| --- | --- |
| `textDocument/didOpen` and `didChange` | Incrementally parse, rebuild affected semantic model parts, publish diagnostics. |
| `textDocument/completion` | Complete keywords, block-local directives, table names, fields, lookup paths, components, handler events, operation targets, flow states, capability keys, APIs, events, package targets, deployment units, LLM providers, and agent skills. |
| `textDocument/hover` | Show keyword docs, symbol summary, field type, relationship target, lookup resolution, handler target, capability catalog metadata, and diagnostic explanation. |
| `textDocument/definition` | Navigate from references to declarations for fields, tables, views, flows, operations, roles, capabilities, APIs, events, packages, and deployment units. |
| `textDocument/references` | Find all references across workspace DSL files and generated contract indexes. |
| `textDocument/documentSymbol` | Return hierarchical outline: app, tables, fields, views, sections, components, handlers, flows, operations, capabilities, packages, deployment. |
| `textDocument/rename` | Rename symbols safely and update references; block unsafe renames when migration impact is ambiguous. |
| `textDocument/codeAction` | Offer quick fixes for missing declarations, typo suggestions, create operation from handler, create event contract, add lookup directive, add permission, remove secret literal, and remove invalid stream/runtime picker fields. |
| `textDocument/formatting` | Call the shared formatter. |
| `workspace/symbol` | Search declarations by name, kind, and catalog metadata. |

### Completion Sources

Completions should be context-aware:

- top-level: language constructs such as `table`, `view`, `flow`, `capability`,
  `composition`, `deploy`, `agent`;
- table body: field snippets, directives, relationship targets, group spreads;
- view body: table fields, lookup paths, component names, handler snippets;
- flow body: state names, directive snippets, operation targets;
- composition body: registered capability keys, versions, APIs, events, commands;
- deploy body: declared units, target kinds, health/check/resource snippets;
- agent body: LLM names, operation targets, permission subjects.

### Code Actions

Required code actions:

- create missing table;
- create missing field;
- create calculated field for unresolved form binding;
- create operation from handler target;
- create flow from handler target;
- create event contract;
- add relationship for lookup path;
- replace typo with nearest symbol;
- add missing permission for agent skill;
- replace secret literal with `env` binding;
- register or import capability manifest;
- add package for app target;
- create smoke test declaration for operation/flow/capability.

## IDE Integration

Two editor surfaces are required.

### VS Code Extension

The VS Code extension should provide:

- language-server activation for `.apg` files;
- syntax highlighting;
- diagnostics panel integration;
- code actions and quick fixes;
- outline tree;
- graph previews;
- generated artifact preview;
- command palette actions for lint, format, graph, explain, compile, and
  package;
- capability catalog browser.

### APG Studio / Monaco

The web IDE should reuse the same language server or a compatible semantic
service.

Required Studio surfaces:

- DSL editor;
- component palette;
- form designer synchronized with DSL;
- database designer synchronized with DSL;
- workflow designer synchronized with DSL;
- capability composition designer synchronized with DSL;
- package/deployment designer synchronized with DSL;
- diagnostics and quick-fix panel;
- graph/explain panel;
- natural-language change planner with DSL diff preview.

Visual designers must never create state that cannot round-trip through the
DSL semantic model.

## Graph Tooling

Graph output must be available from CLI, IDE, tests, and release evidence.

| Graph | Nodes | Edges | Use |
| --- | --- | --- | --- |
| Entity relationship | Tables, fields | Foreign keys | Database review, migration planning. |
| Lookup | Table fields, relationship aliases, lookup paths | Path hops | Form validation and automatic lookup controls. |
| Workflow | States, tasks, timers, operations | Transitions | Workflow review and generated runtime checks. |
| Handler | Views, components, events, operations, flows, agents | Event calls | UI architecture and test generation. |
| Capability | Capabilities, APIs, events, commands | Contract connections | Composition review. |
| Security | Roles, permissions, resources, agents | Grants | Authorization review. |
| Agent | Agents, LLMs, skills, operations, permissions | Tool access | Agent safety review. |
| Deployment | Units, packages, resources, health checks | Runs-on and depends-on | Operations review. |
| Package | Targets, packages, signing, assets | Builds | Release review. |

Formats:

- JSON for tools;
- Mermaid for docs and previews;
- DOT for graph tooling;
- SVG/PNG only as generated artifacts, not as source of truth.

## Migration Planner

The migration planner compares two semantic models.

Inputs:

- previous semantic model;
- current semantic model;
- target backend: `postgresql`, `mysql`, or compatible profile;
- optional rename hints.

Output:

```json
{
  "format": "apg.migration-plan.v1",
  "ok": false,
  "backend": "postgresql",
  "changes": [],
  "destructive": true,
  "requires_approval": true,
  "diagnostics": []
}
```

Required detections:

- added table;
- dropped table;
- renamed table candidate;
- added field;
- dropped field;
- renamed field candidate;
- type change;
- nullability change;
- default change;
- relationship change;
- unique/index/check change;
- calculated-field change;
- capability ownership transfer;
- data backfill requirement.

Executable migration tests now prove first-class change records for table-level
index directives, uniqueness constraints, constraint/check directives, and capability
table ownership transfer. Unknown table directives are still reported as
generic directive changes so generators can remain conservative.

Destructive changes must require explicit approval and should include suggested
safe alternatives when possible.

## Natural-Language Change Planner

Natural language is a development vector, but it must produce DSL diffs first.

Pipeline:

```text
user request
  -> intent classification
  -> bounded edit plan
  -> DSL patch
  -> lint
  -> migration preview
  -> generated test plan
  -> user/agent review
  -> generation
```

Planner output:

```json
{
  "format": "apg.nl-plan.v1",
  "prompt": "Add credit memos",
  "intent": "domain_feature",
  "dsl_patch": "...",
  "affected_symbols": [],
  "lint": {},
  "migration_preview": {},
  "test_plan": [],
  "token_budget_notes": []
}
```

Small-model guidance:

- prefer constrained edit operations over free-form rewriting;
- provide symbol tables and snippets, not whole projects;
- require agents to return patches, not regenerated code blobs;
- run lint after every proposed patch;
- reject plans that cannot be represented as DSL.

Supported edit operations:

- add table;
- add field;
- add relationship;
- add view section;
- add component placement;
- add handler;
- add operation;
- add rule;
- add flow transition;
- add capability include;
- add API/event contract;
- add package/deployment unit;
- add agent skill and permission.

## Package And Verifier Tooling

Release verifiers should generate evidence for each target.

Web verifier:

- app builds;
- routes exist;
- generated forms bind valid fields;
- handler targets resolve;
- smoke tests run.

Mobile verifier:

- package metadata exists;
- signing posture declared;
- offline policy declared where needed;
- permissions are explained;
- generated screens fit target density;
- smoke launch path exists.

Desktop verifier:

- package metadata exists;
- installer/update posture declared;
- splash/startup assets declared where used;
- menus and context menus bind to handlers;
- smoke launch path exists.

Capability verifier:

- manifest validates;
- package artifacts exist;
- owned tables have migrations/models;
- APIs/events/handlers are declared;
- no private cross-capability table mutation;
- self-registration is side-effect-free;
- release evidence exists.

Deployment verifier:

- units declared;
- health checks declared;
- environment variables named;
- secret values absent;
- resource hints present for production units;
- topology graph is connected and explainable.

## Test Strategy

Tooling tests must be fixture-driven and deterministic.

| Test Family | Required Coverage |
| --- | --- |
| Parser golden tests | Valid/invalid DSL examples for every grammar construct, enforced by `apg parser-golden --json` and the `apg.parser-golden-audit.v1` report. |
| Semantic tests | Symbol table, lookup paths, handler targets, capability catalog binding, workflows, packages, deployments. |
| Diagnostic golden tests | Every diagnostic code has at least one fixture and expected JSON output. |
| Formatter tests | Idempotency, comment preservation, modifier ordering, stable output. |
| CLI contract tests | Exit codes, JSON schemas, text summaries, bad arguments. |
| LSP tests | Completion, hover, definition, references, rename, code actions, formatting. |
| Graph tests | ER, lookup, workflow, handler, capability, security, agent, package, deployment graph output. |
| Migration tests | Add/drop/rename/type/nullability/default/relationship/index scenarios. |
| Natural-language planner tests | Prompt-to-DSL patch fixtures, lint integration, rejected unsafe plans. |
| Verifier tests | Web/mobile/desktop/capability/deployment release evidence contracts. |
| Drift tests | CLI, LSP, IDE, generator, and tests consume the same semantic model. |

### Parser Golden Audit

`apg parser-golden --json` is the executable gate for grammar coverage. It is intentionally independent of project files: the command runs the checked-in golden fixture catalog and fails when any valid fixture stops parsing, any invalid fixture starts parsing, or any required grammar construct is not represented by a valid fixture.

The report contract is `apg.parser-golden-audit.v1`:

- `ok`: true only when fixture outcomes and construct coverage pass;
- `constructs_required`: the grammar surface the platform promises to keep covered;
- `constructs_covered`: constructs proven by valid fixtures;
- `missing_constructs`: constructs that need new valid examples before release;
- `fixtures`: per-fixture parse outcome, validity expectation, construct tags, and syntax error text;
- `blocking_gaps`: the exact fixture failures that should block CI.

The required construct set includes application options, table fields, reusable field groups, spreads, derived fields, modifiers, relationships, relationship cardinality, table directives, enums, views, component placement, handlers, flows, workflow directives, roles, permissions, rules, rule expressions, LLM definitions, agents, capabilities, capability composition include/require/expose/connect clauses, audit blocks, deployment units/scale/health/check/resource/env/directives, version blocks, operations, security, APIs, events, jobs, reports, menus, component contracts, packages, and tests.

When a new keyword, block, nested item, or syntax form is added to `spec/apg.g4`, the same change must add or extend a parser golden fixture before the grammar is considered release-ready. Diagnostic golden fixtures are still required for semantic behavior, but parser-golden fixtures prove that the grammar itself accepts and rejects the intended language forms.

## Implementation Phases

### Phase 0: Inventory And Stabilization

- Inventory existing parser, linter, formatter, release-audit, capability catalog, and
  generator code.
- Identify duplicate semantic logic.
- Define JSON schemas for diagnostics and semantic model.
- Add fixture directories and built-in fixture catalogs for parser-golden,
  diagnostic-golden, formatter, semantic drift, graph, migration, and verifier
  tests.

Exit criteria:

- Current behavior documented.
- No new generator behavior required.
- Tooling fixtures can run in CI, including `apg parser-golden --json`,
  `apg diagnostics --audit-fixtures --json`, and `apg drift <file> --json`.

### Phase 1: Shared Semantic Model MVP

- Create shared parser wrapper.
- Create AST conversion layer.
- Build symbol table.
- Resolve tables, fields, relationships, lookup paths, views, handlers, flows,
  operations, capability includes, packages, and deployment units.
- Emit `apg.semantic-model.v1`.

Exit criteria:

- CLI and tests can load the same semantic model.
- Database-backed form field validation uses the shared model.
- capability catalog validation uses the shared model.

### Phase 2: Linter And Formatter

- Implement diagnostic registry.
- Implement linter stage separation.
- Implement formatter idempotency.
- Add JSON/text CLI output.
- Add quick-fix IDs.

Exit criteria:

- All required diagnostic families have fixtures.
- `apg lint --json` and `apg format --check` are stable.
- Existing DSL release audit consumes the new reports.

### Phase 3: CLI And Graph Tooling

- Add subcommands for lint, format, validate, graph, explain, package, capability, and
  natural-language planning.
- Add graph builders.
- Add explain output for symbols and diagnostics.

Exit criteria:

- CI can use command outputs without parsing prose.
- Graph output is available in JSON and Mermaid.

### Phase 4: Language Server

- Implement LSP server using the shared semantic model.
- Add diagnostics, completion, hover, definition, references, document symbols,
  rename, code actions, and formatting.
- Add fixture-based LSP tests.

Exit criteria:

- VS Code can edit `.apg` with live diagnostics and completion.
- Rename/code actions update all references safely in fixtures.

### Phase 5: IDE And Visual Designer Integration

- Integrate Monaco or the LSP semantic service.
- Bind form designer, database designer, workflow designer, capability designer,
  package designer, and deployment designer to semantic-model changes.
- Prove round-trip DSL sync.

Exit criteria:

- Visual edits generate DSL patches.
- DSL edits update visual designers.
- Invalid visual edits are rejected with diagnostics.

### Phase 6: Migration, Natural Language, And Release Verifiers

- Implement migration planner.
- Implement natural-language DSL patch planner.
- Implement package and deployment verifiers.
- Emit release evidence bundles.

Exit criteria:

- Natural-language changes produce linted DSL diffs.
- Migration plans detect destructive changes.
- Web/mobile/desktop/capability/deployment verifiers produce machine-readable evidence.

## Contributor Task Breakdown

Good first implementation tasks:

- define diagnostic dataclasses and JSON schema;
- add diagnostic code registry tests;
- create semantic-model dataclasses;
- write table/field symbol extraction;
- write relationship target resolution;
- write lookup path resolution;
- write view binding validation;
- write handler target validation;
- add `apg lint --json` contract tests;
- add formatter idempotency tests.

Intermediate tasks:

- capability catalog binding in semantic model;
- workflow graph extraction;
- graph output in Mermaid and JSON;
- migration diff detection;
- LSP completion and hover;
- code action application for missing operations and lookup directives.

Advanced tasks:

- safe rename across workspace;
- natural-language patch planner;
- visual designer round-trip engine;
- release evidence bundle verifier;
- cross-tool drift tests.

## Priority Order

1. Shared parser and semantic model.
2. Diagnostic registry and linter.
3. Formatter.
4. CLI JSON contracts.
5. Graph and explain tooling.
6. Language server.
7. VS Code and Monaco integration.
8. Migration planner.
9. Natural-language DSL diff planner.
10. Package and release verifiers.

The shared semantic model is the foundation. Without it, every tool will drift:
the linter, IDE, generator, language server, visual designers, and agents will
eventually disagree about what the language means.
