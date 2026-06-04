# APG Language Guide

APG is a terse application-generation language for describing executable
business applications, capabilities, workflows, AI agents, screens, data
records, rules, and deployment metadata. The current compiler target is
dependency-light Python. APG source is parsed, converted into a semantic model,
validated, and generated into Python application artifacts that expose HTTP
routes, OpenAPI, component manifests, UI pages, runtime helpers, and smoke
tests.

This guide documents the authoring model implemented by `spec/apg.g4`, the
compiler AST, semantic checks, and generated runtime surfaces.

## Core Shape

Most APG code follows one universal pattern:

```apg
entity_type Name {
    property: value;
    field: type;
    field_with_default: type = value;
    nested_contract: {key: value};
}
```

The grammar calls each top-level construct an `entity`. An entity starts with an
entity type keyword, an identifier, optional inheritance, optional semantic
version, and a braced body.

```apg
module customer_ops version 1.0.0 {
    description: "Customer operations";
}

table Customer {
    name: str;
    email: str;
    active: bool = true;
}

capability CustomerCare {
    contract: {
        id: customer_care,
        provides: [case_management],
        configuration: {sla_hours: 24}
    };
}
```

## Files And Modules

An APG file may begin with a module declaration:

```apg
module enterprise_erp_platform version 1.0.0 {
    description: "Composable ERP application";
    author: "Datacraft";
    license: "Proprietary";
}
```

The grammar also accepts imports, includes, and exports:

```apg
import finance.core version >=1.2.0;
import common.audit as audit;
from common.identity import User, Role;
include "shared/contracts.apg";
include <stdlib>;
export const DEFAULT_CURRENCY = "KES";
```

Use modules to give a generated app a stable identity. Use imports and includes
for future package structure, but keep executable examples self-contained unless
the referenced files are committed with the example.

## Comments And Lexical Rules

APG accepts these comment forms:

```apg
// line comment
# line comment
/* block comment */
```

Identifiers start with a letter or underscore and then use letters, digits, or
underscores. Strings may be single quoted, double quoted, or triple quoted.
Booleans accept `true`, `false`, `True`, and `False`. (`yes`, `no`, `on`, `off` were removed as reserved boolean keywords — see Language Design Notes.)

Semicolons are required for field declarations and most configuration items.
Inside contract objects, APG accepts either semicolons or commas as separators:

```apg
contract: {
    id: stock_control,
    provides: [stock_balances],
    configuration: {default_warehouse: "NBO-01"};
};
```

Prefer semicolons at entity-member boundaries and commas inside arrays and
short inline objects.

## Types

APG supports primitive, optional, union, list, dictionary, generic, and
database-oriented types.

```apg
table Invoice {
    number: str;
    issued_at: datetime;
    subtotal: decimal;
    tax: decimal;
    total: float;
    paid: bool;
    external_ref: str?;
    status: str | None;
    tags: List[str];
    metadata: Dict[str, str];
}
```

Common primitive types:

| Type | Meaning |
| --- | --- |
| `str` | text |
| `int` | integer |
| `float` | floating point number |
| `bool` | boolean |
| `bytes` | binary data |
| `datetime` | timestamp |
| `decimal` | financial decimal |
| `Any` | unconstrained value |
| `None` | no value |

Use `?` for a terse optional type, or `| None` when a union reads better.

## Values

APG values include scalars, lists, objects, references, environment references,
model chains, combinations, regular expressions, URLs, durations, and cron-like
strings.

```apg
settings: {
    tenant_scoped: true,
    currency: "KES",
    retry_count: 3,
    timeout: 30s,
    endpoint: https://api.example.com/v1,
    pattern: /INV-[0-9]+/,
    fallback_models: gpt4 -> claude3 -> llama,
    modalities: speech + vision + text,
    secret: env("APG_SECRET")
};
```

Use quoted strings for human prose. Use bare identifiers for APG references,
entity names, rule decisions, layout names, model aliases, and capability ids.

## Entity Types

The grammar intentionally has a broad entity vocabulary. The compiler maps the
most important executable types into specific AST classes and generated runtime
contracts.

| Entity type | Use |
| --- | --- |
| `table` | Generated record model and CRUD-style route surface |
| `db` | Database connection and DBML schema metadata |
| `app`, `application`, `composition` | Application shell over capabilities, agents, routes, screens, workflows, runtime, and deployment metadata |
| `capability` | First-class capability contract with config, rules, UI, theme, screens, i18n, streaming, ERP metadata |
| `agent` | First-class AI agent declaration |
| `team`, `agent_team`, `swarm` | AI agent team or handoff graph |
| `workflow`, `flow` | Executable deterministic workflow metadata |
| `screen`, `view`, `ui`, `component`, `widget` | UI and composition metadata |
| `rule`, `rule_set`, `policy`, `guardrail` | Governance and rule metadata |
| `ledger`, `finance`, `procurement`, `inventory`, `manufacturing`, `crm`, `hr`, `payroll` | ERP-oriented entity vocabularies |
| `stream`, `publisher`, `subscriber` | Streaming and event metadata |
| `agent_runtime`, `agent_tool`, `agent_memory`, `agent_handoff` | AI agent infrastructure metadata |

Other accepted entity types support specialized domains such as robotics,
industrial monitoring, OSINT, analytics, digital twins, caching, gateways, and
developer tooling. When using a specialized type, verify generated behavior
with `apg compile --verify` and the generated `smoke_test.py`.

## Tables

Tables are the fastest path to executable data behavior.

```apg
table Customer {
    customer_number: str;
    legal_name: str;
    email: str;
    active: bool;
}
```

Generated Python apps expose table metadata, in-memory record storage, record
validation, CRUD-style helpers, OpenAPI schemas, and HTTP routes such as:

- `GET /entities`
- `GET /entities/Customer`
- `GET /entities/Customer/records`
- `POST /entities/Customer/records`
- `GET /entities/Customer/records/{id}`
- `PUT /entities/Customer/records/{id}`
- `DELETE /entities/Customer/records/{id}`

Use clear singular table names. Keep fields typed. Add defaults only where the
generated app can safely create a record without caller input.

## Applications

Applications compose capabilities, agents, teams, components, routes, screens,
workflows, policies, theme, runtime, integrations, and deployments.

```apg
app EnterpriseERPPlatform {
    description: "Composable ERP application shell";
    capabilities: [PlatformAudit, EnterpriseFinance, EnterpriseOperations];
    agents: [Planner];
    agent_teams: [FinanceCrew];
    routes: ["/erp/finance", "/erp/operations"];
    components: {
        finance_workbench: {capability: journal_entries, route: "/erp/finance"}
    };
    screens: {
        ExecutiveHome: {route: "/erp", capability: EnterpriseOperations}
    };
    theme: {name: enterprise_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

Generated apps expose application composition through `apg_application.py`,
`describe_application()`, the component manifest, `/applications`, `/ui`, and
route rendering for declared screens and routes.

## Capabilities

A capability is a composable business or platform unit. Every serious
capability should declare a contract with an id, provided services, required
services, configuration, rules, UI, and theme.

```apg
capability EnterpriseFinance {
    contract: {
        id: enterprise_finance,
        provides: [journal_entries, invoice_generation, payment_allocation],
        requires: [audit_events],
        configuration: {currency: "KES", fiscal_calendar: "monthly"},
        rules: [
            {name: "balanced_journal", when: "debits != credits", action: "deny"},
            {name: "invoice_requires_order", when: "order_id missing", action: "deny"}
        ],
        ui: {shell: python, routes: [{name: "Finance", path: "/erp/finance", component: "FinanceWorkbench"}]},
        theme: {name: finance_theme, tokens: {accent: "#126E82"}}
    };
    erp_modules: [finance, general_ledger, accounts_payable];
    approvals: {levels: 2, approvers: [controller, cfo]};
    master_data: {entities: [account, cost_center, financial_period]};
    i18n: {supported_languages: [en, sw, ha, yo, zu, am, rw], default_language: en, fallback_language: en};
    streaming: {processor: bytewax, state: finance_event_state};
}
```

The compiler validates that a capability has a contract, at least one provided
service, no duplicate provides/requires, and named rules. Generated apps expose
capability descriptions, rule evaluation, configuration resolution, approval
planning, language metadata, theme tokens, screen routes, and health reports.

## Rules

Rules can be attached directly to capabilities or inside a `rule_engine`.

```apg
rule_engine: {
    type: deterministic;
    default_decision: allow;
    inputs: [amount, approval_threshold, tenant_id];
    rules: [
        {name: "large_payment_review", when: "amount > approval_threshold", action: "require_review", priority: 10},
        {name: "missing_tenant", when: "tenant_id missing", action: "deny", priority: 100}
    ];
};
```

Generated deterministic rule evaluation supports a focused expression subset:

- equality and comparison such as `severity == critical`
- arithmetic comparisons such as `on_hand - reserved < 0`
- field presence checks such as `order_id missing` and `tenant_id present`
- access to capability configuration values merged into the evaluation context

Rules should be deterministic by default. If a rule requires a model, external
service, historical aggregation, or temporal lookup, model that as an explicit
capability dependency and keep the rule contract clear about the external data.

## Screens And UI Composition

Use `screens:` when a capability or app needs explicit UI composition.

```apg
screens: {
    OperationsDashboard: {
        route: "/erp/operations/dashboard",
        title: "Enterprise Operations",
        layout: dashboard,
        contains: [InventoryKpis, OrderQueue, ExceptionQueue],
        composes: [FulfillmentTable, FinanceKpis],
        binds: [executive_kpis.current],
        actions: [approve, reject, escalate],
        events: [{on: "select", do: "filter", target: FulfillmentTable}],
        relationships: [
            InventoryKpis -> ExceptionQueue,
            {from: OrderQueue, to: FulfillmentTable, via: selection}
        ]
    }
};
```

`contains` means the screen owns or presents an element. `composes` means the
screen assembles another element, often from another capability. `binds` names
data dependencies. `events` describe UI interactions. `relationships` describe
explicit edges between elements.

Generated apps expose screen metadata, UI route indexes, composition graphs,
and browser pages for declared routes.

## AI Agents

AI agents are first-class entities. They describe role, model, runtime, tools,
memory, IO shape, rules, UI, theme, and handoffs.

```apg
agent SupportPlanner {
    role: "support planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Plan customer support follow-up.";
    capabilities: [ticket_triage, response_planning];
    tools: [case_search, customer_history];
    memory: vector support_memory;
    configuration: {temperature: 0.2, max_steps: 4};
}

agent_team SupportCrew {
    agents: [SupportPlanner, QualityReviewer];
    handoffs: SupportPlanner -> QualityReviewer [condition: drafted];
    capabilities: [support_response];
}
```

Known runtime names include `local`, `codex`, `codex_cli`, `claude`,
`claude_code`, `opencode`, `open_code`, `pi`, `openai`, and `ollama`. Unknown
runtime names are accepted as custom runtimes, but semantic analysis warns that
an adapter must be registered.

Generated AI runtime metadata is provider-neutral. Fast-changing tools such as
Codex, Claude Code, OpenCode, Pi, and future runners should integrate through
adapters rather than new grammar keywords.

## Workflows

Workflow entities compile to deterministic generated helpers when they declare
step or stage metadata.

```apg
workflow ProcurementApproval {
    steps: str = "draft -> review -> approved";
    human_tasks: [review];
    assignments: {review: procurement_manager};
    guards: {review: "amount > 0"};
    waits: {approved: approval_received};
    retry_policy: {review: {attempts: 3}};
    compensation: {approved: reverse_reservation};
}
```

Generated apps expose:

- `list_workflows()`
- `describe_workflow(name)`
- `run_workflow(name, payload)`
- `list_workflow_runs()`
- `get_workflow_run(id)`
- `resume_workflow(id, payload)`
- `execute_workflow_compensations(id, payload)`
- HTTP routes under `/workflows`

The generated workflow runtime supports deterministic step chains, guard
evaluation, payload-driven waits, retry attempts, durable run state, resume, and
compensation action recording.

## Streaming

Use Bytewax as the streaming processor in APG runtime metadata.

```apg
streaming: {
    processor: bytewax,
    input: order_events,
    output: fulfillment_alerts,
    state: operations_event_state,
    window: 5min
};
```

The grammar accepts other identifiers for future extension, but APG standards
prefer `bytewax` or `bytewax_streams` for practical runtime implementation.

## Internationalization

Capabilities can declare language support with `i18n` or `localization`.

```apg
i18n: {
    supported_languages: [en, sw, ha, yo, zu, am, rw],
    default_language: en,
    fallback_language: en
};
```

The grammar includes many African language codes including `af`, `ak`, `am`,
`bm`, `bem`, `ber`, `bin`, `din`, `dyu`, `ee`, `ff`, `fon`, `gaa`, `ha`, `ig`,
`kab`, `kam`, `ki`, `kln`, `kg`, `kj`, `kmb`, `kr`, `lg`, `ln`, `loz`, `lu`,
`lua`, `mg`, `mos`, `nd`, `nr`, `nso`, `ny`, `om`, `rn`, `rw`, `sg`, `sn`,
`so`, `ss`, `st`, `sw`, `ti`, `tn`, `ts`, `tum`, `tw`, `ve`, `wo`, `xh`,
`yo`, and `zu`.

## Compilation Model

Use the CLI:

```bash
apg compile examples/20_enterprise_erp_platform/main.apg --output /tmp/apg-erp --verify
python /tmp/apg-erp/smoke_test.py
```

Generated output normally includes:

- `app.py` - standard-library Python HTTP application
- `__init__.py` - importable package surface
- `README.md` - generated runbook
- `requirements.txt` - dependency note
- `Dockerfile`, `.dockerignore`, `.env.example` - deployment scaffold
- `smoke_test.py` - generated runtime smoke test
- `semantic_model.json` - compiler semantic model
- optional `ai_agents.py`, `apg_capabilities.py`, `apg_application.py`

The compiler target is `python`. Avoid older framework target names such as
`flask-appbuilder` or `django` in new APG source and docs.

## Authoring Principles

- Keep APG terse but not cryptic: short names for local concepts, explicit names
  for business contracts.
- Use tables for durable business records.
- Use capabilities for composable behavior, UI, rules, theme, configuration,
  i18n, streaming, and ERP boundaries.
- Use applications for assembly.
- Use screens for UI relationships.
- Use agents for model-backed work and adapter-based runtime integration.
- Use workflows for deterministic business process state.
- Put external systems behind capabilities and adapters, not inline prose.
- Run `apg compile --verify` and the generated `smoke_test.py` before claiming a
  source file is executable.

## Language Design Notes

### `??` Null-Coalescing / Fallback Operator

`??` is the preferred fallback cascade operator for model and service chains:

```apg
settings: {
    model: gpt4 ?? claude3 ?? llama,
    endpoint: env("API_URL") ?? "https://api.example.com"
};
```

`??` evaluates left to right and returns the first non-null value. Use `->` only
for directed flow (handoffs, state transitions, model chains where ordering
matters beyond fallback semantics).

### `|>` Pipeline Operator

`|>` threads a value through a chain of transformations in expression contexts:

```apg
result: raw_data |> normalize |> validate |> enrich;
```

This maps to `enrich(validate(normalize(raw_data)))` in generated Python.

### Physical Unit Literals

Physical measurement literals combine a number with a SI or imperial unit suffix:

```apg
sensor ThermalProbe {
    max_temp: 80°C;
    operating_pressure: 150psi;
    sample_rate: 500Hz;
    tolerance: 0.5mm;
}
```

Supported unit suffixes: `°C`, `°F`, `°K`, `°R`, `μm`, `nm`, `kPa`, `MPa`,
`GPa`, `kHz`, `MHz`, `GHz`, `mV`, `kV`, `MV`, `mA`, `kA`, `kW`, `MW`, `GW`,
`rpm`, `rps`, `ms`, `us`, `ns`, `ps`, `psi`, `bar`, `atm`, `Hz`, `Pa`,
`km`, `cm`, `mm`, `kg`, `mg`.

### `enum` Entity Type

Use `enum` to declare named value sets:

```apg
enum InvoiceStatus {
    draft;
    pending [label: "Awaiting Approval"];
    approved = 2 [label: "Approved", description: "Ready to dispatch"];
    rejected;
    paid;
}
```

Variant syntax: `NAME ('=' (NUMBER | STRING))? ('[' 'label' ':' STRING ... ']')?`

### `statemachine` Entity Type

Use `statemachine`, `state_machine`, or `fsm` to declare state machines. Transitions
use `->` with optional property brackets:

```apg
statemachine OrderLifecycle {
    initial: draft;
    states: [draft, pending, approved, dispatched, delivered, cancelled];

    draft -> pending [on: submit, guard: "amount > 0"];
    pending -> approved [on: approve, action: notify_customer];
    pending -> rejected [on: reject];
    approved -> dispatched [on: dispatch, timeout: 48h];
    dispatched -> delivered [on: delivery_confirmed];
    pending -> cancelled [on: cancel, priority: 10];
}
```

Transition props: `on`, `guard`, `action`, `priority`, `timeout`, plus any
`IDENTIFIER: value` extension.

`initial:` and `states:` are `config_item` key-value pairs — no special
grammar needed.

### Removed Boolean Keywords: `yes`, `no`, `on`, `off`

`yes`, `no`, `on`, and `off` are no longer reserved as BOOLEAN tokens. They
were extremely common as config values, field names, and enum variants, and
their reservation as booleans caused persistent identifier conflicts. Use
`true`/`false` (or `True`/`False`) for all boolean values.

Before:
```apg
notifications: on;
two_factor: yes;
```

After:
```apg
notifications: true;
two_factor: true;
```

### `@annotation: name { ... }` Named-Block Form

Annotation bodies now accept a named-block form for domain annotations that
carry a named configuration sub-object:

```apg
@physics: finite_element {
    mesh: 0.1mm;
    solver: iterative;
}

@compliance: gdpr {
    retention: 90d;
    lawful_basis: legitimate_interest;
}
```

This is equivalent to `@annotation: { ... }` but makes the annotation subtype
explicit in the source, which is useful for tooling and documentation generators.
