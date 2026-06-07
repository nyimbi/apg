# APG Language Manual

**APG** (Application Programming Generation) is a terse declarative language for composing
enterprise applications from capability contracts, data models, AI agents, workflows, and
screens. An APG source file compiles to a runnable Python application, a machine-readable
semantic model, and typed agent stubs — with zero boilerplate.

---

## Table of Contents

1. [Program Structure](#1-program-structure)
2. [Comments](#2-comments)
3. [Types and Values](#3-types-and-values)
4. [Tables — Data Models](#4-tables--data-models)
5. [Capabilities — Composable Contracts](#5-capabilities--composable-contracts)
6. [Rules and Rule Engines](#6-rules-and-rule-engines)
7. [Screens — UI Composition](#7-screens--ui-composition)
8. [Workflows — State Machines](#8-workflows--state-machines)
9. [AI Agents](#9-ai-agents)
10. [Agent Teams](#10-agent-teams)
11. [Applications — Composition Root](#11-applications--composition-root)
12. [ERP Metadata](#12-erp-metadata)
13. [Internationalisation (i18n)](#13-internationalisation-i18n)
14. [Streaming](#14-streaming)
15. [Imports](#15-imports)
16. [The Compiler and CLI](#16-the-compiler-and-cli)
17. [Generated Artefacts](#17-generated-artefacts)
18. [Complete Example](#18-complete-example)
19. [Language Grammar Summary](#19-language-grammar-summary)

---

## 1. Program Structure

An APG file is a sequence of declarations. At the top sits an optional
`module` header, followed by any number of entity declarations in any order.

```apg
module crm_platform version 1.0.0 {
    description: "Composable CRM platform";
    author: "Datacraft";
}

table Contact { ... }
capability CRMCore { ... }
agent SalesAssistant { ... }
workflow LeadQualification { ... }
app CRMPlatform { ... }
```

### Module Header

```apg
module <name> version <semver> {
    description: "<text>";
    author: "<text>";
    license: "<text>";
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `name` | Yes | Dot-separated identifier: `my_app` or `org.crm.platform` |
| `version` | Yes | Semantic version: `1.0.0`, `2.3.1` |
| `description` | No | Free text, quoted |
| `author` | No | Free text, quoted |

The module name appears in generated `manifest.json`, container labels, and
the Python package `__init__.py`.

### Entity Order

Declarations can appear in any order. The compiler resolves references across
the whole file before generating code, so you may declare a `capability` before
or after the `table` it uses.

### Separators

Inside an entity body, statements end with `;`. Inside object literals
(`{ key: value }`), properties are separated by `,`. Both are accepted
anywhere a separator is expected, so existing JSON-style config pastes cleanly.

---

## 2. Comments

```apg
// Single-line comment — rest of line is ignored.
# Python-style single-line comment.
/* Multi-line
   block comment. */
```

Comments may appear anywhere outside a string literal. They are stripped
before the ANTLR parser sees the source, so `//` inside a URL string
(`"http://..."`) is NOT treated as a comment.

---

## 3. Types and Values

### Scalar Types

| Type | Syntax | Notes |
|------|--------|-------|
| String | `str` | UTF-8 text |
| Integer | `int` | 64-bit signed |
| Float | `float` | 64-bit IEEE 754 |
| Decimal | `decimal` | Exact decimal (finance) |
| Boolean | `bool` | `true` or `false` |
| Bytes | `bytes` | Raw binary |
| Date | `date` | ISO-8601 date only |
| Time | `time` | ISO-8601 time only |
| Datetime | `datetime` | ISO-8601 timestamp |
| Any | `Any` | Untyped |

### Optional and Union Types

```apg
external_id: str?;          // optional string (str | None)
status: str | None;         // explicit union
rating: int | float;        // numeric union
```

The `?` suffix is shorthand for `| None`.

### Collection Types

```apg
tags: List[str];
roles: list[str];           // lowercase also accepted
attrs: Dict[str, Any];
pairs: dict[str, int];
nested: List[Dict[str, str]];
```

### Vector

```apg
embeddings: vector;         // embedding vector (for AI memory fields)
```

### Value Literals

```apg
name: "hello";              // double-quoted string
code: 'ABC-001';            // single-quoted string
count: 42;
amount: 3.14;
active: true;               // or: false
created: datetime;          // type-only declaration, no default
external_ref: str?;         // optional, defaults to null

// Environment variable reference
secret: env("DATABASE_URL");
api_key: $MY_API_KEY;

// Duration literals
timeout: 5min;
ttl: 24hour;
window: 30sec;

// Model fallback chain
primary: gpt4 -> claude3 -> llama;     // first-available model chain
preferred: gpt4 ?? claude3 ?? llama;   // null-coalescing chain

// Combination (multi-modal)
modalities: speech + vision + text;

// Object literal
config: {key: value, other: "text"};

// List literal
steps: [a, b, c];
ids: ["001", "002"];
```

### Default Values

```apg
table Product {
    status: str = "active";
    created_at: datetime;        // no default — required at insert
    weight_kg: float = 0.0;
    is_digital: bool = false;
    tags: List[str] = [];
}
```

---

## 4. Tables — Data Models

`table` declares a data-backed entity with typed fields. The compiler
generates a database schema, REST endpoints, and a form projection from
each table.

```apg
table <Name> {
    <field>: <type>;
    <field>: <type> = <default>;
    <field>: <type>?;
}
```

### Example

```apg
table Invoice {
    invoice_number: str;
    customer_id: str;
    issue_date: date;
    due_date: date;
    amount: decimal;
    tax: decimal = 0.0;
    status: str = "draft";
    notes: str?;
}
```

### Field Naming

Field names use `snake_case`. Names must start with a letter or `_` and
contain only `[A-Za-z0-9_]`.

### Relationships (Convention)

APG does not declare foreign-key constraints in field syntax; instead, the
convention is to name linking fields with the pattern `<entity>_id`. The
semantic model and generated code honour this convention for join operations:

```apg
table OrderLine {
    order_id: str;          // links to table Order
    product_id: str;        // links to table Product
    quantity: int;
    unit_price: decimal;
}
```

### Multiple Tables

A program may declare any number of tables. They share no namespace — the
same field name can appear in different tables without conflict.

---

## 5. Capabilities — Composable Contracts

A `capability` is the unit of composition in APG. It declares what it
_provides_ to other capabilities, what it _requires_ from them, how it
is configured, what business rules govern it, what its UI looks like, and
how it streams events. Capabilities are the building blocks that `app`
declarations assemble into a running application.

```apg
capability <Name> {
    contract: {
        id: <identifier>,
        provides: [<service>, ...],
        requires: [<service>, ...],
        configuration: { <key>: <value>, ... },
        configuration_schema: { required: [...] },
        rules: [ { name: "...", when: "...", action: ... }, ... ],
        rule_engine: { type: deterministic, default_decision: allow },
        ui: { shell: python, routes: [...] },
        theme: { name: ..., tokens: { ... }, components: { ... } },
        runtime: { target: python, streaming: { processor: bytewax } }
    };

    // Optional ERP metadata
    erp_modules: [...];
    approvals: { levels: N, approvers: [...] };
    master_data: { entities: [...] };
    business_rules: [...];
    screens: { <ScreenName>: { ... } };

    // Optional streaming config
    streaming: { processor: bytewax, state: <state_name> };

    // Optional i18n config
    i18n: { supported_languages: [...], default_language: en };
}
```

### Contract Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Stable snake_case identifier for this capability |
| `provides` | Yes | Services this capability makes available to dependants |
| `requires` | No | Services this capability depends on |
| `configuration` | No | Runtime configuration values |
| `configuration_schema` | No | Schema for the configuration (required keys, types) |
| `rules` | No | Business rules evaluated at runtime |
| `rule_engine` | No | Rule evaluation strategy |
| `ui` | No | UI shell and route declarations |
| `theme` | No | Visual design tokens |
| `runtime` | No | Execution backend and streaming |

### Provides and Requires

```apg
capability Payroll {
    contract: {
        id: payroll,
        provides: [payroll_runs, payslips, statutory_deductions],
        requires: [hr_employees, general_ledger]
    };
}
```

`provides` identifiers are the public services other capabilities may
`require`. The semantic analyser warns if a `requires` reference cannot be
found in the current module or the system capability manifest.

### Configuration

```apg
contract: {
    id: credit_control,
    configuration: {
        tenant_id: "default",
        default_credit_limit: 50000,
        currency: "KES",
        auto_approve_below: 5000
    },
    configuration_schema: {
        required: ["tenant_id", "currency"]
    }
};
```

Configuration values are available to generated helpers and the rule engine
context. Anything in `configuration_schema.required` must be present for the
capability contract to be considered valid at runtime.

### Themes

```apg
theme: {
    name: crm_theme,
    tokens: {
        "color.primary": "#1565C0",
        "color.accent":  "#FF6D00",
        "border.radius": "6px",
        "density":       "compact"
    },
    components: {
        opportunity_card: { icon: "target", status_indicator: "stage-chip" }
    }
};
```

Theme tokens are passed to the generated frontend shell. Component overrides
let you specialise individual UI components without touching generated code.

### UI and Routes

```apg
ui: {
    shell: python,
    routes: [
        {name: "Dashboard",  path: "/crm",          component: "CRMDashboard",   permission: "crm:view"},
        {name: "Contacts",   path: "/crm/contacts", component: "ContactList",     permission: "crm:contacts"},
        {name: "Pipeline",   path: "/crm/pipeline", component: "PipelineView",    permission: "crm:pipeline"}
    ],
    requires_theme: true
};
```

Shell options: `python` (default server-rendered), `react` (SPA), `mobile`,
`cli`. Routes are registered in the generated app manifest and OpenAPI spec.

---

## 6. Rules and Rule Engines

Rules are declarative guards that the generated rule engine evaluates at
runtime. Each rule has a name, a condition expression, and an action.

### Rule Syntax

```apg
rules: [
    {name: "over_credit_limit",  when: "order_total > credit_limit",  action: deny},
    {name: "manual_review",      when: "risk_score > 0.75",           action: require_review},
    {name: "tenant_required",    when: "tenant_id missing",           action: deny},
    {name: "senior_approval",    when: "amount > 100000",             action: require_review}
];
```

### Rule Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier within the capability |
| `when` | Yes | Condition expression (see below) |
| `action` | Yes | What to do when condition is true |
| `priority` | No | Integer; higher numbers evaluated first |
| `effective_from` | No | ISO-8601 date activation |
| `effective_to` | No | ISO-8601 date expiry |

### Actions

| Action | Meaning |
|--------|---------|
| `allow` | Permit the operation |
| `deny` | Block the operation |
| `require_review` | Route to human approval queue |
| `warn` | Continue but log a warning |
| `audit` | Continue but emit an audit event |
| `reject` | Alias for `deny` |
| Any identifier | Custom action handled by the capability's rule engine |

### Rule Condition Expressions

The `when` field is a boolean expression string parsed at compile time.
The compiler validates that field names exist in the ambient context and
stores the parsed AST in the semantic model.

**Comparison:**
```
amount > 50000
status == "active"
risk_score >= 0.75
discount_pct != 0
count < min_quantity
```

**Existence:**
```
bank_account missing
override_reason not missing
```

**Set membership:**
```
country in [KE, UG, TZ, RW]
agency in [FDA, EMA, HC]
status not in [draft, cancelled]
```

**Boolean combinators:**
```
amount > 50000 and stage == qualification
allergy_detected == true and override_reason missing
amount < 100 or amount > 1000
(a == 1 or b == 2) and c == 3
```

`and` binds more tightly than `or` (same precedence as Python / SQL).

**Operators:**
`==`, `!=`, `<>` (alias for `!=`), `>`, `>=`, `<`, `<=`, `=` (alias for `==`)

**Value types in conditions:**
Unquoted identifiers resolve to field or context references. Quoted strings
are literal string values. Numbers may be integers or floats.

### Rule Engine

```apg
rule_engine: {
    type: deterministic,
    default_decision: allow,
    rules: [...]
};
```

The `rule_engine` block is an alternative to inline `rules` that gives you
explicit control over the evaluation strategy:

| `type` | Behaviour |
|--------|-----------|
| `deterministic` | Evaluate all rules in priority order; first match wins |
| `policy` | Evaluate all rules; combine decisions |
| `ai_assisted` | Pass rules to an LLM for evaluation |

`default_decision` (`allow` or `deny`) is applied when no rule matches.

---

## 7. Screens — UI Composition

Screens declare which UI components a capability exposes and how they relate
to each other. They are defined inside a capability block using the `screens`
property.

```apg
screens: {
    <ScreenName>: {
        route: "<path>",
        title: "<Display Name>",
        layout: <layout>,
        contains: [<Component>, ...],
        composes: [<Component>, ...],
        binds: [<data.binding>, ...],
        actions: [<action>, ...],
        events: [{ on: "<event>", do: "<handler>", target: <Component> }],
        relationships: [
            <A> -> <B>,
            { from: <A>, to: <B>, via: <mechanism>, type: filter }
        ]
    }
};
```

### Screen Fields

| Field | Description |
|-------|-------------|
| `route` | URL path this screen is served at |
| `title` | Human-readable screen title |
| `layout` | Arrangement of contained components |
| `contains` | Components rendered directly inside this screen |
| `composes` | External components imported into this screen's layout |
| `binds` | Data sources wired into the screen at load time |
| `actions` | User actions the screen exposes |
| `events` | Client-side event handlers |
| `relationships` | Directed data-flow edges between components |

### Layouts

`stack`, `grid`, `tabs`, `split`, `wizard`, `dashboard`, `form`, or any custom
identifier.

### Relationships

```apg
relationships: [
    // Arrow shorthand: A filters B
    KpiStrip -> LedgerTable,

    // Object form with metadata
    { from: ApprovalQueue, to: LedgerTable, via: selection, type: filter }
]
```

### Example

```apg
capability OperationsWorkbench {
    contract: { id: ops_workbench, provides: [operations_ui], ui: {shell: python} };

    screens: {
        Dashboard: {
            route: "/ops",
            title: "Operations Dashboard",
            layout: dashboard,
            contains: [KpiStrip, ApprovalQueue],
            composes: [LedgerTable],
            binds: [ledger.entries],
            actions: [approve, reject],
            events: [{on: "select", do: "filter", target: LedgerTable}],
            relationships: [KpiStrip -> LedgerTable, ApprovalQueue -> LedgerTable]
        }
    };
}
```

---

## 8. Workflows — State Machines

A `workflow` declares a named state-machine process with explicit states,
transitions, human task assignments, and business rule guards.

```apg
workflow <Name> {
    steps: str = "<state1> -> <state2> -> ... -> <stateN>";
    human_tasks: [<state>, ...];
    assignments: { <state>: <role_or_user>, ... };
    guards: { <state>: "<condition>", ... };
    timers: { <state>: "<ISO-8601-duration>", ... };
    waits: { <state>: <event_name>, ... };
    retry_policy: { <state>: "<attempts>", ... };
    compensation: { <state>: <action>, ... };
}
```

### steps

The `steps` field declares the ordered list of state names separated by `->`.
The compiler parses this into a typed state graph:

```apg
steps: str = "draft -> budget_review -> procurement_review -> approved";
```

This produces four states and three sequential transitions:
```
draft → budget_review → procurement_review → approved
```

### human_tasks

States listed in `human_tasks` require a human to act before the workflow
advances. The semantic analyser warns if a human_task references a state not
declared in `steps`.

```apg
human_tasks: [budget_review, procurement_review];
```

### assignments

Maps states to the role or user who performs the human task:

```apg
assignments: {
    budget_review:       budget_manager,
    procurement_review:  procurement_lead
};
```

### guards

A guard is a rule condition that must evaluate to true before the workflow
may enter that state. The condition syntax is identical to rule `when`
expressions (see §6).

```apg
guards: {
    budget_review:  "amount > 0",
    approved:       "all_reviews_complete"
};
```

### timers, waits, retry_policy, compensation

```apg
timers: { approved: "PT24H" };                // SLA: must complete within 24h
waits:  { approved: approval_received };       // pause until event
retry_policy: { budget_review: "3" };          // max 3 retry attempts
compensation: { approved: reverse_reservation }; // undo on cancel
```

### Generated Workflow API

Each workflow produces these helpers in the generated `app.py`:

```python
list_workflows()
describe_workflow("ProcurementApproval")
run_workflow("ProcurementApproval", {"amount": 45000})
list_workflow_runs()
get_workflow_run("run-id")
resume_workflow("run-id", {"events": ["approval_received"]})
execute_workflow_compensations("run-id")
```

### Example

```apg
workflow ProcurementApproval {
    steps: str = "draft -> budget_review -> procurement_review -> finance_approval -> approved";
    human_tasks: [budget_review, finance_approval];
    assignments: {
        budget_review:    budget_owner,
        finance_approval: finance_controller
    };
    guards: {
        budget_review:    "amount <= budget_limit",
        finance_approval: "amount > finance_threshold"
    };
    timers: { finance_approval: "PT24H" };
    waits:  { finance_approval: finance_packet_ready };
    compensation: { approved: release_budget_hold };
}
```

---

## 9. AI Agents

An `agent` declaration defines an AI agent with its model, runtime adapter,
system prompt, tools, memory, and capability access. The compiler generates
a typed `AgentBase` subclass in `agent_stubs.py` and a runtime manifest in
`ai_agents.py`.

```apg
agent <Name> {
    role: "<description>";
    model: "<provider>:<model-id>";
    runtime: <runtime>;
    system: "<system-prompt>";
    capabilities: [<capability-service>, ...];
    tools: [<tool-name>, ...];
    memory: <kind> <name>;
    input: <name>;
    output: <name>;
    configuration: { <key>: <value>, ... };
    rules: [...];
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `role` | Yes | Human-readable description of the agent's function |
| `model` | Yes | Model identifier in `provider:model` format |
| `runtime` | No | Runtime adapter (default: `codex`) |
| `system` | No | System prompt string |
| `capabilities` | No | Capability services the agent may invoke |
| `tools` | No | Tool function names the agent may call |
| `memory` | No | Memory backend and variable name |
| `input` | No | Name of the primary input context |
| `output` | No | Name of the primary output context |
| `configuration` | No | Model hyperparameters (temperature, max_turns, etc.) |
| `rules` | No | Pre/post-invocation guards |

### Model Identifiers

```apg
model: "openai:gpt-4.1-mini";
model: "openai:gpt-4.1";
model: "anthropic:claude-opus-4-8";
model: "ollama:llama3.3";
model: "ollama:mistral";
```

The provider prefix determines which runtime adapter is used. Use `ollama:`
for locally-hosted open-weight models.

### Runtimes

| Runtime | Description |
|---------|-------------|
| `codex` | OpenAI Codex / GPT via OpenAI API |
| `claude_code` | Anthropic Claude Code adapter |
| `opencode` | Open source code model adapter |
| `claude` | Anthropic Claude via API |
| `openai` | OpenAI chat completions |
| `ollama` | Local Ollama server |
| `pi` | Inflection Pi |
| `local` | No-op for testing |

Wire up a runtime by setting the environment variable:

```bash
export APG_AGENT_CODEX_PROVIDER_COMMAND='python my_provider.py'
# or globally:
export APG_AGENT_PROVIDER_COMMAND='python my_provider.py'
```

The generated agent stub sends a JSON payload on stdin and reads the
response from stdout:

```json
// stdin (sent to provider):
{"agent": {"name": "SalesAssistant", "role": "sales assistant", "model": "openai:gpt-4.1-mini"},
 "input": "<prompt>",
 "context": {"tenant_id": "acme", "user_id": "u-001"}}

// stdout (expected from provider):
{"output": "<response text>"}
```

### Memory

```apg
memory: vector sales_memory;       // vector store with name sales_memory
memory: redis cache_memory;        // redis-backed memory
memory: {kind: vector, name: support_memory, ttl: 7d};   // structured form
```

### Example

```apg
agent SalesAssistant {
    role: "sales assistant";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Analyse CRM context and suggest next best actions, talking points, and risk factors for deals.";
    capabilities: [contact_lifecycle, opportunity_pipeline];
    tools: [contact_search, deal_analysis, competitor_lookup];
    memory: vector sales_memory;
    input: crm_context;
    output: sales_plan;
    configuration: {temperature: 0.2, max_turns: 6};
    rules: [
        {name: "no_pii_in_output", when: "output_contains_pii == true", action: require_review}
    ];
}
```

---

## 10. Agent Teams

An `agent_team` groups multiple agents into a coordinated crew with
explicit handoff flows.

```apg
agent_team <Name> {
    agents: [<Agent>, ...];
    flow: <A> -> <B> [condition: <state>];
    capabilities: [<service>, ...];
    configuration: { ... };
    rules: [...];
}
```

Alternative keyword: `team` is accepted in addition to `agent_team`.

### Handoff Flow

```apg
flow: Planner -> Writer;                               // sequential
flow: Planner -> Writer, Writer -> Reviewer;           // multi-step
flow: Planner -> Writer [condition: drafted];           // conditional
```

### Example

```apg
agent Planner {
    role: "planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Break the ticket into concrete resolution steps.";
    tools: [tickets.read, docs.search];
}

agent Writer {
    role: "writer";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Write concise customer-facing replies from a plan.";
    tools: [tickets.update];
}

agent_team SupportCrew {
    agents: [Planner, Writer];
    flow: Planner -> Writer;
    capabilities: [support_response];
    configuration: {handoff_mode: sequential};
    rules: [{name: "low_confidence", when: "confidence < 0.6", action: require_review}];
}
```

---

## 11. Applications — Composition Root

An `app` declares the composed application that will actually run. It
assembles capabilities, agents, agent teams, routes, screens, theme, and
runtime configuration.

```apg
app <Name> {
    description: "<text>";
    capabilities: [<Capability>, ...];
    agents: [<Agent>, ...];
    agent_teams: [<AgentTeam>, ...];
    routes: ["<path>", ...];
    components: {
        <slot_name>: {capability: <service>, route: "<path>"}
    };
    screens: {
        <ScreenName>: {route: "<path>", capability: <Capability>}
    };
    theme: { name: ..., tokens: { ... } };
    runtime: {
        target: python,
        deployment: container,
        streaming: {processor: bytewax}
    };
    deployments: {default: local, container: docker, cloud: kubernetes};
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `description` | No | Human-readable app description |
| `capabilities` | Yes | List of `Capability` entity names to include |
| `agents` | No | AI agents registered with this app |
| `agent_teams` | No | Agent team entities registered with this app |
| `routes` | No | Top-level URL path prefixes |
| `components` | No | Named component slots wiring capabilities to routes |
| `screens` | No | Top-level screens served by this app |
| `theme` | No | Global design tokens |
| `runtime` | No | Execution environment |
| `deployments` | No | Deployment target configuration |

### Runtime Targets

```apg
runtime: {
    target: python,                    // Python WSGI/ASGI (default)
    deployment: container,             // container | local | serverless
    streaming: {processor: bytewax}    // event streaming backend
};
```

### Deployments

```apg
deployments: {
    default:   local,
    container: docker,
    cloud:     kubernetes
};
```

The deployment map drives which generated `Dockerfile` and cloud config
templates are emitted.

### Example

```apg
app CRMPlatform {
    description: "Enterprise CRM composed from APG capabilities";
    capabilities: [CRMCore];
    agents: [SalesAssistant];
    routes: ["/crm", "/crm/contacts", "/crm/accounts", "/crm/pipeline"];
    components: {
        contact_desk:  {capability: contact_lifecycle, route: "/crm/contacts"},
        deal_pipeline: {capability: opportunity_pipeline, route: "/crm/pipeline"}
    };
    theme: {
        name: crm_platform_theme,
        tokens: {"accent": "#FF6D00", "border.radius": "6px"}
    };
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

---

## 12. ERP Metadata

Capabilities that represent enterprise modules may declare ERP-specific
metadata. This metadata is consumed by the semantic model, capacity planning
tools, and ERP composition views.

```apg
capability Finance {
    contract: { id: finance, provides: [...], ... };

    erp_modules: [finance, general_ledger, accounts_payable, accounts_receivable];

    approvals: {
        levels: 3,
        thresholds: {level1: 10000, level2: 100000},
        approvers: [finance_manager, controller, cfo],
        segregation_of_duties: true
    };

    master_data: {
        entities: [account, cost_center, financial_period, currency],
        ownership: {account: finance, cost_center: operations}
    };

    business_rules: [
        {name: "balanced_journal",  when: "debits != credits",   action: deny},
        {name: "period_open",       when: "period_status != open", action: deny}
    ];
}
```

### ERP Module Domains

Common domain identifiers:

| Domain | Typical capability |
|--------|--------------------|
| `finance` | Financial management |
| `general_ledger` | Chart of accounts, journals |
| `accounts_payable` | Supplier invoices, payments |
| `accounts_receivable` | Customer invoices, collections |
| `procurement` | Purchase orders, supplier quotes |
| `inventory` | Stock control, movements |
| `warehouse` | Locations, bins, picking |
| `manufacturing` | BOMs, work orders, routing |
| `sales` | Sales orders, pricing |
| `crm` | Contacts, accounts, opportunities |
| `hr` | Employee records, org structure |
| `payroll` | Pay runs, statutory deductions |
| `fixed_assets` | Asset register, depreciation |
| `project_accounting` | Project budgets, time-cost |
| `supply_chain` | End-to-end supply planning |
| `service_management` | Service orders, SLA |
| `reporting` | BI, dashboards, analytics |

---

## 13. Internationalisation (i18n)

```apg
i18n: {
    supported_languages: [en, sw, fr, yo, ha, am],
    default_language: en,
    fallback_language: en
};
```

Language codes are ISO-639-1 / IETF BCP-47 identifiers. APG ships built-in
recognition for major African languages:

```
af ak am ar bm bem ber bin din dyu ee ff fon gaa ha ig kab kam ki kln
kg kj kmb kr lg ln loz lu lua mg mos nd nr nso ny om rn rw sg sn so
ss st sw ti tn ts tum tw ve wo xh yo zu
```

The i18n block may appear inside a `capability` body or standalone
within the module for application-wide locale settings.

---

## 14. Streaming

Streaming configuration declares an event-processing topology. APG uses
**Bytewax** as its standard stream processor.

```apg
streaming: {
    processor: bytewax,
    input: order_event_bus,
    output: fulfillment_alerts,
    state: crm_event_state,
    window: 5min
};
```

| Field | Description |
|-------|-------------|
| `processor` | Stream processor — use `bytewax` |
| `input` | Named event source |
| `output` | Named event sink |
| `state` | Stateful aggregation variable |
| `window` | Tumbling window duration |

A `streaming` block may appear inside a `capability` body, inside a
`contract` block, or inside an `app` runtime block.

---

## 15. Imports

APG supports multi-file programs. An `import` statement pulls entity
declarations from another `.apg` file in the same project:

```apg
import lib.common;                       // import all entities from lib/common.apg
from lib.types import Customer, Address; // import specific entities
```

Module names map to file paths: `lib.common` → `lib/common.apg` relative
to the importing file. Path segments must be simple identifiers (no `..`,
`/`, or special characters).

**Current status**: Import resolution is lenient — missing import files are
silently skipped. An imported entity merges into the importing module's
entity list before semantic analysis. Circular imports are detected and
skipped.

---

## 16. The Compiler and CLI

### Install

```bash
pip install apg
```

### Compile

```bash
apg compile source.apg --output generated/app
apg compile source.apg --output generated/app --verify
```

`--verify` runs the semantic analyser and fails if there are any errors or
unresolvable references.

### Inspect

```bash
apg nl-plan source.apg --prompt "add a table for contracts"
apg studio snapshot source.apg --json
apg studio plan-edit source.apg --edit-json '{"operation":"add_table","name":"Invoice"}'
```

### Capabilities

```bash
apg capabilities list
apg capabilities inspect <capability-id>
apg capabilities scaffold <domain> <code> --name "Display Name"
apg capabilities materialize-packages --root capabilities
```

### Lint and Validate

```bash
apg lint source.apg
apg validate source.apg --target python
```

### Authoring Checklist

Before compiling to production:

- [ ] Module has a version (`version 1.0.0`) and description.
- [ ] Every `table` has at least one typed field.
- [ ] Every `capability` has `id` and `provides`.
- [ ] Rules are named and conditions are valid expressions.
- [ ] `app` lists at least one capability.
- [ ] Agents declare `role` and `model`.
- [ ] Workflows declare `steps` with at least two states.
- [ ] Compile with `--verify`.
- [ ] Run `python generated/app/smoke_test.py`.

---

## 17. Generated Artefacts

Compiling an APG file produces a directory of Python files:

| File | Description |
|------|-------------|
| `app.py` | Runnable WSGI/ASGI application with all route handlers |
| `__init__.py` | Package metadata and version |
| `requirements.txt` | Python dependencies |
| `ai_agents.py` | AI agent runtime manifests and invocation helpers |
| `agent_stubs.py` | Typed `AgentBase` subclasses (one per declared agent) |
| `apg_capabilities.py` | Capability metadata and composability contracts |
| `apg_application.py` | Top-level application descriptor |
| `semantic_model.json` | Machine-readable semantic model (v1 contract) |
| `smoke_test.py` | Self-test script verifying basic route health |
| `Dockerfile` | Container image build instructions |
| `.env.example` | Template for required environment variables |
| `README.md` | Generated documentation for the compiled application |

### Running the Generated Application

```bash
# Development server
python generated/app/app.py --host 127.0.0.1 --port 8080

# Self-test
python generated/app/app.py --self-test

# Smoke test
python generated/app/smoke_test.py

# Container
docker build -t my-app generated/app/
docker run -p 8080:8080 my-app
```

### Generated API Routes

Every compiled application includes these standard routes:

```
GET  /health                              # liveness probe
GET  /self-test                           # readiness probe
GET  /manifest                            # APG application manifest
GET  /component.json                      # component composition map
GET  /openapi.json                        # OpenAPI specification
GET  /ui                                  # UI entry point (if shell configured)

GET  /entities                            # list all entity types
GET  /entities/{Entity}                   # describe entity schema
GET  /entities/{Entity}/records           # list records
POST /entities/{Entity}/records           # create a record

GET  /capabilities                        # capability catalog
GET  /capabilities/{Capability}           # capability contract
POST /capabilities/{Capability}/rules/evaluate  # evaluate rules

GET  /agents                              # list agents
GET  /agents/{Agent}                      # describe agent
POST /agents/{Agent}/invoke               # invoke agent

GET  /workflows                           # list workflow definitions
GET  /workflows/{Workflow}                # describe workflow (states, guards)
POST /workflows/{Workflow}/run            # start a workflow run
GET  /workflows/runs                      # list all runs
GET  /workflows/runs/{id}                 # get run status and trace
POST /workflows/runs/{id}/resume          # resume a paused run
POST /workflows/runs/{id}/compensate      # execute compensations
```

---

## 18. Complete Example

The following APG program defines a composable CRM platform with three
data tables, one capability, an AI sales assistant, two approval workflows,
and the composition root that assembles everything.

```apg
// ============================================================================
// CRM Platform
//
// Composability pattern: Hub-and-Spoke
//   auth, audl, ntfy, wflo underpin CRMCore
//   SalesAssistant AI agent augments the pipeline
// ============================================================================

module crm_platform version 1.0.0 {
    description: "Composable CRM platform";
    author: "Datacraft";
}

// ── Data Model ────────────────────────────────────────────────────────────────

table Contact {
    contact_number: str;
    first_name: str;
    last_name: str;
    email: str;
    phone: str?;
    company: str;
    status: str = "active";
    owner_id: str;
}

table Account {
    account_number: str;
    legal_name: str;
    industry: str;
    tier: str;
    health_score: float = 0.0;
    owner_id: str;
}

table Opportunity {
    opportunity_number: str;
    account_id: str;
    name: str;
    stage: str = "prospecting";
    amount: decimal;
    probability: float = 0.0;
    expected_close: date;
    owner_id: str;
}

// ── Core Capability ───────────────────────────────────────────────────────────

capability CRMCore {
    contract: {
        id: crm_platform_core,
        provides: [contact_lifecycle, account_management, opportunity_pipeline, sales_analytics],
        requires: [auth, audl, ntfy, wflo],
        configuration: {
            tenant_id: "default",
            pipeline_stages: ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"]
        },
        rules: [
            {name: "large_deal_requires_approval", when: "amount > 50000",               action: require_review},
            {name: "discount_cap",                  when: "discount_pct > 25",            action: deny},
            {name: "cross_tenant_denied",           when: "contact_tenant != actor_tenant", action: deny}
        ],
        rule_engine: {type: deterministic, default_decision: allow},
        ui: {
            shell: python,
            routes: [
                {name: "Dashboard",  path: "/crm",          component: "CRMDashboard",  permission: "crm:view"},
                {name: "Contacts",   path: "/crm/contacts", component: "ContactList",   permission: "crm:contacts"},
                {name: "Pipeline",   path: "/crm/pipeline", component: "PipelineView",  permission: "crm:pipeline"},
                {name: "Analytics",  path: "/crm/analytics",component: "SalesAnalytics",permission: "crm:analytics"}
            ]
        },
        theme: {
            name: crm_theme,
            tokens: {"color.primary": "#1565C0", "color.accent": "#FF6D00", "border.radius": "6px"}
        }
    };
    streaming: {processor: bytewax, state: crm_event_state};
}

// ── AI Agent ──────────────────────────────────────────────────────────────────

agent SalesAssistant {
    role: "sales assistant";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Analyse CRM context. Suggest next best actions, talking points, and risk factors.";
    capabilities: [contact_lifecycle, opportunity_pipeline];
    tools: [contact_search, deal_analysis];
    memory: vector sales_memory;
    configuration: {temperature: 0.2, max_turns: 6};
}

// ── Workflows ─────────────────────────────────────────────────────────────────

workflow LeadQualification {
    steps: str = "new_lead -> researched -> contacted -> qualified -> opportunity_created";
    human_tasks: [contacted, qualified];
    assignments: {contacted: sales_rep, qualified: sales_manager};
    guards: {qualified: "budget_confirmed and timeline_defined"};
}

workflow DealApproval {
    steps: str = "submitted -> manager_review -> finance_review -> approved";
    human_tasks: [manager_review, finance_review];
    assignments: {manager_review: sales_manager, finance_review: finance_controller};
    guards: {finance_review: "amount > 100000"};
    timers: {finance_review: "PT48H"};
}

// ── Application ───────────────────────────────────────────────────────────────

app CRMPlatform {
    description: "Enterprise CRM composed from APG capabilities";
    capabilities: [CRMCore];
    agents: [SalesAssistant];
    routes: ["/crm", "/crm/contacts", "/crm/accounts", "/crm/pipeline"];
    theme: {name: crm_platform_theme, tokens: {"accent": "#FF6D00", "border.radius": "6px"}};
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

**Compile and run:**

```bash
apg compile crm_platform.apg --output generated/crm --verify
python generated/crm/smoke_test.py
python generated/crm/app.py --host 127.0.0.1 --port 8080
```

---

## 19. Language Grammar Summary

### Entity Types

| Keyword | Description |
|---------|-------------|
| `table` | Data model (generates schema + REST + form) |
| `capability` | Composable capability contract |
| `agent` | AI agent with model, runtime, tools |
| `agent_team` / `team` | Multi-agent composition with handoffs |
| `workflow` | Named state machine |
| `app` / `application` | Composition root — the runnable application |
| `screen` | Standalone screen declaration |
| `form` | Data entry form |
| `db` / `database` | Explicit database declaration |
| `enum` | Enumeration type |
| `rule` / `rule_set` | Standalone rule declaration |
| Any identifier | User-defined entity kind (extensible via `@decorator`) |

### Built-in Types

`str`, `int`, `float`, `decimal`, `bool`, `bytes`, `date`, `time`, `datetime`,
`Any`, `None`, `vector`, `List[T]`, `Dict[K,V]`, `Optional[T]`, `T?`, `T | U`

### Operators

```
// Comments:   //  #  /* ... */
// Assignment: :  =
// Union:      |
// Optional:   ?
// Arrow:      ->   (transitions, handoffs, model chains)
// Fallback:   ??   (null-coalescing model chain)
// Combine:    +    (modality combination)
// Env var:    $VAR_NAME  or  env("VAR_NAME")
```

### Rule Condition Operators

`==` `!=` `<>` `>` `>=` `<` `<=` `=` (alias: `==`) · `and` · `or` · `not`  
`in [...]` · `not in [...]` · `missing` · `not missing`

### Scope of `//`

`//` is a line-comment delimiter everywhere except inside a quoted string literal.
`"http://example.com"` preserves the `//` — it is not stripped.

---

*Copyright © 2025 Datacraft. Author: Nyimbi Odero.*
