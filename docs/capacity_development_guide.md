# APG Capacity Development Guide

This guide explains how to build new APG capacities: coherent platform or
business abilities that combine APG source, capability packages, configuration,
rules, screens, workflows, AI agents, streaming metadata, generated Python
applications, tests, docs, and release evidence.

Use this guide when the goal is "make APG able to do something new", not merely
"add a file".

## Capacity And Capability

A **capability** is a composable unit with a contract: services provided,
services required, configuration, rules, UI routes, theme, tenant behavior, and
tests.

A **capacity** is an executable ability built from one or more capabilities plus
the language/runtime surfaces needed to demonstrate it.

Example:

```text
Capacity: Procurement approval automation
  Records: PurchaseRequest, Supplier, BudgetCheck
  Capability: ProcurementApproval
  Capability: SupplierMaster
  Capability: PlatformAudit
  Rules: supplier_required, amount_positive, large_request_review
  Screen: ProcurementApprovalWorkbench
  Workflow: draft -> review -> approved -> ordered
  Agent: ProcurementPlanner
  Streaming: Bytewax procurement_events -> procurement_alerts
  Evidence: compile --verify, generated smoke test, capability contract validation
```

Build capacities as vertical slices. Each slice should be executable before the
next slice expands scope.

## Capacity Lifecycle

1. Define the outcome.
2. Choose capability boundaries.
3. Model records.
4. Define capability contracts.
5. Add deterministic rules.
6. Add screens and element relationships.
7. Add workflows for stateful processes.
8. Add AI agents for model-backed work.
9. Add Bytewax streaming metadata where event flow matters.
10. Compose the generated application shell.
11. Compile and verify.
12. Add focused tests.
13. Document the capacity and update the progress log.
14. Commit and push the verified slice.

## 1. Define The Outcome

Write one concrete sentence:

```text
This capacity lets a generated APG app receive purchase requests, validate
supplier and amount rules, route approvals, record audit events, and expose a
procurement workbench screen.
```

Then list:

- primary users;
- records and ownership;
- services provided;
- services required;
- deterministic rules;
- workflows and states;
- screens and actions;
- AI-assisted work;
- streaming events;
- tenant/security boundaries;
- verification evidence.

If the outcome cannot be observed in generated code, CLI output, tests, or docs,
it is not concrete enough.

## 2. Choose Capability Boundaries

Start with one core capability. Split only when ownership, lifecycle, or
dependency direction differs.

Good boundaries:

- `ProcurementApproval`
- `InventoryControl`
- `GeneralLedger`
- `CustomerCare`
- `AuditLog`

Weak boundaries:

- `EverythingERP`
- `AIPlatform`
- `Manager`
- `Helper`
- `Integration`

Boundary questions:

- What services does this capability provide?
- What services does it require?
- What data does it own?
- What rules does it enforce?
- What UI does it expose?
- What theme tokens does it need?
- What events does it publish or consume?
- What tenant context is mandatory?

## 3. Model Records

Use APG tables for durable records:

```apg
table PurchaseRequest {
    request_number: str;
    requester: str;
    supplier_id: int;
    amount: decimal;
    status: str;
}

table Supplier {
    supplier_number: str;
    legal_name: str;
    active: bool;
    risk_rating: str;
}
```

Keep early records small. Add fields when a rule, screen, workflow, API, or
test needs them.

## 4. Define Capability Contracts

For package-backed capabilities, start with the scaffold:

```bash
./.venv/bin/apg capabilities scaffold procurement approvals --name "Procurement Approvals" --json
```

Move or create the package under `capabilities/<domain>/<code>/`, then validate:

```bash
./.venv/bin/python -m pytest -q capabilities/procurement/approvals/tests
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities inspect procurement_approvals --json
```

A minimum APG source contract shape:

```apg
capability ProcurementApproval {
    contract: {
        id: procurement_approval,
        provides: [purchase_request_review, approval_routing],
        requires: [audit_events, supplier_master],
        configuration: {
            approval_threshold: 5000,
            currency: "KES",
            tenant_scoped: true
        },
        configuration_schema: {
            tenant_id: str,
            approval_threshold: decimal,
            currency: str
        },
        rules: [
            {name: "supplier_required", when: "supplier_id missing", action: "deny"},
            {name: "large_request_review", when: "amount > approval_threshold", action: "require_review"}
        ],
        ui: {shell: python},
        theme: {name: procurement_theme, tokens: {accent: "#174EA6", warning: "#B7791F"}}
    };
}
```

Contract quality checks:

- `id` is stable snake_case.
- `provides` names business services.
- `requires` names dependencies, not implementation guesses.
- configuration has safe local defaults.
- configuration schema includes tenant context when tenant-sensitive.
- rules are named and deterministic.
- UI and theme contracts exist.

## 5. Add Rules

Rules should be deterministic and testable:

```apg
rules: [
    {name: "supplier_required", when: "supplier_id missing", action: "deny", priority: 100},
    {name: "amount_positive", when: "amount <= 0", action: "deny", priority: 90},
    {name: "large_request_review", when: "amount > approval_threshold", action: "require_review", priority: 50},
    {name: "inactive_supplier_review", when: "supplier_active == false", action: "require_review", priority: 40}
];
```

Use dependency capabilities for external checks:

- `supplier_master`;
- `budget_control`;
- `audit_events`;
- `identity_access`.

Do not put network calls or external API behavior inside rule strings.

## 6. Add Screens And Relationships

Screens make a capacity inspectable and operable:

```apg
screens: {
    ProcurementApprovalWorkbench: {
        route: "/procurement/approvals",
        title: "Procurement Approvals",
        layout: dashboard,
        contains: [RequestKpis, ApprovalQueue],
        composes: [SupplierRiskPanel, BudgetImpactTable],
        binds: [purchase_request.status, supplier.risk_rating],
        actions: [approve, reject, escalate],
        events: [{on: "select", do: "filter", target: SupplierRiskPanel}],
        relationships: [
            ApprovalQueue -> SupplierRiskPanel,
            ApprovalQueue -> BudgetImpactTable
        ]
    }
};
```

A good screen declares route, title, layout, contained elements, composed
elements, bindings, actions, events, and relationships.

## 7. Add Workflows

Use workflows when process state matters:

```apg
workflow ProcurementApprovalFlow {
    steps: str = "draft -> review -> approved -> ordered";
    human_tasks: [review];
    assignments: {review: procurement_manager};
    guards: {review: "amount > 0"};
    waits: {approved: approval_received};
    retry_policy: {review: {attempts: 3}};
    compensation: {ordered: cancel_purchase_order};
}
```

Generated workflow behavior should expose deterministic step chains, guards,
waits, retries, run state, resume, and compensation recording.

## 8. Add AI Agents

Use AI agents for model-backed planning, review, summarization,
classification, or tool orchestration:

```apg
agent ProcurementPlanner {
    role: "procurement approval analyst";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Summarize procurement approval risks and next actions.";
    capabilities: [approval_routing];
    tools: [supplier_master, budget_control, audit_events];
    memory: vector procurement_memory;
    configuration: {temperature: 0.1, max_steps: 4};
    rules: [
        {name: "request_context_required", when: "request_number missing", action: "deny"}
    ];
}
```

Agent standards:

- declare `model`;
- declare `role` or `system`;
- keep provider details in adapters;
- use runtime identifiers such as `codex`, `claude_code`, `opencode`, `pi`,
  `openai`, or `ollama`;
- keep deterministic governance outside the model where possible.

## 9. Add Bytewax Streaming

Use Bytewax when the capacity needs event processing:

```apg
streaming: {
    processor: bytewax,
    input: procurement_events,
    output: procurement_alerts,
    state: procurement_stream_state,
    window: 5min
};
```

Standards:

- use `processor: bytewax`;
- name input, output, and state explicitly;
- treat external brokers as integration capabilities;
- do not make APG internal stream semantics broker-first.

## 10. Compose The Application

The application shell proves the capacity can be assembled:

```apg
app ProcurementOperationsApp {
    description: "Procurement approval operations";
    capabilities: [ProcurementApproval, SupplierMaster, PlatformAudit];
    agents: [ProcurementPlanner];
    routes: ["/procurement/approvals"];
    components: {
        approval_workbench: {capability: approval_routing, route: "/procurement/approvals"}
    };
    screens: {
        Home: {route: "/", capability: ProcurementApproval, component: "ProcurementApprovalWorkbench"}
    };
    workflows: [ProcurementApprovalFlow];
    theme: {name: procurement_operations_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

## 11. Compile And Verify

Use a temporary output directory:

```bash
./.venv/bin/apg compile path/to/capacity.apg --output /tmp/apg-capacity --verify
./.venv/bin/python /tmp/apg-capacity/smoke_test.py
./.venv/bin/python /tmp/apg-capacity/app.py --self-test
```

Inspect generated surfaces:

```bash
./.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "/tmp/apg-capacity")
import app
print(app.list_entities())
print(app.list_capabilities())
print(app.list_workflows())
print(app.validate_application()["valid"])
PY
```

If the capacity includes packaging or release evidence:

```bash
./.venv/bin/apg evidence path/to/capacity.apg --target container --out /tmp/apg-evidence --json
```

## 12. Add Focused Tests

Choose tests by the layer changed:

```bash
./.venv/bin/apg compile path/to/capacity.apg --output /tmp/apg-capacity --verify
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/python -m pytest -q tests/test_capability_contract_registry.py
./.venv/bin/python -m pytest -q tests/test_generated_workflow_runtime.py
./.venv/bin/apg doctor --json
./.venv/bin/apg hygiene audit --json
./.venv/bin/apg docs audit --json
./.venv/bin/apg tooling audit --json
./.venv/bin/apg lint path/to/capacity.apg --catalog /tmp/apg-capability-catalog.json --json
```

Do not default to the full repository suite for every capacity slice. Run the
smallest commands that prove the changed behavior, then broaden verification
when the slice touches shared compiler or generator contracts.

## 13. Document The Capacity

A capacity document should state:

- outcome;
- users;
- records;
- capabilities;
- provided and required services;
- configuration;
- rules;
- workflows;
- screens and relationships;
- agents;
- streaming;
- generated routes and manifests;
- verification commands;
- known gaps.

If the capacity lives under `examples/`, each example directory needs `main.apg`,
`README.md`, and generated `output/`.

## Parallel Delivery

Capacities can be built in parallel when ownership is clear:

| Lane | Owns | Avoid touching |
| --- | --- | --- |
| Language/compiler | grammar, AST, semantic model, generator | capability package internals unless needed for fixtures |
| Capability package | `capabilities/<domain>/<code>/` | shared compiler files |
| Example/app | `examples/<nn>_*/` | unrelated examples |
| Tests | focused tests for the slice | broad suite rewrites |
| Docs | capacity docs, guide updates, progress log | unrelated historical docs |

Parallel lanes must agree on public names: capability IDs, provided services,
routes, workflow names, agent names, and JSON format keys.

## Acceptance Checklist

A capacity is ready to build on when:

- APG source parses.
- `apg compile --verify` passes.
- generated `smoke_test.py` passes.
- generated app exposes expected entities, capabilities, workflows, screens,
  routes, and manifests.
- capability contracts have named provides, requires, configuration, rules, UI,
  and theme.
- rules are deterministic and named.
- screens declare relationships.
- workflows declare executable steps.
- agents declare model and runtime.
- streaming uses Bytewax when present.
- focused tests cover the changed behavior.
- documentation explains how to use and extend the capacity.
- `docs/progress_log.md` records verification evidence.

## Expansion Order

Expand a capacity in this order:

1. Add records.
2. Add capability contract services.
3. Add rules.
4. Add screens.
5. Add workflows.
6. Add agents.
7. Add streaming.
8. Add app composition.
9. Add package and release evidence.
10. Add docs and tests.

This order keeps each step inspectable and executable.

## Anti-Patterns

Avoid:

- one giant capability for an entire ERP domain;
- rules without names;
- screens without routes;
- agents without models;
- hidden external services;
- grammar changes without semantic-model follow-through;
- generated behavior not visible in OpenAPI, component manifests, or tests;
- docs that describe unimplemented behavior as current.
