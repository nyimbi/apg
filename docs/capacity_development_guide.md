# APG Capacity Development Guide

This guide explains how to build new APG capacities: coherent units of platform
or business ability that may contain one or more APG capabilities, generated
application surfaces, rules, screens, workflows, agents, tests, docs, and
release evidence.

Use this guide when the goal is not merely "add a file" but "make APG able to
do something new."

## Capacity Versus Capability

In this repository:

- A **capability** is a composable APG unit with a contract, provided services,
  dependencies, configuration, rules, UI, theme, and optional i18n/streaming.
- A **capacity** is a larger executable ability APG can demonstrate and evolve.
  A capacity may require several capabilities, language support, generated app
  behavior, CLI tooling, examples, and documentation.

Example:

```text
Capacity: Procurement approval automation
  Capability: Purchase request intake
  Capability: Approval policy engine
  Capability: Supplier audit trail
  Workflow: draft -> review -> approved -> ordered
  Screen: ProcurementApprovalWorkbench
  Agent: ProcurementPlanner
  Example: examples/13_procurement_approval_workbench
  Evidence: compile --verify, generated smoke test, release evidence
```

Build capacities as vertical slices. Each slice should become executable before
the next slice expands scope.

## Capacity Development Lifecycle

1. Define the capacity outcome.
2. Identify the minimum capability set.
3. Model data records.
4. Define capability contracts.
5. Add deterministic rules.
6. Add screens and UI relationships.
7. Add workflows where process state matters.
8. Add AI agents where model-backed work is valuable.
9. Add streaming with Bytewax where event flow matters.
10. Compose the application shell.
11. Compile and verify generated output.
12. Add package/release evidence when ready.
13. Document and log progress.

## Step 1: Define The Outcome

Write one concrete sentence:

```text
This capacity lets a generated APG app receive purchase requests, validate
budget and supplier rules, route approvals, record audit events, and expose a
procurement workbench screen.
```

Then define:

- primary users
- business records
- decisions and rules
- workflows
- screens
- required integrations
- tenant and security boundaries
- success evidence

If the outcome cannot be observed in generated code, CLI output, tests, or docs,
it is not yet concrete enough.

## Step 2: Choose The Capability Boundary

Start with one core capability. Add more only when ownership differs.

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

Ask:

- What services does it provide?
- What services does it require?
- What data does it own?
- What rules does it enforce?
- What UI does it expose?
- What theme tokens does it need?
- What events does it publish or consume?

## Step 3: Model Records

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

Keep early tables small. Add fields when a rule, screen, workflow, or API needs
them.

## Step 4: Define Capability Contracts

For a package-backed capability, start with the scaffold command:

```bash
./.venv/bin/apg capabilities scaffold common demo --name "Demo Capacity" --json
```

That creates a spec-backed package under `capabilities/common/demo/` with
`cap_spec.md`, `capability_contract.py`, dependency-light `models.py`,
`service.py`, `api.py`, `views.py`, and contract tests. The generated contract
is valid against APG's registry shape before you add domain-specific behavior.

Minimum contract:

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
- `configuration` has safe local defaults.
- `configuration_schema` includes tenant context for tenant-sensitive behavior.
- `rules` are named.
- `ui` and `theme` exist.

## Step 5: Add Rules

Prefer deterministic rule strings over prose.

```apg
rules: [
    {name: "supplier_required", when: "supplier_id missing", action: "deny", priority: 100},
    {name: "amount_positive", when: "amount <= 0", action: "deny", priority: 90},
    {name: "large_request_review", when: "amount > approval_threshold", action: "require_review", priority: 50},
    {name: "inactive_supplier_review", when: "supplier_active == false", action: "require_review", priority: 40}
];
```

Move complex external checks behind dependencies:

- `supplier_master`
- `budget_control`
- `audit_events`
- `identity_access`

Do not put external API calls inside rule strings.

## Step 6: Add Screens

Screens make the capacity inspectable and operable.

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

A good screen declares:

- route
- title
- layout
- contained elements
- composed elements
- data bindings
- actions
- events
- relationships

## Step 7: Add Workflows

Use workflows for stateful business processes.

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

Generated workflow behavior supports deterministic step chains, guards, waits,
retries, durable run state, resume, and compensation recording.

## Step 8: Add AI Agents

Use AI agents for work that benefits from model-backed planning, review,
summarization, classification, or code/tool orchestration.

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

- Declare `model`.
- Declare `role` or `system`.
- Keep provider details in adapters.
- Use known runtimes such as `codex`, `claude_code`, `opencode`, `pi`,
  `openai`, `ollama`, or a custom adapter identifier.
- Keep deterministic governance outside the model when possible.

## Step 9: Add Bytewax Streaming

Use streaming when the capacity needs event processing:

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

- Use `processor: bytewax`.
- Name input, output, and state explicitly.
- Treat external brokers as integration capabilities.
- Do not switch APG internal stream semantics to a broker-first runtime.

## Step 10: Compose The App

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

## Step 11: Compile And Verify

Use a temp output directory:

```bash
./.venv/bin/apg compile path/to/capacity.apg --output /tmp/apg-capacity --verify
python /tmp/apg-capacity/smoke_test.py
python /tmp/apg-capacity/app.py --self-test
```

Inspect generated surfaces:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "/tmp/apg-capacity")
import app
print(app.list_entities())
print(app.list_capabilities())
print(app.list_workflows())
print(app.validate_application()["valid"])
PY
```

If the capacity includes packaging/release work:

```bash
./.venv/bin/apg evidence path/to/capacity.apg --target container --out /tmp/apg-evidence --json
```

## Step 12: Add Tests

Choose tests by what changed.

APG source and generated app:

```bash
./.venv/bin/apg compile path/to/capacity.apg --output /tmp/apg-capacity --verify
```

Capability contracts:

```bash
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/python -m pytest -q tests/test_capability_contract_registry.py
```

Generated capability runtime:

```bash
./.venv/bin/python -m pytest -q tests/test_capability_composition_runtime.py
```

Workflow runtime:

```bash
./.venv/bin/python -m pytest -q tests/test_generated_workflow_runtime.py
```

Tooling:

```bash
./.venv/bin/apg tooling audit --json
```

## Step 13: Document The Capacity

A capacity doc should state:

- outcome
- users
- records
- capabilities
- provided services
- required services
- rules
- workflows
- screens
- agents
- streaming
- generated routes
- verification commands
- known gaps

If the capacity lives under `examples/`, each example directory needs a
`README.md` and generated `output/`.

## Example Capacity Skeleton

```apg
module procurement_capacity version 1.0.0 {
    description: "Procurement approval capacity";
}

table PurchaseRequest {
    request_number: str;
    requester: str;
    supplier_id: int;
    amount: decimal;
    status: str;
}

capability ProcurementApproval {
    contract: {
        id: procurement_approval,
        provides: [purchase_request_review, approval_routing],
        requires: [audit_events, supplier_master],
        configuration: {approval_threshold: 5000, tenant_scoped: true},
        rules: [
            {name: "supplier_required", when: "supplier_id missing", action: "deny"},
            {name: "large_request_review", when: "amount > approval_threshold", action: "require_review"}
        ],
        ui: {shell: python},
        theme: {name: procurement_theme, tokens: {accent: "#174EA6"}}
    };
    screens: {
        ProcurementApprovalWorkbench: {
            route: "/procurement/approvals",
            title: "Procurement Approvals",
            layout: dashboard,
            contains: [RequestKpis, ApprovalQueue],
            binds: [purchase_request.status],
            actions: [approve, reject, escalate],
            relationships: [RequestKpis -> ApprovalQueue]
        }
    };
    streaming: {processor: bytewax, state: procurement_stream_state};
}

workflow ProcurementApprovalFlow {
    steps: str = "draft -> review -> approved";
    human_tasks: [review];
    assignments: {review: procurement_manager};
    guards: {review: "amount > 0"};
}

agent ProcurementPlanner {
    role: "procurement analyst";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Summarize procurement approval risks.";
    capabilities: [approval_routing];
}

app ProcurementOperationsApp {
    capabilities: [ProcurementApproval];
    agents: [ProcurementPlanner];
    workflows: [ProcurementApprovalFlow];
    routes: ["/procurement/approvals"];
    runtime: {target: python, streaming: {processor: bytewax}};
}
```

## Capacity Acceptance Checklist

A capacity is ready to build on when:

- APG source parses.
- `apg compile --verify` passes.
- Generated `smoke_test.py` passes.
- Generated app exposes expected entities, capabilities, workflows, screens,
  routes, and manifests.
- Capability contracts have named `provides`, `requires`, rules, UI, and theme.
- Rules are deterministic and named.
- Screens declare relationships.
- Workflows declare executable steps.
- Agents declare model and runtime.
- Streaming uses Bytewax when present.
- Tests cover the changed behavior.
- Documentation explains how to use and extend the capacity.
- `docs/progress_log.md` records verification evidence.

## Capacity Expansion Patterns

Expand a capacity in this order:

1. Add records.
2. Add capability contract services.
3. Add rules.
4. Add screens.
5. Add workflows.
6. Add agents.
7. Add streaming.
8. Add app composition.
9. Add package/release evidence.
10. Add docs and tests.

This order keeps each step inspectable and executable.

## Capacity Anti-Patterns

Avoid:

- one giant capability for an entire ERP domain
- rules without names
- screens without routes
- agents without models
- hidden external services
- grammar changes without codegen or semantic model follow-through
- generated app behavior that is not visible in OpenAPI or component manifests
- docs that describe unimplemented behavior as current
