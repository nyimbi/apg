# APG Cheat Sheet

## Compile

```bash
apg compile source.apg --output generated/app --verify
python generated/app/smoke_test.py
python generated/app/app.py --self-test
python generated/app/app.py --host 127.0.0.1 --port 8080
```

Use target `python`. Avoid old framework target names.

## File Skeleton

```apg
module module_name version 1.0.0 {
    description: "Short description";
}

table RecordName {
    field: str;
}

capability CapabilityName {
    contract: {
        id: capability_id,
        provides: [service_name]
    };
}

app ApplicationName {
    capabilities: [CapabilityName];
    routes: ["/"];
}
```

## Comments

```apg
// line
# line
/* block */
```

## Fields

```apg
name: str;
count: int;
amount: decimal;
active: bool = true;
created_at: datetime;
external_id: str?;
status: str | None;
tags: List[str];
metadata: Dict[str, str];
```

## Values

```apg
text: "quoted";
flag: true;
number: 42;
duration: 5min;
url: https://example.com/api;
regex: /INV-[0-9]+/;
secret: env("APG_SECRET");
chain: gpt4 -> claude3 -> llama;
modalities: speech + vision + text;
object: {key: value, other: "text"};
list: [a, b, c];
```

## Table

```apg
table Customer {
    customer_number: str;
    legal_name: str;
    email: str;
    active: bool = true;
}
```

## Capability

```apg
capability InventoryControl {
    contract: {
        id: inventory_control,
        provides: [stock_balances, reservation_control],
        requires: [audit_events],
        configuration: {default_warehouse: "NBO-01"},
        rules: [
            {name: "no_negative_stock", when: "on_hand - reserved < 0", action: "deny"}
        ],
        ui: {shell: python, routes: [{name: "Inventory", path: "/inventory", component: "InventoryWorkbench"}]},
        theme: {name: inventory_theme, tokens: {accent: "#2F855A"}}
    };
}
```

## Rule Engine

```apg
rule_engine: {
    type: deterministic;
    default_decision: allow;
    rules: [
        {name: "missing_tenant", when: "tenant_id missing", action: "deny", priority: 100},
        {name: "approval_threshold", when: "amount > approval_threshold", action: "require_review"}
    ];
};
```

Rule actions: `allow`, `deny`, `require_review`, `warn`, `audit`, or custom
identifier.

## Screen

```apg
screens: {
    Dashboard: {
        route: "/ops",
        title: "Operations",
        layout: dashboard,
        contains: [KpiStrip, ApprovalQueue],
        composes: [LedgerTable],
        binds: [ledger.entries],
        actions: [approve, reject],
        events: [{on: "select", do: "filter", target: LedgerTable}],
        relationships: [
            KpiStrip -> LedgerTable,
            {from: ApprovalQueue, to: LedgerTable, via: selection}
        ]
    }
};
```

Layouts: `stack`, `grid`, `tabs`, `split`, `wizard`, `dashboard`, `form`, or a
custom identifier/string.

## Application

```apg
app EnterpriseERPPlatform {
    description: "Composable ERP shell";
    capabilities: [PlatformAudit, EnterpriseFinance];
    agents: [Planner];
    agent_teams: [FinanceCrew];
    routes: ["/erp/finance"];
    components: {
        finance_workbench: {capability: journal_entries, route: "/erp/finance"}
    };
    screens: {
        Home: {route: "/", capability: EnterpriseFinance}
    };
    theme: {name: enterprise_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

## Agent

```apg
agent SupportPlanner {
    role: "support planner";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    system: "Plan concise support follow-up.";
    capabilities: [customer_followup];
    tools: [case_search, customer_history];
    memory: vector support_memory;
    configuration: {temperature: 0.2};
}
```

Known runtimes: `local`, `codex`, `codex_cli`, `claude`, `claude_code`,
`opencode`, `open_code`, `pi`, `openai`, `ollama`, or custom identifier/string.

## Agent Team

```apg
agent_team SupportCrew {
    agents: [SupportPlanner, QualityReviewer];
    handoffs: SupportPlanner -> QualityReviewer [condition: drafted];
    capabilities: [support_response];
}
```

## Workflow

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

Generated helpers:

```python
list_workflows()
describe_workflow("ProcurementApproval")
run_workflow("ProcurementApproval", {"amount": 100})
list_workflow_runs()
resume_workflow("workflow-run-1", {"events": ["approval_received"]})
execute_workflow_compensations("workflow-run-1")
```

## ERP Metadata

```apg
erp_modules: [finance, general_ledger, accounts_payable];
approvals: {levels: 2, approvers: [controller, cfo]};
master_data: {entities: [account, cost_center, financial_period]};
business_rules: [
    {name: "balanced_journal", when: "debits != credits", action: "deny"}
];
```

Common domains: `finance`, `general_ledger`, `accounts_payable`,
`accounts_receivable`, `procurement`, `inventory`, `warehouse`, `sales`, `crm`,
`manufacturing`, `hr`, `payroll`, `fixed_assets`, `project_accounting`,
`supply_chain`, `service_management`, `reporting`.

## I18n

```apg
i18n: {
    supported_languages: [en, sw, ha, yo, zu, am, rw],
    default_language: en,
    fallback_language: en
};
```

Built-in African language codes include:

```text
af ak am ar bm bem ber bin din dyu ee ff fon gaa ha ig kab kam ki kln kg kj
kmb kr lg ln loz lu lua mg mos nd nr nso ny om rn rw sg sn so ss st sw ti tn
ts tum tw ve wo xh yo zu
```

## Streaming

```apg
streaming: {
    processor: bytewax,
    input: order_events,
    output: fulfillment_alerts,
    state: operations_event_state,
    window: 5min
};
```

Prefer `bytewax`.

## Generated App Routes

Common routes:

```text
GET  /health
GET  /self-test
GET  /manifest
GET  /component.json
GET  /openapi.json
GET  /ui
GET  /entities
GET  /entities/{Entity}
GET  /entities/{Entity}/records
POST /entities/{Entity}/records
GET  /capabilities
GET  /capabilities/{Capability}
POST /capabilities/{Capability}/rules/evaluate
GET  /workflows
POST /workflows/{Workflow}/run
GET  /workflows/runs
POST /workflows/runs/{id}/resume
POST /workflows/runs/{id}/compensate
```

## Authoring Checklist

- Module has a version and description.
- Tables have typed fields.
- Capabilities have `id`, `provides`, config, rules, UI, and theme.
- Rules are named.
- Screens have routes and explicit relationships.
- Agents declare model and role/system.
- Workflows declare steps or stages.
- Streaming uses Bytewax.
- I18n has default and fallback language.
- Compile with `--verify`.
- Run generated `smoke_test.py`.
