# Standards For Building APG Capabilities

Capabilities are APG's unit of application composition. A capability should be
small enough to reason about, strong enough to execute independently, and clear
enough for another application to compose without reading its internals.

These standards apply to first-class APG `capability` declarations and to
Python capability packages under `capabilities/`.

## Capability Contract

Every executable capability must have a contract. Minimum APG source shape:

```apg
capability InventoryControl {
    contract: {
        id: inventory_control,
        provides: [stock_balances, reservation_control],
        requires: [audit_events],
        configuration: {default_warehouse: "NBO-01"},
        configuration_schema: {tenant_id: str, default_warehouse: str},
        rule_engine: {
            type: deterministic,
            default_decision: allow,
            rules: [
                {name: "no_negative_stock", when: "on_hand - reserved < 0", action: "deny"}
            ]
        },
        ui: {shell: python, routes: [{name: "Inventory", path: "/inventory", component: "InventoryWorkbench"}]},
        theme: {name: inventory_theme, tokens: {accent: "#2F855A"}}
    };
}
```

Required surfaces:

| Surface | Standard |
| --- | --- |
| `id` | Stable snake_case identifier. Do not rename casually. |
| `provides` | Capability services this unit owns. Use concrete business names. |
| `requires` | External services, capabilities, or events needed at runtime. |
| `configuration` | Runtime defaults. Must be tenant-safe. |
| `configuration_schema` | Accepted override shape where practical. |
| `rule_engine` or `rules` | Deterministic governance rules with names. |
| `ui` | UI routes and shell metadata, even if minimal. |
| `theme` | Named visual theme and tokens. |

Generated apps validate contract shape and expose capability runtime helpers.
Platform packages should also expose `capability_contract.py` for registry
discovery.

## Naming

Use three layers of names consistently:

- Entity name: PascalCase, for example `InventoryControl`.
- Contract id: snake_case, for example `inventory_control`.
- Provided services: snake_case business terms, for example `stock_balances`.

Avoid provider names in capability ids unless the capability is explicitly an
adapter. Prefer `payment_processing` over `stripe_payment_processing`; use
`stripe_adapter` as a required dependency when Stripe is the implementation.

## Boundary Standard

A capability owns a coherent business or platform boundary. It should have:

- one clear reason to exist
- explicit inputs and outputs
- named provided services
- named required dependencies
- deterministic rules for governance
- tenant-aware configuration
- observable runtime behavior
- UI metadata when humans operate or inspect it

Split a capability when the rule set, data ownership, or runtime dependency set
diverges. Compose capabilities at the `app` layer rather than making one large
capability that quietly owns unrelated domains.

## Configuration Standard

Configuration should be explicit, tenant-safe, and override-friendly.

```apg
configuration: {
    tenant_scoped: true,
    default_warehouse: "NBO-01",
    reorder_threshold: 25,
    approval_threshold: 5000
};
configuration_schema: {
    tenant_id: str,
    default_warehouse: str,
    reorder_threshold: int,
    approval_threshold: decimal
};
```

Rules:

- Include `tenant_id` in schemas for tenant-sensitive capabilities.
- Do not put secrets in APG source. Reference environment variables or secret
  capability dependencies.
- Defaults should be safe in a generated local app.
- Configuration values used by rules should have obvious names.

## Rule Standard

Capability rules must be named and deterministic unless the contract explicitly
declares another engine.

```apg
rules: [
    {name: "missing_tenant", when: "tenant_id missing", action: "deny", priority: 100},
    {name: "large_payment_review", when: "amount > approval_threshold", action: "require_review", priority: 20},
    {name: "audit_sensitive_change", when: "operation == update", action: "audit", priority: 10}
];
```

Rule standards:

- Give every rule a stable `name`.
- Prefer `when` plus `action` for simple rules.
- Use `decision` when the rule returns a policy decision.
- Use `priority` when multiple rules may match.
- Keep conditions over input fields and configuration values.
- Use `missing` and `present` for field presence.
- Move external lookups into required capabilities.
- Do not hide complex business processes inside one string expression.

Decision vocabulary:

- `allow`
- `deny`
- `require_review`
- `warn`
- `audit`

Custom decisions are allowed, but define them in docs and generated UI copy.

## UI And Screen Standard

Every capability with human interaction should declare UI metadata. Use `ui`
for route-level metadata and `screens` for composition.

```apg
ui: {
    shell: python,
    routes: [
        {name: "Inventory", path: "/inventory", component: "InventoryWorkbench", permission: "inventory.view"}
    ]
};

screens: {
    InventoryDashboard: {
        route: "/inventory",
        title: "Inventory",
        layout: dashboard,
        contains: [StockKpis, ReservationQueue],
        composes: [ReorderTable],
        binds: [stock_balances.current],
        actions: [reserve, release, reorder],
        events: [{on: "select", do: "filter", target: ReorderTable}],
        relationships: [
            StockKpis -> ReservationQueue,
            ReservationQueue -> ReorderTable
        ]
    }
};
```

Screen standards:

- Every screen has a `route`.
- Use `contains` for owned visible elements.
- Use `composes` for elements assembled from another component or capability.
- Use `binds` for data dependencies.
- Use `actions` for user commands.
- Use `events` for interaction behavior.
- Use `relationships` to make dependencies between visible elements explicit.

## Theme Standard

Declare themes as tokens, not hard-coded UI details.

```apg
theme: {
    name: inventory_theme,
    tokens: {
        accent: "#2F855A",
        surface: "#F7FAFC",
        danger: "#B42318"
    },
    allow_tenant_overrides: true
};
```

Minimum theme tokens:

- `accent`
- `surface`
- `danger` when destructive actions exist
- `warning` when review states exist
- `success` when approval or completion states exist

Do not encode page layout into theme tokens. Layout belongs in `screens`.

## I18n Standard

Capabilities that surface user-facing labels or decisions should declare i18n:

```apg
i18n: {
    supported_languages: [en, sw, ha, yo, zu, am, rw],
    default_language: en,
    fallback_language: en
};
```

Standards:

- Always include a `default_language`.
- Always include a `fallback_language`.
- Prefer bare ISO-style language identifiers accepted by the grammar.
- Use strings only when a code is not built into the grammar yet.

## Streaming Standard

Use Bytewax for streaming declarations:

```apg
streaming: {
    processor: bytewax,
    input: inventory_events,
    output: inventory_alerts,
    state: inventory_stream_state,
    window: 5min
};
```

Standards:

- Use `processor: bytewax` unless there is a concrete adapter reason.
- Name state stores explicitly.
- Keep stream input and output names business-readable.
- Treat streaming as a capability dependency when another capability consumes
  the stream.

## ERP Standard

ERP capabilities should declare module membership, master data, approvals, and
business rules.

```apg
erp_modules: [inventory, warehouse, manufacturing];
master_data: {entities: [item, warehouse, lot, bin]};
approvals: {levels: 2, approvers: [warehouse_manager, controller]};
business_rules: [
    {name: "lot_required_for_controlled_item", when: "lot_controlled == true and lot_id missing", action: "deny"}
];
```

Keep ERP module declarations descriptive. Do not use ERP labels as a substitute
for actual capability services in `provides`.

## AI Agent Capability Standard

If a capability depends on AI agent behavior, declare the agent separately and
link by capability service names.

```apg
agent InventoryPlanner {
    role: "inventory planning analyst";
    model: "openai:gpt-4.1-mini";
    runtime: codex;
    capabilities: [reorder_recommendation];
    tools: [stock_history, demand_forecast];
    configuration: {temperature: 0.1};
}

capability InventoryControl {
    contract: {
        id: inventory_control,
        provides: [stock_balances, reorder_recommendation],
        requires: [audit_events]
    };
}
```

Standards:

- Agents must declare a `model`.
- Agents should declare `role` or `system`.
- Runtime names should remain adapter-oriented.
- Do not put provider SDK details in APG source.
- Keep deterministic rules outside the model where possible.

## Python Package Standard

Use the CLI scaffold for a new package-backed capability:

```bash
./.venv/bin/apg capabilities scaffold common demo --name "Demo Capacity" --json
```

The scaffold produces a valid spec-backed contract plus an executable
dependency-light package runtime: tenant-scoped records, rule-guarded
create/update service methods, API helpers, dashboard view metadata, and
focused contract/runtime tests. It also writes `app.py`,
`semantic_model.json`, `package_manifest.json`, and `release_report.json` so
the package can pass `apg capabilities publish-plan <package-dir> --json` and
then `apg capabilities publish-apply <package-dir> --catalog <catalog.json>
--json` when you intentionally want to update a local capability catalog.
Refine that starter behavior into domain-specific operations instead of
replacing it with inert placeholders.

For a Python capability package under `capabilities/<domain>/<code>/`, use this
shape where practical:

```text
capabilities/<domain>/<code>/
  __init__.py
  cap_spec.md
  capability_contract.py
  models.py
  service.py
  api.py
  views.py
  app.py
  semantic_model.json
  package_manifest.json
  release_report.json
  tests/
```

`capability_contract.py` should expose:

```python
def get_capability_contract(tenant_id: str = "default", overrides: dict | None = None) -> dict:
    ...

def evaluate_capability_rules(context: dict) -> dict:
    ...
```

Use existing helpers in `capabilities/capability_contract_factory.py` when the
capability has a `cap_spec.md`.

## Testing Standard

Use focused tests for the changed slice:

- contract registry tests when changing `capability_contract.py`
- compiler and generated app tests when changing APG source generation
- generated smoke tests when changing examples
- service tests when changing runtime service behavior

Battery-conscious minimum for capability source changes:

```bash
python -m py_compile capabilities/<domain>/<code>/*.py
python -m pytest -q capabilities/<domain>/<code>/tests
```

Battery-conscious minimum for APG compiler-facing changes:

```bash
apg compile path/to/source.apg --output /tmp/apg-out --verify
python /tmp/apg-out/smoke_test.py
```

## Documentation Standard

Each capability should document:

- purpose and boundary
- provided services
- required dependencies
- configuration keys
- rule decisions
- UI routes and screens
- theme tokens
- i18n coverage
- streaming contracts
- persistence and state behavior
- verification commands

Keep documentation near the capability and link durable platform-level concepts
from `docs/`.

## Acceptance Checklist

Before a capability is considered composable:

- Contract has `id`, `provides`, `configuration`, `rules` or `rule_engine`,
  `ui`, and `theme`.
- Required dependencies are explicit.
- Rules are named and deterministic.
- Tenant-sensitive configuration includes tenant context.
- UI routes and screen relationships are declared when humans use it.
- Theme tokens exist.
- i18n is declared for user-facing capabilities.
- Streaming uses Bytewax when stream processing is needed.
- Generated app or package runtime has focused tests.
- Documentation states what the capability owns and what it depends on.
