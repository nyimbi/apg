# Screen Composition

APG screens are first-class composition contracts for user-facing application
surfaces. Use them when a capability needs to describe the screens it provides,
the elements those screens contain, the data and actions they bind to, and the
relationships between composed UI parts.

The syntax stays terse: declare `screens:` inside a capability, then give each
screen a route, layout, elements, bindings, actions, events, rules, and
relationships as needed.

```apg
capability OperationsWorkbench {
    contract: {
        id: operations_workbench,
        provides: [operations_ui],
        configuration: {tenant_scoped: true},
        ui: {shell: python},
        theme: {name: ops_theme}
    };

    screens: {
        Dashboard: {
            route: "/ops",
            layout: dashboard,
            contains: [KpiStrip, ApprovalQueue],
            composes: [LedgerTable],
            binds: [ledger.entries],
            actions: [approve, reject],
            events: [{on: "select", do: "filter", target: LedgerTable}],
            relationships: [
                {from: KpiStrip, to: LedgerTable, via: filters},
                ApprovalQueue -> LedgerTable
            ]
        }
    };
}
```

## Fields

- `route` or `path`: URL path exposed by the screen.
- `layout`: screen arrangement such as `dashboard`, `grid`, `tabs`, `split`,
  `wizard`, `form`, or a project-specific layout token.
- `contains`: elements physically contained by the screen.
- `composes`: larger components assembled by the screen.
- `binds`: data contracts, services, entities, streams, or stores the screen
  reads from or writes to.
- `actions`: user or system actions available on the screen.
- `events`: UI events with `on`, `do`, `target`, and optional `when` fields.
- `relationships`: explicit links between composed elements. Use object form
  when you need named relationship metadata, or terse edge form for simple
  element-to-element links.
- `permissions`: permission contract or permission list required to access the
  screen.
- `rules`: deterministic screen-level rule contracts.
- `theme`: screen-specific theme override, otherwise the capability theme is
  used.

## Generated Runtime

Compiling a capability with screens emits the standard dependency-free
`apg_capabilities.py` manifest. That generated module exposes:

- `capability_screens(capability_name)`: normalized screen contracts.
- `ui_route_index()`: route-to-screen lookup for generated shells and routers.
- `composition_graph()`: application graph with screen, component, binding,
  theme, service, ERP-module, and relationship edges.

The generated composition graph includes:

- `capability -> screen` edges for ownership.
- `screen -> component` edges for rendered, contained, and composed elements.
- `screen -> binding` edges for data/service bindings.
- `component -> component` edges for explicit screen relationships.

Legacy `ui.routes` remains supported. If a capability declares only
`ui.routes`, the compiler still exposes those routes through
`capability_screens()` and `ui_route_index()`. New APG should prefer `screens:`
when the relationship between UI elements matters.
