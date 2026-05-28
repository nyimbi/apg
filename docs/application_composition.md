# Application Composition

APG applications are first-class composition units. Use `app`, `application`, or
`composition` when the source needs to assemble capabilities, agents, routes,
screens, deployment choices, and visual theme tokens into one executable
application shell.

```apg
app EnterpriseERPPlatform {
    description: "Composable ERP application shell";
    capabilities: [PlatformAudit, EnterpriseFinance, EnterpriseOperations];
    agents: [Planner];
    agent_teams: [FinanceCrew];
    routes: ["/erp/finance", "/erp/operations"];
    components: {
        finance_workbench: {capability: journal_entries, route: "/erp/finance"},
        operations_workbench: {capability: executive_kpis, route: "/erp/operations"}
    };
    screens: {
        ExecutiveHome: {route: "/erp", capability: EnterpriseOperations}
    };
    theme: {name: enterprise_theme, tokens: {accent: "#174EA6"}};
    runtime: {target: python, deployment: container, streaming: {processor: bytewax}};
    deployments: {default: local, container: docker};
}
```

The compiler emits `apg_application.py` when at least one application
composition is present. Generated applications import that runtime and expose
the composition through:

- `describe_application()["application_composition_descriptions"]`
- `component_manifest()["application_compositions"]`
- `GET /applications`
- `validate_application()["checks"]["application_compositions"]`
- package helpers such as `list_applications()`,
  `application_dependency_graph()`, and
  `validate_application_compositions()`

Application validation checks references against generated capability and agent
runtime catalogs when those catalogs are present. Missing references become
validation errors; external references remain valid when no local catalog is
available.
