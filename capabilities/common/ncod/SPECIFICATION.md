# NCOD Capability Specification

## Identity

- Capability ID: `ncod`
- Display name: No-Code/Low-Code Builder
- Category: common
- Runtime target: Python capability package
- Primary purpose: compose governed APG applications from data models, screens,
  components, workflows, scripts, connectors, AI builder agents, themes,
  publishing gates, and deployment evidence.

## Goals

NCOD must let a person rapidly produce executable APG applications while keeping
enterprise guardrails explicit. The runtime is deliberately deterministic and
dependency-light so it can be tested locally, packaged by the APG compiler, and
connected to production adapters later.

The capability must support:

- App/project lifecycle management.
- Screen and component composition.
- Data model and data source binding.
- Workflow and script extension binding.
- External connector binding.
- Visual theme variants.
- First-class AI builder-agent composition.
- Validation, publish, deploy, retire, and audit workflows.
- Rule-engine decisions for deny/review/allow.
- Bytewax lifecycle stream policy for batch and runtime mutations.
- UI models for operational composition surfaces.

## Lifecycle

1. **Create app**: tenant, name, owner, theme, RBAC policy, data-residency
   policy, accessibility posture, and metadata are captured.
2. **Compose screens**: pages get stable routes, layouts, and optional element
   relationship metadata that describes how composed elements relate.
3. **Place components**: governed component types are attached to screens and
   interactive components require accessible labels.
4. **Define data**: business data models require names, fields, and data policy.
5. **Bind runtime data**: data bindings require valid schemas and source types.
6. **Attach automation**: workflow bindings require triggers, WFLO references,
   and automation policies; script extensions require SCPT policy.
7. **Bind integrations**: connector bindings require connector policies and
   scopes.
8. **Register AI collaborators**: AI builder agents are configured with runtime,
   role, scope, registration status, policy, and contribution disclosure.
9. **Validate app**: readiness checks cover owner, page, component, data model,
   theme, accessibility, RBAC, data residency, data model validity, data binding
   validity, workflow policy, script policy, connector policy, AI agent
   registration, AI contribution disclosure, and theme approval.
10. **Publish app**: publication requires approval, passing validation, script
    policy, connector policy, and production review for production targets.
11. **Deploy release**: deployment requires supported runtime target, target
    reference, deployment approval, and rollback plan.
12. **Change state**: lifecycle state changes require reason and audit evidence.

## Domain Model

- `BuilderApp`: tenant-scoped application/project.
- `BuilderPage`: screen or form canvas.
- `BuilderComponent`: component instance on a page.
- `DataModelDefinition`: governed business entity with fields.
- `DataBinding`: source binding exposed to pages/components.
- `WorkflowBinding`: WFLO automation attached to an app.
- `ThemeVariant`: tenant-approved theme token set.
- `ScriptExtension`: approved SCPT-backed low-code extension.
- `ConnectorBinding`: external integration binding.
- `BuilderAgent`: AI agent collaborator such as Codex, Claude Code, OpenCode,
  or Pi.
- `ValidationResult`: readiness gate result.
- `PublishRelease`: governed app publication.
- `DeploymentRecord`: governed release deployment.
- `NcodAuditEvent`: local audit event for all builder mutations.

## Rule Engine

The deterministic rule engine evaluates a context dictionary and returns:

- `allow` when no guardrails match.
- `require_review` when policy requires human or external review.
- `deny` when the operation is not allowed until required evidence is present.

The rule set covers tenant context, app ownership/name/theme/RBAC/residency,
screen routes, screen relationship metadata, component placement,
accessibility, data models, data bindings, workflow policies, publishing,
validation, scripts, connectors, production changes, deployments, AI builder
agents, state-change audit, cross-tenant access, and Bytewax batch-mutation
requirements.

## UI Contract

NCOD exposes APG Python UI model routes:

- `/ncod/dashboard`
- `/ncod/apps`
- `/ncod/builder`
- `/ncod/pages`
- `/ncod/data-models`
- `/ncod/components`
- `/ncod/workflows`
- `/ncod/publishing`
- `/ncod/deployments`
- `/ncod/connectors`
- `/ncod/agents`
- `/ncod/audit`
- `/ncod/analytics`
- `/ncod/settings`

The view models are data-only and framework-neutral so they can be rendered by
the generated Python target or by later richer UI adapters.

## Theming

The default theme is `ncod_app_builder`. It uses compact builder-oriented tokens
and component-specific visual contracts for app libraries, page composer,
component catalog, data modeler, workflow designer, publish center, deployment
center, and AI agent panel.

## Adapter Boundaries

NCOD does not call external systems directly. Production integrations should be
attached through:

- `wflo` for workflow execution.
- `scpt` for scripts, validation hooks, and automation.
- `conn` for connector credentials, APIs, and external services.
- `auth` for RBAC and tenant policy.
- `audl` for durable audit.
- `them` for tenant theme registries.
- `accs` for accessibility checks.
- `moni` for metrics and observability.
- `bytewax` for lifecycle event streaming and batch mutation orchestration.

## Non-Goals For This Packet

- Live visual drag-and-drop rendering.
- Live connector execution.
- Live external AI CLI invocation.
- Production database persistence.
- Browser-rendered UI verification.
- Full APG compiler integration across every target.

Those belong in later passes once the coherent executable capability spine is
stable.

