# Central Configuration Management Specification

## Intent

Central Configuration Management makes configuration a governed APG composition primitive. It lets generated applications share configuration namespaces, schemas, values, releases, templates, drift evidence, AI-agent review, UI models, and Bytewax lifecycle events.

## Functional Requirements

- Register tenant-scoped namespaces by environment, owner, path prefix, and capability id.
- Create configuration values only inside namespace path boundaries.
- Redact secret values in returned records.
- Require secret references for secret configurations.
- Require schemas for restricted configurations.
- Require validation evidence before activation.
- Require production approval for production deployments.
- Require canary evidence for high-impact deployments.
- Require Bytewax routing for deployments, rollbacks, and batch changes.
- Record drift between expected and observed configuration versions.
- Create templates with variable schemas and review for shared use.
- Register first-class configuration agents for Codex, Claude Code, OpenCode, and Pi.
- Expose dashboard, namespace, editor, release, template, drift, agent, and settings UI models.

## Rule Engine

The deterministic rule engine enforces tenant context, policy attachment, schema requirements, secret references, validation evidence, production approvals, canary evidence, Bytewax routing, template review, agent runtime and role support, and human approval for privileged agent actions.

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, rules, UI, theme, and Bytewax streaming metadata.
- Package import exposes `CompositionConfigService`, `CentralConfigurationService`, records, contract helpers, and registration metadata without optional framework imports.
- Service supports namespace, configuration, validation, activation, update, deployment, rollback, template, drift, agent, batch, and audit operations.
- API helpers and view models expose the same lifecycle surfaces.
- Semantic model includes `config_agents`, required dependencies, route metadata, rules, theme, and Bytewax stream metadata.
- Focused tests cover lifecycle success paths and guardrail failures.
