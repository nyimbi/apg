# APG Central Configuration Management Capability Specification

## Capability Metadata

- **Capability Code:** COMPOSITION_CONFIG
- **Capability Name:** Central Configuration Management
- **Version:** 2.1.0
- **Category:** Composition
- **Runtime Target:** python
- **Primary Stream Processor:** Bytewax

## Purpose

Central Configuration Management is the configuration plane for APG-composed applications. It gives every capability a shared way to declare namespaces, values, schemas, releases, templates, drift records, AI-agent review, UI surfaces, visual theme hooks, and Bytewax-backed lifecycle events.

## Scope

The capability owns these executable surfaces:

- Namespace registration by tenant, environment, owner, path prefix, and capability boundary.
- Configuration value creation, validation, activation, update, deployment, rollback, and drift capture.
- Schema and secret-reference guardrails for restricted and secret values.
- Release workflows for development, test, staging, and production environments.
- Template library for repeatable configuration bundles.
- AI-agent registration for Codex, Claude Code, OpenCode, and Pi review lanes.
- UI contracts for dashboard, namespaces, configurations, releases, templates, drift, agents, and settings.
- Bytewax lifecycle stream metadata for deployment and rollback evidence.

## Lifecycle

1. Register a namespace with tenant, environment, owner, capability id, and path prefix.
2. Create configuration values inside namespace boundaries.
3. Attach schemas to restricted values and secret references to secret values.
4. Validate configuration values and record validation evidence.
5. Activate validated configurations.
6. Deploy active configurations with production approvals and canary evidence for high-impact releases.
7. Record drift when observed runtime versions differ from expected versions.
8. Roll back deployments with reason and Bytewax event routing.
9. Register configuration agents for review and validation lanes.
10. Audit every state change.

## Guardrails

- Tenant context is mandatory.
- Namespaces require owners and environments.
- Writes require policy attachment.
- Restricted configurations require schemas.
- Secret configurations require secret references.
- Activation requires validation evidence.
- Production deployments require approval.
- High-impact deployments require canary evidence and review.
- Deployment events require Bytewax stream routing.
- Rollbacks require reasons and Bytewax stream routing.
- Shared templates require review.
- Batch changes require Bytewax stream routing.
- Configuration agents require supported runtime and role.
- Privileged agent configuration actions require human approval.

## UI Contract

The capability exposes these APG routes:

- `/composition-config/dashboard`
- `/composition-config/namespaces`
- `/composition-config/configurations`
- `/composition-config/releases`
- `/composition-config/templates`
- `/composition-config/drift`
- `/composition-config/agents`
- `/composition-config/settings`

## Event Stream

- **Processor:** `bytewax`
- **Stream:** `apg.composition.config.lifecycle`
- **Key:** `tenant_id`
- **Events:** namespace registered, configuration created, configuration validated, configuration activated, configuration deployed, configuration rolled back, template created, drift detected, config agent registered.

## Integration Requirements

- Requires `auth`, `audl`, `ntfy`, `registry`, and `composition_access`.
- Provides namespace registry, configuration lifecycle, schema validation, release workflows, template library, drift monitoring, and config agents.
- Uses APG Python runtime surfaces: `service.py`, `api.py`, `views.py`, and `app.py`.
