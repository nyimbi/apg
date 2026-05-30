# APG Access Control Integration Hub Capability Specification

## Capability Metadata

- **Capability Code:** COMPOSITION_ACCESS
- **Capability Name:** Access Control Integration Hub
- **Version:** 2.1.0
- **Category:** Composition
- **Runtime Target:** python
- **Primary Stream Processor:** Bytewax

## Purpose

Access Control Integration Hub is the composition-layer security capability for APG. It gives composed applications a shared way to register identity providers, protected resources, policies, grants, session risk checks, access decisions, and security-focused AI agents. The capability is intentionally terse at the APG language level while preserving strong operational guardrails in the generated runtime.

## Scope

The capability owns these executable surfaces:

- Identity-provider registration and activation for local, OIDC, SAML, LDAP, API-key, and JWT providers.
- Protected-resource registration across APG capabilities and composed applications.
- Policy creation and activation with sensitive-resource conditions and high-risk simulation evidence.
- Access-grant lifecycle with privileged approval, expiry, justification, and separation-of-duties checks.
- Session risk evaluation with adaptive step-up enforcement.
- Access-decision recording through a Bytewax lifecycle stream.
- AI-agent registration for security review roles across Codex, Claude Code, OpenCode, and Pi runtimes.
- UI contracts for dashboard, providers, resources, policies, grants, decisions, sessions, agents, audit, and settings.
- Theming hooks for compact operational security consoles.

## Non-Goals

- The package does not implement vendor-specific OAuth, SAML, LDAP, or vault drivers directly.
- The package does not replace APG `auth`; it composes and governs access-control integration around it.
- The package does not run a live Bytewax topology in local package tests. It declares and enforces the stream contract that deployment binds to an operating topology.

## Domain Model

### Access Provider

Represents an identity provider attached to a tenant. Providers require an owner, type, validation evidence, and secret reference when external.

### Access Resource

Represents a protected APG resource, route, workflow, screen, dataset, or composed application boundary. Resources require an owner, capability id, and one or more scopes.

### Access Policy

Represents an allow or deny rule for a resource. Sensitive resources require conditions. High-risk policy activation requires simulation evidence and review attribution.

### Access Grant

Represents subject access to a resource scope. Privileged grants require approval, expiry, justification, and independent approval.

### Access Session

Represents a runtime session risk evaluation. High-risk sessions require adaptive step-up authentication before the session can be trusted.

### Access Decision

Represents a recorded authorization decision. Decisions must be emitted through Bytewax so downstream audit, monitoring, and rule-tuning pipelines receive the event.

### Access Agent

Represents a first-class AI agent that can review, recommend, and explain access-control actions. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.

## Lifecycle

1. Register an identity provider with an accountable owner.
2. Activate the provider only after metadata validation, secret-reference attachment, and test evidence.
3. Register protected resources with owner, capability id, and scopes.
4. Create policies for resources, using explicit conditions for sensitive resources.
5. Activate high-risk policies only after simulation evidence and review.
6. Create grants for subjects; privileged grants require approval, expiry, justification, and separation of duties.
7. Evaluate sessions continuously; high-risk sessions require step-up authentication.
8. Record decisions through the Bytewax event stream.
9. Register access agents for review lanes, with human approval required for privileged agent proposals.
10. Audit all state changes and expose them through APG UI models.

## Guardrails

- Tenant context is mandatory for all operations.
- Providers require accountable ownership.
- Provider activation requires metadata evidence.
- External providers require a secret reference.
- Resources require ownership and scopes.
- Policies require accountable ownership.
- Sensitive-resource policies require explicit conditions.
- High-risk policy activation requires simulation evidence and review.
- Privileged grants require approval.
- Privileged grants require expiry.
- Privileged grants require separation of duties.
- Grants require business justification.
- High-risk sessions require adaptive step-up authentication.
- Access decisions require Bytewax stream routing.
- Batch grant changes require Bytewax stream routing.
- Access agents must use supported runtimes and roles.
- Privileged access actions proposed by agents require human approval.

## UI Contract

The capability exposes these APG routes:

- `/composition-access/dashboard`
- `/composition-access/providers`
- `/composition-access/resources`
- `/composition-access/policies`
- `/composition-access/grants`
- `/composition-access/decisions`
- `/composition-access/sessions`
- `/composition-access/agents`
- `/composition-access/audit`
- `/composition-access/settings`

The UI is theme-aware and uses compact operational surfaces: provider console, policy studio, grant workbench, decision explorer, session monitor, and agent workbench.

## Event Stream

- **Processor:** `bytewax`
- **Stream:** `apg.composition.access.lifecycle`
- **Key:** `tenant_id`
- **Events:** provider registered, provider activated, resource registered, policy created, policy activated, grant created, grant revoked, session evaluated, access decision recorded, access agent registered.

## Integration Requirements

- Requires `auth`, `audl`, `ntfy`, `conf`, and `registry`.
- Provides identity-provider composition, resource-access registry, policy orchestration, grant lifecycle, session risk control, access-decision audit, and access agents.
- Uses APG Python runtime surfaces: `service.py`, `api.py`, `views.py`, and `app.py`.
