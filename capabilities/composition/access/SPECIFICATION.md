# Access Control Integration Hub Specification

## Intent

Access Control Integration Hub makes identity, policy, grants, runtime decisions, and access-review agents composable APG primitives. It is the security boundary that lets APG applications combine screens, workflows, data, services, and AI agents without scattering authorization logic across every generated app.

## Users

- Application builders who need to attach access controls to composed applications.
- Security administrators who manage providers, resource scopes, policies, and grants.
- Auditors who need decision evidence and state-change history.
- Operations teams who need session risk and grant lifecycle visibility.
- AI agents that review policies, grants, sessions, and audit evidence under human-governed boundaries.

## Functional Requirements

### Identity Providers

- Register local, OIDC, SAML, LDAP, API-key, and JWT providers.
- Require a provider owner.
- Keep providers in draft until activation evidence is complete.
- Require metadata validation before activation.
- Require secret-reference attribution for external providers.
- Require test evidence before activation.

### Protected Resources

- Register capability resources, screens, workflows, datasets, routes, and composed application boundaries.
- Require a resource owner.
- Require one or more scopes.
- Track the owning capability id.
- Mark sensitive resources so stricter policy rules apply.

### Policy Lifecycle

- Create allow and deny policies for registered resources.
- Require a policy owner.
- Require explicit conditions for sensitive resources.
- Require simulation evidence and review for high-risk activation.
- Expose policies through a policy-studio UI model.

### Grant Lifecycle

- Create grants for subjects against registered resource scopes.
- Validate grant scopes against resource scopes.
- Require justification for every grant.
- Require approval, expiry, and separation of duties for privileged grants.
- Support revocation with audit evidence.

### Session Risk

- Evaluate session risk per provider and subject.
- Require adaptive step-up when risk exceeds the configured threshold.
- Store session evaluation outcomes for operations review.

### Decision Recording

- Record allow, deny, and review decisions.
- Require decisions to be routed through Bytewax.
- Attach subject, resource, action, reason, and policy ids.
- Emit audit events for decision recording.

### AI Agents

- Register access agents as first-class runtime records.
- Support `codex`, `claude_code`, `opencode`, and `pi`.
- Support architect, policy-review, grant-review, risk-review, session-review, and audit-review roles.
- Require human approval for privileged access actions proposed by agents.

### UI and Theme

- Provide APG route contracts for dashboard, providers, resources, policies, grants, decisions, sessions, agents, audit, and settings.
- Expose compact security-operation view models.
- Provide theme tokens and component-level visual contracts for policy, grant, decision, session, and agent surfaces.

### Eventing

- Use Bytewax as the lifecycle-stream processor.
- Emit provider, resource, policy, grant, session, decision, and agent events keyed by tenant id.

## Rule Engine

The deterministic rule engine is the source of truth for runtime guardrails. Service methods invoke rules before mutating records. Rules use exact matches plus comparison suffixes such as `_gt`, `_gte`, `_lt`, `_lte`, and `_ne`.

## Acceptance Criteria

- `get_capability_contract()` returns a valid APG contract with configuration, schema, rules, UI, theme, and Bytewax streaming metadata.
- Package import exposes `CompositionAccessService`, contract helpers, models, and registration metadata without external service dependencies.
- Service supports provider, resource, policy, grant, session, decision, agent, batch, and audit operations.
- Privileged grants cannot bypass approval, expiry, justification, or separation-of-duties checks.
- High-risk sessions cannot pass without step-up authentication.
- Decisions and batch grant operations reject non-Bytewax stream routing.
- API helpers and view models expose the same capability surfaces.
- Tests cover contract shape, rules, service lifecycle, API/view surfaces, semantic model, and package self-test.
