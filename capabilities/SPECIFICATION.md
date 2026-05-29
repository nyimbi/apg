# APG Capability System Specification

This specification defines the required functionality, quality bar, and build
contract for every package-backed capability under `capabilities/`.

## Purpose

APG capabilities are first-class application components. They allow APG authors
to compose enterprise applications from stable, inspectable, package-owned
services instead of rewriting business logic for each generated application.

The capability system must support:

- domain-specific runtime behavior;
- deterministic governance and rule evaluation;
- tenant-safe configuration and execution;
- UI and theme composition;
- AI agent, workflow, event, and integration boundaries;
- side-effect-free package inspection and publish planning;
- release evidence suitable for review and automation.

## Functional Requirements

### Contract Registry

The registry must discover every `capability_contract.py` under
`capabilities/`, normalize legacy UI shells to `apg_python`, validate contract
shape, and expose structured reports for CLI and generated-app consumers.

Every contract must provide:

- `capability`: stable ID;
- `display_name`: human-readable name;
- `configuration`: tenant-safe runtime defaults;
- `configuration_schema`: required configuration shape;
- `rule_engine`: deterministic rules with decisions and required actions;
- `ui`: route metadata, shell, API prefix, template roots, and theme
  requirement;
- `theme`: named visual theme with tokens and component-level metadata.

### Runtime Package

Every package-backed capability must provide runtime code that can execute
without live provider credentials:

- domain records in `models.py` or a focused runtime module;
- package-owned service lifecycle methods in `service.py`;
- dependency-light API helpers in `api.py`;
- route/view models in `views.py`;
- package entrypoint and self-test in `app.py`;
- serializable semantic model and package manifest;
- release evidence that can be loaded by publish-plan tooling.

### Rules And Governance

Rules must be deterministic by default and return one of the APG decision
families:

- `allow`;
- `deny`;
- `require_review`;
- `warn`;
- `audit`.

Each rule must have a stable `name`, explicit `condition`, explicit `effect`,
denial or review `reason`, and `required_action` where applicable.

Service code must enforce the same rules that the contract exposes. It is not
acceptable for a rule to appear only in metadata while service methods bypass
the decision.

### UI And Theme

Every human-operated capability must expose route metadata and view models.
Routes must include names, paths, components, permissions, and navigation
grouping where useful.

Themes must use semantic tokens and component metadata. Theme definitions must
avoid hard-coded layout implementation details; layout belongs in view models
and screen composition.

### Adapter Boundaries

Live providers and external systems must remain behind explicit adapter
boundaries. Local package proof must not depend on external credentials,
network services, broker processes, AI providers, payment providers, identity
providers, or hardware devices.

Common adapter boundaries include:

- AI model runtimes and agent tools;
- identity providers and MFA providers;
- payment networks and banking rails;
- Bytewax deployment infrastructure and event bridges;
- document stores, object stores, search indexes, and SIEM systems;
- device, media, and sensor integrations.

### Documentation

Each actively developed capability must keep these documents aligned:

- `SPECIFICATION.md`: target functionality and acceptance criteria;
- `PLAN.md`: current implementation/review sequence;
- `cap_spec.md`: current executable runtime behavior and proof commands;
- tests and progress-log entries when readiness changes.

## Quality Requirements

World-class APG capabilities must be:

- **Composable**: usable by generated applications through stable contracts.
- **Executable**: domain lifecycles run locally without placeholder behavior.
- **Governed**: important business/security rules are enforced in service code.
- **Tenant-safe**: tenant context is explicit and checked in write paths.
- **Auditable**: decisions and important state changes produce evidence.
- **Themeable**: UI surfaces expose semantic theme tokens and components.
- **Configurable**: runtime defaults are safe and schema-backed.
- **Adapter-ready**: external integrations are explicit and replaceable.
- **Tested**: positive lifecycles and negative guardrails are covered.
- **Reviewable**: docs and commits explain constraints, proof, and non-goals.

## Acceptance Gates

A capability package is acceptable only when these gates pass:

```bash
./.venv/bin/pytest -q capabilities/<category>/<code>/test_capability_contract.py capabilities/<category>/<code>/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/<category>/<code> --json
./.venv/bin/apg capabilities publish-plan capabilities/<category>/<code> --json
```

Repository-wide capability readiness requires:

```bash
./.venv/bin/apg capabilities validate-contracts --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg tooling audit --json
```

The repository target is:

- 109/109 contracts valid and operable;
- 109/109 package artifacts complete;
- 109/109 packages classified as domain-specific;
- 0 materialized-baseline packages;
- 0 mixed or contract-only packages;
- 0 package gaps, errors, or warnings.

## Development Order

For each capability, proceed in this order:

1. Read the current contract, package spec, service, API, views, and tests.
2. Write or update `SPECIFICATION.md`.
3. Write or update `PLAN.md`.
4. Implement one coherent lifecycle or guardrail packet.
5. Run focused proof.
6. Perform code review and fix emergent problems.
7. Update `cap_spec.md` and progress evidence.
8. Commit and push the verified slice.
9. Move to the next capability.

Parallel execution is safe when agents own different capability package roots
and do not change shared contract registry behavior without coordination.
