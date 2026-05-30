# Capability Registry Specification

## Objective

Build `composition_registry` as an executable APG capability that generated applications can compose for capability cataloging, dependency graph validation, composition blueprint governance, version compatibility, marketplace publication preparation, and AI-assisted registry review.

## Functional Surface

- Capability registration with owner, category, version, provides, requires, and executable contract reference.
- Dependency graph management with target capability, dependency type, version constraint, and cycle detection.
- Composition blueprint creation and validation.
- Composition publication with validation evidence.
- Version release with compatibility evidence.
- Deprecation with migration plan.
- Marketplace publication preparation with documentation and review.
- Deterministic registry rule evaluation.
- UI routes, view models, and theme tokens for generated applications.
- AI agent registration and privileged-action validation for `codex`, `claude_code`, `opencode`, and `pi`.

## Non-Goals

- Owning live search index infrastructure.
- Owning auth, audit, notification, or event infrastructure.
- Requiring FastAPI, Flask-AppBuilder, SQLAlchemy, Redis, or mobile service dependencies for package loading.

## Acceptance Criteria

- `get_capability_contract()` validates through `capabilities.capability_contract_registry.validate_contract_shape`.
- `CompositionRegistryService` can register capabilities, add dependencies, create and publish compositions, release versions, prepare marketplace publications, register agents, and summarize registry state without optional infrastructure.
- Rules deny incomplete catalog records, unsafe dependency records, unsafe publication/release actions, unsupported agents, and non-Bytewax registry coordination.
- `app.semantic_model()` exposes provides/requires, configuration, rules, screens, theme, streaming, and agent team metadata.
- Focused package tests compile and pass.
- `apg capabilities inspect`, `publish-plan`, and `implementation-audit` succeed for the package.
