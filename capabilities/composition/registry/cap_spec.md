# APG Capability Registry Specification

- **Capability Name**: Capability Registry
- **Category**: Composition
- **Version**: 2.1.0
- **Capability ID**: `composition_registry`

## Purpose

Capability Registry is the catalog and governance authority for APG capability composition. It lets generated applications register capabilities, validate dependency graphs, define composition blueprints, govern version releases, prepare marketplace publication, expose operator screens, emit Bytewax lifecycle events, and include AI agents as first-class reviewers.

## Capability Boundaries

The capability owns catalog records, dependency edges, composition blueprint records, version release evidence, marketplace publication preparation, deterministic registry rules, and registry-agent records. It does not own authentication, audit persistence, notification delivery, search infrastructure, or event infrastructure; those remain adapter dependencies.

## Provides

- `capability_catalog_lifecycle`
- `dependency_graph_management`
- `composition_blueprint_validation`
- `version_compatibility_governance`
- `marketplace_publication_governance`
- `registry_discovery`
- `registry_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `composition_access`

## Lifecycle

1. Register a capability with owner, category, version, provided surfaces, required surfaces, and executable contract reference.
2. Add dependency edges with target capability, dependency type, version constraint, and cycle detection.
3. Create composition blueprints from registered capabilities.
4. Validate blueprints for missing capabilities and unmet dependencies.
5. Publish validated blueprints with validation evidence.
6. Release capability versions with compatibility evidence.
7. Deprecate capabilities only with a migration plan.
8. Prepare marketplace publication with documentation and review evidence.
9. Register AI agents for registry curation, dependency review, composition review, version review, marketplace review, and security review.

## Rule Engine

The deterministic rule engine denies missing tenant context, writes without policy, incomplete capability records, invalid dependency records, invalid composition blueprints, releases without compatibility evidence, deprecations without migration plans, marketplace publication without documentation, registry imports without Bytewax, unsupported registry-agent runtime or role, and privileged agent actions without human approval. Marketplace publication without review requires review.

## UI Contract

The capability exposes screens for dashboard, catalog, dependencies, compositions, versions, marketplace, rules, agents, and settings. Theme metadata defines compact operational surfaces for catalog quality, dependency graph edges, composition validation, version compatibility, marketplace publication review, rule grids, and agent approval lanes.

## Streaming

Registry lifecycle events use the Bytewax processor and stream `apg.composition.registry.lifecycle`. The stream key is `tenant_id`. Events include capability registration, dependency addition, composition creation, composition validation, version release, capability deprecation, marketplace publication preparation, and registry agent registration.

## AI Agent Composition

Registry agents are first-class capability records. Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported roles are capability curator, dependency reviewer, composition reviewer, version reviewer, marketplace reviewer, and security reviewer. Privileged registry actions require recorded human approval.
