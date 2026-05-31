# Data Virtualization Capability Summary

- **Capability name**: Data Virtualization
- **Capability ID**: `dvrl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

DVRL gives APG applications a governed data virtualization control plane. It
lets generated applications register virtual data sources, discover and review
schemas, publish virtual tables, evaluate federated read-query requests, manage
cache decisions, review policy changes, retire sources, register governed
virtualization agents, validate Bytewax lifecycle batches, and emit audit
evidence.

The current package keeps two boundaries clear:

- `DVRLLifecycleService` is the dependency-light generated-application control
  plane used by package tests, semantic-model evidence, and composed apps.
- `DVRLService` remains the production-oriented federation runtime for physical
  connectors, query parsing, connector orchestration, cache metadata, NLP
  assistance, Singer integration, and APG service integration.

## Provided Services

- `dvrl_operations`
- `data_virtualization`
- `federated_query_lifecycle`
- `virtualization_agent_composition`
- virtual source lifecycle governance
- schema refresh and virtual table publication workflows
- federated query guardrail evaluation
- query cache lifecycle decisions
- virtualization policy review tracking
- provider-neutral AI/automation agent participation for Codex, Claude Code,
  OpenCode, Pi, and future runtimes
- Bytewax lifecycle-batch validation
- generated-application UI route and theme metadata

## Required Services

- `mdm` for governed data domains
- `etlp` for pipeline and transformation context
- `meta` for catalog, classification, and lineage
- tenant context from the generated application
- adapter-bound `keym`, `auth`, `audl`, and `cach` services when production
  deployments bind physical source credentials, RBAC, audit sinks, or cache
  stores
- Bytewax-backed event streaming for lifecycle batch processing

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The contract includes source, schema, query,
cache, governance, optimization, adapter, agent, streaming, UI, and theme
sections.

## Rules

DVRL ships deterministic guardrails for:

- tenant context
- source owner, type, credentials, encryption, and activation approval
- stale schema refresh review
- virtual table owner and classification evidence
- read-only, parameterized, RBAC-authorized, lineage-captured query requests
- sensitive result cache blocking
- high-cost and cross-source join review
- result-row limits and cache TTL limits
- policy review
- source retirement impact review
- supported virtualization-agent runtime and role
- virtualization-agent scope, owner, purpose, contribution disclosure, and
  privileged-role human approval
- Bytewax-only lifecycle-batch routing

## UI

The generated-application UI contract is exposed through `view_models.py` and
includes dashboard, query, sources, schemas, virtual tables, federation,
policies, cache, metrics, adapters, agents, lifecycle, audit, and settings
routes.

## Runtime Boundary

Dependency-light lifecycle tests do not open physical database, SaaS,
object-store, streaming, or Singer tap connections. Production deployments bind
the connector registry, query planner, execution engine, metadata catalog,
cache store, credential vault, audit sink, and Bytewax runtime through APG
configuration and capability adapters. AI/automation tools are also adapters:
the lifecycle packet records their runtime, role, scope, owner, purpose, and
approval posture, but it does not embed vendor-specific clients.
