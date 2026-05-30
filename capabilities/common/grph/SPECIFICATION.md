# GRPH Capability Specification

## Purpose

Graph Data Management is the APG capability for modeling, governing, querying,
and visualizing connected data. It provides a first-class graph runtime that can
be embedded in generated applications and composed with knowledge, search,
lineage, analytics, and governance capabilities.

## Scope

GRPH must support:

- Tenant-scoped graph schemas.
- Property, lineage, knowledge, and dependency graph kinds.
- Node and edge lifecycle management.
- Schema-constrained node types and edge types.
- Relationship classification and review.
- Bounded traversal, lineage paths, impact analysis, and neighborhood views.
- Quality reports and operational summaries.
- Audit evidence for mutations, review decisions, and state changes.
- UI route metadata and view models.
- Visual theming for graph-oriented generated applications.
- Bytewax event-stream adapter evidence.

## Configuration

The contract exposes configuration sections for:

- `schemas`: graph kind allowlists, schema naming, source assets, and retirement.
- `nodes`: node identity, ownership, type, label, and property controls.
- `edges`: relationship identity, endpoint, type, owner, and classification
  controls.
- `traversal`: max depth, review thresholds, query types, and result windows.
- `lineage`: lineage graph requirements and source asset enforcement.
- `quality`: orphan, missing-owner, restricted-edge, and density thresholds.
- `security`: tenant isolation, RBAC filters, and restricted relationship
  access.
- `governance`: audit and review requirements.
- `observability`: metrics, traces, audit, quality metrics, and Bytewax stream
  evidence.
- `adapters`: generated runtime, helper runtime, API, metadata, event stream,
  search, knowledge graph, AI, auth, audit, cache, and monitoring adapters.
- `ui`: generated-app screen switches.
- `theme`: visual token and component configuration.

## Rule Engine

GRPH uses deterministic rules. Rules must cover:

- tenant context,
- schema required fields,
- allowed graph kinds,
- lineage source asset checks,
- node identity, ownership, type, labels, and properties,
- edge identity, endpoint, type, owner, classification, and schema membership,
- restricted relationship review,
- tenant-local endpoint checks,
- traversal start, depth, result-window, review, and RBAC checks,
- lineage query source asset checks,
- quality threshold reviews,
- batch/event-stream requirements,
- schema retirement review, and
- audit evidence for state changes.

Rules that require review must be executable by supplying explicit review
evidence. They must not behave as permanent denials unless the rule explicitly
declares a deny decision.

## Runtime Behavior

The service runtime must:

- create schemas,
- create nodes,
- create edges,
- run traversals,
- run lineage paths,
- generate neighborhood and impact views,
- compute quality reports,
- record audit events,
- expose list and dashboard surfaces,
- provide APG `create_record` compatibility, and
- enforce the contract guardrails before mutating state.

## UI

The generated UI manifest must include routes for:

- dashboard,
- explorer,
- schemas,
- nodes,
- edges,
- traversal,
- lineage,
- impact,
- quality,
- governance,
- audit, and
- settings.

Each view model should be serializable, tenant-scoped, and directly usable by a
generated Python application shell.

## Package Evidence

The package must include:

- `README.md`,
- `SPECIFICATION.md`,
- `PLAN.md`,
- `cap_spec.md`,
- contract, runtime, API, views, and helper modules,
- refreshed `semantic_model.json`,
- refreshed `release_report.json`, and
- refreshed `package_manifest.json`.

Focused verification must prove contract shape, package evidence, docs
existence, runtime lifecycle behavior, review-evidence paths, UI model behavior,
and APG compatibility.

## Out of Scope

This packet does not require a live graph database, rendered browser UI, live
Bytewax pipeline execution, or external adapter calls. Those integrations are
represented as explicit adapter surfaces and can be bound by larger generated
applications.
