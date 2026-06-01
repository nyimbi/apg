# Intelligence Reporting Capability Specification

`intel_reporting` is the APG Intelligence Reporting capability. It turns
analytical findings into governed templates, report products, sections,
citations, approvals, distributions, publications, reviews, UI models, Bytewax
lifecycle events, and provider-neutral AI-agent composition surfaces.

## Capability Summary

- Capability ID: `intel_reporting`
- Display name: Intelligence Reporting
- Target: Python executable capability package
- Event processor: Bytewax
- Event stream: `apg.intel.reporting.lifecycle`
- Theme: `intel_reporting_control`
- Agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

## Composition Interfaces

The package provides authority, workspace, template, product, section,
citation, approval, distribution, publication, review, and AI-agent workflows.
It requires APG auth, audit, notification, NLP, graph, RAG, and geospatial
capabilities so generated applications can compose reporting with identity,
evidence, source citation, retrieval, graph context, map context, and
dissemination.

## Runtime Shape

The service keeps tenant-scoped in-memory records for the executable baseline
while leaving persistent document stores, rendering engines, notification
delivery, case-file writes, export pipelines, graph/RAG writes, and durable
Bytewax workers behind adapter boundaries.

## Governance

Every write path evaluates deterministic rules before mutation. The rules
require tenant context, policy attachment, lawful authority, evidence,
classification, template/product ownership, citation evidence, approvals for
distribution and publication, Bytewax routing, and human approval for
privileged AI-agent scopes. Uncited claims, classification downgrades, source
fabrication, privacy bypasses, autonomous publication, and unapproved
distribution are denied.

