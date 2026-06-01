# Threat Intelligence Capability Specification

`intel_threats` is the APG Threat Intelligence capability. It turns governed
collection and analysis into threat workspaces, source lineage, indicators,
actors, campaigns, assessments, reports, mitigations, reviews, UI models,
Bytewax lifecycle events, and provider-neutral AI-agent composition surfaces.

## Capability Summary

- Capability ID: `intel_threats`
- Display name: Threat Intelligence
- Target: Python executable capability package
- Event processor: Bytewax
- Event stream: `apg.intel.threats.lifecycle`
- Theme: `intel_threats_control`
- Agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

## Composition Interfaces

The package provides authority, workspace, source, indicator, actor, campaign,
assessment, report, mitigation, review, and AI-agent workflows. It requires
APG auth, audit, notification, NLP, graph, RAG, and geospatial capabilities so
generated applications can compose threat intelligence with identity,
evidence, enrichment, link analysis, retrieval, map context, and downstream
notifications.

## Runtime Shape

The service keeps tenant-scoped in-memory records for the executable baseline
while leaving persistent storage, enrichment providers, graph writes, RAG
indexing, reporting renderers, notification delivery, case-management writes,
and durable Bytewax workers behind adapter boundaries.

## Governance

Every write path evaluates deterministic rules before mutation. The rules
require tenant context, policy attachment, lawful authority, evidence,
classification, source lineage, custodian ownership, confidence scoring,
analyst attribution, approval for reports and mitigations, Bytewax routing, and
human approval for privileged AI-agent scopes. Unsupported attribution,
fabricated indicators, source tampering, privacy bypasses, autonomous
mitigation, and unapproved publication are denied.

