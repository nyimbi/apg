# Geospatial Intelligence Build Plan

## Packet 1: Contract

- Define capability metadata, dependencies, configuration, deterministic rules,
  UI routes, theme tokens, and Bytewax lifecycle metadata.
- Make lawful authority, area/source governance, retention, classification,
  confidence, evidence, release control, and review guardrails explicit.

## Packet 2: Runtime

- Add tenant-keyed in-memory models for authorities, areas, sources, collection
  plans, observations, features, changes, assessments, dissemination, reviews,
  and agents.
- Implement service methods that evaluate rules before state mutation.
- Add dependency-light API helpers.

## Packet 3: Composition

- Add dashboard, GEOINT console, and agent workbench view models.
- Add app entrypoint with self-test, semantic model, and component manifest.
- Generate release evidence and package metadata.

## Packet 4: Verification And Review

- Run focused package tests and APG package audits.
- Review tenant isolation, area/source authority relationships, retention,
  confidence bounds, release control, Bytewax-only lifecycle routing, adapter
  boundaries, and AI-agent guardrails.
- Update catalog and progress evidence, then commit and push the verified slice.
