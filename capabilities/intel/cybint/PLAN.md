# Cyber Intelligence Build Plan

## Packet 1: Contract

- Define capability metadata, dependencies, configuration, deterministic rules,
  UI routes, theme tokens, and Bytewax lifecycle metadata.
- Make defensive authority, indicator lineage, TLP, confidence, evidence, risk,
  release control, and review guardrails explicit.

## Packet 2: Runtime

- Add tenant-keyed in-memory models for authorities, indicators, sightings,
  enrichments, profiles, risks, incident links, dissemination, reviews, and
  agents.
- Implement service methods that evaluate rules before state mutation.
- Add dependency-light API helpers.

## Packet 3: Composition

- Add dashboard, CYBINT console, and agent workbench view models.
- Add app entrypoint with self-test, semantic model, and component manifest.
- Generate release evidence and package metadata.

## Packet 4: Verification And Review

- Run focused package tests and APG package audits.
- Review tenant isolation, authority relationships, TLP/confidence handling,
  incident linkage, release control, Bytewax-only lifecycle routing, adapter
  boundaries, and AI-agent guardrails.
- Update catalog and progress evidence, then commit and push the verified slice.
