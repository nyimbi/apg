# GEOS Development Plan

## Objective

Build GEOS into a coherent APG geo-spatial lifecycle and guardrail packet that
can be packaged and tested without heavyweight mapping, routing, warehouse, or
streaming dependencies while leaving clear adapter boundaries for production
location infrastructure.

## Implementation Slices

1. **Contract**
   - Expand configuration for geofencing, events, analytics, AI location
     agents, governance, observability, adapters, UI, theme, and Bytewax.
   - Expand deterministic guardrails for consent, data residency, geometry,
     source registration, accuracy, privacy review, spatial index, aggregation
     privacy, agents, state changes, tenant isolation, and Bytewax.

2. **Runtime Facade**
   - Keep the dependency-light `GeosService` as the generated-application
     surface.
   - Tenant-qualify package state so duplicate IDs across tenants cannot
     collide.
   - Add location-agent registration and geofence state changes.

3. **API And Views**
   - Add package helpers for event source, geofence, location event, and agent
     operations.
   - Add view models for geofences, territories, agents, audit, and settings.

4. **Documentation**
   - Add README and full specification.
   - Replace older broad docs with current adapter-boundary notes.

5. **Generated Evidence**
   - Refresh package app, semantic model, manifest, and release report after
     contract changes.

6. **Verification**
   - Run focused `py_compile`.
   - Run GEOS contract/package tests only.
   - Run generated app self-test.
   - Run APG implementation audit and publish plan for GEOS.
   - Search GEOS for stale markers, unsupported overclaims, unfinished
     scaffolding, and banned stream choices.

## Review Checklist

- The contract exposes provides, requires, rules, UI routes, theme, adapters,
  and Bytewax streaming.
- Runtime imports remain usable in the current environment.
- Event sources enforce tenant context, data residency, and sensitive-location
  review.
- Location events enforce source registration, consent, accuracy, and privacy
  review.
- Geofences enforce owner, active rule, geometry, and large-polygon review.
- Spatial analytics enforce spatial index and aggregation privacy.
- AI location agents are first-class records with runtime, role, scope, policy,
  and disclosure.
- Generated package evidence is refreshed after contract changes.
