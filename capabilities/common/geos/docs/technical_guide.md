# GEOS Technical Guide

GEOS exposes a dependency-light package facade for generated APG applications.
Use `GeosService` for local lifecycle execution and attach production providers
through adapters declared by `capability_contract.py`.

Current technical guarantees:

- tenant-qualified event source, geofence, event, territory, analytics, agent,
  and audit state;
- deterministic guardrail evaluation;
- Bytewax lifecycle stream metadata;
- framework-neutral view models;
- generated app self-test and semantic model evidence.

Provider integrations such as H3 indexes, route engines, map rendering, data
warehouses, and edge ingestion remain outside this local package proof.
