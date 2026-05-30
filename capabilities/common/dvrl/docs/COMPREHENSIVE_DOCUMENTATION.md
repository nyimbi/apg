# DVRL Comprehensive Documentation

This document summarizes the current APG Data Virtualization packet. It should
be read with the root `README.md`, `SPECIFICATION.md`, and `PLAN.md`.

## Architecture

DVRL has two execution surfaces:

- `DVRLLifecycleService`: dependency-light generated-application lifecycle
  control plane.
- `DVRLService`: production-oriented federation runtime for adapter-backed
  connector, query, cache, NLP, Singer, and APG integration work.

## Lifecycle Records

- `DVRLSourceRecord`
- `DVRLSchemaRecord`
- `DVRLVirtualTableRecord`
- `DVRLQueryRecord`
- `DVRLCacheRecord`
- `DVRLPolicyRecord`
- `DVRLAuditEventRecord`

## Contract Sections

- `sources`
- `schemas`
- `queries`
- `cache`
- `governance`
- `optimization`
- `adapters`
- `ui`
- `theme`

## Rule Coverage

DVRL evaluates deterministic rules for tenant context, source ownership,
source type support, vaulted credentials, encrypted source connections,
activation approval, stale schema review, virtual table ownership,
classification, query parameterization, write-query blocking, restricted-data
RBAC, sensitive-result cache blocking, lineage capture, high-cost query review,
cross-source join review, row-limit enforcement, cache TTL enforcement, policy
review, and source retirement impact review.

## UI Models

`view_models.py` exposes generated UI data models for dashboard, sources,
schemas, virtual tables, query workbench, federation topology, cache, policies,
metrics, adapter health, audit, and settings.

## Verification

Focused verification should compile the DVRL Python files, run the two DVRL
contract test modules, run APG implementation audit, and run APG publish-plan
evidence. Full live connector, browser, performance, and Bytewax verification
are deferred to integration windows.
