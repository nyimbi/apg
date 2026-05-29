# Data Virtualization Capability Specification

- **Capability Name**: Data Virtualization
- **Capability ID**: `dvrl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

DVRL is APG's package-backed data virtualization capability. It provides
tenant-scoped virtual source registration, federated query parsing and planning,
schema discovery, source adapters, Singer tap integration, natural-language
query assistance, APG service integration, connection health handling,
governance rule evaluation, UI route metadata, semantic-model publication, and
publish-plan evidence.

The package is not only a generated contract shell: `service.py`, `models.py`,
`connectors.py`, `adapters.py`, `singer_integration.py`, `nlp_integration.py`,
`apg_integrations.py`, `error_handling.py`, and `real_implementations.py`
provide executable runtime behavior for data-source management, federated query
analysis, connector orchestration, cache metadata, lineage capture, integration
boundaries, and production validation.

## Provided Services

- `dvrl_operations`

## Required Services

- `tenant_context`
- `keym` or an equivalent credential vault for source secrets
- `auth`/RBAC for restricted query authorization
- `audl` for query and source audit trails
- `cach` for query result cache integration when enabled

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `source_registration_requires_credentials`
- `restricted_query_requires_rbac`
- `sensitive_results_block_cache`
- `query_requires_lineage_capture`
- `high_cost_query_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The UI contract covers the DVRL dashboard, query workbench, virtual source
manager, schema browser, federation map, policy console, metrics, and settings
surfaces.

## Theme

The package uses the `dvrl_federation_console` APG theme contract.

## External Runtime Boundary

DVRL keeps live database, SaaS, object-store, streaming, and Singer tap
connections behind connector/adaptor boundaries. Capability tests and publish
evidence exercise deterministic package behavior without requiring live
credentials or external systems. Production deployments can bind the connector
manager to PostgreSQL, MySQL, MongoDB, Snowflake, BigQuery, S3, REST, GraphQL,
Bytewax, Singer, or other configured source adapters through APG configuration
and credential-vault services.
