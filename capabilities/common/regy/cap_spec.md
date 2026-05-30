# API/Service Registry (REGY) Capability Summary

REGY provides APG applications with a governed registry control plane for
service registration, instance registration, service discovery, version
governance, gateway publication, retirement, and audit evidence.

## Runtime Shape

- `registry_runtime.py` is the dependency-light generated-app lifecycle runtime.
- `service.py` is the production-oriented async registry implementation.
- `api.py` exposes the Flask REST blueprint and generated-app helper functions.
- `views.py` keeps the legacy Flask-AppBuilder UI runtime.
- `view_models.py` exposes dependency-light UI data for generated apps.
- `app.py` derives semantic model and package evidence from the contract.

## Lifecycle

1. Register a service with tenant, owner, API version, schema, health endpoint,
   and routing metadata.
2. Register one or more service instances with endpoint, allowed region, health
   probe, health status, and load-balancing weight.
3. Discover services in tenant scope, preferring healthy instances.
4. Record versions and compatibility evidence for contract changes.
5. Publish only healthy registered services with routing metadata to gateway
   adapters.
6. Retire services only after impact review and gateway unpublish evidence.
7. Emit audit events for lifecycle decisions.

## Guardrails

REGY guardrails deny missing tenant context, owner, health endpoint, API
version, schema, unique name, instance endpoint, health probe, allowed region,
positive weight, gateway registration, healthy instance, routing metadata,
migration notes, future EOL date, health override incident reference,
retirement impact review, gateway unpublish evidence, and production tracing.

REGY guardrails require review for production registrations, high discovery
limits, breaking changes, and owner transfers.

## Adapter Boundaries

Generated-app REGY does not run a live service mesh, APG gateway, cache,
monitoring backend, audit sink, or Bytewax worker. Production adapters for
`auth`, `conf`, `moni`, `audl`, `apig`, `cach`, and Bytewax must honor REGY
decisions before side effects.
