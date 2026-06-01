# API/Service Registry (REGY) Capability Summary

REGY provides APG applications with a governed registry control plane for
service registration, instance registration, service discovery, version
governance, gateway publication, first-class registry-agent composition,
Bytewax lifecycle batches, retirement, and audit evidence.
It also preserves durable review evidence so generated registry consoles can
compose pending queues and explain every allow, review, or denial decision.

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
6. Register governed AI and automation agents with runtime, role, scope, owner,
   purpose, contribution disclosure, and human approval for privileged roles.
7. Validate registry lifecycle mutation batches through Bytewax before adapter
   side effects.
8. Retire services only after impact review and gateway unpublish evidence.
9. Preserve policy decisions, matched rules, review reasons, and review
   evidence on lifecycle records, pending queues, and audit events.
10. Emit audit events for lifecycle decisions.

## Guardrails

REGY guardrails deny missing tenant context, owner, health endpoint, API
version, schema, unique name, instance endpoint, health probe, allowed region,
positive weight, gateway registration, healthy instance, routing metadata,
migration notes, future EOL date, health override incident reference,
retirement impact review, gateway unpublish evidence, and production tracing.
They also deny unsupported registry-agent runtimes, unsupported agent roles,
missing agent scope, missing owner, missing purpose, missing contribution
disclosure, and non-Bytewax registry lifecycle batches.

REGY guardrails require review for production registrations, high discovery
limits, breaking changes, owner transfers, and privileged registry-agent roles
that lack human approval evidence.

## Adapter Boundaries

Generated-app REGY does not run a live service mesh, APG gateway, cache,
monitoring backend, audit sink, external AI runtime, or Bytewax worker.
Production adapters for `auth`, `conf`, `moni`, `audl`, `apig`, `cach`,
agent runtimes, and Bytewax must honor REGY decisions before side effects.
