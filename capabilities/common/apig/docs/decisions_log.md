# APIG Decisions Log

## Current Packet Boundary

APIG uses a dependency-light generated-application control plane for package
composition and focused proof. Live gateway execution remains adapter-backed.

Decision: keep `gateway_runtime.ApigService` independent from reverse proxies,
Kubernetes, WebAssembly engines, cache stores, AI providers, and APG service
clients.

Rationale: generated applications need deterministic route, traffic, security,
deployment, retirement, and audit decisions before runtime side effects.

## View Model Convention

Decision: expose generated UI models from `view_models.py` and keep `views.py`
as a compatibility re-export.

Rationale: recent APG capability packets use `view_models.py` as the generated
application UI contract.

## Event Stream Adapter

Decision: use `bytewax` as the event stream adapter in the APIG contract.

Rationale: APG's current architecture direction is Bytewax rather than Kafka
for capability event flows.
