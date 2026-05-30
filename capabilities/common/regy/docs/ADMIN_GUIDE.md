# REGY Administrator Guide

REGY administrators manage registry policy, adapter wiring, audit evidence,
and lifecycle exceptions.

## Required Controls

- Tenant context is required for every operation.
- Service registrations require owner, health endpoint, API version, and schema
  evidence.
- Production services require production review and trace propagation evidence.
- Instances require endpoint, health probe, allowed region, and positive
  weight.
- Gateway publication requires a registered service, healthy instance, and
  routing metadata.
- Service retirement requires impact review and gateway unpublish evidence.

## Adapter Responsibilities

Production adapters for `auth`, `conf`, `moni`, `audl`, `apig`, `cach`, and
Bytewax must call REGY guardrails before side effects. If an adapter is
unavailable, generated-app REGY can still produce deterministic lifecycle
records, but live external effects must be treated as not executed.

## Review Queues

The generated-app runtime records review objects for production registration,
compatibility review, and high discovery result limits. Production UIs should
surface those records with reviewer, decision, notes, and audit evidence.

## Operations

Use `registry_summary()` and the UI view models in `view_models.py` for
lightweight operational state. Use `list_audit_events()` for package-local
decision evidence.
