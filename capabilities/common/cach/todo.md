# CACH Backlog

CACH's current coherent packet is the dependency-light cache governance control
plane with first-class cache-agent composition and Bytewax lifecycle validation.

## Next Useful Slices

- Add a live Bytewax adapter that executes validated lifecycle batches.
- Add Redis and Valkey adapter contract tests that prove cache backends honor
  CACH admission, TTL, encryption, freshness, and invalidation decisions.
- Add persistent storage for namespace policies, review queues, cache-agent
  registrations, and audit events.
- Add generated UI screens backed by `view_models.py` for cache agents,
  lifecycle batches, warming, eviction, namespace policy, and adapter health.
- Add cross-capability examples composing CACH with AUTH, AUDL, CONF, MQEB,
  MONI, SECU, ENCR, and KEYM.

Keep each future item as a specification-first, plan-first, implementation,
review, and focused-verification packet.
