# MQEB Backlog

MQEB's current coherent packet is the dependency-light event fabric with
first-class event-agent composition and Bytewax lifecycle validation.

## Next Useful Slices

- Add adapter implementations that run MQEB lifecycle batches through real
  Bytewax workers while preserving the dependency-light package contract.
- Add schema-registry adapter tests for regulated topic publish decisions.
- Add dead-letter triage workflows that can assign an approved event agent and
  record operator evidence.
- Add generated UI screens backed by `view_models.py` for event-agent roster,
  lifecycle batches, replay, quota review, and delivery reliability.
- Add cross-capability examples showing MQEB composed with AUTH, AUDL, SECU,
  ENCR, KEYM, and CONF.

Keep each future item as a specification-first, plan-first, implementation,
review, and focused-verification packet.
