# ONTO Capability Specification Pointer

The active ONTO specification is `SPECIFICATION.md`.

The executable APG contract is `capability_contract.py`. The dependency-light generated-app runtime is implemented by `service.OntoService`, with helper functions in `ontology_runtime.py`, endpoint helpers in `api.py`, UI models in `views.py`, and package evidence in `app.py`. ONTO also owns provider-neutral ontology-agent composition and Bytewax lifecycle-batch validation contracts.

Review-required lifecycle outcomes are durable: duplicate terms, breaking curation requests, deprecations, low-confidence mappings, issue-bearing validation reports, and privileged ontology-agent registrations persist as `pending_review` records with `decision`, `matched_rules`, `review_reasons`, and `audit_evidence`. Hard deny outcomes still raise `PermissionError`.
