# CONF Capability Specification Pointer

The authoritative APG Configuration Management specification is
`SPECIFICATION.md` in this directory.

Use these package files as the source of truth for generated applications and
APG composition:

- `README.md`
- `SPECIFICATION.md`
- `PLAN.md`
- `capability_contract.py`
- `models.py`
- `service.py`
- `api.py`
- `views.py`
- `app.py`
- `semantic_model.json`
- `package_manifest.json`
- `release_report.json`

The active CONF packet covers tenant-scoped configuration records, governed
change approval, production deployment guardrails, drift remediation review,
configuration review agents, compact UI theming, deterministic rules, and
Bytewax lifecycle stream metadata.

The current review-evidence packet also covers durable pending-review and deny
evidence for production changes, drift remediations, privileged configuration
agents, configuration lifecycle batches, and audit events.
