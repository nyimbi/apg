# Digital Forms and eSign Capability Runtime Spec

The active capability specification is `SPECIFICATION.md`.

This file remains as a compatibility pointer for APG tooling and older package
readers that still look for `cap_spec.md`. Runtime behavior is defined by:

- `capability_contract.py`
- `models.py`
- `signing_engine.py`
- `service.py`
- `api.py`
- `views.py`
- `app.py`

The current packet includes first-class signing-agent metadata and Bytewax
lifecycle-batch stream metadata in the executable contract.

Use the focused verification commands in `PLAN.md` after changing this package.
