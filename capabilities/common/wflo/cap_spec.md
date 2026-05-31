# Workflow Orchestration Capability Runtime Spec

The active capability specification is `SPECIFICATION.md`.

This file remains as a compatibility pointer for APG tooling and older package
readers that still look for `cap_spec.md`. Runtime behavior is defined by:

- `capability_contract.py`
- `models.py`
- `workflow_runtime.py`
- `service.py`
- `api.py`
- `views.py`
- `app.py`

The current packet adds first-class provider-neutral workflow agents and
Bytewax lifecycle-batch guardrails on top of the existing workflow definition,
execution, task, approval, compensation, event, audit, UI, and theming runtime.

Use the focused verification commands in `PLAN.md` after changing this package.
