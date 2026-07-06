# Rationale

## Decision

Ship a workflow intelligence layer inside `workflow_wizard.html.j2` with live duration estimates, rollback links, save-as-template, and a per-step estimate ledger. Compute the context from existing workflow steps and `WORKFLOW_RUNS` traces.

## Why this beats the benchmark

Temporal, Retool, and Zapier are strong at workflow visibility, but they do not automatically blend schema-driven data entry with runtime run intelligence in a generated offline UI. APG can because the compiler owns the wizard, entity fields, generated run records, and local browser behavior.

## Rejected alternatives

- Server-side template library: rejected to avoid introducing new persistence and permission policy.
- True compensation rollback from the wizard: rejected because rollback execution belongs to the debug/run surface where compensations are visible.
- Branch graph canvas: rejected because current generated workflows are sequential; a ledger is clearer and lighter.

## Validation target

Generated workflow HTML must still advance sequentially, record completed runs, and expose the existing Inspect/Open actions while adding duration estimates, rollback links, save-as-template, and an estimate ledger.
