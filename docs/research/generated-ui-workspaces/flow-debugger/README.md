# Flow Debugger

Date: 2026-07-04

## Best-In-Class Patterns

- Temporal-style workflow history makes run metadata, event history, pending work, and related execution details visible from one run page.
- Airflow-style debugging gives operators quick run status, step/task state, and drill-ins to logs or details from the run list.
- Camunda Operate-style process monitoring emphasizes incidents, variables, and operational actions for stuck process instances.
- Trace tools such as Datadog use timelines and event/span detail to make execution order and context inspectable.

## Live Audit

Representative app: `examples/01_minimal_customer_records/output/app.py`.

Before server: `127.0.0.1:20903`.

Observed defects:

- Empty debugger state had no guidance beyond `No workflow runs yet`.
- A completed generated UI workflow showed only a plain step list with step numbers and status badges.
- The run detail did not show payload, created record, entity, event id, duration, or journal count.
- The workflow journal endpoint returned `events: []` for UI-created workflow runs, leaving no audit trail.
- Recent runs did not include entity context or created-record context.

After server: `127.0.0.1:20904`.

After verification:

- UI-created workflow runs append journal events for run start, each completed step, record creation, and run completion.
- Run detail renders summary metrics, run context, a step timeline, payload snapshot, created-record snapshot, and event journal.
- `/workflows/runs/workflow-run-1/journal` returns the same populated event sequence shown in the UI.
- The recent-runs table includes entity context and links directly to run detail.

## Fix List

Must-fix:

- Populate the workflow journal for generated UI wizard completions.
- Render a real run-detail page with timeline, context, payload, record, and journal information.
- Preserve the JSON journal endpoint as a useful audit source.

High-value polish:

- Add run summary metrics and duration.
- Add field lists and field counts per step.
- Keep snapshots and journal data inspectable but secondary through details/table affordances.

## Validation

- Regenerated all 20 numbered examples.
- Live after audit: `assets/after-debug-run.html` and `assets/after-journal.json`.
- Targeted tests: `3 passed` across flow debugger regression, template route coverage, and CSS class coverage.
- Full suite: `1484 passed, 1 skipped, 3 warnings in 729.14s`.
