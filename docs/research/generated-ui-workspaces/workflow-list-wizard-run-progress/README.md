# Workflow List, Wizard, And Run Progress

Date: 2026-07-04

## Best-In-Class Patterns

- Temporal Web UI makes workflow executions inspectable first: run state, metadata, history, and debugging access are central, not hidden behind a raw API.
- Airflow's UI treats recent runs and task state as the primary operational view, with fast scanning of state across runs and direct drill-in when a run needs attention.
- Camunda Operate emphasizes active/completed process visibility, incident resolution, and process variables, which maps to APG's generated workflow payloads and debugger.
- Linear keeps workflow status opinionated and low-friction: a clear default status flow, strong keyboard paths, and minimal ceremony around moving work forward.

## Live Audit

Representative app: `examples/01_minimal_customer_records/output/app.py`.

Before server: `127.0.0.1:20892`.

Observed defects:

- The wizard form for step 1 posted to `/step/1`, while the POST handler also incremented to the next step. Result: the user moved from step 1 directly to step 3.
- Completing the generated UI wizard did not create an entry in `WORKFLOW_RUNS`; `/workflows/runs` stayed empty and `/ui/debug` had no run to inspect.
- Completion could not link to a run or trace because no run id existed.
- Structured list/object fields entered as JSON text remained strings through record coercion, causing final validation failure for entities with required structured fields.
- The workflow list did not communicate run history or recent activity.

After server: `127.0.0.1:20894`.

After verification:

- The generated `Customer` wizard advanced through steps 1, 2, 3, 4, 5, and 6 in order.
- The final wizard POST created one run: `workflow-run-1`.
- The recorded run has a six-step trace and appears in `/workflows/runs`, `/ui/workflows`, and `/ui/debug/workflow-run-1`.
- Completion links now include the created record and run inspector.

## Fix List

Must-fix:

- Correct wizard POST action so each form submits the current step.
- Record successful UI wizard completions in the shared workflow run store.
- Parse array/object JSON strings during record coercion so wizard and form submissions can create structured fields.

High-value polish:

- Add run counts and a recent-runs panel to the workflow list.
- Add completion-state run metadata and direct links to the created record and debugger.

## Validation

- Regenerated all 20 numbered examples.
- Targeted tests: `UV_CACHE_DIR=$PWD/.uv-cache uv run pytest tests/test_generated_ui_dashboard.py -q` -> `8 passed in 15.99s`.
- Full-suite gate pending before commit.
