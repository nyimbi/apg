# Rationale

## Decisions

- Use the existing `WORKFLOW_RUNS` store for generated UI wizard completions. This keeps `/workflows/runs`, `/ui/debug`, persistence, and future live bindings aligned.
- Store UI wizard runs with the same core keys as DSL workflow runs: `id`, `workflow`, `status`, `steps`, `completed_steps`, `pending_steps`, `trace`, `payload`, and `record`.
- Keep POST advancement in `_ui_workflow_step_post()` and render form actions to the current step. This removes the double-increment bug without changing URL structure.
- Parse JSON strings for array/object field types in `_coerce_value_for_type()`. This benefits workflow submissions and the generated create/edit form paths that submit structured textareas.
- Add workflow-list run counts and recent-runs drill-ins instead of adding another run-history page.

## Rejected Alternatives

- Calling `run_workflow()` from the UI wizard: rejected because generated UI workflows are entity-field wizards, not DSL workflow declarations.
- Recording only a browser-side progress event: rejected because it would still leave `/workflows/runs` and `/ui/debug` empty after refresh.
- Adding a second workflow-run store just for UI wizards: rejected because it would split debugger behavior and complicate persistence.
- Leaving structured field parsing to each form handler: rejected because `coerce_record_types()` is already the shared typed boundary.
