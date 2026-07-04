# Rationale

## Decisions

- Write workflow journal events from `_record_ui_workflow_run()` so UI-generated runs have the same audit surface as runtime workflows.
- Add one journal entry for run start, each completed wizard step, record creation, and run completion.
- Preserve the existing `/workflows/runs/<id>/journal` endpoint and make the UI render the same event source.
- Show run context, payload snapshot, created-record snapshot, and timeline on the run detail page.
- Keep circuit breakers and event subscriptions in the debugger, but move run-specific evidence above them when a run is selected.

## Rejected Alternatives

- A client-side waterfall/trace library: rejected because the generated app must remain self-contained and this workspace can be solved with accessible server-rendered markup.
- Storing debugger-only run state: rejected because it would diverge from `WORKFLOW_RUNS` and `WORKFLOW_EVENT_JOURNAL`.
- Removing raw payload/record JSON: rejected because debuggers need exact evidence; details disclosures keep it available without dominating the page.
