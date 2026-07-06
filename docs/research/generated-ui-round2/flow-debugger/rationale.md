# Flow Debugger Rationale

## Decisions

- Named Temporal Web UI as the leader because durable workflow debugging depends on workflow execution metadata, event history, and replay/reset-oriented operations.
- Implemented replay as a derived UI rail instead of a backend replay engine. APG generated runs already store step trace, duration, touched fields, and journal events, which is enough to make an operator-facing replay view truthful.
- Implemented breakpoints as local investigation state. This avoids changing generated runtime contracts while still giving users the breakpoint workflow missing from the current debug page.
- Implemented variable inspection from payload, record, event id, run id, and workflow identifiers. These variables are stable across generated examples and are the values most often inspected when debugging form-backed workflow runs.

## Rejected alternatives

- Full deterministic replay engine: rejected because generated APG workflow traces are not SDK command histories and pretending otherwise would be misleading.
- New JavaScript dependency for debugger controls: rejected by the no-dependency and JS budget ground rules.
- Server-mutating breakpoint persistence: rejected because it expands the generated API surface and is unnecessary for the current operator task.
- Log search panel: rejected for this pass because APG journal events are already short and structured; replay plus variables gives a stronger differentiator.

## Budget note

The implementation uses template-local vanilla JavaScript only. It adds no generated Python dependencies and no external runtime URLs.
