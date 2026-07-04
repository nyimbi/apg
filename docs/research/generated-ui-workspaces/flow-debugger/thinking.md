# Raw Reasoning

The flow debugger should explain a workflow run without requiring the user to reconstruct it from a record and a step count. A useful debugger needs run identity, status, timeline, event history, payload, outputs, and failure or circuit-breaker context.

The before-state had enough structure to be a launch point, but the important evidence was missing. UI-created workflow runs were recorded in `WORKFLOW_RUNS`, yet no journal events were written. That made the `/workflows/runs/<id>/journal` endpoint effectively dead for the very workflow path users are most likely to exercise from the generated UI.

Best references converge on chronology plus context. Temporal exposes workflow history and metadata; Airflow makes task state and logs the main path for debugging; Camunda Operate focuses on incidents and variables; trace UIs use ordered event/spans to make execution understandable.

The strict fix is to write journal events at the same time the generated UI wizard creates a run, then render those events in the debugger. This keeps the existing API contract and avoids a separate debug-only data model. The event sequence is deterministic: `run_started`, one `step_completed` per wizard step, `record_created`, and `run_completed`.

Rejected: creating a new workflow-run persistence subsystem. The generated app already has `WORKFLOW_RUNS` and `WORKFLOW_EVENT_JOURNAL`; this pass should connect them. Rejected: adding a client-side trace visualization dependency. A server-rendered timeline and table are sufficient, accessible, and self-contained.
