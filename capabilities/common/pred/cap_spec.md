# PRED Capability Summary

PRED is specified in the root capability documents:

- `README.md` explains generated-app usage, files, guardrails, and focused
  verification.
- `SPECIFICATION.md` defines the functional contract for the current lifecycle
  packet.
- `PLAN.md` records the implementation and review sequence.

The executable source of truth is `capability_contract.py`, supported by
`service.py`, `predictive_runtime.py`, `views.py`, and `app.py`.

The current packet adds review-evidence lifecycle state to the first-class AI
prediction-agent composition and Bytewax lifecycle guardrails. PRED can now
represent governed agents from `codex`, `claude_code`, `opencode`, and `pi`;
expose agent, lifecycle, and pending-review UI models; reject lifecycle
mutations that are not routed through Bytewax; and retain review-required
models, feature sets, forecasts, and drift reports as `pending_review` records
with matched rules and review reasons.
