# CVSN Capability Summary

CVSN is specified in the root capability documents:

- `README.md` explains generated-app usage, files, guardrails, and focused
  verification.
- `SPECIFICATION.md` defines the functional contract for the current lifecycle
  packet.
- `PLAN.md` records the implementation and review sequence.

The executable source of truth is `capability_contract.py`, supported by
`cvsn_runtime.py`, `view_models.py`, and `app.py`.

The current packet adds processing-review evidence to the first-class AI
vision-agent composition and Bytewax lifecycle guardrails. CVSN can now
represent governed agents from `codex`, `claude_code`, `opencode`, and `pi`;
expose agent, lifecycle, and processing-review UI models; reject lifecycle
mutations that are not routed through Bytewax; and retain review-required
vision jobs as `pending_review` records with matched rules and review reasons.
