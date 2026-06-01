# NLPC Capability Summary

NLPC is specified in the root capability documents:

- `README.md` explains generated-app usage, files, guardrails, and focused
  verification.
- `SPECIFICATION.md` defines the functional contract for the current lifecycle
  packet.
- `PLAN.md` records the implementation and review sequence.

The executable source of truth is `capability_contract.py`, supported by
`nlpc_runtime.py`, `view_models.py`, and `app.py`.

Current packet additions:

- Processing review governance that stores low-confidence and
  budget-incomplete runs as `pending_review` evidence with `decision`,
  `matched_rules`, and `review_reasons`.
- Review queues in processing and human-review view models.
- First-class NLP agents for provider-neutral text-governance roles.
- Runtime codes for Codex, Claude Code, OpenCode, and Pi through AICR adapter
  contracts.
- Bytewax `nlpc.lifecycle` batch validation with accepted and denied evidence.
- UI route metadata for `/nlpc/agents` and `/nlpc/lifecycle`.
- Deterministic rules for NLP-agent registration and lifecycle stream
  guardrails.
