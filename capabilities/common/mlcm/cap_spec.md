# MLCM Capability Summary

AI Model Lifecycle Management governs tenant-scoped model registration,
versioning, evaluation, promotion, deployment, monitoring, drift response,
rollback, retirement, audit evidence, UI composition, and theming.
It also treats model lifecycle agents and Bytewax lifecycle batches as
first-class composition state.

The full packet definition is maintained in `SPECIFICATION.md`; the execution
sequence is maintained in `PLAN.md`.

## Current Packet

- Runtime: `service.MlcmService`
- API helpers: `api.py`
- UI view models: `views.py`
- Contract: `capability_contract.py`
- Package evidence: `app.py`, `semantic_model.json`, `package_manifest.json`,
  and `release_report.json`
- Event stream adapter: Bytewax
- First-class lifecycle agents: `codex`, `claude_code`, `opencode`, `pi`
- Lifecycle processor: Bytewax

## Lifecycle

`model -> version -> evaluation -> promotion -> deployment -> monitoring ->
drift review -> rollback -> retirement -> model lifecycle agent -> Bytewax
lifecycle batch -> audit`

## Guardrail Additions

- Version lineage enforcement: missing artifacts and non-development model-card
  gaps are denied; missing training or baseline lineage becomes pending-review
  evidence with matched rule names.
- Evaluation evidence enforcement: missing baselines are denied; missing
  evidence, fairness review, or explainability review becomes pending-review
  evidence with matched rule names.
- Durable review evidence: pending versions, pending evaluations, privileged
  lifecycle agents, denied lifecycle batches, and audit events expose matched
  rules, review reasons, and audit evidence.
- Model lifecycle-agent runtime, role, scope, owner, purpose, contribution
  disclosure, and privileged approval status.
- Bytewax lifecycle batch processing for model, version, evaluation, promotion,
  deployment, drift, rollback, retirement, and agent batches.

Generated applications can compose review queues from `list_pending_reviews()`
or from dashboard, version manager, evaluation console, lifecycle-agent roster,
lifecycle-batch, and governance view models without re-running policy rules.
