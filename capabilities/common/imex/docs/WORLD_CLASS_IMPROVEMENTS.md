# IMEX Improvement Backlog

This file records forward-looking improvement areas for IMEX. The current
verified packet is the governed transfer lifecycle in `imex_runtime.py`; the
items below require separate implementation and proof before they should be
described as delivered behavior.

## Candidate Enhancements

- Adaptive schema drift detection tied to source profile history.
- Bytewax-backed transfer event stream for live progress and replay signals.
- ETLP plan generation from mapping profiles.
- Policy-aware export destination approvals.
- Connector health gating through CONN before transfer execution.
- Retention policy reconciliation against artifact stores.
- Transfer cost estimates before execution.
- Incremental checkpoint persistence across process restarts.
- Browser-rendered transfer designer verification.
- Load and throughput benchmarks on representative datasets.

Each enhancement should add contract rules, runtime behavior, UI model support,
focused tests, and progress-log evidence when implemented.
