# APG Data Correlation

`intel_correlation` is an executable APG capability for governed,
evidence-backed cross-source data correlation. It can be composed into
generated APG applications that need entity resolution, link analysis, fraud
correlation, threat correlation, public-safety correlation, incident
correlation, or operational correlation.

## What It Provides

- Authority, workspace, source, entity, observation, rule, run, cluster,
  decision, referral, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, source
  lineage, evidence, thresholds, approvals, Bytewax lifecycle routing, and
  AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/correlation/app.py
./.venv/bin/pytest -q capabilities/intel/correlation/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_correlation --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.correlation import DataCorrelationService

service = DataCorrelationService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "mission_order",
    "correlation-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is evidence-led and compliance-first. It does not implement
unapproved identity merge, source tampering, privacy bypass, evidence
fabrication, autonomous referral, unreviewed high-impact matches, or
cross-tenant correlation. AI-agent actions that request those scopes are denied
by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/correlation/*.py capabilities/intel/correlation/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/correlation/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/correlation --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/correlation --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** World-Class Improvements: intel_correlation
- **I2.** Streaming Correlation Pipeline (Bytewax Integration)
- **I3.** Graph-Native Entity Resolution with NetworkX / igraph
- **I4.** Jaccard + MinHash Approximate Similarity for Text References
- **I5.** Temporal Pattern Mining with TFIDF-like Burst Detection
- **I6.** Attribution Chain Tracking (Kill-Chain Mapping)
- **I7.** Adaptive Confidence Decay (Staleness Weighting)
- **I8.** Cross-Tenant Federated Correlation (Privacy-Preserving)
- **I9.** Causal Bayesian Network Scoring
- **I10.** Anomaly Detection on Correlation Score Distributions
- **I11.** Rule Hot-Reload Without Service Restart
- **I12.** Distributed Locking for Concurrent Cluster Merge
- **I13.** Explainability Layer: Correlation Rationale Generation
- **I14.** Pagination and Cursor-Based Result Streaming
- **I15.** Persistent Storage Adapter (PostgreSQL via asyncpg)

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
