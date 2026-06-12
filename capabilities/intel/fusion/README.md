# APG Intelligence Fusion

`intel_fusion` is an executable APG capability for lawful, evidence-led
intelligence fusion. It can be composed into generated APG applications that
need cross-source operational pictures, threat fusion, fraud fusion,
public-safety fusion, incident fusion, strategic assessments, or analyst
workspaces.

## What It Provides

- Authority, workspace, source, artifact, correlation, hypothesis, assessment,
  referral, dissemination, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, source
  lineage, evidence, approvals, Bytewax lifecycle routing, and AI-agent
  guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/fusion/app.py
./.venv/bin/pytest -q capabilities/intel/fusion/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_fusion --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.fusion import IntelligenceFusionService

service = IntelligenceFusionService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "mission_order",
    "fusion-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is evidence-led and compliance-first. It does not implement
source tampering, evidence fabrication, privacy bypass, unsupported identity
resolution, autonomous dissemination, unapproved attribution, or cross-tenant
fusion. AI-agent actions that request those scopes are denied by the rule
engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/fusion/*.py capabilities/intel/fusion/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/fusion/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/fusion --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/fusion --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** Intelligence Fusion — World-Class Improvements
- **I2.** Probabilistic Fusion with Dempster-Shafer Theory
- **I3.** Temporal Decay Model for Intelligence Staleness
- **I4.** Structured Analytic Technique — Red Team / Devil's Advocate Automation
- **I5.** Cross-Domain Semantic Deduplication
- **I6.** Automated Assessment Quality Scoring Pipeline
- **I7.** Intelligence Gap Tracking
- **I8.** Streaming Event Replay and Audit Trail
- **I9.** Multi-Hypothesis Conflict Resolution Protocol
- **I10.** Source Reliability and Information Credibility (SRCC) Framework
- **I11.** Product Versioning and Lineage Tracking
- **I12.** Geospatial Fusion Support
- **I13.** Confidence Decay on Challenge Events
- **I14.** Batch Ingestion with Deconfliction
- **I15.** Analyst Performance Metrics

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
