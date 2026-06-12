# APG Dark Web Monitoring

`intel_darkweb` is an executable APG capability for lawful, defensive
dark-web-monitoring workflows. It can be composed into generated APG
applications that need exposure monitoring, fraud-market intelligence,
brand-protection review, threat actor tracking, incident response, or
compliance evidence.

## What It Provides

- Authority, monitoring program, source, observation, exposure indicator,
  marketplace risk, threat actor, referral, dissemination, review, and AI-agent
  workflows.
- Deterministic rules that enforce tenant context, lawful authority, evidence,
  access review, approvals, Bytewax lifecycle routing, and AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/darkweb/app.py
./.venv/bin/pytest -q capabilities/intel/darkweb/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_darkweb --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.darkweb import DarkWebMonitoringService

service = DarkWebMonitoringService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "security_monitoring_authority",
    "incident-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is defensive and compliance-first. It does not implement live
network access, marketplace participation, credential use, exploit procurement,
contraband transactions, evasion, account automation, identity resolution, or
doxxing workflows. AI-agent actions that request those scopes are denied by the
rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/darkweb/*.py capabilities/intel/darkweb/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/darkweb/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/darkweb --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/darkweb --json
```

---

## World-Class Enhancements (v2.0)

- **I1.** Dark Web Intelligence — World-Class Improvements
- **I2.** Streaming Intelligence Pipeline (Bytewax integration)
- **I3.** Graph-Based Threat Actor Attribution
- **I4.** Onion Address Active Probing with Tor Circuit Rotation
- **I5.** Automated Credential Deduplication via k-Anonymity
- **I6.** MITRE ATT&CK TTP Enrichment
- **I7.** Cryptocurrency Transaction Monitoring
- **I8.** LLM-Assisted Threat Summarisation
- **I9.** Automated STIX 2.1 Export
- **I10.** Confidence Score Bayesian Updating
- **I11.** Parallel Onion Network Crawl Scheduler
- **I12.** Brand Impersonation Detection
- **I13.** Geopolitical Risk Context Layer
- **I14.** Time-Series Anomaly Detection on Monitor Metrics
- **I15.** Secure Evidence Chain of Custody

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
