# Audit Management

## Overview

Audit Management provides a world-class, standalone-deployable implementation of audit management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`grc_aud`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| `audit_program_lifecycle` | Audit Program Lifecycle workflow |
| `audit_finding_lifecycle` | Audit Finding Lifecycle workflow |
| `audit_evidence_workflow` | Audit Evidence Workflow workflow |
| `audit_report_workflow` | Audit Report Workflow workflow |
| `audit_dashboard_service` | Audit Dashboard Service workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `mten` | Mten services |
| `conf` | Conf services |
| `ntfy` | Ntfy services |


## Installation

```bash
pip install apg-grc-aud
```

## Standalone Usage

```python
from apg_grc_aud import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # grc_aud
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-grc-aud --port 8080

# With PostgreSQL persistence
apg-grc-aud --db-url postgresql+asyncpg://user:pass@localhost/aud --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-aud/dashboard` | `grc_aud:view` |
| audits | `/grc-aud/audits` | `grc_aud:manage_audits` |
| audit_detail | `/grc-aud/audits/:id` | `grc_aud:view` |
| findings | `/grc-aud/findings` | `grc_aud:manage_findings` |
| finding_detail | `/grc-aud/findings/:id` | `grc_aud:view` |
| evidence | `/grc-aud/evidence` | `grc_aud:manage_evidence` |
| reports | `/grc-aud/reports` | `grc_aud:manage_reports` |
| calendar | `/grc-aud/calendar` | `grc_aud:view` |


## HTTP Endpoints

```
GET  /health           Liveness probe
GET  /contract         Full capability contract JSON
POST /evaluate         Evaluate governance rules
GET  /api/v1/...       Domain-specific REST API
```

## Composability

This capability integrates with the APG platform via the `apg.capabilities` entry-point group. It is auto-discovered by the capability registry when installed.

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["grc_aud"].contract
```

## Development

```bash
# Run tests
pytest tests/ -q

# Build wheel
python -m build --wheel .

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## License

Proprietary — © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## World-Class Enhancements (v2.0)

- **I1.** Audit Management — World-Class Improvements
- **I2.** AI-Driven Risk Scoring for Audit Universe Prioritisation
- **I3.** Continuous Control Monitoring (CCM) Integration
- **I4.** Structured Finding Root-Cause Taxonomy
- **I5.** Remediation SLA Escalation Engine
- **I6.** Sampling Engine with Statistical Confidence Intervals
- **I7.** Dual-Approval Workpaper Sign-Off with Digital Signatures
- **I8.** Benchmark Comparative Analytics Against Industry Peers
- **I9.** Automated Regulatory Change Impact Assessment
- **I10.** Heatmap-Ready Risk Matrix for Executive Dashboards
- **I11.** Cross-Engagement Finding Correlation and Systemic Risk Detection
- **I12.** Whistleblower Case Management with Chain-of-Custody Tracking
- **I13.** Engagement Time-Budget Tracking with Earned-Value Analysis
- **I14.** Integrated Control Testing Library with Test Steps
- **I15.** Audit Report Version Control with Diff Tracking

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
