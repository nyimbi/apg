# Financial Intelligence — User Guide

**Capability ID**: `intel_finint` | **Domain**: `intel` | **Version**: `2.0.0`

## Overview

`intel_finint` is the APG governed financial-intelligence capability. It provides
transaction monitoring, beneficial ownership tracing, financial network analysis,
sanctions/PEP screening, AML compliance, and case lifecycle management in a
tenant-scoped, policy-enforced service layer.

All async methods are safe to call concurrently with `asyncio.gather`. Every
state-changing operation emits to the `apg.intel.finint.lifecycle` Bytewax stream.

## Installation

```bash
pip install apg-intel-finint
```

## Quick Start

```python
import asyncio
from capabilities.intel.finint import FinancialIntelligenceService

svc = FinancialIntelligenceService(tenant_id="acme", actor_id="analyst-1")

authority = svc.record_authority(
    "auth-1", "acme", "regulatory_authority",
    "scope://aml", "confidential",
    "approver-1", "2027-12-31", "evidence://auth-1",
)
svc.register_source("src-1", "acme", "bank_feed", "KE", "owner-1", "auth-1", "evidence://src-1")
svc.record_subject("subj-1", "acme", "individual", "ref://subj-1", "high", "auth-1", "evidence://subj-1")
svc.record_transaction("tx-1", "acme", "src-1", "subj-1", "ref://tx-1", 9500.00, "USD", "cash_deposit", "2026-06-01T10:00:00Z", "evidence://tx-1")

async def investigate():
    return await asyncio.gather(
        svc.pep_screening("subj-1", "Jane Doe", "KE"),
        svc.placement_detection("subj-1"),
        svc.layering_detection("subj-1", lookback_days=30),
        svc.aml_compliance_check("subj-1"),
        svc.currency_exposure_report("subj-1"),
    )

asyncio.run(investigate())
```

## Core Workflow

Every FININT operation requires the authority chain first. The service enforces
`source_subject_authority_match` — source and subject must reference the same authority.

After transactions are recorded: patterns -> risk -> referral/dissemination.

## Case Management FSM

```
OPEN -> UNDER_REVIEW -> ESCALATED -> SAR_FILED -> CLOSED
                    \                            /
                     DISMISSED
```

```python
await svc.case_lifecycle_transition(
    "case-42", "OPEN", "UNDER_REVIEW", "analyst-1", "High-risk pattern detected"
)
```

Attempting an invalid transition raises `ValueError`.

## Detection Methods

### Placement Detection

```python
result = await svc.placement_detection("subj-1")
# placement_suspected if placement_score >= 0.25
# indicators: STRUCTURING_NEAR_CTR_THRESHOLD, HIGH_CASH_PLACEMENT_VOLUME, etc.
```

### Layering Detection

```python
result = await svc.layering_detection("subj-1", lookback_days=30)
# layering_suspected if layering_score >= 0.5
# indicators: DUPLICATE_AMOUNT_TRANSFERS, MULTI_CURRENCY_CONVERSION_CHAIN, etc.
```

### Illicit Finance Screening

```python
result = await svc.illicit_finance_detection(["tx-1", "tx-2", "tx-3"])
# Checks structuring near 10k/15k/50k thresholds, round amounts, velocity
```

### Typology Matching

```python
result = await svc.typology_match(
    ["tx-1", "tx-2"],
    typology_codes=["SMURFING", "HAWALA", "CRYPTO_ML"],
)
# composite_risk_score normalised across all typologies
```

Full typology codes: `SMURFING`, `CUCKOO_SMURFING`, `ROUND_TRIPPING`,
`LOAN_BACK`, `PAYABLE_THROUGH_ACCOUNTS`, `TRADE_BASED_ML`, `REAL_ESTATE_ML`,
`CRYPTO_ML`, `HAWALA`, `SHELL_COMPANY_ML`.

### Velocity Analysis

```python
result = await svc.transaction_velocity_analysis("subj-1", window_hours=24)
# transaction_rate_per_hour, amount_coefficient_of_variation, velocity_flags
```

## Screening Methods

### PEP Screening

```python
result = await svc.pep_screening("subj-1", "Jane Doe", "KE")
# pep_category: DIRECT_PEP | RCA | HISTORICAL_PEP | NO_PEP_MATCH
# enhanced_due_diligence_required: bool
```

### Wire Transfer Screening (FATF R.16)

```python
result = await svc.wire_transfer_screening(
    "tx-1", "Alice Corp Ltd", "Bob International",
    correspondent_bank="CorrespondentX",
)
# hold_required: True if any SANCTIONS_HIT deficiency
```

### Correspondent Bank Risk

```python
result = await svc.correspondent_bank_risk("corr-1", "AF", aml_rating="UNSATISFACTORY")
# recommendation: STANDARD_MONITORING | ENHANCED_MONITORING | TERMINATE
```

### AML Compliance

```python
result = await svc.aml_compliance_check("subj-1")
# compliance_flags: PEP_MATCH | SANCTIONS_LIST_HIT | ADVERSE_MEDIA_DETECTED | HIGH_RISK_TIER
```

### Bulk Subject Screening

```python
result = await svc.bulk_subject_risk_screening(["subj-1", "subj-2"])
# per-subject: risk_tier, sanctions_flag, pep_flag, high_risk
```

## Ownership & Network Analysis

```python
trace = await svc.beneficial_ownership_trace("entity-1")
# ubo_confidence, high_opacity, fatf_jurisdictions

result = await svc.beneficial_ownership_compliance("entity-1")
# compliant: bool; compliance_issues list

result = await svc.shell_company_identification("entity-1")
# is_likely_shell if shell_score >= 0.5

result = await svc.financial_network_map(["e1", "e2", "e3"])
# adjacency, edge_volumes, density

result = await svc.asset_tracing("subj-1")
# unexplained_wealth_flag (>500k for individuals)
```

## Currency & Correspondent Banking

```python
result = await svc.currency_exposure_report("subj-1")
# herfindahl_hirschman_index (1.0 = single currency)
# hawala_corridor_exposure_pct; high_hawala_exposure if >= 30%

result = await svc.hawala_detection([
    {"amount": 5000, "currency": "AED", "counterparty_jurisdiction": "AF",
     "settlement_mechanism": "cash", "offsetting_ref": "ref-42"},
])
# hawala_suspected if hawala_score >= 0.3
```

## Reporting & Audit

```python
sar = await svc.suspicious_activity_report("subj-1", "12 structured deposits over 30 days")
# sar_type: ABBREVIATED | FULL

report = await svc.finint_report("case-42")
# summary: total_transactions, high_risk_subjects, illicit_finance_alerts

bulletin = await svc.financial_intelligence_bulletin("2026-Q2")
# top_risk_types, hawala_suspected_cases, sanctions_evasion_cases

referral = await svc.inter_agency_referral("case-42", "FIU-NAIROBI", priority="URGENT")
# priority: ROUTINE | PRIORITY | URGENT | FLASH

verify = await svc.audit_chain_verify()
# chain_intact: bool; broken_link_count; terminal_hash
```

## Provides

- `finint_authority_workflow`
- `finint_source_workflow`
- `finint_subject_workflow`
- `finint_transaction_workflow`
- `finint_pattern_workflow`
- `finint_placement_detection`
- `finint_layering_detection`
- `finint_pep_screening`
- `finint_wire_screening`
- `finint_typology_matching`
- `finint_case_lifecycle`
- `finint_correspondent_bank_risk`
- `finint_currency_exposure`
- `finint_audit_chain_verify`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-finint/dashboard` | `intel_finint:view` | Overview |
| `/intel-finint/authorities` | `intel_finint:authorities` | Governance |
| `/intel-finint/sources` | `intel_finint:sources` | Data |
| `/intel-finint/subjects` | `intel_finint:subjects` | Data |
| `/intel-finint/transactions` | `intel_finint:transactions` | Intelligence |
| `/intel-finint/patterns` | `intel_finint:patterns` | Analysis |
| `/intel-finint/risk` | `intel_finint:risk` | Analysis |
| `/intel-finint/referrals` | `intel_finint:referrals` | Release |

## Error Handling

| Exception | Cause |
|-----------|-------|
| `PermissionError` | Policy rule denied (missing authority, tenant mismatch) |
| `ValueError` | Invalid FSM transition, unsupported enum, constraint violation |
| `KeyError` | Referenced entity not found in tenant store |
| `AssertionError` | Required parameter absent or out of range |

## Configuration

Tenant-scoped. Set via `conf` capability or `INTEL_FININT_` env vars.

| Key | Default | Description |
|-----|---------|-------------|
| `INTEL_FININT_CTR_THRESHOLD` | `10000` | Cash Transaction Report threshold (USD) |
| `INTEL_FININT_HAWALA_SCORE_THRESHOLD` | `0.3` | Minimum score to flag hawala |
| `INTEL_FININT_VELOCITY_WINDOW_HOURS` | `24` | Default velocity analysis window |

## Interoperability

```apg
use intel_finint;
```

Composes with: `auth`, `audl`, `ntfy`, `nlpc`, `grph`.

## Further Reading

- `service.py` — 35 async methods, 10 sync CRUD methods
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference and full method table
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvement roadmap
- `tests/` — Unit, integration, and composition test suites
