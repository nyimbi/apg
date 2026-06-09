# Regulatory Compliance

**Capability ID**: `pharma_rec` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages pharmaceutical regulatory compliance obligations across multiple frameworks (FDA, EMA, GMP, ICH), including gap assessments, inspection readiness, label change management, post-market surveillance, regulatory intelligence dissemination, and regulatory commitment tracking. Enforces inspection response timelines, label QP approval, and overdue commitment escalation.

## Installation

```bash
pip install apg-pharma-rec
```

## Provides

- `regulatory_compliance_monitoring_workflow`
- `inspection_readiness_workflow`
- `label_management_workflow`
- `post_market_surveillance_workflow`
- `regulatory_intelligence_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-rec/dashboard` | `pharma_rec:view` | Overview |
| `/pharma-rec/compliance` | `pharma_rec:compliance` | Compliance |
| `/pharma-rec/compliance/gap` | `pharma_rec:gap_assessment` | Compliance |
| `/pharma-rec/inspections` | `pharma_rec:inspections` | Inspections |
| `/pharma-rec/inspections/<id>` | `pharma_rec:inspections` | Inspections |
| `/pharma-rec/labeling` | `pharma_rec:labeling` | Labeling |
| `/pharma-rec/pms` | `pharma_rec:pms` | Post-Market |
| `/pharma-rec/intelligence` | `pharma_rec:intel` | Intelligence |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_compliance()`
- `list_frameworks()`
- `create_gap_assessment()`
- `close_gap_assessment()`
- `list_gap_assessments()`
- `record_inspection()`
- `record_inspection_outcome()`
- `respond_to_inspection()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_rec` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_rec;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_REC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
