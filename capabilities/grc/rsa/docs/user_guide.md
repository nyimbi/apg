# Risk and Security Assessment

**Capability ID**: `grc_rsa` | **Domain**: `grc` | **Version**: `1.0.0`

## Description

Risk and Security Assessment provides a world-class, standalone-deployable implementation of risk and security assessment capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-grc-rsa
```

## Provides

- `security_assessment_lifecycle`
- `vulnerability_finding_workflow`
- `remediation_tracking_workflow`
- `vendor_risk_assessment_workflow`
- `threat_modelling_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/grc-rsa/dashboard` | `grc_rsa:view` | Overview |
| `/grc-rsa/assessments` | `grc_rsa:manage_assessments` | Assessments |
| `/grc-rsa/assessments/:id` | `grc_rsa:view` | Assessments |
| `/grc-rsa/findings` | `grc_rsa:manage_findings` | Findings |
| `/grc-rsa/findings/:id` | `grc_rsa:view` | Findings |
| `/grc-rsa/remediation` | `grc_rsa:manage_remediation` | Remediation |
| `/grc-rsa/vendor-risk` | `grc_rsa:manage_vendor_risk` | Vendor Risk |
| `/grc-rsa/threat-model` | `grc_rsa:view` | Threat Intelligence |

## Key Service Methods

- `_audit_event()`
- `_get_risk()`
- `risk_register_entry()`
- `risk_assessment()`
- `inherent_risk_score()`
- `residual_risk_score()`
- `update_residual_score()`
- `risk_heat_map()`
- `control_assessment()`
- `control_gap()`

_(See `service.py` for complete API.)_

## Interoperability

`grc_rsa` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use grc_rsa;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GRC_RSA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
