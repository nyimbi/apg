# Healthcare Regulatory

**Capability ID**: `healthcare_reg` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Regulatory compliance management covering facility and professional licensing with expiry tracking, accreditation management (Joint Commission, DNV, CAP, etc.), incident reporting with sentinel event workflow enforcement, regulatory submission management (CMS IQR/OQR, HIPAA breach, FDA MDR), and corrective action tracking. Sentinel event closure requires a completed root cause analysis reference.

## Installation

```bash
pip install apg-healthcare-reg
```

## Provides

- `facility_licensing_management`
- `accreditation_management`
- `incident_reporting`
- `hipaa_compliance_tracking`
- `regulatory_submission_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-reg/dashboard` | `healthcare_reg:view` | Overview |
| `/healthcare-reg/licenses` | `healthcare_reg:licenses` | Licensing |
| `/healthcare-reg/licenses/<id>` | `healthcare_reg:licenses` | Licensing |
| `/healthcare-reg/accreditation` | `healthcare_reg:accreditation` | Accreditation |
| `/healthcare-reg/incidents` | `healthcare_reg:incidents` | Incidents |
| `/healthcare-reg/incidents/new` | `healthcare_reg:incidents_write` | Incidents |
| `/healthcare-reg/incidents/<id>` | `healthcare_reg:incidents` | Incidents |
| `/healthcare-reg/submissions` | `healthcare_reg:submissions` | Submissions |

## Key Service Methods

- `describe()`
- `add_license()`
- `facility_licence_apply()`
- `licence_renewal()`
- `get_license()`
- `list_licenses()`
- `get_expiring_licenses()`
- `add_accreditation()`
- `accreditation_application()`
- `update_accreditation_status()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_reg` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_reg;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_REG_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
