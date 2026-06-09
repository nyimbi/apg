# Telemedicine

**Capability ID**: `healthcare_tel` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Full-featured telemedicine capability covering virtual consultation booking, video session management with consent and E-911 disclosure enforcement, remote patient monitoring enrollment, electronic prescription transmission, and telehealth-specific billing code management. Schedule II/III prescription transmission is blocked without a prior in-person visit.

## Installation

```bash
pip install apg-healthcare-tel
```

## Provides

- `virtual_consultation_booking`
- `video_session_management`
- `remote_patient_monitoring`
- `prescription_transmission`
- `telehealth_billing`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-tel/dashboard` | `healthcare_tel:view` | Overview |
| `/healthcare-tel/schedule` | `healthcare_tel:schedule` | Scheduling |
| `/healthcare-tel/schedule/new` | `healthcare_tel:schedule_write` | Scheduling |
| `/healthcare-tel/schedule/<id>` | `healthcare_tel:schedule` | Scheduling |
| `/healthcare-tel/sessions` | `healthcare_tel:sessions` | Sessions |
| `/healthcare-tel/sessions/<id>/room` | `healthcare_tel:sessions` | Sessions |
| `/healthcare-tel/monitoring` | `healthcare_tel:monitoring` | Monitoring |
| `/healthcare-tel/monitoring/<patient_id>` | `healthcare_tel:monitoring` | Monitoring |

## Key Service Methods

- `describe()`
- `book_consultation()`
- `book_teleconsult()`
- `cancel_consultation()`
- `get_consultation()`
- `list_consultations()`
- `create_session()`
- `video_session_start()`
- `video_session_end()`
- `complete_session()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_tel` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_tel;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_TEL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
