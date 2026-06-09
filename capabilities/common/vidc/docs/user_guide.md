# Video Conferencing

**Capability ID**: `vidc` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`vidc` provides APG's common capability for tenant-scoped video meetings. It composes meeting rooms, accountable hosts, waiting-room controls, participants, encrypted recordings, caption artifacts, AI meeting agents, first-class provider-neutral video agents, audit events, UI routes, visual theming, and Bytewax lifecycle guardrails into a generated-application packet that runs without live media infrastructure.

## Installation

```bash
pip install apg-common-vidc
```

## Provides

_(see capability contract)_

## Requires

_(none)_

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/vidc/dashboard` | `vidc:view` | Overview |
| `/vidc/meetings` | `vidc:schedule` | Meetings |
| `/vidc/rooms` | `vidc:moderate` | Meetings |
| `/vidc/participants` | `vidc:moderate` | Meetings |
| `/vidc/recordings` | `vidc:manage_recordings` | Artifacts |
| `/vidc/captions` | `vidc:view` | Artifacts |
| `/vidc/agents` | `vidc:moderate` | Meetings |
| `/vidc/lifecycle` | `vidc:admin` | Operations |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_room()`
- `start_meeting()`
- `add_participant()`
- `create_recording()`
- `generate_captions()`
- `register_meeting_agent()`
- `register_video_agent()`
- `validate_vidc_lifecycle_batch()`

_(See `service.py` for complete API.)_

## Interoperability

`vidc` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use vidc;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `VIDC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
