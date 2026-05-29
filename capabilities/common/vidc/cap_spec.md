# Video Conferencing Capability Specification

- **Capability Name**: Video Conferencing
- **Capability ID**: `vidc`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `vidc`.
It gives composed applications a deterministic video collaboration surface for
meeting room creation, accountable hosts, waiting rooms, external guest policy,
participant tracking, recording consent, encrypted recordings, retention
evidence, caption artifacts, large meeting review, UI route metadata,
semantic-model publication, and publish-plan
evidence.

## Provided Services

- `meeting_rooms`
- `video_meetings`
- `participant_tracking`
- `recording_governance`
- `caption_artifacts`
- `meeting_audit_events`

## Required Services

- `tenant_context`
- `collaboration_bus`
- `event_messaging`
- `computer_vision_services`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `meeting_requires_host`
- `external_guest_requires_policy`
- `recording_requires_consent`
- `recording_requires_encryption`
- `large_meeting_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, meeting console, room manager, participant
panel, recording library, caption workbench, analytics, and settings models.

## Theme

The package uses the `vidc_meeting_room` APG theme contract.

## Runtime Behavior

`VidcService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `create_room()` for tenant-scoped meeting rooms with owner, guest policy,
  moderation policy, and waiting-room metadata.
- `start_meeting()` for accountable host, participant count, external guest,
  recording consent, encryption, and capacity review guardrails.
- `add_participant()` for host, cohost, participant, guest, and observer
  membership tracking.
- `create_recording()` for encrypted recording records with consent and
  retention policy evidence.
- `generate_captions()` for language-specific transcript and caption artifacts.
- `end_meeting()` for lifecycle closure and audit evidence.
- `dashboard_summary()` and list helpers for API and UI composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind realtime media transports, collaboration buses, event
messaging, recording object stores, captioning engines, computer-vision assist,
retention enforcement, and audit sinks at the APG composition layer without
changing the deterministic package contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/vidc/__init__.py capabilities/common/vidc/models.py capabilities/common/vidc/video_runtime.py capabilities/common/vidc/service.py capabilities/common/vidc/api.py capabilities/common/vidc/views.py capabilities/common/vidc/capability_contract.py capabilities/common/vidc/app.py capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/vidc/test_capability_contract.py capabilities/common/vidc/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/vidc --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/vidc --json`
