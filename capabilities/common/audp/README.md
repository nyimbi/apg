# AUDP Audio Processing Capability

AUDP is APG's governed audio-processing capability. It gives generated
applications a dependency-light way to compose transcription, synthesis,
voice-cloning consent, audio analysis, model policy, human review,
AI audio-agent coordination, tenant isolation, visual theming, and Bytewax
lifecycle events.

The local package is intentionally executable without live speech providers,
GPU workers, media stores, web servers, or stream processors. Production audio
engines attach through adapters declared by the capability contract.

## What AUDP Provides

- Tenant-scoped recording consent for captured audio.
- Tenant-scoped voice-owner consent for voice cloning and custom voices.
- Audio model-policy attachments for transcription, synthesis, analysis, and
  voice-cloning work.
- Transcription jobs with confidence thresholds and review requirements.
- Synthetic-audio jobs with watermark and release-review requirements.
- Voice-cloning jobs blocked until voice-owner consent exists.
- Audio-analysis jobs with consent, model-policy, and retention evidence.
- First-class AI audio agents for Codex, Claude Code, OpenCode, Pi, and future
  runtimes.
- Governance event evidence for consent, model policy, job, review, agent, and
  state-change decisions.
- Framework-neutral API helpers and UI view models.
- Visual theme metadata and Bytewax lifecycle stream metadata.

## Main Files

- `SPECIFICATION.md`: functional requirements, lifecycle, rules, UI, and
  adapter boundaries.
- `PLAN.md`: implementation sequencing and review checklist.
- `capability_contract.py`: executable configuration, rules, UI routes, theme,
  supported audio-agent runtimes, and Bytewax stream metadata.
- `models.py`: package records for consent, policy, jobs, reviews, agents, and
  governance events.
- `audio_runtime.py`: dependency-light `AudpService` lifecycle facade.
- `api_helpers.py`: callable helpers for generated APG applications.
- `view_models.py`: dashboard, console, review, agent, audit, analytics, and
  settings models.

## Basic Usage

```python
from capabilities.common.audp.audio_runtime import AudpService

service = AudpService()
service.record_consent(
	"consent-call-1",
	"tenant-a",
	"recording",
	"call-1",
	"participant-1",
	"signed-consent://call-1",
)
service.attach_model_policy(
	"policy-audio",
	"tenant-a",
	"audio-model",
	"Approved audio model policy",
	["transcription", "synthesis", "analysis"],
	"governor",
)
job = service.request_transcription(
	"transcribe-call-1",
	"tenant-a",
	"call-1",
	"operator",
	"audio-model",
	confidence=0.5,
)
```

Low-confidence transcripts produce a pending review. Synthetic audio requires
watermark evidence and release review. Voice cloning requires active voice-owner
consent.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/audp/__init__.py capabilities/common/audp/capability_contract.py capabilities/common/audp/models.py capabilities/common/audp/audio_runtime.py capabilities/common/audp/api_helpers.py capabilities/common/audp/view_models.py capabilities/common/audp/test_capability_contract.py capabilities/common/audp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/audp/test_capability_contract.py capabilities/common/audp/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.audp import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/audp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/audp --json
```
