# Audio Processing

**Capability ID**: `audp` | **Domain**: `common` | **Version**: `1.0.0`

## Description

AUDP is APG's governed audio-processing capability. It gives generated applications a dependency-light way to compose transcription, synthesis, voice-cloning consent, audio analysis, model policy, human review,

## Installation

```bash
pip install apg-common-audp
```

## Provides

- `audio_transcription`
- `voice_synthesis`
- `audio_analysis`
- `speaker_diarization`
- `audio_enhancement`

## Requires

- `aicr`
- `nlpc`
- `mlcm`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/audp/dashboard` | `audp:view` | Overview |
| `/audp/transcription` | `audp:transcribe` | Processing |
| `/audp/synthesis` | `audp:synthesize` | Processing |
| `/audp/analysis` | `audp:analyze` | Analysis |
| `/audp/sessions` | `audp:view` | Runtime |
| `/audp/models` | `audp:manage_models` | Models |
| `/audp/consents` | `audp:govern` | Governance |
| `/audp/reviews` | `audp:review` | Governance |

## Key Service Methods

- `create_transcription_job()`
- `start_transcription_job()`
- `_process_transcription_job()`
- `_get_transcription_model()`
- `_transcribe_with_whisper()`
- `_transcribe_with_google()`
- `_transcribe_with_azure()`
- `_transcribe_with_deepgram()`
- `_transcribe_with_custom_model()`
- `_process_transcription_results()`

_(See `service.py` for complete API.)_

## Interoperability

`audp` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use audp;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `AUDP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
