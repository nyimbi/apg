# AUDP Capability Specification

## Identity

- Capability ID: `audp`
- Display name: Audio Processing
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `audp_audio_intelligence`

## Purpose

AUDP is the tenant-scoped audio intelligence capability for APG applications.
It governs audio recording consent, transcription, transcript review, voice
synthesis, synthetic-audio release review, voice cloning consent, audio
analysis, model policy enforcement, retention posture, synthetic-audio
watermark evidence, and scoped AI audio-agent participation.

The package must remain usable without live speech-to-text providers, text-to-
speech providers, GPU workers, object storage, streaming infrastructure, or
web servers. Those systems remain adapter boundaries. Local package proof
focuses on deterministic audio governance, lifecycle state, tenant isolation,
and composition behavior.

## Users And Outcomes

- Application builders can declare audio transcription, synthesis, and analysis
  workflows as first-class APG components.
- Meeting, call-center, education, and field-service applications can require
  recording consent before processing captured speech.
- Compliance teams can prove voice-owner consent before voice cloning or custom
  voice-model use.
- Reviewers can approve low-confidence transcripts before downstream workflow
  automation consumes them.
- Safety teams can require synthetic-audio watermarks, explicit release review,
  and model policies before generated speech leaves the system.
- Generated APG applications can compose AUDP with AICR, NLPC, MLCM, AUDL,
  AUTH, NTFY, COLB, CACH, and WFLO without binding to one provider.

## Domain Model

AUDP owns these package-level records:

- `AudioConsentRecord`: recording or voice-owner consent evidence.
- `AudioProcessingJobRecord`: governed transcription, synthesis, analysis, or
  enhancement job state.
- `AudioTranscriptReviewRecord`: human review for low-confidence transcripts.
- `AudioSynthesisReviewRecord`: release-review evidence for synthetic audio
  output.
- `AudioModelPolicyRecord`: policy attachment for model-backed audio work.
- `AudioAgentRecord`: AI audio-agent registration, runtime, role, scope,
  disclosure, and policy reference.
- `AudioGovernanceEvent`: tenant-scoped evidence event for AUDP decisions.

All mutable package-level state must be tenant-qualified so duplicate IDs in
different tenants cannot collide.

## Lifecycle

The focused lifecycle is:

1. Register recording consent for an audio source or participant.
2. Register voice-owner consent before voice cloning.
3. Attach model policy for model-backed processing.
4. Request transcription against consented audio.
5. Create transcript-review state when confidence is below the configured
   threshold.
6. Approve or reject transcript review with reviewer notes.
7. Request voice synthesis only when watermarking and model policy are present.
8. Create synthetic-audio release-review state and require explicit reviewer
   approval or rejection before completion.
9. Request voice cloning only when voice-owner consent is present.
10. Request audio analysis with consent, model policy, and retention metadata.
11. Register AI audio agents with supported runtime, role, scope, disclosure,
    and policy evidence.
12. Change job state only with reason and audit evidence.
13. Emit tenant-scoped governance events for consent, model policy, job,
    review, synthesis, cloning, analysis, agent, and state-change lifecycle
    changes.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: operations require tenant context.
- `recording_consent_required`: recording processing requires consent.
- `voice_cloning_requires_consent`: voice cloning requires voice-owner consent.
- `synthetic_audio_requires_watermark`: synthetic audio requires watermarking.
- `synthetic_audio_requires_release_review`: synthetic audio requires explicit
  release review before completion.
- `audio_model_requires_policy`: model-backed processing requires policy.
- `low_transcription_confidence_requires_review`: low-confidence transcripts
  require human review.
- `audio_retention_policy_required`: audio jobs require retention policy
  evidence.
- `audio_agent_*`: AI audio agents require registration, supported runtime,
  explicit scope, and contribution disclosure.
- `audio_state_change_*`: state changes require reason and audit evidence.
- `cross_tenant_audio_access_denied`: tenant boundaries must not be crossed.
- `batch_audio_mutation_requires_bytewax`: batch mutations require Bytewax
  event streams.

Service methods must enforce these rules and expose the same decisions through
API helpers and view models.

## UI And Theme

AUDP exposes route and view-model surfaces for:

- dashboard summary;
- transcription console;
- synthesis studio;
- analysis workbench;
- audio sessions;
- model policy registry;
- consent center;
- review queue;
- quality/governance center;
- AI audio-agent panel;
- audit timeline;
- analytics summary;
- settings.

The `audp_audio_intelligence` theme must provide semantic tokens and component
metadata for waveform viewers, transcript panels, synthesis studios, analysis
grids, consent banners, transcript-review queues, and synthetic-watermark
status chips, audio-agent panels, and audit timelines.

## Adapter Boundaries

These integrations remain replaceable:

- speech-to-text providers such as Whisper, Deepgram, Google, Azure, and custom
  models;
- text-to-speech providers and voice-cloning engines;
- GPU/audio worker pools, media stores, and transcoding pipelines;
- Bytewax stream processors and low-latency audio streaming adapters;
- speaker diarization, sentiment, topic, and acoustic-analysis providers;
- audit, notification, collaboration, workflow, and model-lifecycle services.

Local package tests must not require those systems.

## Acceptance Gates

Focused AUDP proof:

```bash
./.venv/bin/pytest -q capabilities/common/audp/test_capability_contract.py capabilities/common/audp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/audp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/audp --json
git diff --check -- capabilities/common/audp
```
