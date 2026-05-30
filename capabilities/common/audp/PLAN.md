# AUDP Capability Development Plan

## Current State

AUDP has production-oriented async transcription, synthesis, analysis, model,
workflow, FastAPI, and Pydantic view-model surfaces. It also has a capability
contract, package evidence, and contract tests. The package-level composition
gap is a dependency-light lifecycle facade that can execute audio consent,
model-policy, review, synthesis, analysis, AI audio-agent, and state-change
governance without live audio providers or web servers.

## Packet 1: Governed Audio Processing Lifecycle

Deliver a focused lifecycle packet:

- add package-level consent, model-policy, processing-job, transcript-review,
  synthesis-review, audio-agent, and governance-event records;
- add a dependency-light `AudpService` runtime facade;
- register recording and voice-owner consent;
- attach tenant-scoped audio model policy;
- request transcription with consent, model-policy, and confidence review;
- decide transcript reviews with reviewer evidence;
- request voice synthesis only with watermark and model policy;
- require synthetic-audio release review before completion;
- request voice cloning only with voice-owner consent;
- request audio analysis with consent, model-policy, and retention metadata;
- register scoped AI audio agents with runtime, role, policy, and disclosure;
- require reason and audit evidence for job state changes;
- declare Bytewax lifecycle stream metadata and batch-mutation guardrails;
- expose API-helper and view-model surfaces for dashboard, transcription,
  synthesis, analysis, sessions, models, consent, reviews, quality, and audit
  evidence;
- replace stale generated-package test naming with package contract tests;
- replace stale embedded semantic evidence with contract-derived package
  evidence;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AudioConsentRecord`, `AudioModelPolicyRecord`,
   `AudioProcessingJobRecord`, `AudioTranscriptReviewRecord`,
   `AudioSynthesisReviewRecord`, and `AudioGovernanceEvent`.
2. Add `audio_runtime.py` with the dependency-light `AudpService` facade.
3. Add `api_helpers.py` for generated APG applications.
4. Add `view_models.py` for dependency-light AUDP UI model surfaces.
5. Update contract metadata with consent, review, agent, audit, analytics, and
   settings UI routes plus theme components.
6. Update capability registration metadata with consent/review/model-policy
   services and endpoints.
7. Replace stale package semantic evidence with contract-derived evidence.
8. Extend package tests with positive transcription-review-synthesis-analysis
   lifecycle coverage and negative consent, watermark, model-policy,
   voice-cloning consent, missing reviewer, tenant-mismatch, and duplicate-ID
   coverage.
9. Rename generated-package tests to package contract naming.
10. Update `cap_spec.md` with current executable lifecycle and proof commands.
11. Run focused package proof, implementation audit, publish-plan, and diff
    checks.

## Review Checklist

- Consent, model policy, job, transcript review, synthesis review, and
  governance state is tenant-qualified.
- Recording processing requires recording consent.
- Voice cloning requires voice-owner consent.
- Synthetic audio requires watermark evidence.
- Synthetic audio requires explicit release review before completion.
- Model-backed audio processing requires attached model policy.
- Low-confidence transcripts require review before activation.
- Review decisions require reviewer identity and notes.
- AI audio agents require supported runtime, role, scope, and disclosure.
- Batch audio mutations require Bytewax.
- API helpers expose the same behavior as service methods.
- View models expose dashboard, transcription, synthesis, analysis, session,
  model-policy, consent, review, quality, theme, and governance-event state.
- Live providers, GPU workers, media stores, stream processors, and UI servers
  remain adapter boundaries.
